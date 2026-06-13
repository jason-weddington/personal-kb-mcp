#!/usr/bin/env bash
# Per-machine setup for personal-kb tooling. Idempotent — safe to re-run.
#
#   1. Symlinks the map-building workflows into ~/.claude/workflows/ so they
#      resolve as /recommend-maps and /author-chunky-entries on this machine.
#   2. Installs the personal-kb-hook (uv tool) from the canonical git remote.
#   3. Prompts for the KB API key and writes it to a machine-local file.
#
# WHY a key FILE and not the hook's settings.json env block: ~/.claude is a git
# repo synced to vm01, so a key inlined in settings.json would be committed to
# git AND shared across every machine (defeating per-machine key revocation).
# The hook command reads this local file via $(cat ...) instead. The key never
# enters synced config. (The hook's settings.json *wiring* DOES live in the
# synced ~/.claude — pull that repo on a new machine; this script handles only
# the machine-local pieces.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF_DIR="$HOME/.claude/workflows"
KEY_FILE="$HOME/.personal_kb_hook_key"
KB_URL="${PERSONAL_KB_URL:-http://kb-host-1:8000}"
HOOK_SPEC="personal-kb-hook @ git+ssh://git@git-host/home/git/repos/personal_kb#subdirectory=packages/personal-kb-hook"

echo "== workflows -> $WF_DIR =="
mkdir -p "$WF_DIR"
for wf in "$SCRIPT_DIR"/*.workflow.js; do
  [ -e "$wf" ] || continue
  ln -sfn "$wf" "$WF_DIR/$(basename "$wf")"
  echo "  /$(basename "$wf" .workflow.js)  ->  $wf"
done

echo "== personal-kb-hook =="
if command -v uv >/dev/null 2>&1; then
  uv tool install --force "$HOOK_SPEC" 2>&1 | tail -1
else
  echo "  uv not found. Install it (https://astral.sh/uv), then re-run this script." >&2
fi

echo "== KB API key ($KEY_FILE) =="
if [ -s "$KEY_FILE" ]; then
  echo "  present — leaving as-is (delete the file and re-run to replace it)."
else
  echo "  The hook authenticates to the KB service with a per-machine API key."
  echo "  Get one: open  $KB_URL  ->  Settings  ->  API Keys  ->  create (shown once)."
  printf "  Paste API key (hidden; leave blank to skip): "
  read -rs KEY; echo
  if [ -n "${KEY:-}" ]; then
    printf '%s' "$KEY" > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
    echo "  wrote $KEY_FILE (chmod 600)."
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 \
      -X POST "$KB_URL/api/kb/search" \
      -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
      -d '{"query":"setup probe","limit":1}' 2>/dev/null || true)
    case "$code" in
      200) echo "  verified against $KB_URL (200 OK)." ;;
      401) echo "  WARNING: $KB_URL rejected the key (401). Re-check the paste." >&2 ;;
      *)   echo "  note: could not verify against $KB_URL (got '$code'); key written anyway." ;;
    esac
  else
    echo "  skipped. The hook stays in local/degraded mode until a key is set."
  fi
fi

echo
echo "Done. Reminder: the hook's settings.json wiring lives in the synced ~/.claude"
echo "repo — pull it on a new machine. See KB kb-01784 for the full runbook."
