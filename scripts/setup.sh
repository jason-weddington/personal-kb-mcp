#!/usr/bin/env bash
# Per-machine setup for personal-kb tooling. Idempotent — safe to re-run.
#
#   1. Symlinks the map-building workflows into ~/.claude/workflows/ so they
#      resolve as /recommend-maps and /author-chunky-entries on this machine.
#   2. Installs the personal-kb-hook (uv tool) from the canonical git remote.
#   3. Prompts for a KB mode, then configures auth for that mode:
#        - Remote — hosted KB service: prompt for a per-machine API key and
#          write it to a machine-local file (steps below).
#        - Local  — auto-spawned no-auth daemon: no key prompt, no key file,
#          no remote probe; just guidance for the synced settings.json wiring.
#
# WHY a key FILE and not the hook's settings.json env block (REMOTE mode):
# ~/.claude is a git repo synced to vm01, so a key inlined in settings.json
# would be committed to git AND shared across every machine (defeating
# per-machine key revocation). The hook command reads this local file via
# $(cat ...) instead. The key never enters synced config.
#
# LOCAL mode has no per-machine secret: the loopback URL and a fixed
# non-secret SENTINEL key are identical on every machine, so they live in the
# synced settings.json wiring directly (this script only prints them).
#
# (The hook's settings.json *wiring* DOES live in the synced ~/.claude — pull
# that repo on a new machine; this script handles only the machine-local
# pieces.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF_DIR="$HOME/.claude/workflows"
KEY_FILE="$HOME/.personal_kb_hook_key"
HOOK_SPEC="personal-kb-hook @ git+ssh://git@git-host/home/git/repos/personal_kb#subdirectory=packages/personal-kb-hook"

# LOCAL-mode daemon endpoint. The port literal is the canonical local port
# documented by the daemon-spawn code (src/personal_kb/daemon.py, parse_port
# example) and locked by kb-01807. The MCP server's ensure_daemon() parses
# this port from PERSONAL_KB_URL and spawns `kb-service serve --port <port>`.
# The SENTINEL key is a fixed, non-secret constant: the hook gates BOTH HTTP
# paths on a NON-EMPTY PERSONAL_KB_API_KEY (http_index.py: `if not url or not
# key`; listener.py is_listener_enabled `bool(url) and bool(key)`), so a blank
# key would silently disable the hook. The no-auth daemon ignores the value.
LOCAL_KB_PORT="8765"
LOCAL_KB_URL="http://localhost:${LOCAL_KB_PORT}"
LOCAL_SENTINEL_KEY="local-no-auth"

echo "== workflows -> $WF_DIR =="
mkdir -p "$WF_DIR"
for wf in "$SCRIPT_DIR"/*.workflow.js; do
  [ -e "$wf" ] || continue
  # Force a readable mode on the source file before linking, so a restrictive
  # local umask (or prior manual chmod) on this machine can't leave the
  # symlinked workflow unreadable on the next one. Matches the file's
  # git-tracked mode (100644); idempotent.
  # Best-effort: a read-only or root-owned clone must not abort the whole
  # installer before the symlink and the hook install happen.
  chmod 644 "$wf" 2>/dev/null || true
  ln -sfn "$wf" "$WF_DIR/$(basename "$wf")"
  echo "  /$(basename "$wf" .workflow.js)  ->  $wf"
done

echo "== personal-kb-hook =="
if command -v uv >/dev/null 2>&1; then
  uv tool install --force "$HOOK_SPEC" 2>&1 | tail -1
else
  echo "  uv not found. Install it (https://astral.sh/uv), then re-run this script." >&2
fi

echo "== KB mode =="
echo "  Local  — auto-spawned no-auth daemon on this machine (no API key)."
echo "  Remote — hosted KB service (per-machine API key)."
printf "  Choose mode [Local/Remote] (default: Remote): "
read -r MODE || MODE=""
case "${MODE:-}" in
  [Ll] | [Ll]ocal) MODE="local" ;;
  *)               MODE="remote" ;;
esac

if [ "$MODE" = "local" ]; then
  echo "== KB mode: local (auto-spawned no-auth daemon) =="
  echo "  No API key needed. setup.sh writes no key file and runs no remote probe;"
  echo "  the MCP server auto-spawns 'kb-service serve --port $LOCAL_KB_PORT' on first use."
  echo "  Wire these into the hook env in your synced ~/.claude settings.json:"
  echo "    PERSONAL_KB_URL=$LOCAL_KB_URL"
  echo "    PERSONAL_KB_API_KEY=$LOCAL_SENTINEL_KEY   # fixed sentinel; non-secret, safe to sync"
  echo "  The sentinel is non-empty on purpose: a blank key disables the hook"
  echo "  (maps-index fetch + listener both gate on a non-empty key). The no-auth"
  echo "  daemon ignores the value."
  echo "  (No verify probe yet — the local daemon's runtime endpoint ships separately.)"
else
  KB_URL="${PERSONAL_KB_URL:-http://kb-host-1:8000}"
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
fi

echo
echo "Done. Reminder: the hook's settings.json wiring lives in the synced ~/.claude"
echo "repo — pull it on a new machine. Remote mode reads the per-machine key file"
echo "via \$(cat $KEY_FILE); local mode uses the fixed non-secret sentinel key above"
echo "(safe to keep in synced config). See KB kb-01784 for the full runbook."
