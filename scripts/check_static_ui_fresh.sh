#!/usr/bin/env bash
# Fail unless the packaged UI's .source-hash matches the current frontend source.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
hash_file="${STATIC_UI_HASH_FILE:-$root/packages/kb-service/src/kb_service/static/.source-hash}"
current="$("$root/scripts/_static_ui_hash.sh")"
recorded="$(cat "$hash_file" 2>/dev/null || true)"
if [ "$current" != "$recorded" ]; then
  echo "Packaged UI is stale: recorded ${recorded:-<none>}, source is $current." >&2
  echo "Run scripts/build_static_ui.sh and commit." >&2
  exit 1
fi
echo "Packaged UI is fresh ($current)."
