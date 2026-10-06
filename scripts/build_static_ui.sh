#!/usr/bin/env bash
# Build the kb-service web UI and install it as the packaged static dir.
# Run at release time (release.sh calls it); work users have no Node.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
fe="$root/packages/kb-service/frontend"
static="$root/packages/kb-service/src/kb_service/static"

(cd "$fe" && npm ci && npm run build)

rm -rf "$static"
mkdir -p "$static"
cp -R "$fe/dist/." "$static/"

# Repo whitespace gate wants a final newline on text files; minified output
# often lacks one. Appending a newline is harmless to HTML/JS/CSS.
while IFS= read -r -d '' f; do
  if grep -Iq . "$f" && [ "$(tail -c 1 "$f" | wc -l)" -eq 0 ]; then
    printf '\n' >> "$f"
  fi
done < <(find "$static" -type f -print0)

"$root/scripts/_static_ui_hash.sh" > "$static/.source-hash"
echo "Built UI into $static (source hash $(cat "$static/.source-hash"))"
