#!/usr/bin/env bash
# Print a deterministic hash of the kb-service frontend source inputs.
# Sorted file list, content-only (no mtimes): src/, index.html,
# package-lock.json and the vite/ts configs.
set -euo pipefail
cd "${STATIC_UI_FRONTEND_DIR:-$(dirname "$0")/../packages/kb-service/frontend}"
export LC_ALL=C
{
  find src -type f
  ls index.html package-lock.json vite.config.ts tsconfig*.json
} | sort | while IFS= read -r f; do
  printf '%s  %s\n' "$(sha256sum < "$f" | cut -d' ' -f1)" "$f"
done | sha256sum | cut -d' ' -f1
