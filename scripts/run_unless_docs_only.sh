#!/usr/bin/env bash
# Usage: scripts/run_unless_docs_only.sh <cmd> [args...]
#
# Pre-push wrapper: skips <cmd> when the pushed range touches only docs
# (docs/**, proposals/**, top-level *.md). Fail-safe: any uncertainty, an empty
# file list, or any other file means the command runs.
set -u

docs_only=0
from="${PRE_COMMIT_FROM_REF:-}"
to="${PRE_COMMIT_TO_REF:-}"

if [ -n "$to" ]; then
    range_ok=1
    case "$from" in
        '' | *[!0]*) ;;
        *) from='' ;; # all zeros: new branch
    esac
    if [ -z "$from" ]; then
        from=$(git merge-base origin/main "$to" 2>/dev/null) || range_ok=0
    fi
    if [ "$range_ok" = 1 ] && [ -n "$from" ]; then
        if files=$(git diff --name-only "$from" "$to" 2>/dev/null) && [ -n "$files" ]; then
            if ! printf '%s\n' "$files" | grep -Evq '^(docs/|proposals/|[^/]*\.md$)'; then
                docs_only=1
            fi
        fi
    fi
fi

if [ "$docs_only" = 1 ]; then
    echo "pre-push: docs-only push — skipping $(basename "$1")"
    exit 0
fi
exec "$@"
