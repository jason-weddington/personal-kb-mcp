#!/usr/bin/env bash
#
# Check-only replacement for the `trailing-whitespace` and `end-of-file-fixer`
# pre-commit hooks. It REPORTS and never REWRITES.
#
# Why this exists (kb-03099): a fixer hook mutates the tree at commit time and
# exits non-zero. Claude Code's commit-retry loop absorbs that, but the talos
# dispatch worker commits once, so a mutating hook destroys a completed,
# gate-green unit of work and the run returns `failed` before pushing. Upstream
# pre-commit-hooks ships no check-only variant of either hook, hence this file.
#
# Fixing is the agent's job, not the hook's. On failure this prints the offending
# file:line and the one-liner that repairs it.
set -uo pipefail

status=0
trailing_hits=""
eof_hits=""

for file in "$@"; do
    # Skip anything that vanished (a rename or delete staged in the same commit)
    # and anything that is not a regular file.
    [ -f "$file" ] || continue

    # Binary files have no meaningful trailing whitespace or final newline.
    if ! grep -Iq . "$file" 2>/dev/null; then
        continue
    fi

    # [[:blank:]] is POSIX for exactly {space, tab}. Do NOT write [ \t] here:
    # GNU grep treats a bracket expression literally, so [ \t] is
    # {space, backslash, t} and matches every line ending in the letter "t".
    # (This shipped as a bug for ten minutes. It was invisible in ad-hoc
    # testing because `grep` in an interactive Claude Code session is a shell
    # function routing to ugrep, which DOES read \t as a tab — so the
    # interactive test and the script disagreed.)
    if hits=$(grep -nE '[[:blank:]]+$' "$file" 2>/dev/null); then
        while IFS= read -r line; do
            trailing_hits+="  ${file}:${line%%:*}"$'\n'
        done <<<"$hits"
        status=1
    fi

    # An empty file is fine; a non-empty one must end in exactly one newline.
    if [ -s "$file" ] && [ "$(tail -c 1 "$file" | wc -l)" -eq 0 ]; then
        eof_hits+="  ${file}"$'\n'
        status=1
    fi
done

if [ -n "$trailing_hits" ]; then
    printf 'Trailing whitespace (this hook does not fix it):\n%s' "$trailing_hits"
    printf "  fix: sed -i 's/[[:blank:]]*\$//' <file>\n\n"
fi

if [ -n "$eof_hits" ]; then
    printf 'Missing final newline (this hook does not fix it):\n%s' "$eof_hits"
    printf "  fix: printf '\\\\n' >> <file>\n\n"
fi

exit "$status"
