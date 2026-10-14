#!/usr/bin/env bash
# Pre-push: run the maintainer-local leak guard before a commit leaves this
# machine, not only at release time.
#
# ./release.guard.local.sh is gitignored and optional (see release.sh). It
# scans the tree and every commit the github push would eventually publish.
# Run at release time only, a leak merged mid-week surfaces once the commit
# is already shared history on origin, where removing it means a force-push.
# Running it on every push to origin catches it while the commit can still be
# amended locally.
#
# It runs in --tree-only mode: no github remote or network needed, about a
# second, and it catches the merge-time case (a private name in a tracked
# file). History-only cases (a name added and removed between releases, or one
# in a commit message) are still caught by the full scan release.sh runs.
#
# Absent or non-executable guard: skip (dispatch clones and other users have
# none). A push to github is skipped too: release.sh runs the guard itself
# before that push. Any other non-zero exit blocks the push.
# pre-commit exports the remote name as PRE_COMMIT_REMOTE_NAME; a direct hook
# invocation passes it as $1.
remote="${PRE_COMMIT_REMOTE_NAME:-${1:-}}"
[ "$remote" = "github" ] && exit 0
[ -x ./release.guard.local.sh ] || exit 0
./release.guard.local.sh --tree-only
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "Push blocked by the leak guard (exit $rc). Fix the hits in the commits being pushed (amend them before they reach origin), or have the guard's owner allow-list a genuine false positive." >&2
fi
exit "$rc"
