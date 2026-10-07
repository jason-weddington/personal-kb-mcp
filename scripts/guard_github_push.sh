#!/usr/bin/env bash
# Pre-push guard: the public `github` remote only receives release pushes.
#
# Users run this package by uvx-ing it straight from GitHub, so a push there
# is a deploy. Day-to-day work goes to `origin` only; ./release.sh is the one
# path to github, and it sets KB_RELEASE_PUSH=1 for exactly that push.
# pre-commit exports the remote name as PRE_COMMIT_REMOTE_NAME; a direct
# hook invocation passes it as $1.
remote="${PRE_COMMIT_REMOTE_NAME:-${1:-}}"
if [ "$remote" = "github" ] && [ "${KB_RELEASE_PUSH:-}" != "1" ]; then
  echo "Refusing to push to 'github' outside a release: github only receives ./release.sh pushes. Push day-to-day work to origin." >&2
  exit 1
fi
exit 0
