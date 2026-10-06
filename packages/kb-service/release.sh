#!/usr/bin/env bash
# release.sh — cut a version and roll it out to every KB instance.
#
# This repo is a PRIVATE home-lab service, not a published package: its only
# remote is `origin` (the home-lab VM). So unlike personal_kb — where GitHub
# main is production for the team and release.sh is the promote-to-GitHub
# step — releasing here means "stamp a version, tag it, and put that exact
# commit on all three Pis". There is no public history to protect.
#
# Day-to-day development squash-merges to main, pushes to origin, and runs
# ./deploy.sh per host freely. Run this at a meaningful boundary, when you
# want the version shown in the app's settings page to move.
set -euo pipefail

# Every KB instance. deploy.sh handles ONE host via KB_DEPLOY_HOST, so the
# release loops: a release that leaves an instance behind is worse than no
# release, because the settings page then reports a version some hosts do
# not run.
HOSTS=(kb-host-1 kb-host-2 kb-host-3)

# 1. Preconditions: on main, clean working tree.
branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "main" ]; then
  echo "release.sh must run on main (currently on '$branch')." >&2
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is not clean — commit or stash before releasing." >&2
  exit 1
fi

# 2. Bump version + CHANGELOG + uv.lock and tag locally.
#    --no-push:        we push explicitly below.
#    --no-vcs-release: no GitHub Release object — there is no GitHub remote.
#    NOTE: `push` is not a valid python-semantic-release v9+ config key; it is
#    silently ignored, which is why suppression has to be a CLI flag.
uv run semantic-release version --no-push --no-vcs-release

version=$(uv run python -c 'import tomllib,pathlib;print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])')
echo "[release] version is now ${version}"

# 3. Publish to the one remote we have.
git push origin main --tags

# 4. Roll it out everywhere. Each deploy.sh run gates on /api/health and exits
#    non-zero if the instance does not come back, so `set -e` stops the rollout
#    at the first unhealthy host rather than marching on and leaving a mixed
#    fleet behind a green summary line.
for host in "${HOSTS[@]}"; do
  echo "[release] deploying ${version} to ${host}"
  KB_DEPLOY_HOST="$host" ./deploy.sh
done

echo "[release] ${version} released and live on: ${HOSTS[*]}"
