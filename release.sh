#!/usr/bin/env bash
# release.sh — cut a version and publish to BOTH remotes.
#
# GitHub `main` is PRODUCTION: the 60-person team's MCP config uvx's the
# package straight from github, so nothing reaches github except through this
# script, at a deliberate, vetted release boundary.
#
# Day-to-day development squash-merges to local `main` and pushes to `origin`
# (the home-lab VM) freely for testing. This script is the separate
# promote-to-github step — run it only when local `main` is verified good.
set -euo pipefail

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
#    --no-push:        we push explicitly to both remotes below.
#    --no-vcs-release:  don't create a GitHub Release object (tags are enough).
uv run semantic-release version --no-push --no-vcs-release

# 3. Publish to BOTH remotes: home-lab origin first, then the team-facing github.
git push origin main --tags
git push github main --tags

# 4. Optional local deploy (restart the home-lab service) if a deploy script exists.
if [ -x ./deploy.sh ]; then
  ./deploy.sh
fi

echo "Released $(git describe --tags --abbrev=0) to origin + github."
