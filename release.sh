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

# 2a. Lock the standalone personal-kb-hook package to the same release version
#     as the main package, and fold that into the release commit so the single
#     repo tag covers BOTH packages.
#
#     Why this works as a post-hoc amend + retag:
#       - We just ran semantic-release with --no-push, so the new commit and
#         tag are local only — nothing has been published yet, so editing the
#         commit and moving the tag onto it is safe.
#       - The hook is consumed via `uv tool install --from
#         "git+...#subdirectory=packages/personal-kb-hook"` — pulled straight
#         out of the git tree at a tag. It is NOT a separately published
#         wheel, so semantic-release's build_command does not need to build
#         it. Only the version line that lands in the tagged commit matters.
#       - Decision: one repo version for both packages — simplest. Revisit
#         independent versioning only if a real need appears.
python3 scripts/stamp_hook_version.py
new_version=$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -1)
new_tag="v${new_version}"
if ! git diff --quiet -- packages/personal-kb-hook/pyproject.toml; then
  # The hook is a uv workspace member, so its version also appears in
  # uv.lock — re-lock so the lockfile entry tracks the new version, then
  # fold both files into the release commit.
  uv lock
  git add packages/personal-kb-hook/pyproject.toml uv.lock
  # Keep the conventional `chore(release): v{version}` message, then move
  # the annotated tag onto the new commit. The tag message mirrors what
  # semantic-release writes by default ("v{version}").
  git commit --amend --no-edit
  git tag -d "$new_tag"
  git tag -a "$new_tag" -m "$new_tag"
fi

# 3. Publish to BOTH remotes: home-lab origin first, then the team-facing github.
git push origin main --tags
git push github main --tags

# 4. Optional local deploy (restart the home-lab service) if a deploy script exists.
if [ -x ./deploy.sh ]; then
  ./deploy.sh
fi

echo "Released $(git describe --tags --abbrev=0) to origin + github."
