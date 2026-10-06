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
#
# Usage: ./release.sh [--no-publish]
#
# Optional maintainer-local publish hook
# --------------------------------------
# After the release commit and tag exist locally, and BEFORE anything is
# pushed, release.sh builds four wheels into ./dist (personal-kb, kb-core,
# personal-kb-web-service, personal-kb-hook, all at the release version) and,
# if an executable ./release.local.sh exists (gitignored, never committed),
# runs it from the repo root as:
#
#     ./release.local.sh <version> <absolute-dist-dir>
#
# Exit 0 means the artifacts are published (e.g. uploaded to a private
# package index); exit 10 means they are published but a downstream deploy
# did not finish (the release still pushes, then exits 3); any other exit
# aborts the release. Publishing happens before
# the push so a remote never advertises a version whose artifacts did not ship.
# On any abort the local release tag is deleted and nothing has been pushed.
# With no hook present the release aborts too, unless --no-publish is given,
# which skips publishing (loudly) and pushes anyway.
set -euo pipefail

publish=1
for arg in "$@"; do
  case "$arg" in
    --no-publish) publish=0 ;;
    *)
      echo "Unknown argument: $arg (usage: ./release.sh [--no-publish])" >&2
      exit 2
      ;;
  esac
done

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

# 1. (gate) Work-user upgrade smoke test. GitHub is production: ~60 people
#    uvx personal-kb from it against a local SQLite KB with no config. This
#    installs the build they run today, seeds a KB, upgrades to THIS commit
#    with the identical environment, and checks reads, row counts, writes,
#    the web UI and rollback. A failure here means the release would break
#    them, so it blocks the release. Skip only deliberately:
#    SKIP_UPGRADE_SMOKE=1 ./release.sh
if [ "${SKIP_UPGRADE_SMOKE:-}" = "1" ]; then
  echo "WARNING: SKIP_UPGRADE_SMOKE=1, so the work-user upgrade smoke test was NOT run." >&2
else
  uv run python scripts/smoke_work_user_upgrade.py \
    --new-spec "personal-kb @ git+file://$PWD@$(git rev-parse HEAD)"
fi

# 1a. Build the packaged web UI and refuse to release if it is stale relative to
#     the frontend source. The fresh build is stashed in a temp dir and removed
#     from the tree so semantic-release sees a clean repo; step 2a restores it
#     and folds it into the single release commit.
static_dir=packages/kb-service/src/kb_service/static
./scripts/build_static_ui.sh
./scripts/check_static_ui_fresh.sh
ui_tmp=$(mktemp -d)
cp -R "$static_dir/." "$ui_tmp/"
git checkout -- "$static_dir"
git clean -fdq -- "$static_dir"

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

# Restore the freshly built UI, then re-verify it against the source.
rm -rf "$static_dir"
mkdir -p "$static_dir"
cp -R "$ui_tmp/." "$static_dir/"
rm -rf "$ui_tmp"
./scripts/check_static_ui_fresh.sh

# kb-service is a workspace member too, so its version is also in uv.lock.
uv lock
git add -A "$static_dir" pyproject.toml packages/personal-kb-hook/pyproject.toml \
  packages/kb-service/pyproject.toml packages/kb-core/pyproject.toml uv.lock
if ! git diff --cached --quiet; then
  # Keep the conventional `chore(release): v{version}` message, then move
  # the annotated tag onto the new commit.
  git commit --amend --no-edit
  git tag -d "$new_tag"
  git tag -a "$new_tag" -m "$new_tag"
fi

# Abort the release: drop the local tag so a re-run starts clean. Nothing has
# been pushed at this point.
abort_release() {
  echo "$1" >&2
  git tag -d "$new_tag" >/dev/null 2>&1 || true
  echo "Local tag $new_tag deleted; nothing was pushed." >&2
  exit 1
}

# 2b. Build all four wheels and verify exactly those, at the release version.
rm -rf dist
uv build --all-packages -o dist
expected_wheels="kb_core personal_kb personal_kb_hook personal_kb_web_service"
actual_wheels=$(find dist -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')
[ "$actual_wheels" = "4" ] || abort_release "Expected exactly 4 wheels in dist/, found $actual_wheels."
for name in $expected_wheels; do
  [ -f "dist/${name}-${new_version}-py3-none-any.whl" ] ||
    abort_release "Missing wheel dist/${name}-${new_version}-py3-none-any.whl."
done

# 2c. Publish hook (see header). Runs before any push.
#     Exit codes: 0 = published (and deployed, if the hook deploys);
#     10 = artifacts ARE published but a downstream deploy did not finish. A
#     published version is immutable, so aborting would leave the release
#     un-rerunnable: push the tags (the artifacts shipped), then fail loudly so
#     the operator finishes the deploy. Any other code = nothing published:
#     abort.
deploy_incomplete=0
if [ -x ./release.local.sh ]; then
  set +e
  ./release.local.sh "$new_version" "$PWD/dist"
  hook_rc=$?
  set -e
  case "$hook_rc" in
    0) ;;
    10) deploy_incomplete=1 ;;
    *) abort_release "release.local.sh failed (exit $hook_rc); aborting release." ;;
  esac
elif [ "$publish" = "1" ]; then
  abort_release "No executable ./release.local.sh publish hook found. It is an optional maintainer-local script, invoked as: release.local.sh <version> <abs-dist-dir>; exit 0 means artifacts are published. Create it, or re-run with --no-publish to release without publishing artifacts."
else
  echo "!!! NO ARTIFACTS PUBLISHED: --no-publish given and no ./release.local.sh hook; pushing tags anyway. !!!" >&2
fi

# 3. Publish to BOTH remotes: home-lab origin first, then the team-facing github.
git push origin main --tags
git push github main --tags

# 4. Optional local deploy (restart the home-lab service) if a deploy script exists.
if [ -x ./deploy.sh ]; then
  ./deploy.sh
fi

echo "Released $(git describe --tags --abbrev=0) to origin + github."
if [ "$deploy_incomplete" = "1" ]; then
  echo "!!! ${new_tag} is published and pushed, but the publish hook reported an INCOMPLETE deploy (exit 10). Finish the deploy before relying on it. !!!" >&2
  exit 3
fi
