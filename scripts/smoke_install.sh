#!/usr/bin/env bash
# scripts/smoke_install.sh — pre-release packaging-and-clean-install verification.
#
# The in-workspace test suite CANNOT catch missing-runtime-dep / packaging
# breaks: `uv sync` installs ALL workspace members (and every dependency-group),
# so a sibling package can be missing from `[project] dependencies` and every
# in-workspace test still passes. The real deploy artifact only fails when
# someone `uvx`s or `pip install`s it from a clean clone. We learned this the
# hard way in kb-01738 — the W6 deploy crashed with `ModuleNotFoundError: kb_core`
# despite 1089 green tests.
#
# This script reproduces those clean-install paths.
#
# Part A — kb-core BUILT-WHEEL smoke (deterministic, runs anywhere)
#   1. `uv build --package kb-core` to produce a wheel.
#   2. Create a fresh venv under /tmp (OUTSIDE the workspace).
#   3. `pip install` the built kb-core wheel into the fresh venv.
#   4. Run a tiny round-trip script: import kb_core, open create_sqlite() on a
#      temp DB, store an entry, search it back. Asserts the PUBLISHED artifact
#      stands on its own with zero workspace.
#
# Part B — personal-kb DEPLOY-PATH smoke (release-time only; documented here)
#   This MUST run AFTER the release commit has been pushed to the production
#   remote so `uvx --from "...@<sha>"` can clone it. It is therefore a
#   RELEASE-TIME check (run from `release.sh` or by hand right before
#   publishing), not a pre-commit gate.
#
#   Example (run after pushing):
#       SHA="$(git rev-parse HEAD)"
#       env -u KB_DATABASE_URL \
#         KB_DB_PATH=/tmp/personal_kb_smoke.db \
#         timeout 20 uvx --from \
#           "personal-kb @ git+ssh://git@git-host.example.com/path/to/personal_kb@${SHA}" \
#           personal-kb </dev/null 2>&1 \
#         | tee /tmp/personal_kb_smoke.log
#       grep -q "Starting MCP server" /tmp/personal_kb_smoke.log
#       ! grep -q "ModuleNotFoundError" /tmp/personal_kb_smoke.log
#
#   The MCP server reads stdin; piping `</dev/null` makes it exit cleanly
#   after the startup banner. We assert the FastMCP "Starting MCP server"
#   banner is present and that there is NO `ModuleNotFoundError` anywhere
#   in the captured stderr. This is the exact failure shape from kb-01738
#   that the in-workspace gate missed.
#
# See ./release.sh for the release machinery this script supports.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '[smoke_install] %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# Part A — kb-core built-wheel smoke
# ---------------------------------------------------------------------------
log "Part A: kb-core built-wheel smoke"

# Build the kb-core wheel. uv places artifacts in <pkg>/dist/.
log "Building kb-core wheel via 'uv build --package kb-core'..."
uv build --package kb-core >&2

# `uv build --package kb-core` places artifacts under the workspace-root `dist/`
# (modern uv) — fall back to the per-package `packages/kb-core/dist/` for older
# uv versions.
WHEEL="$(ls -1t "$REPO_ROOT/dist"/kb_core-*.whl 2>/dev/null | head -n1 || true)"
if [ -z "$WHEEL" ]; then
    WHEEL="$(ls -1t "$REPO_ROOT/packages/kb-core/dist"/kb_core-*.whl 2>/dev/null | head -n1 || true)"
fi
if [ -z "$WHEEL" ]; then
    log "FAIL: no kb-core wheel produced (looked in dist/ and packages/kb-core/dist/)"
    exit 1
fi
log "Built wheel: $WHEEL"

# Build the smoke venv OUTSIDE the workspace so workspace editable installs
# cannot leak in via path resolution.
SMOKE_DIR="$(mktemp -d -t kb_core_smoke.XXXXXX)"
trap 'rm -rf "$SMOKE_DIR"' EXIT
log "Smoke venv root: $SMOKE_DIR"

VENV_DIR="$SMOKE_DIR/venv"
# --seed gets us a pip in the venv (uv venvs are bare by default). We install
# with pip — same install path an end user gets — to keep this representative
# of a real `pip install kb-core` from PyPI.
uv venv --python 3.13 --seed "$VENV_DIR" >&2

# Install ONLY the built wheel. No editable workspace, no extras.
log "Installing built wheel into clean venv..."
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
"$VENV_DIR/bin/python" -m pip install --quiet "$WHEEL"

# Tiny round-trip: import + create_sqlite + store + search.
SMOKE_SCRIPT="$SMOKE_DIR/smoke.py"
cat >"$SMOKE_SCRIPT" <<'PY'
"""Minimal kb-core round-trip — run in a CLEAN venv with only kb-core installed."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import kb_core
from kb_core import KnowledgeBase, create_sqlite
from kb_core.models.search import SearchQuery


async def main() -> int:
    # Sanity: the public surface is importable.
    assert isinstance(kb_core.__name__, str)
    assert KnowledgeBase is not None
    assert create_sqlite is not None

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "smoke.db"
        async with await create_sqlite(db_path) as kb:
            entry = await kb.store(
                short_title="kb-core smoke entry",
                long_title="The clean-install smoke wrote this",
                knowledge_details=(
                    "If you can search this back, the published kb-core wheel "
                    "imports cleanly and the SQLite + FTS5 nucleus works with "
                    "zero workspace."
                ),
                enrich=False,
            )
            print(f"stored entry id={entry.id}")

            results, total = await kb.search(SearchQuery(query="clean-install smoke"))
            if not results:
                print("FAIL: stored entry was not searchable", file=sys.stderr)
                return 2
            top = results[0].entry
            if top.id != entry.id:
                print(
                    f"FAIL: top result {top.id!r} != stored {entry.id!r}",
                    file=sys.stderr,
                )
                return 3
            print(
                f"searched ok: top={top.id} score={results[0].score:.3f} "
                f"match_source={results[0].match_source}"
            )
    print("kb-core clean-install smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
PY

log "Running round-trip script..."
"$VENV_DIR/bin/python" "$SMOKE_SCRIPT"

log "Part A passed — kb-core wheel imports and round-trips in a clean venv."

# ---------------------------------------------------------------------------
# Part B — personal-kb deploy-path smoke (documented; runs at release time)
# ---------------------------------------------------------------------------
log "Part B is RELEASE-TIME ONLY — requires the commit to be pushed first."
log "See the header of this script for the exact uvx-from-git invocation."

log "smoke_install.sh complete."
