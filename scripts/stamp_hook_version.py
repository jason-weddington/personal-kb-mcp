"""Stamp ``packages/personal-kb-hook/pyproject.toml`` to the main repo version.

Called from ``release.sh`` right after ``uv run semantic-release version`` has
bumped the root ``pyproject.toml``.  We keep the standalone ``personal-kb-hook``
package locked to the same version as the main ``personal-kb`` package so the
single repo tag cut by semantic-release covers BOTH packages (decision: one
repo version for both — simplest; revisit independent versioning only if a real
need appears).

The hook is consumed via ``uv tool install --from
"git+...#subdirectory=packages/personal-kb-hook"`` — i.e. pulled out of the git
tree at a tag.  It is NOT a separately published wheel, so semantic-release's
``build_command`` does not need to build it; only the ``[project] version``
line that lands in the tagged commit has to be right.

Stdlib-only on purpose: this runs in the release shell before ``uv build`` and
must not pull in any extra dependencies.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
DEFAULT_HOOK_PYPROJECT = REPO_ROOT / "packages" / "personal-kb-hook" / "pyproject.toml"
DEFAULT_SERVICE_PYPROJECT = REPO_ROOT / "packages" / "kb-service" / "pyproject.toml"
DEFAULT_CORE_PYPROJECT = REPO_ROOT / "packages" / "kb-core" / "pyproject.toml"

# Matches the FIRST ``version = "X.Y.Z"`` line at column 0 (i.e. the
# ``[project]`` table's version, not e.g. ``requires-python``-adjacent
# version literals deeper in the file).
_VERSION_LINE_RE = re.compile(r'^version = "([^"]+)"', re.MULTILINE)


def read_project_version(pyproject_path: Path) -> str:
    """Return the ``[project] version`` string from a pyproject.toml file."""
    text = pyproject_path.read_text()
    match = _VERSION_LINE_RE.search(text)
    if match is None:
        raise ValueError(f'No top-level `version = "..."` line found in {pyproject_path}')
    return match.group(1)


def stamp_hook_version(hook_pyproject_path: Path, new_version: str) -> bool:
    """Set the hook package's ``[project] version`` to ``new_version``.

    Returns True if the file changed, False if it already matched.  Raises
    ``ValueError`` if the file has zero or more than one matching line.
    """
    text = hook_pyproject_path.read_text()
    matches = _VERSION_LINE_RE.findall(text)
    if len(matches) != 1:
        raise ValueError(
            f'Expected exactly one top-level `version = "..."` line in '
            f"{hook_pyproject_path}; found {len(matches)}"
        )
    current = matches[0]
    if current == new_version:
        return False
    new_text, count = _VERSION_LINE_RE.subn(f'version = "{new_version}"', text, count=1)
    if count != 1:
        raise ValueError(f"Failed to substitute version in {hook_pyproject_path}")
    hook_pyproject_path.write_text(new_text)
    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root-pyproject",
        type=Path,
        default=DEFAULT_ROOT_PYPROJECT,
        help="Path to the main repo pyproject.toml (source of truth for the version).",
    )
    parser.add_argument(
        "--hook-pyproject",
        type=Path,
        default=DEFAULT_HOOK_PYPROJECT,
        help="Path to packages/personal-kb-hook/pyproject.toml (target).",
    )
    parser.add_argument(
        "--service-pyproject",
        type=Path,
        default=DEFAULT_SERVICE_PYPROJECT,
        help="Path to packages/kb-service/pyproject.toml (also stamped).",
    )
    parser.add_argument(
        "--core-pyproject",
        type=Path,
        default=DEFAULT_CORE_PYPROJECT,
        help="Path to packages/kb-core/pyproject.toml (also stamped).",
    )
    args = parser.parse_args(argv)

    new_version = read_project_version(args.root_pyproject)
    for target in (args.hook_pyproject, args.service_pyproject, args.core_pyproject):
        changed = stamp_hook_version(target, new_version)
        display = _display_path(target)
        if changed:
            print(f"Stamped {display} to version {new_version}")
        else:
            print(f"{display} already at version {new_version}; no change.")
    return 0


def _display_path(path: Path) -> Path:
    """Return ``path`` relative to the repo root when possible, else absolute."""
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


if __name__ == "__main__":
    sys.exit(main())
