"""Scope resolver — walk up from ``cwd`` to the first ``.kb_project`` file.

The committed ``.kb_project`` is the v1 scope anchor for both hooks. It is
portable across machines and users (it lives in git), needs no per-user
config, and avoids depending on git-origin data. The first non-blank,
non-comment line of the file is the project_ref the hook surfaces.
"""

from __future__ import annotations

from pathlib import Path

_KB_PROJECT_FILENAME = ".kb_project"


def resolve_project(cwd: str | None) -> str | None:
    """Resolve the project_ref by walking up from ``cwd``.

    Walks from ``Path(cwd)`` through each parent up to the filesystem root,
    looking for a ``.kb_project`` file. The project_ref is the FIRST non-blank
    line that does not start with ``#``, stripped of surrounding whitespace.

    Returns ``None`` for any of: falsy ``cwd``, no ``.kb_project`` anywhere
    on the walk, unreadable file, empty file, comment-only file. Never raises.
    """
    if not cwd:
        return None

    try:
        start = Path(cwd)
    except (TypeError, ValueError):
        return None

    candidates: list[Path] = []
    try:
        candidates.append(start)
        candidates.extend(start.parents)
    except (OSError, ValueError):
        return None

    for directory in candidates:
        marker = directory / _KB_PROJECT_FILENAME
        try:
            if not marker.is_file():
                continue
            text = marker.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            return line
        # File existed but had no usable line; stop here — a higher .kb_project
        # would belong to a different (outer) project.
        return None

    return None
