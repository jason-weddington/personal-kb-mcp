"""Read the on-disk JSONL maps index.

The hook reads ONLY these files — no MCP call, no SQLite connection. The reader
is intentionally tolerant: a missing file is empty, a corrupt line is skipped,
and nothing here ever raises into the hook entry point.

When ``read_index`` is called with no argument, it globs every role-keyed
``maps_index.<role>.jsonl`` file in the maps-index directory (one per server
instance — ``maps_index.default.jsonl``, ``maps_index.personal.jsonl``,
``maps_index.team.jsonl``) and merges them into a single ``{project_ref:
[maps]}`` dict. The legacy suffix-less ``maps_index.jsonl`` (written by older
builds) is DELIBERATELY excluded — it is frozen cruft that would otherwise
shadow the live role-keyed files (it sorts after ``maps_index.default.jsonl``).
Merge policy is **last-wins** in ASCENDING sorted filename order: if the same
``project_ref`` appears in multiple files, the later-sorted file's maps
overwrite the earlier file's. In practice ``maps_index.team.jsonl`` shadows
``maps_index.personal.jsonl`` / ``maps_index.default.jsonl`` for a shared
project_ref.
"""

from __future__ import annotations

import glob as _glob
import json
import logging
from pathlib import Path
from typing import TypedDict

from personal_kb_hook.paths import get_maps_index_path

logger = logging.getLogger(__name__)


class MapEntry(TypedDict):
    """One mental_map row in the on-disk index."""

    id: str
    short_title: str
    long_title: str


def _parse_one_file(target: Path) -> dict[str, list[MapEntry]]:
    """Parse a single JSONL index file. Tolerant; never raises."""
    try:
        if not target.exists():
            return {}
        raw = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError) as exc:
        logger.debug("maps_index unreadable at %s: %s", target, exc)
        return {}

    result: dict[str, list[MapEntry]] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        project_ref = obj.get("project_ref")
        maps = obj.get("maps")
        if not isinstance(project_ref, str) or not project_ref:
            continue
        if not isinstance(maps, list):
            continue
        cleaned: list[MapEntry] = []
        for raw_map in maps:
            if not isinstance(raw_map, dict):
                continue
            entry_id = raw_map.get("id")
            short_title = raw_map.get("short_title")
            long_title = raw_map.get("long_title", "")
            if not isinstance(entry_id, str) or not entry_id:
                continue
            if not isinstance(short_title, str):
                continue
            if long_title is None:
                long_title = ""
            if not isinstance(long_title, str):
                continue
            cleaned.append(MapEntry(id=entry_id, short_title=short_title, long_title=long_title))
        result[project_ref] = cleaned
    return result


def read_index(path: Path | None = None) -> dict[str, list[MapEntry]]:
    """Read the JSONL index and return ``project_ref -> [maps]``.

    Tolerant of missing files (returns empty mapping), corrupt or partial
    lines (skipped), and unexpected shapes (skipped). Never raises.

    When ``path`` is None (the default), all role-keyed
    ``maps_index.<role>.jsonl`` files in the maps-index directory are globbed
    and merged in ASCENDING sorted filename order with LAST-WINS on
    ``project_ref`` collision. The legacy suffix-less ``maps_index.jsonl`` is
    excluded so a frozen old-build file can't shadow the live role-keyed ones.

    When ``path`` is given explicitly, only that one file is read (back-compat
    for callers that pinned a single path, including tests).
    """
    if path is not None:
        return _parse_one_file(path)

    default = get_maps_index_path()
    parent = default.parent
    try:
        # Sorted ascending so later (alphabetically) files overwrite earlier
        # ones on collision (team > personal > default). The role suffix is
        # REQUIRED ("maps_index.<role>.jsonl") so the legacy suffix-less
        # maps_index.jsonl — frozen cruft from older builds — is excluded; it
        # sorts AFTER maps_index.default.jsonl and would otherwise shadow it.
        matched = sorted(_glob.glob(str(parent / "maps_index.*.jsonl")))
    except (OSError, ValueError) as exc:
        logger.debug("maps_index glob failed at %s: %s", parent, exc)
        return {}

    merged: dict[str, list[MapEntry]] = {}
    for filename in matched:
        partial = _parse_one_file(Path(filename))
        # Last-wins on collision (later file overwrites earlier file).
        merged.update(partial)
    return merged
