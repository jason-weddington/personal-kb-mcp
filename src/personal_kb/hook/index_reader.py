"""Read the on-disk JSONL maps index.

The hook reads ONLY this file — no MCP call, no SQLite connection. The reader
is intentionally tolerant: a missing file is empty, a corrupt line is skipped,
and nothing here ever raises into the hook entry point.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, TypedDict

from personal_kb.config import get_maps_index_path

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class MapEntry(TypedDict):
    """One mental_map row in the on-disk index."""

    id: str
    short_title: str
    long_title: str


def read_index(path: Path | None = None) -> dict[str, list[MapEntry]]:
    """Read the JSONL index and return ``project_ref -> [maps]``.

    Tolerant of missing files (returns empty mapping), corrupt or partial
    lines (skipped), and unexpected shapes (skipped). Never raises.
    """
    target = path or get_maps_index_path()
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
