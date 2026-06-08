"""MCP-side writer for the on-disk JSONL maps index.

Called from ``kb_store`` and ``kb_store_batch`` after a successful
``mental_map`` create/update/deactivate. Re-queries the active maps for the
project using the EXACT same predicate as :func:`personal_kb.preflight._maps_sql`
(including ``ORDER BY created_at DESC LIMIT 5``), then rewrites the JSONL file
atomically: drop the target project's existing line (if any) and append a
fresh record — or omit the line entirely if the project now has zero maps.

The writer is strictly best-effort: an unwritable filesystem, a closed DB,
or any other error must NEVER fail the store path. Callers wrap it in a
``try / except`` mirroring the existing ``_build_graph`` wrapper.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import TYPE_CHECKING

from personal_kb.config import get_maps_index_path
from personal_kb.preflight import _maps_sql

if TYPE_CHECKING:
    from pathlib import Path

    from personal_kb.db.backend import Database

logger = logging.getLogger(__name__)


async def _fetch_maps_for_project(
    db: Database, project_ref: str, team: str | None
) -> list[dict[str, str]]:
    """Run the preflight._maps_sql query and return rows as dicts."""
    sql, has_team = _maps_sql(team)
    params: list[str] = [project_ref]
    if has_team and team is not None:
        params.append(team)
    cursor = await db.execute(sql, params)
    rows = await cursor.fetchall()
    maps: list[dict[str, str]] = []
    for row in rows:
        entry_id = row[0]
        short_title = row[1] or ""
        long_title = row[2] or ""
        maps.append(
            {
                "id": str(entry_id),
                "short_title": str(short_title),
                "long_title": str(long_title),
            }
        )
    return maps


def _read_existing_lines(path: Path) -> list[str]:
    """Read existing lines from the JSONL index, tolerant of a missing file."""
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("maps_index unreadable at %s: %s", path, exc)
        return []
    return [line for line in raw.splitlines() if line.strip()]


def _line_project_ref(line: str) -> str | None:
    """Return the ``project_ref`` on a JSONL line, or None if unparseable."""
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    project_ref = obj.get("project_ref")
    return project_ref if isinstance(project_ref, str) else None


def _atomic_write(path: Path, content: str) -> None:
    """Atomically write ``content`` to ``path`` via temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as fh:
        tmp_path = fh.name
        fh.write(content)
    os.replace(tmp_path, path)


async def write_project_maps(
    db: Database,
    project_ref: str,
    *,
    team: str | None = None,
    path: Path | None = None,
) -> None:
    """Refresh the JSONL index entry for ``project_ref``.

    Re-queries active mental_maps for the project (same predicate +
    LIMIT 5 as preflight._maps_sql), then atomically rewrites the index
    file: drop the target project's old line and append a fresh record
    (or omit the line entirely if the project now has zero active maps).

    Best-effort: a failure here logs and returns; it must not propagate.
    """
    if not project_ref:
        return
    target = path or get_maps_index_path()
    try:
        maps = await _fetch_maps_for_project(db, project_ref, team)
    except Exception:
        logger.warning("maps_index query failed for project %s", project_ref, exc_info=True)
        return

    try:
        existing = _read_existing_lines(target)
        # Drop any prior line for this project_ref.
        kept = [line for line in existing if _line_project_ref(line) != project_ref]

        if maps:
            record = json.dumps({"project_ref": project_ref, "maps": maps})
            kept.append(record)

        new_content = ("\n".join(kept) + "\n") if kept else ""
        _atomic_write(target, new_content)
    except OSError:
        logger.warning("maps_index write failed at %s", target, exc_info=True)
