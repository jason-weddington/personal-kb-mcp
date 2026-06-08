"""MCP-side writer for the on-disk JSONL maps index.

Called from ``kb_store`` and ``kb_store_batch`` after a successful
``mental_map`` create/update/deactivate. Re-queries the active maps for the
project using the EXACT same predicate as :func:`personal_kb.preflight._maps_sql`
(``ORDER BY created_at DESC``, no limit — maps are a small curated set), then
rewrites the JSONL file
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


async def rebuild_all_projects(
    db: Database,
    *,
    team: str | None = None,
    path: Path | None = None,
) -> None:
    """Rebuild the entire maps index from scratch by re-querying every project.

    Used at server startup and on listener reconnect to converge any drift
    introduced while this instance wasn't running (e.g. another server wrote
    rows; this server missed the NOTIFY). One whole-file atomic rewrite via
    ``_atomic_write`` — no per-project line-edit logic.

    Discovery query: ``SELECT DISTINCT project_ref FROM knowledge_entries
    WHERE is_active = 1 AND entry_type = 'mental_map' AND project_ref IS NOT
    NULL``. The static SQL is portable across SQLite and Postgres (no ``?``
    params). Per project, we reuse :func:`_fetch_maps_for_project` so the
    same predicate + ``ORDER BY created_at DESC`` (no limit) + optional team
    clause as the incremental writer is applied — no drift between
    incremental and rebuild output for any given project.

    Projects whose ``_fetch_maps_for_project`` returns ``[]`` are SKIPPED:
    no line is emitted (mirrors ``write_project_maps``'s "drop the line"
    semantics).

    Best-effort: any failure logs a warning and returns; it must never
    abort startup or a reconnect.
    """
    target = path or get_maps_index_path()
    try:
        cursor = await db.execute(
            "SELECT DISTINCT project_ref FROM knowledge_entries "
            "WHERE is_active = 1 AND entry_type = 'mental_map' "
            "AND project_ref IS NOT NULL"
        )
        rows = await cursor.fetchall()
    except Exception:
        logger.warning("maps_index rebuild discovery query failed", exc_info=True)
        return

    lines: list[str] = []
    for row in rows:
        project_ref = row[0]
        if not isinstance(project_ref, str) or not project_ref:
            continue
        try:
            maps = await _fetch_maps_for_project(db, project_ref, team)
        except Exception:
            logger.warning(
                "maps_index rebuild: per-project fetch failed for %s",
                project_ref,
                exc_info=True,
            )
            continue
        if not maps:
            # Skip zero-map projects — no line emitted.
            continue
        lines.append(json.dumps({"project_ref": project_ref, "maps": maps}))

    new_content = ("\n".join(lines) + "\n") if lines else ""
    try:
        _atomic_write(target, new_content)
    except OSError:
        logger.warning("maps_index rebuild write failed at %s", target, exc_info=True)


async def write_project_maps(
    db: Database,
    project_ref: str,
    *,
    team: str | None = None,
    path: Path | None = None,
) -> None:
    """Refresh the JSONL index entry for ``project_ref``.

    Re-queries active mental_maps for the project (same predicate as
    preflight._maps_sql — all maps, no limit), then atomically rewrites the index
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
