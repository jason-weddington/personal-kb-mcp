"""Project context primer — "bring me up to speed on X".

Pure SQL queries that return a compact table-of-contents of relevant KB
entries for a project: expiring entries, recent decisions/lessons, and
active conventions. No LLM calls.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from personal_kb.db.backend import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL queries (pure SQL, no LLM)
# ---------------------------------------------------------------------------

_TEAM_CLAUSE = " AND (team = ? OR team IS NULL)"


def _expiring_sql(team: str | None) -> tuple[str, bool]:
    """Build expiring entries SQL. Returns (sql, has_team_param)."""
    sql = (
        "SELECT id, entry_type, short_title, expires_at "
        "FROM knowledge_entries "
        "WHERE is_active = 1 AND project_ref = ? "
        "AND expires_at IS NOT NULL "
        "AND expires_at > ? "
        "AND expires_at < ? "
    )
    if team:
        sql += _TEAM_CLAUSE
    sql += "ORDER BY expires_at ASC LIMIT 5"
    return sql, team is not None


def _recent_sql(team: str | None, *, has_since: bool) -> tuple[str, bool]:
    """Build recent decisions/lessons SQL. Returns (sql, has_team_param)."""
    sql = (
        "SELECT id, entry_type, short_title "
        "FROM knowledge_entries "
        "WHERE is_active = 1 AND project_ref = ? "
        "AND entry_type IN ('decision', 'lesson_learned') "
    )
    if has_since:
        sql += "AND created_at >= ? "
    if team:
        sql += _TEAM_CLAUSE
    sql += "ORDER BY created_at DESC LIMIT 5"
    return sql, team is not None


def _conventions_sql(team: str | None) -> tuple[str, bool]:
    """Build conventions SQL. Returns (sql, has_team_param)."""
    sql = (
        "SELECT id, entry_type, short_title "
        "FROM knowledge_entries "
        "WHERE is_active = 1 AND project_ref = ? "
        "AND entry_type = 'pattern_convention' "
    )
    if team:
        sql += _TEAM_CLAUSE
    sql += "ORDER BY created_at DESC LIMIT 5"
    return sql, team is not None


_MAX_RELATED = 5


async def _graph_related(
    db: Database,
    project_ref: str,
    exclude_ids: set[str],
    *,
    team: str | None = None,
) -> list[tuple[str, str, str, str]]:
    """Find decisions/lessons from other projects via shared tags.

    Traverses: project entries → tag nodes → other entries.
    Only expands tags that appear on 2+ of this project's entries (signal).
    Returns rows of (id, entry_type, short_title, tag_name).
    """
    # Step 1: find tags connected to 2+ entries in this project.
    # graph_edges: entry --has_tag--> tag:X
    tag_sql = (
        "SELECT e.target AS tag_node, COUNT(DISTINCT e.source) AS cnt "
        "FROM graph_edges e "
        "JOIN knowledge_entries ke ON ke.id = e.source "
        "WHERE e.edge_type = 'has_tag' "
        "AND ke.is_active = 1 "
        "AND ke.project_ref = ? "
        "GROUP BY e.target "
        "HAVING COUNT(DISTINCT e.source) >= 2"
    )
    tag_cursor = await db.execute(tag_sql, [project_ref])
    tag_rows = await tag_cursor.fetchall()

    if not tag_rows:
        return []

    # Build tag node IDs and a map to display names (strip "tag:" prefix)
    tag_node_ids = []
    tag_names: dict[str, str] = {}
    for row in tag_rows:
        node_id = row[0]
        tag_node_ids.append(node_id)
        tag_names[node_id] = node_id.removeprefix("tag:")

    # Step 2: find entries from OTHER projects connected to these tags,
    # filtered to decisions/lessons.
    placeholders = ", ".join("?" for _ in tag_node_ids)
    related_sql = (
        "SELECT ke.id, ke.entry_type, ke.short_title, e.target AS tag_node "
        "FROM graph_edges e "
        "JOIN knowledge_entries ke ON ke.id = e.source "
        "WHERE e.edge_type = 'has_tag' "
        "AND e.target IN (" + placeholders + ") "
        "AND ke.is_active = 1 "
        "AND (ke.project_ref != ? OR ke.project_ref IS NULL) "
        "AND ke.entry_type IN ('decision', 'lesson_learned') "
    )
    params: list[str | int] = [*tag_node_ids, project_ref]

    if team:
        related_sql += _TEAM_CLAUSE
        params.append(team)

    related_sql += "ORDER BY ke.created_at DESC "
    related_sql += "LIMIT ?"
    # Fetch extra to account for dedup against exclude_ids
    params.append(_MAX_RELATED + len(exclude_ids))

    cursor = await db.execute(related_sql, params)
    rows = await cursor.fetchall()

    # Dedup against entries already in the ToC + dedup by entry ID
    seen: set[str] = set(exclude_ids)
    result = []
    for row in rows:
        entry_id = row[0]
        if entry_id in seen:
            continue
        seen.add(entry_id)
        # Replace tag_node with display name
        tag_node = row[3]
        result.append((row[0], row[1], row[2], tag_names.get(tag_node, tag_node)))
        if len(result) >= _MAX_RELATED:
            break

    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_expiry_badge(expires_at_str: str) -> str:
    """Format an expiry badge from an ISO datetime string."""
    try:
        exp = datetime.fromisoformat(expires_at_str)
        if not exp.tzinfo:
            exp = exp.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return ""
    now = datetime.now(UTC)
    if now >= exp:
        ago = (now - exp).days
        if ago == 0:
            return " [EXPIRED today]"
        return f" [EXPIRED {ago}d ago]"
    remaining = exp - now
    if remaining.days > 0:
        return f" [EXPIRES {remaining.days}d]"
    hours = int(remaining.total_seconds() / 3600)
    return f" [EXPIRES {hours}h]"


def _format_toc_line(row: object, include_expiry: bool = False) -> str:
    """Format a single ToC line: [kb-00123] type — Title [EXPIRES 3d]."""
    entry_id = row[0]  # type: ignore[index]
    entry_type = row[1]  # type: ignore[index]
    short_title = row[2]  # type: ignore[index]
    line = f"[{entry_id}] {entry_type} — {short_title}"
    if include_expiry:
        expires_at = row[3]  # type: ignore[index]
        if expires_at:
            line += _format_expiry_badge(expires_at)
    return line


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def build_project_context(
    db: Database,
    project_ref: str,
    *,
    team: str | None = None,
    since: timedelta | None = None,
) -> str:
    """Build a compact project context primer.

    Returns a formatted ToC string with expiring entries, recent decisions/
    lessons, and active conventions. Returns "No entries found." if empty.

    *since* narrows decisions/lessons to those created within that window.
    Expiring entries and conventions are unaffected by *since*.
    """
    now = datetime.now(UTC)
    grace_start = (now - timedelta(days=7)).isoformat()
    horizon_end = (now + timedelta(days=30)).isoformat()

    since_iso: str | None = None
    if since is not None:
        since_iso = (now - since).isoformat()

    # Build SQL with optional team + since filters
    exp_sql, exp_has_team = _expiring_sql(team)
    rec_sql, rec_has_team = _recent_sql(team, has_since=since_iso is not None)
    conv_sql, conv_has_team = _conventions_sql(team)

    exp_params: list[str] = [project_ref, grace_start, horizon_end]
    if exp_has_team:
        exp_params.append(team)  # type: ignore[arg-type]

    rec_params: list[str] = [project_ref]
    if since_iso is not None:
        rec_params.append(since_iso)
    if rec_has_team:
        rec_params.append(team)  # type: ignore[arg-type]

    conv_params: list[str] = [project_ref]
    if conv_has_team:
        conv_params.append(team)  # type: ignore[arg-type]

    # Run all three queries
    exp_cursor = await db.execute(exp_sql, exp_params)
    expiring = await exp_cursor.fetchall()

    rec_cursor = await db.execute(rec_sql, rec_params)
    recent = await rec_cursor.fetchall()

    conv_cursor = await db.execute(conv_sql, conv_params)
    conventions = await conv_cursor.fetchall()

    # Graph expansion: find decisions/lessons from other projects connected
    # via shared tags (tags that appear on 2+ of this project's entries).
    project_entry_ids = {r[0] for r in expiring}
    project_entry_ids |= {r[0] for r in recent}
    project_entry_ids |= {r[0] for r in conventions}

    related = await _graph_related(db, project_ref, project_entry_ids, team=team)

    if not expiring and not recent and not conventions and not related:
        return f"No entries found for project '{project_ref}'."

    total = len(expiring) + len(recent) + len(conventions) + len(related)
    lines = [f"{project_ref} ({total} entries)"]
    lines.append("Use kb_get to read full details for any entry below.")

    if expiring:
        lines.append("\nExpiring:")
        for row in expiring:
            lines.append(f"  - {_format_toc_line(row, include_expiry=True)}")

    if recent:
        label = "Recent decisions & lessons"
        if since is not None:
            label += f" (last {_format_duration(since)})"
        lines.append(f"\n{label}:")
        for row in recent:
            lines.append(f"  - {_format_toc_line(row)}")

    if conventions:
        lines.append("\nConventions:")
        for row in conventions:
            lines.append(f"  - {_format_toc_line(row)}")

    if related:
        lines.append("\nRelated (via graph):")
        for entry_id, entry_type, short_title, via_tag in related:
            lines.append(f"  - [{entry_id}] {entry_type} — {short_title} (via #{via_tag})")

    return "\n".join(lines)


def _format_duration(td: timedelta) -> str:
    """Format a timedelta as a human-readable string."""
    days = td.days
    if days >= 7 and days % 7 == 0:
        return f"{days // 7}w"
    if days > 0:
        return f"{days}d"
    hours = int(td.total_seconds() / 3600)
    return f"{hours}h"
