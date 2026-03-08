"""CWD-based project context injection.

At server startup, detect the project from the working directory and inject
a compact table-of-contents of relevant KB entries into the MCP server
instructions. No LLM calls — pure SQL queries.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from personal_kb.db.backend import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

_SEP_RE = re.compile(r"[-_.\s]+")


def _normalize(name: str) -> str:
    """Strip separators and lowercase for fuzzy comparison."""
    return _SEP_RE.sub("", name.lower())


def detect_project(cwd: str, known_projects: list[str]) -> list[str]:
    """Fuzzy-match the leaf directory of *cwd* against *known_projects*.

    Returns a list of matching project_ref values (may be empty or multiple).
    Biased toward recall — it's fine to return extra matches since the output
    is a lightweight ToC the agent can choose to ignore.
    """
    dirname = os.path.basename(os.path.normpath(cwd))
    if not dirname:
        return []

    norm_dir = _normalize(dirname)
    if not norm_dir:
        return []

    matches: list[str] = []
    for proj in known_projects:
        norm_proj = _normalize(proj)
        if not norm_proj:
            continue

        # Tier 1: exact match after normalization
        if norm_dir == norm_proj:
            matches.append(proj)
            continue

        # Tier 2: substring match with length guard
        shorter, longer = (
            (norm_dir, norm_proj) if len(norm_dir) <= len(norm_proj) else (norm_proj, norm_dir)
        )
        if len(shorter) >= 3 and shorter in longer and len(shorter) / len(longer) >= 0.5:
            matches.append(proj)
            continue

        # Tier 3: fuzzy similarity for typos/variations
        if difflib.SequenceMatcher(None, norm_dir, norm_proj).ratio() >= 0.75:
            matches.append(proj)

    return matches


# ---------------------------------------------------------------------------
# SQL queries (pure SQL, no LLM)
# ---------------------------------------------------------------------------

_EXPIRING_SQL = (
    "SELECT id, entry_type, short_title, expires_at "
    "FROM knowledge_entries "
    "WHERE is_active = 1 AND project_ref = ? "
    "AND expires_at IS NOT NULL "
    "AND expires_at > ? "
    "AND expires_at < ? "
    "ORDER BY expires_at ASC "
    "LIMIT 5"
)

_RECENT_SQL = (
    "SELECT id, entry_type, short_title "
    "FROM knowledge_entries "
    "WHERE is_active = 1 AND project_ref = ? "
    "AND entry_type IN ('decision', 'lesson_learned') "
    "ORDER BY created_at DESC "
    "LIMIT 5"
)

_CONVENTIONS_SQL = (
    "SELECT id, entry_type, short_title "
    "FROM knowledge_entries "
    "WHERE is_active = 1 AND project_ref = ? "
    "AND entry_type = 'pattern_convention' "
    "ORDER BY created_at DESC "
    "LIMIT 5"
)


async def _get_known_projects(db: Database) -> list[str]:
    """Get all distinct project_refs from active entries."""
    cursor = await db.execute(
        "SELECT DISTINCT project_ref FROM knowledge_entries "
        "WHERE is_active = 1 AND project_ref IS NOT NULL"
    )
    rows = await cursor.fetchall()
    return [row[0] for row in rows]


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


async def _build_project_section(db: Database, project_ref: str) -> str | None:
    """Build the ToC section for one project. Returns None if empty."""
    now = datetime.now(UTC)
    grace_start = (now - timedelta(days=7)).isoformat()
    horizon_end = (now + timedelta(days=30)).isoformat()

    # Run all three queries
    exp_cursor = await db.execute(_EXPIRING_SQL, [project_ref, grace_start, horizon_end])
    expiring = await exp_cursor.fetchall()

    rec_cursor = await db.execute(_RECENT_SQL, [project_ref])
    recent = await rec_cursor.fetchall()

    conv_cursor = await db.execute(_CONVENTIONS_SQL, [project_ref])
    conventions = await conv_cursor.fetchall()

    if not expiring and not recent and not conventions:
        return None

    total = len(expiring) + len(recent) + len(conventions)
    lines = [f"## {project_ref} ({total} entries)"]

    if expiring:
        lines.append("Expiring:")
        for row in expiring:
            lines.append(f"  - {_format_toc_line(row, include_expiry=True)}")

    if recent:
        lines.append("Recent decisions & lessons:")
        for row in recent:
            lines.append(f"  - {_format_toc_line(row)}")

    if conventions:
        lines.append("Conventions:")
        for row in conventions:
            lines.append(f"  - {_format_toc_line(row)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def build_preflight_context(db: Database, cwd: str) -> str:
    """Build compact project context from the working directory.

    Returns a string to append to server instructions, or empty string
    if no matching project is found.
    """
    try:
        known = await _get_known_projects(db)
    except Exception:
        logger.debug("Preflight: failed to query projects", exc_info=True)
        return ""

    if not known:
        return ""

    matches = detect_project(cwd, known)
    if not matches:
        return ""

    sections: list[str] = []
    for proj in matches:
        try:
            section = await _build_project_section(db, proj)
            if section:
                sections.append(section)
        except Exception:
            logger.debug("Preflight: failed to build section for %s", proj, exc_info=True)

    if not sections:
        return ""

    header = (
        "\n\n# Project Context (auto-detected from working directory)\n"
        "Use kb_get to read full details for any entry below.\n\n"
    )
    return header + "\n\n".join(sections)
