"""Read-only browse queries over the KB data DB (new, top, stale, projects).

Every function runs one portable entry SELECT (``?`` placeholders, no
ORDER BY, no comparison on a TEXT column) and does all parsing, filtering
and ordering in Python over parsed UTC datetimes, so SQLite and Postgres
return identical pages. Inactive and superseded entries never appear.
Rows with an unknown entry type or an unparseable timestamp are skipped,
counted in ``skipped`` and logged once per call as ``browse-skip``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from kb_core.confidence.decay import (
    STALE_FILTER_THRESHOLD,
    compute_effective_confidence,
    half_life_for,
)
from kb_core.models.entry import EntryType

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from kb_core.db.backend import Database

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 25
DEFAULT_PROJECT_PAGE_SIZE = 24
MAX_PAGE_SIZE = 100

RecentSort = Literal["created", "updated"]
TopSignal = Literal["degree", "read"]
StaleKind = Literal["decayed", "expired"]
ProjectSort = Literal["newest", "updated", "confidence"]

_ENTRY_SELECT = (
    "SELECT id, project_ref, short_title, long_title, entry_type, confidence_level,"
    " created_at, updated_at, last_accessed, expires_at FROM knowledge_entries"
    " WHERE is_active = 1 AND superseded_by IS NULL"
)

_CURSOR_RE = re.compile(r"[0-9]+", re.ASCII)

_ENTRY_TYPE_VALUES = frozenset(t.value for t in EntryType)


@dataclass(frozen=True)
class BrowseRow:
    """One entry as shown in a browse list, with derived decay fields."""

    id: str
    short_title: str
    long_title: str
    entry_type: str
    project_ref: str | None
    created_at: datetime
    updated_at: datetime
    last_accessed: datetime | None
    expires_at: datetime | None
    confidence_level: float
    effective_confidence: float
    half_life_days: float | None
    anchor_age_days: float
    expired: bool
    signal_value: int | None = None


@dataclass(frozen=True)
class BrowsePage:
    """A page of browse rows."""

    items: list[BrowseRow]
    next_cursor: str | None
    total: int
    scanned: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class ProjectSummary:
    """Per-project counts for the projects grid."""

    project_ref: str
    entry_count: int
    counts_by_type: dict[str, int]
    last_activity: datetime


@dataclass(frozen=True)
class ProjectPage:
    """A page of project summaries."""

    items: list[ProjectSummary]
    next_cursor: str | None
    total: int
    scanned: int = 0
    skipped: int = 0


class BrowseParamError(ValueError):
    """An invalid cursor or limit."""


class ProjectNotFound(LookupError):  # noqa: N818 - name pinned by the browse API
    """The project has no active, non-superseded entries."""


def decode_cursor(cursor: str | None) -> int:
    """Decode an offset cursor.

    Args:
        cursor: A decimal offset string, or None/'' for the first page.

    Returns:
        The offset.

    Raises:
        BrowseParamError: When the cursor is not an ASCII decimal string.
    """
    if not cursor:
        return 0
    if not _CURSOR_RE.fullmatch(cursor):
        raise BrowseParamError("invalid cursor")
    return int(cursor)


def _check_limit(limit: int) -> None:
    if limit < 1 or limit > MAX_PAGE_SIZE:
        raise BrowseParamError(f"limit must be between 1 and {MAX_PAGE_SIZE}")


def _parse_instant(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_optional(raw: str | None) -> datetime | None:
    return None if raw is None else _parse_instant(raw)


@dataclass
class _Loaded:
    rows: list[BrowseRow]
    scanned: int
    skipped: int


async def _load_rows(
    db: Database,
    fn_name: str,
    now: datetime,
    *,
    project_ref: str | None = None,
    require_project: bool = False,
    entry_type: str | None = None,
) -> _Loaded:
    sql = _ENTRY_SELECT
    params: list[Any] = []
    if project_ref is not None:
        sql += " AND project_ref = ?"
        params.append(project_ref)
    elif require_project:
        sql += " AND project_ref IS NOT NULL"
    if entry_type is not None:
        sql += " AND entry_type = ?"
        params.append(entry_type)
    cursor = await db.execute(sql, tuple(params))
    raw_rows = await cursor.fetchall()

    rows: list[BrowseRow] = []
    n_unknown = 0
    n_bad = 0
    skipped_ids: list[str] = []
    for raw in raw_rows:
        entry_id = str(raw["id"])
        type_value = raw["entry_type"]
        if type_value not in _ENTRY_TYPE_VALUES:
            n_unknown += 1
            skipped_ids.append(entry_id)
            continue
        try:
            created_at = _parse_instant(raw["created_at"])
            updated_at = _parse_instant(raw["updated_at"])
            last_accessed = _parse_optional(raw["last_accessed"])
            expires_at = _parse_optional(raw["expires_at"])
        except (ValueError, TypeError):
            n_bad += 1
            skipped_ids.append(entry_id)
            continue
        etype = EntryType(type_value)
        confidence = float(raw["confidence_level"])
        anchor = updated_at
        if last_accessed is not None and last_accessed > anchor:
            anchor = last_accessed
        rows.append(
            BrowseRow(
                id=entry_id,
                short_title=raw["short_title"],
                long_title=raw["long_title"],
                entry_type=etype.value,
                project_ref=raw["project_ref"],
                created_at=created_at,
                updated_at=updated_at,
                last_accessed=last_accessed,
                expires_at=expires_at,
                confidence_level=confidence,
                effective_confidence=compute_effective_confidence(
                    confidence, etype, updated_at, now, last_accessed=last_accessed
                ),
                half_life_days=half_life_for(etype),
                anchor_age_days=round(max(0.0, (now - anchor).total_seconds() / 86400), 1),
                expired=expires_at is not None and now >= expires_at,
            )
        )

    skipped = n_unknown + n_bad
    if skipped:
        logger.warning(
            "browse-skip fn=%s skipped=%d unknown_type=%d bad_timestamp=%d ids=%s",
            fn_name,
            skipped,
            n_unknown,
            n_bad,
            ",".join(skipped_ids[:10]),
        )
    return _Loaded(rows=rows, scanned=len(raw_rows), skipped=skipped)


def _ordered(
    rows: Sequence[BrowseRow], keys: list[tuple[Callable[[BrowseRow], Any], bool]]
) -> list[BrowseRow]:
    """Sort by the given (key, descending) pairs, most significant first, ties by id asc."""
    result = sorted(rows, key=lambda r: r.id)
    for key, descending in reversed(keys):
        result.sort(key=key, reverse=descending)
    return result


def _page(rows: list[BrowseRow], offset: int, limit: int, loaded: _Loaded) -> BrowsePage:
    total = len(rows)
    end = offset + limit
    return BrowsePage(
        items=rows[offset:end],
        next_cursor=str(end) if end < total else None,
        total=total,
        scanned=loaded.scanned,
        skipped=loaded.skipped,
    )


def _now(now: datetime | None) -> datetime:
    return datetime.now(UTC) if now is None else now


async def browse_recent(
    db: Database,
    *,
    sort: RecentSort = "created",
    entry_type: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
    now: datetime | None = None,
) -> BrowsePage:
    """List entries by creation time, or updated entries by update time.

    Args:
        db: The KB data database.
        sort: 'created' (created_at desc) or 'updated' (only entries
            updated after creation, updated_at desc).
        entry_type: Optional exact entry-type filter.
        limit: Page size, 1 to 100.
        cursor: Offset cursor from a previous page.
        now: Clock for decay fields; defaults to the current time.

    Returns:
        The requested page.
    """
    _check_limit(limit)
    offset = decode_cursor(cursor)
    loaded = await _load_rows(db, "browse_recent", _now(now), entry_type=entry_type)
    if sort == "updated":
        rows = _ordered(
            [r for r in loaded.rows if r.updated_at > r.created_at],
            [(lambda r: r.updated_at, True)],
        )
    else:
        rows = _ordered(loaded.rows, [(lambda r: r.created_at, True)])
    return _page(rows, offset, limit, loaded)


async def _degrees(db: Database) -> Counter[str]:
    cursor = await db.execute("SELECT source, target FROM graph_edges")
    counts: Counter[str] = Counter()
    for row in await cursor.fetchall():
        source = row["source"]
        target = row["target"]
        counts[source] += 1
        if target != source:
            counts[target] += 1
    return counts


async def browse_top(
    db: Database,
    *,
    signal: TopSignal = "degree",
    entry_type: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
    now: datetime | None = None,
) -> BrowsePage:
    """List the most connected or most recently read entries.

    Args:
        db: The KB data database.
        signal: 'degree' (graph connection count desc) or 'read'
            (last_accessed desc).
        entry_type: Optional exact entry-type filter.
        limit: Page size, 1 to 100.
        cursor: Offset cursor from a previous page.
        now: Clock for decay fields; defaults to the current time.

    Returns:
        The requested page.
    """
    _check_limit(limit)
    offset = decode_cursor(cursor)
    loaded = await _load_rows(db, "browse_top", _now(now), entry_type=entry_type)
    if signal == "read":
        rows = _ordered(
            [r for r in loaded.rows if r.last_accessed is not None],
            [(lambda r: r.last_accessed, True)],
        )
    else:
        degrees = await _degrees(db)
        with_degree = [
            replace(r, signal_value=degrees[r.id]) for r in loaded.rows if degrees[r.id] > 0
        ]
        rows = _ordered(
            with_degree,
            [(lambda r: r.signal_value, True), (lambda r: r.created_at, True)],
        )
    return _page(rows, offset, limit, loaded)


async def browse_stale(
    db: Database,
    *,
    kind: StaleKind = "decayed",
    entry_type: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
    now: datetime | None = None,
) -> BrowsePage:
    """List decayed or expired entries.

    Args:
        db: The KB data database.
        kind: 'decayed' (effective confidence below STALE_FILTER_THRESHOLD,
            lowest first) or 'expired' (TTL passed, earliest first).
        entry_type: Optional exact entry-type filter.
        limit: Page size, 1 to 100.
        cursor: Offset cursor from a previous page.
        now: Clock for decay fields; defaults to the current time.

    Returns:
        The requested page.
    """
    _check_limit(limit)
    offset = decode_cursor(cursor)
    loaded = await _load_rows(db, "browse_stale", _now(now), entry_type=entry_type)
    if kind == "expired":
        rows = _ordered(
            [r for r in loaded.rows if r.expired],
            [(lambda r: r.expires_at, False)],
        )
    else:
        rows = _ordered(
            [r for r in loaded.rows if r.effective_confidence < STALE_FILTER_THRESHOLD],
            [(lambda r: r.effective_confidence, False), (lambda r: r.anchor_age_days, True)],
        )
    return _page(rows, offset, limit, loaded)


async def browse_projects(
    db: Database,
    *,
    q: str | None = None,
    limit: int = DEFAULT_PROJECT_PAGE_SIZE,
    cursor: str | None = None,
) -> ProjectPage:
    """List projects with entry counts, most recently active first.

    Args:
        db: The KB data database.
        q: Optional case-insensitive substring filter on project_ref.
        limit: Page size, 1 to 100.
        cursor: Offset cursor from a previous page.

    Returns:
        The requested page of project summaries.
    """
    _check_limit(limit)
    offset = decode_cursor(cursor)
    loaded = await _load_rows(db, "browse_projects", datetime.now(UTC), require_project=True)
    grouped: dict[str, list[BrowseRow]] = {}
    for row in loaded.rows:
        if row.project_ref is not None:
            grouped.setdefault(row.project_ref, []).append(row)

    needle = (q or "").strip().lower()
    summaries: list[ProjectSummary] = []
    for project_ref, rows in grouped.items():
        if needle and needle not in project_ref.lower():
            continue
        type_counts = Counter(r.entry_type for r in rows)
        summaries.append(
            ProjectSummary(
                project_ref=project_ref,
                entry_count=len(rows),
                counts_by_type={
                    t.value: type_counts[t.value] for t in EntryType if type_counts[t.value] >= 1
                },
                last_activity=max(r.updated_at for r in rows),
            )
        )
    summaries.sort(key=lambda s: s.project_ref)
    summaries.sort(key=lambda s: s.last_activity, reverse=True)

    total = len(summaries)
    end = offset + limit
    return ProjectPage(
        items=summaries[offset:end],
        next_cursor=str(end) if end < total else None,
        total=total,
        scanned=loaded.scanned,
        skipped=loaded.skipped,
    )


async def browse_project(
    db: Database,
    project_ref: str,
    *,
    sort: ProjectSort = "newest",
    entry_type: str | None = None,
    q: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
    now: datetime | None = None,
) -> BrowsePage:
    """List one project's entries, mental maps first.

    Args:
        db: The KB data database.
        project_ref: The project to list.
        sort: 'newest' (created_at desc), 'updated' (updated_at desc) or
            'confidence' (effective confidence desc).
        entry_type: Optional exact entry-type filter.
        q: Optional case-insensitive substring filter on id and titles.
        limit: Page size, 1 to 100.
        cursor: Offset cursor from a previous page.
        now: Clock for decay fields; defaults to the current time.

    Returns:
        The requested page.

    Raises:
        ProjectNotFound: When the project has no active entries.
    """
    _check_limit(limit)
    offset = decode_cursor(cursor)
    loaded = await _load_rows(db, "browse_project", _now(now), project_ref=project_ref)
    if loaded.scanned == 0:
        raise ProjectNotFound(project_ref)
    rows = loaded.rows
    if entry_type is not None:
        rows = [r for r in rows if r.entry_type == entry_type]
    needle = (q or "").strip().lower()
    if needle:
        rows = [r for r in rows if needle in f"{r.id} {r.short_title} {r.long_title}".lower()]
    sort_key: Callable[[BrowseRow], Any]
    if sort == "updated":
        sort_key = lambda r: r.updated_at  # noqa: E731
    elif sort == "confidence":
        sort_key = lambda r: r.effective_confidence  # noqa: E731
    else:
        sort_key = lambda r: r.created_at  # noqa: E731
    ordered = _ordered(
        rows,
        [(lambda r: r.entry_type == EntryType.MENTAL_MAP.value, True), (sort_key, True)],
    )
    return _page(ordered, offset, limit, loaded)
