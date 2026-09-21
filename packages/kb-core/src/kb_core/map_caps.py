"""Per-night map-creation caps — the count primitive the map-op write path rests on.

The design's per-night caps: one new map per project, three per KB.
They turn a clustering mistake into a one-map event rather than a corpus-wide one.
They are prompt instructions today.
Enforced server-side, a bug in the loop cannot exceed them.
Nor must the loop carry its own creation count across a crash.

This module ships the per-night count and the per-project map-write summary.
Comparing the returned int against a limit and refusing the write is the web service's item.
Ranking the summary's rows (never-mapped first, then oldest write first) is the consumer's item.
Pinned in `docs/somnus-functional-spec.md` at "The write path — POST /api/kb/map-op".

The web service resolves kb-core from `rev = main` over git+ssh and cannot see feature-branch code.
So kb-core lands first, then the lead bumps the lock, then the endpoint item runs.

Column-type audit, read from `kb_core.db.schema` before the WHERE clause was written.
`knowledge_entries.created_at` is `TEXT NOT NULL` in `SCHEMA_SQL`.
It is `created_at TEXT NOT NULL` in `postgres_backend._apply_schema_locked` too.
So the kb-core data DB stores instants as ISO-8601 strings on BOTH backends.
The service auth DB is a different schema and was not consulted for this.

Because that column is TEXT, not a real timestamp type, the `>=` is a string comparison.
So the Postgres leg carries `COLLATE "C"` on the `created_at` term.
That is this repo's pinned convention for a TEXT column that is ordered or range-compared.
(See the `embedding_retry_queue.next_attempt_at` pin and the `map_eligibility_counts` ORDER BYs.)
glibc's default collation does not order these strings byte-wise.

`updated_at` was re-verified against `kb_core.db.schema` for the summary query below, not assumed.
It is `updated_at TEXT NOT NULL` in `SCHEMA_SQL` and in `postgres_backend._apply_schema_locked` too.
So `MAX(updated_at)` is an ordering operation over ISO-8601 strings, and the same pin applies.
The `ORDER BY project_ref` term is pinned for the same reason — `project_ref` is TEXT as well.

SQLite needs no pin: its default BINARY collation already orders byte-wise.
That is the same semantics `COLLATE "C"` buys on Postgres.

All SQL uses `?` placeholders only; the Postgres backend rewrites them to `$N` at execute time.
The summary query below takes no parameters, so nothing is ever interpolated into its string.
The count's boundary is INCLUSIVE: an entry created at exactly `since` IS counted.
`contributor` is matched exactly, and `is_active = 1` excludes deactivated maps.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kb_core.db.backend import Database


# The one dialect difference between the two backends' legs: a TEXT column
# that is ordered or range-compared needs `COLLATE "C"` on Postgres (see the
# module docstring's column-type audit). SQLite's default BINARY collation
# already orders byte-wise, so its legs stay bare. One branch serves every
# pinned term in this module — `created_at >= ?`, `MAX(updated_at)` and the
# `ORDER BY project_ref` — so a third backend adds one branch here, not a
# second mechanism somewhere else.
def _byte_ordered_text(db: Database, term: str) -> str:
    """Return *term* with the byte-wise ordering pin *db*'s dialect needs.

    Branches on the concrete backend, the same backend-conditional decision
    :class:`kb_core.knowledge_base.KnowledgeBase` already makes with
    ``isinstance(self._db, SQLiteBackend)``: the Postgres backend is the only
    implementation whose default collation is not byte-wise, so it is the only
    one that needs the pin. A future backend gets the bare term, which is
    correct wherever the default ordering is byte-wise; if it is not, that
    backend needs its own branch here.
    """
    # Local import so importing this module does not pay for loading the
    # Postgres backend, mirroring how ``knowledge_base`` imports backends
    # lazily.
    from kb_core.db.postgres_backend import PostgresBackend

    if isinstance(db, PostgresBackend):
        return f'{term} COLLATE "C"'
    return term


def _parse_stored_instant(raw: str) -> datetime:
    """Parse a stored ISO-8601 instant, treating a NAIVE value as UTC.

    The naive case is not hypothetical and not a legacy-schema worry — it is in
    the live corpus today. ``db/queries.py`` renders ``entry.updated_at`` with
    ``.isoformat()``, so an entry whose timestamp reached the store naive is
    written WITHOUT an offset, and the personal KB currently holds one such map
    project (``2026-06-15T10:15:16``, no suffix) along 27 maps that all carry
    ``+00:00``.

    Returning that row naive would hand the caller a list it cannot sort:
    comparing a naive and an aware ``datetime`` raises ``TypeError``, so the
    worklist ranking would crash on real data while every fixture-backed test
    passed. Attaching UTC is the correct repair rather than a papering-over —
    kb-core writes instants in UTC, so a missing offset means UTC was dropped
    on the way in, not that the instant was local.
    """
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iso(dt: datetime) -> str:
    """Render an aware ``datetime`` as a UTC ISO-8601 string (house convention)."""
    return dt.astimezone(UTC).isoformat()


def _require_aware(since: datetime) -> None:
    """Raise ``ValueError`` if ``since`` is naive — a naive value would compare wrong."""
    if since.tzinfo is None:
        msg = "map_caps: `since` must be timezone-aware"
        raise ValueError(msg)


async def count_maps_created_since(
    db: Database,
    *,
    contributor: str,
    since: datetime,
    project_ref: str | None = None,
) -> int:
    """Count ACTIVE ``mental_map`` entries ``contributor`` created at or after ``since``.

    One SQL ``COUNT(*)`` over existing ``knowledge_entries`` columns.
    No new table, no new index, no state for the caller to carry across a crash.

    It matches ``entry_type = 'mental_map'``, so a same-instant ``factual_reference`` counts zero.
    It matches ``contributor`` exactly — no prefix or substring matching.
    It matches ``is_active = 1``, so a deactivated map consumes no cap slot.

    ``created_at`` is a TEXT column of ISO-8601 strings on both backends.
    The inclusive ``>=`` is therefore a string comparison, not a timestamp one.
    The Postgres leg carries ``COLLATE "C"`` on that term.
    That is this repo's convention for a range-compared TEXT column.
    The SQLite leg stays bare because its default BINARY collation already orders byte-wise.

    ``since`` must be timezone-aware and is normalized to UTC ISO-8601.
    That is the normalization the store applies when it writes ``created_at``.
    A naive ``since`` raises ``ValueError`` rather than comparing a differently-shaped string.

    ``project_ref`` scopes the count to one project when given.
    ``None`` leaves it unscoped, across every project — the two calls the caps need.
    A ``project_ref`` with no qualifying maps returns 0.

    Args:
        db: Any :class:`kb_core.db.backend.Database` — SQLite or Postgres.
        contributor: Exact contributor string to count (the machine principal).
        since: Timezone-aware inclusive lower bound on ``created_at``.
        project_ref: Project to scope to, or ``None`` to count across projects.

    Returns:
        The number of active ``mental_map`` entries by ``contributor`` with ``created_at >= since``.
        Scoped to ``project_ref`` when one is given, unscoped when it is ``None``.
    """
    _require_aware(since)

    # S608 fires on the string construction below, but every value is a
    # bound `?` — the only interpolation is a module constant fragment.
    sql = (
        "SELECT COUNT(*) AS n FROM knowledge_entries"  # noqa: S608
        " WHERE is_active = 1"
        " AND entry_type = 'mental_map'"
        " AND contributor = ?"
        f" AND {_byte_ordered_text(db, 'created_at')} >= ?"
    )
    params: list[str] = [contributor, _iso(since)]
    if project_ref is not None:
        sql += " AND project_ref = ?"
        params.append(project_ref)

    cursor = await db.execute(sql, tuple(params))
    row = await cursor.fetchone()
    if row is None:
        return 0
    return int(row["n"])


@dataclasses.dataclass(frozen=True)
class MapWriteSummary:
    """One project's map-write facts, as the somnus worklist ranking needs them.

    ``project_ref`` is the project's own ref, ``map_count`` its number of
    active ``mental_map`` entries, and ``latest_map_written_at`` the most
    recent ``updated_at`` among them.

    These are FACTS, not a worklist: the ranking — never-mapped projects
    first, then oldest last write first, then ``project_ref`` ascending — is
    the consumer's, and a copy of it here would be one more thing to drift.
    """

    project_ref: str
    map_count: int
    latest_map_written_at: datetime


async def map_write_summary(db: Database) -> list[MapWriteSummary]:
    """Return per-project active-map facts, ordered by ``project_ref`` ascending.

    One SQL ``GROUP BY`` over existing ``knowledge_entries`` columns.
    No new table, no new index, no state for the caller to carry across a crash.

    It matches ``entry_type = 'mental_map'`` and ``is_active = 1``, so a
    ``factual_reference`` and a deactivated map contribute to NEITHER field.
    It does NOT filter on ``contributor``: a map written by anyone still
    makes that project's maps fresh, which is exactly what the count above
    must not do — the two queries measure different things.

    ``latest_map_written_at`` is ``MAX(updated_at)`` over those maps, NOT
    ``MAX(created_at)``: a map that received a pointer today has been worked
    today, however long ago it was created, and ranking it stale would send
    the loop straight back to the project it just finished.

    A project with ZERO active maps is ABSENT from the result, not present
    with ``map_count = 0``. The consumer reads absence as never-mapped and
    sorts those projects first; do not "fix" this into a LEFT JOIN over every
    project in the KB — that would hand the caller rows it must filter away
    to recover what the SQL can simply not return.

    Entries with a NULL or EMPTY ``project_ref`` are excluded — they belong
    to no project and cannot be ranked as one.

    ``updated_at`` is a TEXT column of ISO-8601 strings on both backends
    (re-verified against ``kb_core.db.schema``, not assumed to match
    ``created_at``). A ``MAX()`` over TEXT is an ordering operation, so the
    Postgres leg pins ``COLLATE "C"`` on that term through the same
    dialect-branching helper the count uses; the SQLite leg stays bare.
    The ``ORDER BY project_ref`` term is pinned for the same reason.
    The returned instant is always an AWARE ``datetime`` in UTC, including for
    the rows the live corpus stores without an offset — see
    :func:`_parse_stored_instant`. Every element being aware is what makes the
    list sortable at all, so it is a property of the contract rather than an
    implementation detail.

    Rows come back ordered by ``project_ref`` ascending so the function is
    deterministic. The RANKING — never-mapped first, then oldest write
    first — belongs to the consumer and is NOT applied here.

    Args:
        db: Any :class:`kb_core.db.backend.Database` — SQLite or Postgres.

    Returns:
        One :class:`MapWriteSummary` per project that has at least one
        active ``mental_map`` entry, ordered by ``project_ref`` ascending.
    """
    # S608 fires on the string construction below, but the query binds NO
    # parameters at all — the only interpolation is a module constant fragment.
    sql = (
        "SELECT project_ref, COUNT(*) AS map_count,"  # noqa: S608
        f" MAX({_byte_ordered_text(db, 'updated_at')}) AS latest_map_written_at"
        " FROM knowledge_entries"
        " WHERE is_active = 1"
        " AND entry_type = 'mental_map'"
        " AND project_ref IS NOT NULL"
        " AND project_ref <> ''"
        " GROUP BY project_ref"
        f" ORDER BY {_byte_ordered_text(db, 'project_ref')} ASC"
    )

    cursor = await db.execute(sql)
    rows = await cursor.fetchall()
    return [
        MapWriteSummary(
            project_ref=str(row["project_ref"]),
            map_count=int(row["map_count"]),
            latest_map_written_at=_parse_stored_instant(str(row["latest_map_written_at"])),
        )
        for row in rows
    ]
