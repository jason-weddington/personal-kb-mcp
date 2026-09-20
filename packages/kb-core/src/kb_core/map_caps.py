"""Per-night map-creation caps — the count primitive the map-op write path rests on.

The design's per-night caps: one new map per project, three per KB.
They turn a clustering mistake into a one-map event rather than a corpus-wide one.
They are prompt instructions today.
Enforced server-side, a bug in the loop cannot exceed them.
Nor must the loop carry its own creation count across a crash.

This module ships only the count.
Comparing the returned int against a limit and refusing the write is the web service's item.
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

SQLite needs no pin: its default BINARY collation already orders byte-wise.
That is the same semantics `COLLATE "C"` buys on Postgres.

All SQL uses `?` placeholders only; the Postgres backend rewrites them to `$N` at execute time.
The boundary is INCLUSIVE: an entry created at exactly `since` IS counted.
`contributor` is matched exactly, and `is_active = 1` excludes deactivated maps.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kb_core.db.backend import Database

# The one dialect difference between the two backends' legs: a TEXT column
# that is range-compared needs `COLLATE "C"` on Postgres (see the module
# docstring's column-type audit). SQLite's default BINARY collation already
# orders byte-wise, so its leg stays bare.
_CREATED_AT_GTE_SQLITE = "created_at >= ?"
_CREATED_AT_GTE_POSTGRES = 'created_at COLLATE "C" >= ?'


def _iso(dt: datetime) -> str:
    """Render an aware ``datetime`` as a UTC ISO-8601 string (house convention)."""
    return dt.astimezone(UTC).isoformat()


def _require_aware(since: datetime) -> None:
    """Raise ``ValueError`` if ``since`` is naive — a naive value would compare wrong."""
    if since.tzinfo is None:
        msg = "map_caps: `since` must be timezone-aware"
        raise ValueError(msg)


def _created_at_gte_sql(db: Database) -> str:
    """Return the ``created_at >= ?`` fragment in *db*'s dialect.

    Branches on the concrete backend, the same backend-conditional decision
    :class:`kb_core.knowledge_base.KnowledgeBase` already makes with
    ``isinstance(self._db, SQLiteBackend)``: the Postgres backend is the only
    implementation whose default collation is not byte-wise, so it is the only
    one that needs the pin. A future backend gets the bare fragment, which is
    correct wherever the default ordering is byte-wise; if it is not, that
    backend needs its own branch here.
    """
    # Local import so importing this module does not pay for loading the
    # Postgres backend, mirroring how ``knowledge_base`` imports backends
    # lazily.
    from kb_core.db.postgres_backend import PostgresBackend

    if isinstance(db, PostgresBackend):
        return _CREATED_AT_GTE_POSTGRES
    return _CREATED_AT_GTE_SQLITE


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
        f" AND {_created_at_gte_sql(db)}"
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
