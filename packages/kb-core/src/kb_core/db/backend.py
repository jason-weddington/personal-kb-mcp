"""Database backend protocol — thin abstraction over async DB connections.

Application code programs against these protocols. Each backend (SQLite,
Postgres, ...) provides a concrete implementation. SQL dialect differences
are handled inside the backend, not in application code.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# The WHERE clause that defines "mappable" for the whole map-maintenance
# feature (nightly map maintenance, Loop 1 eligibility): an active,
# non-mental_map entry that belongs to a project. Both backends MUST
# interpolate this constant into their ``map_eligibility_counts`` SQL
# rather than repeat the literal — the clause appears nowhere else in
# either repo today, and the later nightly loop needs the same entry set
# again. Written as a parenthesized single-element string so the physical
# line stays under 100 columns.
MAPPABLE_ENTRY_WHERE_SQL = (
    "is_active = 1 AND entry_type <> 'mental_map' AND project_ref IS NOT NULL"
)


class MapEligibilityCounts(NamedTuple):
    """Per-project counts feeding the map-eligibility classifier.

    A NamedTuple rather than a bare tuple so a transposed column is
    impossible at either backend's implementation site.
    """

    project_ref: str
    mappable: int
    ingested: int
    top_prefix_count: int
    top_prefix: str
    maps: int


@runtime_checkable
class Row(Protocol):
    """A database row supporting both named and positional access."""

    def __getitem__(self, key: str | int) -> Any:
        """Get a column value by name or position."""
        ...

    def keys(self) -> Any:
        """Return column names."""
        ...


@runtime_checkable
class Cursor(Protocol):
    """Async cursor returned by Database.execute()."""

    @property
    def rowcount(self) -> int:
        """Number of rows affected by the last operation."""
        ...

    async def fetchone(self) -> Row | None:
        """Fetch the next row, or None if exhausted."""
        ...

    async def fetchall(self) -> list[Row]:
        """Fetch all remaining rows."""
        ...


@runtime_checkable
class Database(Protocol):
    """Async database backend.

    All application SQL uses ``?`` placeholders and SQLite-flavored syntax.
    Non-SQLite backends translate at execute time (``?`` → ``$N``,
    ``INSERT OR IGNORE`` → ``ON CONFLICT DO NOTHING``, etc.).
    """

    async def execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> Cursor:
        """Execute a single SQL statement and return a cursor."""
        ...

    async def executemany(self, sql: str, params_seq: list[tuple[Any, ...] | list[Any]]) -> None:
        """Execute a SQL statement for each set of parameters."""
        ...

    async def executescript(self, sql: str) -> None:
        """Execute multiple SQL statements (DDL, migrations, VACUUM)."""
        ...

    async def commit(self) -> None:
        """Commit the current transaction."""
        ...

    async def close(self) -> None:
        """Close the database connection."""
        ...

    async def fts_search(
        self,
        query: str,
        *,
        limit: int = 20,
        project_ref: str | None = None,
        entry_type: str | None = None,
        tags: list[str] | None = None,
        contributor: str | None = None,
        team: str | None = None,
    ) -> list[tuple[str, float]]:
        """Full-text search. Returns (entry_id, score) — lower = better."""
        ...

    async def vector_store(self, entry_id: str, embedding: list[float]) -> None:
        """Upsert an embedding vector."""
        ...

    async def vector_search(
        self,
        embedding: list[float],
        limit: int = 20,
        *,
        project_ref: str | None = None,
        entry_type: str | None = None,
        tags: list[str] | None = None,
        contributor: str | None = None,
        team: str | None = None,
    ) -> list[tuple[str, float]]:
        """KNN search. Returns (entry_id, cosine distance).

        Optional metadata filters restrict results to entries matching
        the given project_ref / entry_type / tags / contributor / team
        (and only ``is_active = 1`` entries). Filters are applied at
        the SQL level so that hybrid RRF callers can trust the vector
        leg honors the same scoping as the FTS leg — see
        ``kb_core.search.hybrid``.
        """
        ...

    async def vector_delete(self, entry_id: str) -> None:
        """Delete embedding for an entry."""
        ...

    async def delete_llm_edges(self, entry_id: str) -> None:
        """Delete LLM-enriched graph edges."""
        ...

    async def delete_deterministic_edges(self, entry_id: str) -> None:
        """Delete graph edges for a source that are NOT LLM-enriched.

        Used to clear the deterministic edge set (has_tag, in_project,
        supersedes, references, extracted_from, ...) ahead of a rebuild
        without destroying edges the enricher previously derived.
        """
        ...

    async def map_eligibility_counts(self) -> list[MapEligibilityCounts]:
        """Per-project mappable-entry counts for map-eligibility classification.

        Exactly ONE SQL statement with ZERO bind parameters — a backend-
        internal full scan, so it bypasses the ``?`` translator on
        Postgres. Returns one row per ``project_ref`` that has at least
        one mappable entry (``MAPPABLE_ENTRY_WHERE_SQL``), so
        ``project_ref IS NULL`` entries are excluded and a project with
        only ``mental_map`` entries is absent entirely — such a project
        reports ``maps = 0`` in the synthesized override-only row built
        by ``kb_core.map_eligibility.resolve_eligibility``. The JSON-array
        unnest of ``ingested_files.entry_ids``, the title-prefix
        expression and its tie-break are backend-owned because the two
        dialects diverge (``json_each`` vs
        ``jsonb_array_elements_text(entry_ids::jsonb)``,
        ``instr``/``substr`` vs ``split_part``). ``maps`` is
        reporting-only and affects no verdict.
        """
        ...

    async def vacuum(self) -> str:
        """Backend-specific optimization. Returns status string."""
        ...

    async def next_sequence_value(self) -> int:
        """Atomically get and increment the entry ID sequence. Returns the current value."""
        ...

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Begin a transaction.

        All execute() calls within this context use the same connection
        and are committed atomically. On normal exit: commits.
        On exception: rolls back. Outside a transaction: execute()
        auto-commits as before. Nested calls use savepoints so inner
        failures don't poison the outer transaction.
        """
        yield  # pragma: no cover

    async def apply_schema(self, *, embedding_dim: int = 1024) -> None:
        """Apply all DDL for this backend."""
        ...
