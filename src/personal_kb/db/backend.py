"""Database backend protocol — thin abstraction over async DB connections.

Application code programs against these protocols. Each backend (SQLite,
Postgres, ...) provides a concrete implementation. SQL dialect differences
are handled inside the backend, not in application code.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable


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
        ``personal_kb.search.hybrid``.
        """
        ...

    async def vector_delete(self, entry_id: str) -> None:
        """Delete embedding for an entry."""
        ...

    async def delete_llm_edges(self, entry_id: str) -> None:
        """Delete LLM-enriched graph edges."""
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

    async def notify_maps_changed(self, project_ref: str) -> None:
        """Notify other server instances that a project's maps changed.

        On Postgres this fires ``SELECT pg_notify('kb_maps_changed', $1)``
        with ``project_ref`` as the payload. On SQLite this is a no-op
        (there are no peer instances to notify). Best-effort: failures
        must log and not raise — the local file write has already
        succeeded.
        """
        ...

    async def start_maps_listener(
        self,
        on_change: Callable[[str], Awaitable[None]],
        on_reconnect: Callable[[], Awaitable[None]],
    ) -> Callable[[], Awaitable[None]]:
        """Subscribe to NOTIFY('kb_maps_changed') from peer instances.

        Returns an async teardown callable. Awaiting the teardown stops the
        listener task and closes the dedicated long-lived connection used
        for ``LISTEN`` (Postgres). The caller is responsible for awaiting
        the teardown during lifespan cleanup before ``db.close()``.

        Backend behavior:

        * On Postgres: a dedicated ``asyncpg.connect`` (NOT a pooled
          connection — a held pool conn would starve the pool) is opened
          and ``LISTEN kb_maps_changed`` is issued. ``on_change(payload)``
          is invoked for each NOTIFY payload (the project_ref string).
          On every successful (re)connect of that dedicated connection,
          ``on_reconnect()`` is invoked so the app can trigger a full
          rebuild and re-sync events missed while disconnected. Reconnect
          retries indefinitely with a 5.0s sleep on failure; the loop
          exits cleanly only on cancellation (teardown).
        * On SQLite: no task is started, the callbacks are never invoked,
          and the returned teardown is a no-op closure.

        The backend MUST NOT import ``maps_index_writer`` — the reconnect
        rebuild is driven via the app-supplied ``on_reconnect`` callback
        from ``server.py``. (Layering: the db layer cannot depend on the
        writer/tools layer.)
        """
        ...
