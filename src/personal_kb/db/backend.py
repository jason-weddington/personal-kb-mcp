"""Database backend protocol — thin abstraction over async DB connections.

Application code programs against these protocols. Each backend (SQLite,
Postgres, ...) provides a concrete implementation. SQL dialect differences
are handled inside the backend, not in application code.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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
