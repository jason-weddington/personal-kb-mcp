"""SQLite implementation of the Database protocol.

Thin wrapper around aiosqlite.Connection — no SQL translation needed
since application code already uses SQLite-flavored SQL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiosqlite

    from personal_kb.db.backend import Cursor, Row


class SQLiteCursor:
    """Wraps aiosqlite.Cursor to satisfy the Cursor protocol."""

    def __init__(self, cursor: aiosqlite.Cursor) -> None:
        """Initialize with an aiosqlite cursor."""
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        """Number of rows affected by the last operation."""
        rc = self._cursor.rowcount
        return rc if rc is not None else -1

    async def fetchone(self) -> Row | None:
        """Fetch the next row, or None if exhausted."""
        return await self._cursor.fetchone()

    async def fetchall(self) -> list[Row]:
        """Fetch all remaining rows."""
        return list(await self._cursor.fetchall())


class SQLiteBackend:
    """SQLite implementation of the Database protocol.

    Passes all calls through to the underlying aiosqlite.Connection.
    The raw connection is exposed as ``_conn`` for SQLite-specific
    operations (extension loading, PRAGMA, etc.) that only run during
    connection setup.
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        """Initialize with an aiosqlite connection."""
        self._conn = conn

    async def execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> Cursor:
        """Execute a single SQL statement and return a cursor."""
        cursor = await self._conn.execute(sql, params)
        return SQLiteCursor(cursor)

    async def executemany(self, sql: str, params_seq: list[tuple[Any, ...] | list[Any]]) -> None:
        """Execute a SQL statement for each set of parameters."""
        await self._conn.executemany(sql, params_seq)

    async def executescript(self, sql: str) -> None:
        """Execute multiple SQL statements (DDL, migrations, VACUUM)."""
        await self._conn.executescript(sql)

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        await self._conn.close()
