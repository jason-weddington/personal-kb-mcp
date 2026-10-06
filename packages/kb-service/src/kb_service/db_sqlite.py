"""SQLite implementation of the ``DbPool`` protocol for local (no-auth) mode.

When ``KB_SERVICE_DATABASE_URL`` is unset and ``KB_AUTH_MODE=none``, the
service tables (app_config, listener_decisions, whisper_telemetry, ...) live in
a local SQLite file instead of Postgres. This module adapts aiosqlite to the
small asyncpg-shaped surface the service code uses:

* ``fetch`` / ``fetchrow`` / ``fetchval`` / ``execute`` on both the pool and an
  acquired connection, plus ``pool.acquire()`` and ``conn.transaction()``.
* asyncpg ``$N`` placeholders are rewritten to SQLite's numbered ``?N`` form,
  which keeps positional binding AND reuse of the same ``$N`` (the telemetry
  upsert binds ``$8`` twice). Postgres-only ``FOR UPDATE`` row locks are
  dropped (SQLite locks the whole database for a write transaction anyway).
* Rows come back as plain ``dict`` objects, so ``row["col"]`` and
  ``row_to_dict(row)`` both work exactly as with an asyncpg ``Record``.
* ``execute`` returns an asyncpg-style status string (``"DELETE 1"``,
  ``"UPDATE 0"``, ``"INSERT 0 1"``) so callers that parse it keep working.

The daemon is single-process but CLI commands may open the same file, so the
connection runs in WAL mode with a busy timeout. A single aiosqlite connection
is shared and guarded by an ``asyncio.Lock`` — ``acquire()`` holds the lock for
the whole ``async with`` block so a ``transaction()`` cannot interleave with
another request's statements.
"""

import asyncio
import re
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

_PLACEHOLDER_RE = re.compile(r"\$(\d+)")
_FOR_UPDATE_RE = re.compile(r"\s+FOR\s+UPDATE\b", re.IGNORECASE)

BUSY_TIMEOUT_MS = 5000


def translate_sql(sql: str) -> str:
    """Rewrite asyncpg-flavoured SQL into SQLite-flavoured SQL.

    ``$N`` becomes ``?N`` (SQLite numbered parameters bind positionally from
    the same args tuple, so reuse of one ``$N`` keeps working) and any
    ``FOR UPDATE`` clause is removed.
    """
    return _FOR_UPDATE_RE.sub("", _PLACEHOLDER_RE.sub(r"?\1", sql))


def _status(sql: str, rowcount: int) -> str:
    """Build an asyncpg-style command status string for *sql*."""
    words = sql.split(None, 1)
    verb = words[0].upper() if words else ""
    count = max(rowcount, 0)
    if verb == "INSERT":
        return f"INSERT 0 {count}"
    if verb in {"UPDATE", "DELETE"}:
        return f"{verb} {count}"
    return verb


class SqliteConnection:
    """asyncpg-``Connection``-shaped wrapper over one aiosqlite connection."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        """Wrap an open aiosqlite connection (in autocommit mode)."""
        self._conn = conn

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        """Fetch all matching rows as plain dicts."""
        async with self._conn.execute(translate_sql(sql), args) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        """Fetch the first matching row as a plain dict, or None."""
        async with self._conn.execute(translate_sql(sql), args) as cursor:
            row = await cursor.fetchone()
        return None if row is None else dict(row)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        """Fetch the first column of the first row, or None."""
        async with self._conn.execute(translate_sql(sql), args) as cursor:
            row = await cursor.fetchone()
        return None if row is None else row[0]

    async def execute(self, sql: str, *args: Any) -> str:
        """Execute a statement and return an asyncpg-style status string."""
        async with self._conn.execute(translate_sql(sql), args) as cursor:
            rowcount = cursor.rowcount
        return _status(sql, rowcount)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Run the block in one transaction; roll back if it raises."""
        await self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            await self._conn.execute("ROLLBACK")
            raise
        await self._conn.execute("COMMIT")


class SqlitePool:
    """asyncpg-``Pool``-shaped wrapper over a single shared SQLite connection."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        """Wrap an already-open aiosqlite connection. Use :func:`open` instead."""
        self._raw = conn
        self._conn = SqliteConnection(conn)
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, path: Path) -> "SqlitePool":
        """Open (creating if needed) the SQLite file at *path*.

        The parent directory is created, autocommit is enabled (explicit
        ``transaction()`` blocks issue their own BEGIN/COMMIT), and the
        connection is switched to WAL with a busy timeout so a concurrent CLI
        process can share the file with the daemon.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        await conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")
        return cls(conn)

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        """Fetch all matching rows."""
        async with self._lock:
            return await self._conn.fetch(sql, *args)

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        """Fetch a single row, or None."""
        async with self._lock:
            return await self._conn.fetchrow(sql, *args)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        """Fetch a single value, or None."""
        async with self._lock:
            return await self._conn.fetchval(sql, *args)

    async def execute(self, sql: str, *args: Any) -> str:
        """Execute a SQL statement."""
        async with self._lock:
            return await self._conn.execute(sql, *args)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[SqliteConnection]:
        """Hold the shared connection exclusively for the ``async with`` block."""
        async with self._lock:
            yield self._conn

    async def close(self) -> None:
        """Close the underlying connection."""
        await self._raw.close()
