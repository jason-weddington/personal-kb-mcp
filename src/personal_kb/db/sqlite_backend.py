"""SQLite implementation of the Database protocol.

Thin wrapper around aiosqlite.Connection — no SQL translation needed
since application code already uses SQLite-flavored SQL.
"""

from __future__ import annotations

import asyncio
import logging
import os
import struct
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import aiosqlite

    from personal_kb.db.backend import Cursor, Row

# Tracks whether we're inside a transaction() context.
_in_txn: ContextVar[bool] = ContextVar("_in_txn", default=False)
_savepoint_counter: ContextVar[int] = ContextVar("_savepoint_counter", default=0)

logger = logging.getLogger(__name__)


def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a list of floats to a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _escape_fts_query(query: str) -> str:
    """Convert a natural language query to a safe FTS5 query.

    Wraps each token in quotes to avoid FTS5 syntax errors from special chars.
    """
    tokens = query.split()
    if not tokens:
        return ""
    cleaned = [t.replace('"', "") for t in tokens]
    return " ".join(f'"{t}"' for t in cleaned if t)


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

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Begin an atomic transaction.

        Suppresses intermediate commit() calls so the whole block commits
        at the end. Nested calls use SAVEPOINTs for rollback isolation.
        """
        if _in_txn.get(False):
            # Nested — use savepoint
            count = _savepoint_counter.get(0) + 1
            token_c = _savepoint_counter.set(count)
            sp = f"sp_{id(asyncio.current_task())}_{count}"
            await self._conn.execute(f"SAVEPOINT [{sp}]")
            try:
                yield
                await self._conn.execute(f"RELEASE [{sp}]")
            except BaseException:
                await self._conn.execute(f"ROLLBACK TO [{sp}]")
                raise
            finally:
                _savepoint_counter.reset(token_c)
            return

        token = _in_txn.set(True)
        try:
            yield
            await self._conn.commit()
        except BaseException:
            try:
                await self._conn.execute("ROLLBACK")
            except Exception:
                logger.warning("ROLLBACK failed", exc_info=True)
            raise
        finally:
            _in_txn.reset(token)

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
        """Commit the current transaction. Suppressed inside transaction() context."""
        if _in_txn.get(False):
            return  # transaction context manager handles commit
        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        await self._conn.close()

    # -- FTS5 search --

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
        """Full-text search via FTS5 BM25.

        Returns (entry_id, bm25_score) pairs. Lower scores are better
        (FTS5 returns negative scores where more negative = better match).
        """
        fts_query = _escape_fts_query(query)
        if not fts_query:
            return []

        sql = """
            SELECT e.id, bm25(knowledge_fts) as score
            FROM knowledge_fts f
            JOIN knowledge_entries e ON e.rowid = f.rowid
            WHERE knowledge_fts MATCH ?
            AND e.is_active = 1
        """
        params: list[str | int] = [fts_query]

        if project_ref:
            sql += " AND e.project_ref = ?"
            params.append(project_ref)
        if entry_type:
            sql += " AND e.entry_type = ?"
            params.append(entry_type)
        if tags:
            for tag in tags:
                sql += " AND (' ' || e.tags || ' ') LIKE ?"
                params.append(f"% {tag} %")
        if contributor:
            sql += " AND e.contributor = ?"
            params.append(contributor)
        if team:
            sql += " AND e.team = ?"
            params.append(team)

        sql += " ORDER BY score LIMIT ?"
        params.append(limit)

        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [(row[0], row[1]) for row in rows]

    # -- Vector operations (sqlite-vec) --

    async def vector_store(self, entry_id: str, embedding: list[float]) -> None:
        """Upsert an embedding in the vec0 table."""
        blob = _serialize_f32(embedding)
        # vec0 doesn't support ON CONFLICT — delete then insert
        await self._conn.execute("DELETE FROM knowledge_vec WHERE entry_id = ?", (entry_id,))
        await self._conn.execute(
            "INSERT INTO knowledge_vec (entry_id, embedding) VALUES (?, ?)",
            (entry_id, blob),
        )

    async def vector_search(
        self, embedding: list[float], limit: int = 20
    ) -> list[tuple[str, float]]:
        """KNN search via sqlite-vec cosine distance. Returns (entry_id, distance) pairs."""
        blob = _serialize_f32(embedding)
        cursor = await self._conn.execute(
            """SELECT entry_id, distance
            FROM knowledge_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?""",
            (blob, limit),
        )
        rows = await cursor.fetchall()
        return [(row[0], row[1]) for row in rows]

    async def vector_delete(self, entry_id: str) -> None:
        """Delete embedding for an entry."""
        await self._conn.execute("DELETE FROM knowledge_vec WHERE entry_id = ?", (entry_id,))

    # -- Graph helpers --

    async def delete_llm_edges(self, entry_id: str) -> None:
        """Remove all LLM-derived edges for a given source entry."""
        await self._conn.execute(
            "DELETE FROM graph_edges"
            " WHERE source = ? AND json_extract(properties, '$.source') = 'llm'",
            (entry_id,),
        )

    # -- Maintenance --

    async def vacuum(self) -> str:
        """Run PRAGMA optimize and VACUUM. Returns status string."""
        await self._conn.execute("PRAGMA optimize")
        await self._conn.executescript("VACUUM;")

        size_info = ""
        cursor = await self._conn.execute("PRAGMA database_list")
        db_row = await cursor.fetchone()
        if db_row and db_row[2]:
            try:
                size = os.path.getsize(db_row[2])
                if size < 1024 * 1024:
                    size_info = f" Database size: {size / 1024:.1f} KB"
                else:
                    size_info = f" Database size: {size / (1024 * 1024):.1f} MB"
            except OSError:
                pass

        return f"Vacuum complete.{size_info}"

    # -- Sequence --

    async def next_sequence_value(self) -> int:
        """Get and increment the entry ID sequence. Safe for SQLite (single-writer)."""
        cursor = await self._conn.execute("SELECT next_id FROM entry_id_seq")
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError("entry_id_seq table is empty")
        next_id: int = row[0]
        await self._conn.execute("UPDATE entry_id_seq SET next_id = ?", (next_id + 1,))
        return next_id

    # -- Schema --

    async def apply_schema(self, *, embedding_dim: int = 1024) -> None:
        """Apply all SQLite DDL: tables, FTS5, graph, ingest, vec0, migrations."""
        from personal_kb.db.schema import (
            _migrate_v2_multi_user,
            apply_graph_schema,
            apply_ingest_schema,
            apply_schema,
            apply_vec_schema,
        )

        await apply_schema(self)
        await apply_graph_schema(self)
        await apply_ingest_schema(self)

        # vec0 may not be available if sqlite-vec failed to load
        try:
            await apply_vec_schema(self, dim=embedding_dim)
        except Exception:
            logger.warning("sqlite-vec schema not applied — vector search disabled")

        # Multi-user migration (must run after all tables exist)
        await _migrate_v2_multi_user(self)
        await self.commit()
