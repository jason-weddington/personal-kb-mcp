"""PostgreSQL implementation of the Database protocol.

Uses asyncpg for async access, pgvector for embeddings, and tsvector/GIN
for full-text search. All application SQL uses ``?`` placeholders — this
backend translates them to ``$N`` at execute time.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg

    from personal_kb.db.backend import Cursor, Row

logger = logging.getLogger(__name__)

# Pre-compiled regex for placeholder translation
_PLACEHOLDER_RE = re.compile(r"\?")


def _translate_placeholders(sql: str) -> str:
    """Convert ``?`` placeholders to ``$1, $2, ...`` for asyncpg."""
    counter = 0

    def _replace(_match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        return f"${counter}"

    return _PLACEHOLDER_RE.sub(_replace, sql)


class PostgresRow:
    """Wraps asyncpg.Record to satisfy the Row protocol."""

    def __init__(self, record: asyncpg.Record) -> None:
        """Initialize with an asyncpg Record."""
        self._record = record

    def __getitem__(self, key: str | int) -> Any:
        """Get a column value by name or position."""
        return self._record[key]

    def keys(self) -> list[str]:
        """Return column names."""
        return list(self._record.keys())


class PostgresCursor:
    """Wraps a list of asyncpg.Record as a Cursor.

    asyncpg returns results eagerly — there's no server-side cursor for
    simple queries. This wraps the result list to match the Cursor protocol.
    """

    def __init__(self, rows: list[asyncpg.Record], status: str | None = None) -> None:
        """Initialize with result rows and optional status string."""
        self._rows = rows
        self._index = 0
        self._rowcount = self._parse_rowcount(status)

    @property
    def rowcount(self) -> int:
        """Number of rows affected by the last operation."""
        return self._rowcount

    async def fetchone(self) -> Row | None:
        """Fetch the next row, or None if exhausted."""
        if self._index >= len(self._rows):
            return None
        row = PostgresRow(self._rows[self._index])
        self._index += 1
        return row

    async def fetchall(self) -> list[Row]:
        """Fetch all remaining rows."""
        remaining: list[Row] = [PostgresRow(r) for r in self._rows[self._index :]]
        self._index = len(self._rows)
        return remaining

    @staticmethod
    def _parse_rowcount(status: str | None) -> int:
        """Parse affected row count from asyncpg status string.

        Examples: "INSERT 0 1" → 1, "UPDATE 3" → 3, "DELETE 0" → 0.
        """
        if not status:
            return -1
        parts = status.split()
        if len(parts) >= 2:
            try:
                return int(parts[-1])
            except ValueError:
                pass
        return -1


class PostgresBackend:
    """PostgreSQL implementation of the Database protocol.

    Each ``execute()`` call acquires a connection from the pool, translates
    ``?`` → ``$N`` placeholders, and releases the connection after.
    ``commit()`` is a no-op — asyncpg auto-commits each statement.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize with an asyncpg connection pool."""
        self._pool = pool

    @classmethod
    async def create(cls, url: str) -> PostgresBackend:
        """Create a PostgresBackend from a connection URL."""
        import asyncpg as _asyncpg

        pool = await _asyncpg.create_pool(url, min_size=2, max_size=10)
        return cls(pool)

    async def execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> Cursor:
        """Execute a single SQL statement and return a cursor."""
        pg_sql = _translate_placeholders(sql)
        async with self._pool.acquire() as conn:
            # asyncpg.fetch returns list of Records for SELECT
            # asyncpg.execute returns status string for INSERT/UPDATE/DELETE
            stmt = await conn.prepare(pg_sql)
            if stmt.get_attributes():
                # Query returns rows
                rows = await conn.fetch(pg_sql, *params)
                return PostgresCursor(rows)
            else:
                # DML — returns status string
                status = await conn.execute(pg_sql, *params)
                return PostgresCursor([], status=status)

    async def executemany(self, sql: str, params_seq: list[tuple[Any, ...] | list[Any]]) -> None:
        """Execute a SQL statement for each set of parameters."""
        pg_sql = _translate_placeholders(sql)
        async with self._pool.acquire() as conn:
            await conn.executemany(pg_sql, params_seq)

    async def executescript(self, sql: str) -> None:
        """Execute multiple SQL statements."""
        async with self._pool.acquire() as conn:
            await conn.execute(sql)

    async def commit(self) -> None:
        """No-op — asyncpg auto-commits each statement."""

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()

    # -- FTS (tsvector + GIN) --

    async def fts_search(
        self,
        query: str,
        *,
        limit: int = 20,
        project_ref: str | None = None,
        entry_type: str | None = None,
        tags: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Full-text search via tsvector + ts_rank_cd.

        Returns (entry_id, score) pairs. Scores are negated so that
        lower = better, matching the FTS5/bm25 convention.
        """
        sql = """
            SELECT e.id, -ts_rank_cd(e.search_vector, plainto_tsquery('english', $1)) as score
            FROM knowledge_entries e
            WHERE e.search_vector @@ plainto_tsquery('english', $1)
            AND e.is_active = 1
        """
        params: list[Any] = [query]
        param_idx = 2

        if project_ref:
            sql += f" AND e.project_ref = ${param_idx}"
            params.append(project_ref)
            param_idx += 1
        if entry_type:
            sql += f" AND e.entry_type = ${param_idx}"
            params.append(entry_type)
            param_idx += 1
        if tags:
            for tag in tags:
                sql += f" AND (' ' || e.tags || ' ') LIKE ${param_idx}"
                params.append(f"% {tag} %")
                param_idx += 1

        sql += f" ORDER BY score LIMIT ${param_idx}"
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [(row["id"], row["score"]) for row in rows]

    # -- Vector operations (pgvector) --

    async def vector_store(self, entry_id: str, embedding: list[float]) -> None:
        """Upsert an embedding vector."""
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO knowledge_vec (entry_id, embedding)
                   VALUES ($1, $2::vector)
                   ON CONFLICT (entry_id) DO UPDATE SET embedding = EXCLUDED.embedding""",
                entry_id,
                vec_str,
            )

    async def vector_search(
        self, embedding: list[float], limit: int = 20
    ) -> list[tuple[str, float]]:
        """KNN search via pgvector L2 distance. Returns (entry_id, distance)."""
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT entry_id, embedding <-> $1::vector as distance
                   FROM knowledge_vec
                   ORDER BY distance
                   LIMIT $2""",
                vec_str,
                limit,
            )
            return [(row["entry_id"], row["distance"]) for row in rows]

    async def vector_delete(self, entry_id: str) -> None:
        """Delete embedding for an entry."""
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM knowledge_vec WHERE entry_id = $1", entry_id)

    # -- Graph helpers --

    async def delete_llm_edges(self, entry_id: str) -> None:
        """Remove all LLM-derived edges for a given source entry."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM graph_edges WHERE source = $1 AND properties->>'source' = 'llm'",
                entry_id,
            )

    # -- Maintenance --

    async def vacuum(self) -> str:
        """Run ANALYZE (Postgres equivalent of PRAGMA optimize + VACUUM)."""
        async with self._pool.acquire() as conn:
            await conn.execute("ANALYZE")
        return "Vacuum complete (ANALYZE)."

    # -- Schema --

    async def apply_schema(self, *, embedding_dim: int = 1024) -> None:
        """Apply all PostgreSQL DDL."""
        async with self._pool.acquire() as conn:
            # Enable pgvector
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id TEXT PRIMARY KEY,
                    project_ref TEXT,
                    short_title TEXT NOT NULL,
                    long_title TEXT NOT NULL,
                    knowledge_details TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    source_context TEXT,
                    confidence_level REAL NOT NULL DEFAULT 0.9,
                    tags TEXT NOT NULL DEFAULT '[]',
                    hints TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed TEXT,
                    superseded_by TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    has_embedding INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    search_vector tsvector
                )
            """)

            # Indexes
            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_entries_project ON knowledge_entries(project_ref)",
                "CREATE INDEX IF NOT EXISTS idx_entries_type ON knowledge_entries(entry_type)",
                "CREATE INDEX IF NOT EXISTS idx_entries_active ON knowledge_entries(is_active)",
                "CREATE INDEX IF NOT EXISTS idx_entries_fts"
                " ON knowledge_entries USING gin(search_vector)",
            ]:
                await conn.execute(idx_sql)

            # tsvector trigger
            await conn.execute("""
                CREATE OR REPLACE FUNCTION knowledge_entries_search_trigger() RETURNS trigger AS $$
                BEGIN
                    NEW.search_vector :=
                        setweight(to_tsvector('english', COALESCE(NEW.short_title, '')), 'A') ||
                        setweight(to_tsvector('english', COALESCE(NEW.long_title, '')), 'B') ||
                        setweight(to_tsvector('english',
                            COALESCE(NEW.knowledge_details, '')), 'C') ||
                        setweight(to_tsvector('english', COALESCE(NEW.tags, '')), 'D');
                    RETURN NEW;
                END
                $$ LANGUAGE plpgsql
            """)

            # Drop and recreate trigger to ensure it's current
            await conn.execute("DROP TRIGGER IF EXISTS tsvector_update ON knowledge_entries")
            await conn.execute("""
                CREATE TRIGGER tsvector_update BEFORE INSERT OR UPDATE
                ON knowledge_entries FOR EACH ROW
                EXECUTE FUNCTION knowledge_entries_search_trigger()
            """)

            # Entry versions
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entry_versions (
                    id SERIAL PRIMARY KEY,
                    entry_id TEXT NOT NULL REFERENCES knowledge_entries(id),
                    version_number INTEGER NOT NULL,
                    knowledge_details TEXT NOT NULL,
                    change_reason TEXT,
                    confidence_level REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(entry_id, version_number)
                )
            """)

            # Entry ID sequence
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entry_id_seq (
                    next_id INTEGER NOT NULL DEFAULT 1
                )
            """)
            await conn.execute("""
                INSERT INTO entry_id_seq (next_id)
                SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM entry_id_seq)
            """)

            # Graph tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    properties TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
            """)
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_type ON graph_nodes(node_type)"
            )

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id SERIAL PRIMARY KEY,
                    source TEXT NOT NULL REFERENCES graph_nodes(node_id),
                    target TEXT NOT NULL REFERENCES graph_nodes(node_id),
                    edge_type TEXT NOT NULL,
                    properties TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE(source, target, edge_type)
                )
            """)
            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source)",
                "CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target)",
                "CREATE INDEX IF NOT EXISTS idx_edges_type ON graph_edges(edge_type)",
            ]:
                await conn.execute(idx_sql)

            # Ingest table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ingested_files (
                    id SERIAL PRIMARY KEY,
                    relative_path TEXT NOT NULL UNIQUE,
                    content_hash TEXT NOT NULL,
                    note_node_id TEXT NOT NULL,
                    entry_ids TEXT NOT NULL DEFAULT '[]',
                    summary TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_extension TEXT NOT NULL,
                    project_ref TEXT,
                    redactions TEXT NOT NULL DEFAULT '[]',
                    ingested_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                )
            """)

            # Vector table (pgvector)
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS knowledge_vec (
                    entry_id TEXT PRIMARY KEY,
                    embedding vector({embedding_dim})
                )
            """)

            # Schema version init
            row = await conn.fetchrow("SELECT version FROM schema_version")
            if row is None:
                await conn.execute("INSERT INTO schema_version (version) VALUES ($1)", 1)
