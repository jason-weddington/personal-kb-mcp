"""PostgreSQL implementation of the Database protocol.

Uses asyncpg for async access, pgvector for embeddings, and tsvector/GIN
for full-text search. All application SQL uses ``?`` placeholders — this
backend translates them to ``$N`` at execute time.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import ssl as ssl_module
    from collections.abc import AsyncIterator, Callable

    import asyncpg

    from personal_kb.db.backend import Cursor, Row

# When set, all execute() calls route to this connection instead of the pool.
_txn_conn: ContextVar[asyncpg.Connection | None] = ContextVar("_txn_conn", default=None)

logger = logging.getLogger(__name__)


def _translate_placeholders(sql: str) -> str:
    """Convert ``?`` placeholders to ``$1, $2, ...`` for asyncpg.

    Skips ``?`` inside SQL string literals (single-quoted). Handles
    escaped quotes (``''``) correctly.
    """
    if "?" not in sql:
        return sql

    out: list[str] = []
    counter = 0
    in_quote = False
    i = 0
    length = len(sql)

    while i < length:
        ch = sql[i]
        if ch == "'":
            if in_quote and i + 1 < length and sql[i + 1] == "'":
                # Escaped quote ('') — emit both, stay in quote
                out.append("''")
                i += 2
                continue
            in_quote = not in_quote
            out.append(ch)
        elif ch == "?" and not in_quote:
            counter += 1
            out.append(f"${counter}")
        else:
            out.append(ch)
        i += 1

    return "".join(out)


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

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[asyncpg.Connection]:
        """Get the transaction connection if inside a transaction, else acquire from pool."""
        existing = _txn_conn.get(None)
        if existing is not None:
            yield existing
        else:
            async with self._pool.acquire() as conn:
                yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Begin an atomic transaction.

        All execute() calls within this context use the same pooled connection.
        Nested calls create savepoints (asyncpg does this automatically).
        """
        existing = _txn_conn.get(None)
        if existing is not None:
            # Nested — asyncpg creates a savepoint automatically
            async with existing.transaction():
                yield
            return

        async with self._pool.acquire() as conn:
            token = _txn_conn.set(conn)
            try:
                async with conn.transaction():
                    yield
            finally:
                _txn_conn.reset(token)

    @classmethod
    async def create(
        cls,
        url: str,
        *,
        pool_min: int = 2,
        pool_max: int = 10,
        password: Callable[[], str] | None = None,
        ssl: ssl_module.SSLContext | bool | None = None,
    ) -> PostgresBackend:
        """Create a PostgresBackend from a connection URL."""
        import asyncpg as _asyncpg

        kwargs: dict[str, Any] = {"min_size": pool_min, "max_size": pool_max}
        if password is not None:
            kwargs["password"] = password
        if ssl is not None:
            kwargs["ssl"] = ssl
        pool = await _asyncpg.create_pool(url, **kwargs)
        return cls(pool)

    async def execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> Cursor:
        """Execute a single SQL statement and return a cursor."""
        pg_sql = _translate_placeholders(sql)
        async with self._conn() as conn:
            stmt = await conn.prepare(pg_sql)
            if stmt.get_attributes():
                rows = await conn.fetch(pg_sql, *params)
                return PostgresCursor(rows)
            else:
                status = await conn.execute(pg_sql, *params)
                return PostgresCursor([], status=status)

    async def executemany(self, sql: str, params_seq: list[tuple[Any, ...] | list[Any]]) -> None:
        """Execute a SQL statement for each set of parameters."""
        pg_sql = _translate_placeholders(sql)
        async with self._conn() as conn:
            await conn.executemany(pg_sql, params_seq)

    async def executescript(self, sql: str) -> None:
        """Execute multiple SQL statements."""
        async with self._conn() as conn:
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
        contributor: str | None = None,
        team: str | None = None,
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
        if contributor:
            sql += f" AND e.contributor = ${param_idx}"
            params.append(contributor)
            param_idx += 1
        if team:
            sql += f" AND e.team = ${param_idx}"
            params.append(team)
            param_idx += 1

        sql += f" ORDER BY score LIMIT ${param_idx}"
        params.append(limit)

        async with self._conn() as conn:
            rows = await conn.fetch(sql, *params)
            return [(row["id"], row["score"]) for row in rows]

    # -- Vector operations (pgvector) --

    async def vector_store(self, entry_id: str, embedding: list[float]) -> None:
        """Upsert an embedding vector."""
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
        async with self._conn() as conn:
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
        """KNN search via pgvector cosine distance. Returns (entry_id, distance)."""
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
        async with self._conn() as conn:
            rows = await conn.fetch(
                """SELECT entry_id, embedding <=> $1::vector as distance
                   FROM knowledge_vec
                   ORDER BY distance
                   LIMIT $2""",
                vec_str,
                limit,
            )
            return [(row["entry_id"], row["distance"]) for row in rows]

    async def vector_delete(self, entry_id: str) -> None:
        """Delete embedding for an entry."""
        async with self._conn() as conn:
            await conn.execute("DELETE FROM knowledge_vec WHERE entry_id = $1", entry_id)

    # -- Graph helpers --

    async def delete_llm_edges(self, entry_id: str) -> None:
        """Remove all LLM-derived edges for a given source entry."""
        async with self._conn() as conn:
            await conn.execute(
                "DELETE FROM graph_edges WHERE source = $1 AND properties->>'source' = 'llm'",
                entry_id,
            )

    # -- Sequence --

    async def next_sequence_value(self) -> int:
        """Atomically get and increment the entry ID sequence."""
        async with self._conn() as conn:
            row = await conn.fetchrow(
                "UPDATE entry_id_seq SET next_id = next_id + 1 RETURNING next_id - 1 AS val"
            )
            if row is None:
                raise RuntimeError("entry_id_seq table is empty")
            val: int = row["val"]
            return val

    # -- Maintenance --

    async def vacuum(self) -> str:
        """Run ANALYZE (Postgres equivalent of PRAGMA optimize + VACUUM)."""
        async with self._conn() as conn:
            await conn.execute("ANALYZE")
        return "Vacuum complete (ANALYZE)."

    # -- Schema --

    async def apply_schema(self, *, embedding_dim: int = 1024) -> None:
        """Apply all PostgreSQL DDL. Uses advisory lock for migration safety."""
        async with self._pool.acquire() as conn:
            # Advisory lock ensures only one server instance runs migrations
            await conn.execute("SELECT pg_advisory_lock(42)")
            try:
                await self._apply_schema_locked(conn, embedding_dim=embedding_dim)
            finally:
                await conn.execute("SELECT pg_advisory_unlock(42)")

    async def _apply_schema_locked(
        self, conn: asyncpg.Connection, *, embedding_dim: int = 1024
    ) -> None:
        """Apply all PostgreSQL DDL inside an advisory lock."""
        # Enable pgvector (must be outside transaction on some PG versions)
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
                contributor TEXT,
                team TEXT,
                updated_by TEXT,
                sensitivity TEXT,
                expires_at TEXT,
                search_vector tsvector
            )
        """)

        # Indexes (contributor/team indexes are in _migrate_multi_user_columns
        # because the columns may not exist yet on pre-migration databases)
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
                contributor TEXT,
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
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON graph_nodes(node_type)")

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
                contributor TEXT,
                ingested_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
        """)

        # Search telemetry
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS search_events (
                id SERIAL PRIMARY KEY,
                query_text TEXT NOT NULL,
                result_count INTEGER NOT NULL,
                top_score REAL,
                match_source TEXT NOT NULL,
                contributor TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Agent feedback
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_feedback (
                id SERIAL PRIMARY KEY,
                feedback_type TEXT NOT NULL
                    CHECK(feedback_type IN ('missing', 'unhelpful', 'friction')),
                tool_name TEXT,
                query_or_params TEXT,
                detail TEXT,
                contributor TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Audit events
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id SERIAL PRIMARY KEY,
                event_type TEXT NOT NULL,
                entry_id TEXT,
                contributor TEXT,
                detail TEXT,
                created_at TEXT NOT NULL
            )
        """)
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_audit_entry ON audit_events(entry_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_events(created_at)",
        ]:
            await conn.execute(idx_sql)

        # Deployment config
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS deployment_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                set_at TEXT NOT NULL
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

        # Run migrations for existing databases
        await self._migrate_multi_user_columns(conn)

    @staticmethod
    async def _migrate_multi_user_columns(conn: asyncpg.Connection) -> None:
        """Add multi-user columns to existing tables if missing."""
        _alter = "ALTER TABLE {} ADD COLUMN {} TEXT"
        migrations = [
            ("knowledge_entries", "contributor"),
            ("knowledge_entries", "team"),
            ("knowledge_entries", "updated_by"),
            ("knowledge_entries", "sensitivity"),
            ("knowledge_entries", "expires_at"),
            ("entry_versions", "contributor"),
            ("search_events", "contributor"),
            ("agent_feedback", "contributor"),
            ("agent_feedback", "team"),
            ("ingested_files", "contributor"),
        ]
        for table, column in migrations:
            alter_sql = _alter.format(table, column)
            row = await conn.fetchrow(
                "SELECT column_name FROM information_schema.columns"
                " WHERE table_schema = current_schema()"
                " AND table_name = $1 AND column_name = $2",
                table,
                column,
            )
            if row is None:
                await conn.execute(alter_sql)

        # Indexes for new columns (idempotent)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_contributor ON knowledge_entries(contributor)"
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_team ON knowledge_entries(team)")
