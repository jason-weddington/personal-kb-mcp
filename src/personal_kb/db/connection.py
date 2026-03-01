"""Database connection management with sqlite-vec and FTS5."""

import logging
from pathlib import Path

import aiosqlite
import sqlite_vec

from personal_kb.config import get_database_url, get_db_path
from personal_kb.db.backend import Database
from personal_kb.db.sqlite_backend import SQLiteBackend

logger = logging.getLogger(__name__)


async def create_connection(
    db_path: Path | str | None = None, *, embedding_dim: int = 1024
) -> Database:
    """Create and initialize a database connection.

    Dispatches to SQLite or PostgreSQL based on KB_DATABASE_URL.
    For in-memory SQLite databases, pass ":memory:".
    """
    # Explicit ":memory:" always uses SQLite (used by tests)
    if db_path == ":memory:":
        return await _create_sqlite(":memory:", embedding_dim=embedding_dim)
    url = get_database_url()
    if url and url.startswith("postgresql"):
        return await _create_postgres(url, embedding_dim=embedding_dim)
    return await _create_sqlite(db_path or get_db_path(), embedding_dim=embedding_dim)


async def _create_sqlite(db_path: Path | str, *, embedding_dim: int = 1024) -> Database:
    """Create a SQLite backend with sqlite-vec and FTS5."""
    db_path = str(db_path)

    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row

    # Enable WAL mode for better concurrent read performance
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")

    # Load sqlite-vec extension using its native load() API
    try:

        def _load_vec() -> None:
            conn._conn.enable_load_extension(True)
            sqlite_vec.load(conn._conn)
            conn._conn.enable_load_extension(False)

        await conn._execute(_load_vec)  # type: ignore[no-untyped-call]
        logger.debug("sqlite-vec extension loaded")
    except Exception:
        logger.warning("sqlite-vec extension not available — vector search disabled")

    # Wrap in backend and apply schema
    db = SQLiteBackend(conn)
    await db.apply_schema(embedding_dim=embedding_dim)

    return db


async def _create_postgres(url: str, *, embedding_dim: int = 1024) -> Database:
    """Create a PostgreSQL backend with pgvector."""
    from personal_kb.db.postgres_backend import PostgresBackend

    db = await PostgresBackend.create(url)
    await db.apply_schema(embedding_dim=embedding_dim)
    return db
