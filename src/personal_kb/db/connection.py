"""Database connection management with sqlite-vec and FTS5."""

import logging
from pathlib import Path

import aiosqlite
import sqlite_vec

from personal_kb.db.backend import Database
from personal_kb.db.schema import (
    apply_graph_schema,
    apply_ingest_schema,
    apply_schema,
    apply_vec_schema,
)
from personal_kb.db.sqlite_backend import SQLiteBackend

logger = logging.getLogger(__name__)


async def create_connection(db_path: Path | str, *, embedding_dim: int = 1024) -> Database:
    """Create and initialize a database connection.

    Loads sqlite-vec, applies schema, and enables WAL mode.
    For in-memory databases, pass ":memory:".
    """
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
        vec_available = True
    except Exception:
        logger.warning("sqlite-vec extension not available — vector search disabled")
        vec_available = False

    # Wrap in backend before applying schema
    db = SQLiteBackend(conn)

    # Apply base schema (FTS5 + tables)
    await apply_schema(db)

    # Apply graph schema (nodes + edges)
    await apply_graph_schema(db)

    # Apply ingest schema (ingested_files)
    await apply_ingest_schema(db)

    # Apply vec schema if extension loaded
    if vec_available:
        await apply_vec_schema(db, dim=embedding_dim)

    return db
