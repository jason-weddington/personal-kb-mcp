"""Database connection management with sqlite-vec and FTS5."""

import logging
from pathlib import Path

import aiosqlite
import sqlite_vec

from personal_kb.db.schema import apply_schema, apply_vec_schema

logger = logging.getLogger(__name__)


async def create_connection(
    db_path: Path | str, *, embedding_dim: int = 1024
) -> aiosqlite.Connection:
    """Create and initialize a database connection.

    Loads sqlite-vec, applies schema, and enables WAL mode.
    For in-memory databases, pass ":memory:".
    """
    db_path = str(db_path)

    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row

    # Enable WAL mode for better concurrent read performance
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")

    # Load sqlite-vec extension using its native load() API
    try:

        def _load_vec() -> None:
            db._conn.enable_load_extension(True)
            sqlite_vec.load(db._conn)
            db._conn.enable_load_extension(False)

        await db._execute(_load_vec)  # type: ignore[no-untyped-call]
        logger.debug("sqlite-vec extension loaded")
        vec_available = True
    except Exception:
        logger.warning("sqlite-vec extension not available — vector search disabled")
        vec_available = False

    # Apply base schema (FTS5 + tables)
    await apply_schema(db)

    # Apply vec schema if extension loaded
    if vec_available:
        await apply_vec_schema(db, dim=embedding_dim)

    return db
