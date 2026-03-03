"""Database connection management with sqlite-vec and FTS5."""

import logging
from pathlib import Path

import aiosqlite
import sqlite_vec

from personal_kb.config import (
    get_database_url,
    get_db_path,
    get_pg_pool_max,
    get_pg_pool_min,
    get_pg_region,
    is_pg_iam_auth,
)
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
    from typing import Any

    from personal_kb.db.postgres_backend import PostgresBackend

    create_kwargs: dict[str, Any] = {
        "pool_min": get_pg_pool_min(),
        "pool_max": get_pg_pool_max(),
    }

    if is_pg_iam_auth():
        from personal_kb.db.iam_auth import make_ssl_context, make_token_factory, parse_dsn

        dsn = parse_dsn(url)
        region = get_pg_region()
        create_kwargs["password"] = make_token_factory(dsn.host, dsn.port, dsn.username, region)
        create_kwargs["ssl"] = make_ssl_context()
        logger.info("Postgres IAM auth enabled: host=%s region=%s", dsn.host, region)

    db = await PostgresBackend.create(url, **create_kwargs)
    await db.apply_schema(embedding_dim=embedding_dim)

    await _check_deployment_config(db, embedding_dim=embedding_dim)

    return db


async def _check_deployment_config(db: Database, *, embedding_dim: int) -> None:
    """Check/store deployment config for embedding model consistency."""
    from personal_kb.config import get_embedding_model

    model = get_embedding_model()
    dim = str(embedding_dim)
    now_iso = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()

    cursor = await db.execute("SELECT key, value FROM deployment_config")
    rows = await cursor.fetchall()
    stored = {row["key"]: row["value"] for row in rows}

    if not stored:
        # First startup: seed the deployment config
        await db.execute(
            "INSERT INTO deployment_config (key, value, set_at) VALUES (?, ?, ?)"
            " ON CONFLICT (key) DO NOTHING",
            ("embedding_model", model, now_iso),
        )
        await db.execute(
            "INSERT INTO deployment_config (key, value, set_at) VALUES (?, ?, ?)"
            " ON CONFLICT (key) DO NOTHING",
            ("embedding_dim", dim, now_iso),
        )
        await db.commit()
        return

    # Check for mismatches
    stored_dim = stored.get("embedding_dim")
    if stored_dim and stored_dim != dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: DB has {stored_dim}, local is {dim}. "
            "Vectors are incompatible — cannot mix dimensions in the same database."
        )

    stored_model = stored.get("embedding_model")
    if stored_model and stored_model != model:
        logger.warning(
            "Embedding model mismatch: DB was initialized with '%s', "
            "local is '%s'. Vectors may be incompatible.",
            stored_model,
            model,
        )
