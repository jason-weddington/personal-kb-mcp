"""Async service-auth database (PostgreSQL connection pool).

This pool backs the service's OWN tables (users, api_keys, invites,
password_resets) and is opened from ``KB_SERVICE_DATABASE_URL``. It is
DISTINCT from the kb-core data pool (opened from ``KB_DATABASE_URL`` via
``create_postgres``) — two separate asyncpg pools to two separate databases.
"""

import os
from typing import Any

from kb_service.db_types import DbPool

_pool: DbPool | None = None

# Each statement must be executed individually (asyncpg has no executescript).
# Column types are TEXT/INTEGER on purpose: the auth code writes timestamps via
# ``.isoformat()`` strings and relies on Pydantic int->bool / str->datetime
# coercion when building ``User(**row_to_dict(row))``.
_SCHEMA_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        is_admin INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS api_keys (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id),
        key_hash TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash)",
    "CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)",
    """
    CREATE TABLE IF NOT EXISTS invites (
        token TEXT PRIMARY KEY,
        issued_by TEXT NOT NULL REFERENCES users(id),
        note TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        used_at TEXT,
        used_by TEXT REFERENCES users(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS password_resets (
        token TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id),
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        used_at TEXT
    )
    """,
]


async def get_db() -> DbPool:
    """Return the service-auth connection pool, creating it lazily if needed.

    Opens an asyncpg pool from ``KB_SERVICE_DATABASE_URL``. This service is
    Postgres-only — there is no SQLite fallback.

    Raises:
        RuntimeError: If ``KB_SERVICE_DATABASE_URL`` is not set.
    """
    global _pool
    if _pool is None:
        dsn = os.environ.get("KB_SERVICE_DATABASE_URL")
        if not dsn:
            raise RuntimeError(
                "KB_SERVICE_DATABASE_URL is not set; the service-auth database "
                "is required (Postgres-only)."
            )
        import asyncpg

        _pool = await asyncpg.create_pool(dsn)
    return _pool


async def init_db() -> None:
    """Create the service-auth tables if they don't exist."""
    pool = await get_db()
    async with pool.acquire() as conn:
        for stmt in _SCHEMA_STATEMENTS:
            await conn.execute(stmt)


async def close_db() -> None:
    """Close the service-auth connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a Record to a plain dict."""
    return dict(row)
