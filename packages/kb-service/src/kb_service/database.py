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
    """
    CREATE TABLE IF NOT EXISTS app_config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        updated_by TEXT REFERENCES users(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id),
        title TEXT NOT NULL,
        mode TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_chat ON chat_messages(chat_id)",
    "CREATE INDEX IF NOT EXISTS idx_chats_user_updated"
    " ON chats(user_id, updated_at DESC)",
    # whisper_telemetry: SERVICE/AUTH DB sink for whisper-efficacy telemetry.
    # Composite PRIMARY KEY (session_id, surface, map_id) — no surrogate id;
    # the composite doubles as the ON CONFLICT conflict target so a Stop-flush
    # carrying an updated consumed flag wins via DO UPDATE. trigger_context is
    # a TEXT json.dumps() string (matches the module-wide TEXT/INTEGER
    # convention — there is no asyncpg jsonb codec registered).
    """
    CREATE TABLE IF NOT EXISTS whisper_telemetry (
        session_id TEXT NOT NULL,
        host TEXT NOT NULL,
        surface TEXT NOT NULL CHECK (surface IN ('roster', 'listener')),
        map_id TEXT NOT NULL,
        source_kb TEXT NOT NULL,
        cwd_project TEXT,
        trigger_context TEXT NOT NULL DEFAULT '{}',
        emitted_ts TEXT NOT NULL,
        consumed INTEGER NOT NULL DEFAULT 0,
        consumed_ts TEXT,
        build_engine TEXT,
        flushed_at TEXT NOT NULL,
        PRIMARY KEY (session_id, surface, map_id)
    )
    """,
    # emit_count / last_emitted_ts: added idempotently for the ALREADY-DEPLOYED
    # table (three instances, 2000+ live rows) — ADD COLUMN IF NOT EXISTS is
    # safe to re-run on every init_db(). emit_count NOT NULL DEFAULT 1 backfills
    # existing rows to 1 as part of the ALTER itself (Postgres 11+ fast default).
    # last_emitted_ts starts NULL for pre-existing rows; the UPDATE below
    # backfills it to the existing emitted_ts exactly once (subsequent runs are
    # no-ops since the WHERE clause only matches unbackfilled rows).
    "ALTER TABLE whisper_telemetry"
    " ADD COLUMN IF NOT EXISTS emit_count INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE whisper_telemetry ADD COLUMN IF NOT EXISTS last_emitted_ts TEXT",
    "UPDATE whisper_telemetry SET last_emitted_ts = emitted_ts"
    " WHERE last_emitted_ts IS NULL",
    # listener_decisions: SERVICE/AUTH DB sink recording the listener gate's
    # decision on EVERY request, including declines (kill-switch, rule-A/B
    # drops, no-LLM, non-unanimous votes) that today leave no durable trace.
    # Insert-only (one row per listener request); no composite PK / upsert —
    # unlike whisper_telemetry there is nothing to conflict on or update.
    """
    CREATE TABLE IF NOT EXISTS listener_decisions (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        session_id TEXT,
        cwd_project TEXT,
        source_kb TEXT NOT NULL,
        decided_ts TEXT NOT NULL,
        candidates_considered INTEGER NOT NULL,
        decision TEXT NOT NULL CHECK (decision IN ('whisper', 'declined')),
        reason TEXT NOT NULL CHECK (reason IN (
            'kill-switch', 'no-candidates', 'rule-a', 'rule-b', 'no-llm',
            'vote-split', 'vote-none', 'whispered'
        ))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_listener_decisions_decided_ts"
    " ON listener_decisions(decided_ts)",
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
