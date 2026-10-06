"""Async service-auth database (PostgreSQL pool, or local SQLite).

This pool backs the service's OWN tables (users, api_keys, invites,
password_resets, app_config, telemetry, ...). It is DISTINCT from the kb-core
data DB (``KB_DATABASE_URL`` / ``KB_DB_PATH``).

* ``KB_SERVICE_DATABASE_URL`` set -> an asyncpg pool to that Postgres DSN
  (hosted mode; unchanged).
* ``KB_SERVICE_DATABASE_URL`` unset AND ``KB_AUTH_MODE=none`` -> a local
  SQLite file, ``service.db`` next to the data DB (directory of
  ``KB_DB_PATH``, default ``~/.local/share/personal_kb/``).
* ``KB_SERVICE_DATABASE_URL`` unset in any other auth mode -> RuntimeError.
  A hosted deployment that lost its env var must fail loudly rather than
  silently start against an empty, userless SQLite auth DB.
"""

import os
import re
from pathlib import Path
from typing import Any

from kb_service.db_sqlite import SqlitePool
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
            'vote-split', 'vote-none', 'whispered', 'fallback-direct'
        ))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_listener_decisions_decided_ts"
    " ON listener_decisions(decided_ts)",
    # 'fallback-direct' added idempotently for the ALREADY-DEPLOYED table: the
    # CREATE TABLE above only fires on a brand-new DB (IF NOT EXISTS), so an
    # existing instance's reason CHECK still lacks the new member until this
    # DROP+ADD runs. Postgres has no ADD-a-value-to-CHECK shortcut, so the
    # named constraint (default autogenerated name
    # ``<table>_<column>_check``) is dropped and re-added with the extended
    # set — safe to re-run on every init_db() call, mirroring the
    # ADD COLUMN IF NOT EXISTS idempotency used for whisper_telemetry above.
    "ALTER TABLE listener_decisions"
    " DROP CONSTRAINT IF EXISTS listener_decisions_reason_check",
    "ALTER TABLE listener_decisions ADD CONSTRAINT listener_decisions_reason_check"
    " CHECK (reason IN ("
    "'kill-switch', 'no-candidates', 'rule-a', 'rule-b', 'no-llm',"
    " 'vote-split', 'vote-none', 'whispered', 'fallback-direct'"
    "))",
    # vote_shape (GTD 66ea1fe4): added idempotically for the ALREADY-DEPLOYED
    # table, same ADD COLUMN IF NOT EXISTS pattern as whisper_telemetry above.
    # json.dumps of the 3 voters' raw candidate-id sets, e.g.
    # '[["kb-1"],["kb-1","kb-2"],[]]'; '' (default) on every branch that
    # never reached the LLM gate. Lets the reframed set-returning vote's
    # effect on whisper rate be measured directly off this table.
    "ALTER TABLE listener_decisions"
    " ADD COLUMN IF NOT EXISTS vote_shape TEXT NOT NULL DEFAULT ''",
    # candidate_signal (GTD be964e94): added idempotently for the ALREADY-
    # DEPLOYED table, same ADD COLUMN IF NOT EXISTS pattern as vote_shape
    # above. Records WHICH candidate-retrieval signal produced the surfaced
    # candidate on a 'whisper' decision:
    #   'lexical'  — the new project_ref/title substring path (be964e94),
    #                 including maps ALSO found by detail-matching (lexical
    #                 is the higher-precision signal so it wins attribution)
    #   'detail'   — the primary detail-match retrieval path (bf40d4f1)
    #   'fallback' — that same retrieval's own direct-mental_map-search
    #                fallback (used when detail-matching resolves zero
    #                candidate maps)
    # '' (default) on every decline branch, and on 'whisper' rows written
    # before this migration. Answers "how many whispers came from the
    # lexical path vs detail matching vs fallback" with:
    #   SELECT candidate_signal, COUNT(*) FROM listener_decisions
    #   WHERE decision = 'whisper' GROUP BY candidate_signal;
    "ALTER TABLE listener_decisions"
    " ADD COLUMN IF NOT EXISTS candidate_signal TEXT NOT NULL DEFAULT ''",
    # Same DROP+ADD CONSTRAINT idempotency pattern used for the 'reason'
    # CHECK above — safe to re-run on every init_db() call.
    "ALTER TABLE listener_decisions"
    " DROP CONSTRAINT IF EXISTS listener_decisions_candidate_signal_check",
    "ALTER TABLE listener_decisions"
    " ADD CONSTRAINT listener_decisions_candidate_signal_check"
    " CHECK (candidate_signal IN ('', 'lexical', 'detail', 'fallback'))",
    # Listener telemetry (GTD 268e2af3): added idempotently for the
    # ALREADY-DEPLOYED table, same ADD COLUMN IF NOT EXISTS pattern as
    # vote_shape above — three live instances hold this table and existing
    # rows must survive, backfilling to the defaults below.
    #
    # candidate_ids: json.dumps of the RETRIEVED candidate pool, captured
    # BEFORE rule A / rule B filter it (the whole point of this column —
    # a rule-a/rule-b decline must still show a non-empty list here).
    # whispered_ids: json.dumps of the ids actually surfaced to the caller
    # (empty list on every declined decision). Contrast with the existing
    # vote_shape column, which stores each voter's raw CHOSEN set, not "the
    # pool we voted on" or "what we whispered".
    # n_retrieved / n_after_a / n_after_b: per-stage candidate counts, so
    # rule-A and rule-B attrition are measurable independently instead of
    # being thrown away after the route computes them.
    # retrieval_path: whether the direct-mental_map-search FALLBACK ran
    # (see _retrieve_candidate_maps) — split out of `reason` so the
    # granular decline cause (rule-a, rule-b, vote-none, ...) is never
    # masked by "fallback-direct" again.
    "ALTER TABLE listener_decisions"
    " ADD COLUMN IF NOT EXISTS candidate_ids TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE listener_decisions"
    " ADD COLUMN IF NOT EXISTS whispered_ids TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE listener_decisions"
    " ADD COLUMN IF NOT EXISTS n_retrieved INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE listener_decisions"
    " ADD COLUMN IF NOT EXISTS n_after_a INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE listener_decisions"
    " ADD COLUMN IF NOT EXISTS n_after_b INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE listener_decisions"
    " ADD COLUMN IF NOT EXISTS retrieval_path TEXT NOT NULL DEFAULT ''",
]


# Mirrors kb_service.main._DEFAULT_KB_DB_PATH / personal_kb.config.get_db_path.
_DEFAULT_KB_DB_PATH = "~/.local/share/personal_kb/knowledge.db"
SERVICE_DB_FILENAME = "service.db"

_ADD_COLUMN_RE = re.compile(
    r"^\s*ALTER TABLE (\w+) ADD COLUMN IF NOT EXISTS (\w+) (.*)$", re.DOTALL
)
_PG_IDENTITY = "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY"
_SQLITE_IDENTITY = "INTEGER PRIMARY KEY AUTOINCREMENT"


def sqlite_service_db_path() -> Path:
    """Return the local service-DB path: ``service.db`` beside ``KB_DB_PATH``.

    ``KB_DB_PATH`` is read at call time (not import time) so tests and the
    CLI can point it elsewhere.
    """
    data_path = Path(os.environ.get("KB_DB_PATH", _DEFAULT_KB_DB_PATH)).expanduser()
    return data_path.parent / SERVICE_DB_FILENAME


def is_sqlite_pool(pool: Any) -> bool:
    """Return True when *pool* is the local SQLite service DB."""
    return isinstance(pool, SqlitePool)


async def get_db() -> DbPool:
    """Return the service-auth connection pool, creating it lazily if needed.

    Opens an asyncpg pool from ``KB_SERVICE_DATABASE_URL`` when it is set.
    When it is unset, local no-auth mode (``KB_AUTH_MODE=none``) opens the
    SQLite file from :func:`sqlite_service_db_path`; every other auth mode
    fails closed.

    Raises:
        RuntimeError: If ``KB_SERVICE_DATABASE_URL`` is not set and
            ``KB_AUTH_MODE`` is not ``none``.
    """
    global _pool
    if _pool is None:
        # Local import: kb_service.auth imports this module at load time.
        from kb_service.auth import _auth_mode

        dsn = os.environ.get("KB_SERVICE_DATABASE_URL")
        if dsn:
            import asyncpg

            _pool = await asyncpg.create_pool(dsn)
        elif _auth_mode() == "none":
            _pool = await SqlitePool.open(sqlite_service_db_path())
        else:
            raise RuntimeError(
                "KB_SERVICE_DATABASE_URL is not set; the service-auth database "
                "is required unless KB_AUTH_MODE=none (local mode)."
            )
    return _pool


async def _init_sqlite(pool: SqlitePool) -> None:
    """Apply ``_SCHEMA_STATEMENTS`` to SQLite, bridging the dialect gaps.

    * ``BIGINT GENERATED ALWAYS AS IDENTITY`` -> ``INTEGER PRIMARY KEY
      AUTOINCREMENT``.
    * ``ADD COLUMN IF NOT EXISTS`` (unsupported in SQLite) -> checked against
      ``PRAGMA table_info`` first, so re-running is a no-op.
    * ``DROP/ADD CONSTRAINT`` (unsupported in SQLite) are skipped: the fresh
      ``CREATE TABLE`` already carries the full ``reason`` CHECK, and there is
      no pre-existing SQLite table to migrate.
    """
    async with pool.acquire() as conn:
        for stmt in _SCHEMA_STATEMENTS:
            if "DROP CONSTRAINT" in stmt or "ADD CONSTRAINT" in stmt:
                continue
            match = _ADD_COLUMN_RE.match(stmt)
            if match:
                table, column, definition = match.groups()
                cols = await conn.fetch(f"PRAGMA table_info({table})")
                if any(c["name"] == column for c in cols):
                    continue
                await conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                )
                continue
            await conn.execute(stmt.replace(_PG_IDENTITY, _SQLITE_IDENTITY))


async def init_db() -> None:
    """Create the service-auth tables if they don't exist."""
    pool = await get_db()
    if isinstance(pool, SqlitePool):
        await _init_sqlite(pool)
        return
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
