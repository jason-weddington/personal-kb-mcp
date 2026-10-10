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

import logging
import os
import re
from pathlib import Path
from typing import Any

from kb_service.db_sqlite import SqliteConnection, SqlitePool
from kb_service.db_types import DbPool

logger = logging.getLogger(__name__)

_pool: DbPool | None = None

# Surprise-capture shapes allowed by the shape CHECK on surprise_candidates,
# surprise_detections and surprise_distillations (surprise_dry_runs has none).
SURPRISE_SHAPES: tuple[int, ...] = (1, 2, 3, 4, 5)
_SURPRISE_SHAPE_CHECK = (
    "CHECK (shape IN (" + ", ".join(str(s) for s in SURPRISE_SHAPES) + "))"
)
_SURPRISE_SHAPE_TABLES = (
    "surprise_candidates",
    "surprise_detections",
    "surprise_distillations",
)

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
    # api_keys.surface: the write-policy surface of a key (kb_service.write_policy).
    # NULL follows KB_WRITE_POLICY_DEFAULT_SURFACE; set it with
    # `kb-service set-key-surface`.
    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS surface TEXT CHECK (surface IS NULL"
    " OR surface IN ('interactive', 'headless', 'autonomous'))",
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
    # text_source: where the hook got the judged text
    # ('last_assistant_message' | 'transcript'); NULL on old rows/hooks.
    "ALTER TABLE listener_decisions ADD COLUMN IF NOT EXISTS text_source TEXT",
    # failure_events: SERVICE DB sink for the failure-cue index. One row per
    # recorded post_tool failure (POST /api/kb/event); event_id is the
    # idempotency key (ON CONFLICT DO NOTHING). cue_key and every normalized
    # column come from kb_core.cues.build_cue — see that module for the rules
    # and CUE_NORMALIZER_VERSION. Record-only: nothing is delivered back yet.
    "CREATE TABLE IF NOT EXISTS failure_events ("
    "id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
    "event_id TEXT NOT NULL UNIQUE, "
    "cue_key TEXT NOT NULL, "
    "normalizer_version INTEGER NOT NULL, "
    "session_id TEXT NOT NULL, "
    "harness TEXT NOT NULL, "
    "mode TEXT NOT NULL CHECK (mode IN ('interactive', 'headless')), "
    "engine TEXT, "
    "host TEXT, "
    "hook_version TEXT, "
    "host_class TEXT NOT NULL, "
    "project TEXT NOT NULL DEFAULT '', "
    "project_source TEXT NOT NULL, "
    "tool TEXT NOT NULL, "
    "target TEXT NOT NULL DEFAULT '', "
    "target_class TEXT NOT NULL DEFAULT '', "
    "normalized_error TEXT NOT NULL, "
    "error_rule TEXT NOT NULL, "
    "anomaly TEXT, "
    "raw_error_excerpt TEXT NOT NULL, "
    "is_interrupt INTEGER NOT NULL DEFAULT 0, "
    "ts TEXT NOT NULL, "
    "received_ts TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_failure_events_cue_ts"
    " ON failure_events(cue_key, ts)",
    "CREATE INDEX IF NOT EXISTS idx_failure_events_session"
    " ON failure_events(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_failure_events_received_ts"
    " ON failure_events(received_ts)",
    # gate_decisions: SERVICE DB sink for the prevention soft gate. One row
    # per hook-recorded decision (POST /api/kb/prevention/decisions);
    # decision_id is the idempotency key. See routes/prevention_routes.py.
    "CREATE TABLE IF NOT EXISTS gate_decisions ("
    "id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
    "decision_id TEXT NOT NULL UNIQUE, "
    "session_id TEXT NOT NULL, "
    "harness TEXT NOT NULL, "
    "mode TEXT NOT NULL CHECK (mode IN ('interactive', 'headless')), "
    "engine TEXT, "
    "host TEXT, "
    "hook_version TEXT, "
    "project TEXT NOT NULL DEFAULT '', "
    "resolution_id TEXT NOT NULL DEFAULT '', "
    "resolution_updated_at TEXT, "
    "tool TEXT NOT NULL, "
    "target TEXT NOT NULL DEFAULT '', "
    "target_class TEXT NOT NULL DEFAULT '', "
    "decision TEXT NOT NULL CHECK (decision IN ('denied', 'would_deny', "
    "'skipped_already_denied', 'skipped_cap', 'retry', 'armed', 'summary', "
    "'failure_context', 'failure_context_repeat', 'failure_context_error', "
    "'rearmed', 'overridden')), "
    "shadow INTEGER NOT NULL DEFAULT 0, "
    "reason_excerpt TEXT, "
    "retry_changed_command INTEGER, "
    "prior_target TEXT, "
    "observed_once INTEGER NOT NULL DEFAULT 0, "
    "index_len INTEGER, "
    "slice_len INTEGER, "
    "pre_tool_calls INTEGER, "
    "pre_tool_errors INTEGER, "
    "last_error_type TEXT, "
    "tool_use_id TEXT, "
    "ts TEXT NOT NULL, "
    "received_ts TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_gate_decisions_session"
    " ON gate_decisions(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_gate_decisions_resolution_ts"
    " ON gate_decisions(resolution_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_gate_decisions_received_ts"
    " ON gate_decisions(received_ts)",
    # Widen an ALREADY-DEPLOYED Postgres gate_decisions decision CHECK with the
    # PostToolUseFailure failure-context decisions and the soft gate's
    # rearmed / overridden decisions, mirroring the
    # listener_decisions_reason_check DROP+ADD above (idempotent on re-run).
    # SQLite skips these and rebuilds the table instead
    # (_rebuild_sqlite_table via _SQLITE_CHECK_REBUILDS).
    "ALTER TABLE gate_decisions"
    " DROP CONSTRAINT IF EXISTS gate_decisions_decision_check",
    "ALTER TABLE gate_decisions ADD CONSTRAINT gate_decisions_decision_check"
    " CHECK (decision IN ('denied', 'would_deny', 'skipped_already_denied',"
    " 'skipped_cap', 'retry', 'armed', 'summary', 'failure_context',"
    " 'failure_context_repeat', 'failure_context_error', 'rearmed',"
    " 'overridden'))",
    # turn_events: surprise-capture turn digests posted by the hook at Stop.
    # event_id is exactly '<session_id>:<turn_index>' (idempotency key, first
    # write wins). items and redactions hold json.dumps() TEXT. processed_at
    # NULL means pending (not yet consumed by the surprise drain) and is the
    # ONLY pending marker on this table. capture_mode records the
    # KB_SURPRISE_CAPTURE value at ingest and is informational only: consumers
    # decide shadow vs on from the CURRENT env value at processing time.
    "CREATE TABLE IF NOT EXISTS turn_events ("
    "id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
    "event_id TEXT NOT NULL UNIQUE, "
    "session_id TEXT NOT NULL, "
    "harness TEXT NOT NULL, "
    "mode TEXT NOT NULL CHECK (mode IN ('interactive', 'headless')), "
    "engine TEXT, "
    "host TEXT, "
    "hook_version TEXT, "
    "project TEXT NOT NULL DEFAULT '', "
    "turn_index INTEGER NOT NULL, "
    "ts TEXT NOT NULL, "
    "user_prompt TEXT, "
    "items TEXT NOT NULL DEFAULT '[]', "
    "final_message TEXT, "
    "truncated INTEGER NOT NULL DEFAULT 0, "
    "redactions TEXT NOT NULL DEFAULT '[]', "
    "anomaly TEXT, "
    "capture_mode TEXT NOT NULL CHECK (capture_mode IN ('shadow', 'on')), "
    "processed_at TEXT, "
    "received_ts TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_turn_events_session_turn"
    " ON turn_events(session_id, turn_index)",
    "CREATE INDEX IF NOT EXISTS idx_turn_events_received_ts"
    " ON turn_events(received_ts)",
    # surprise_candidates: corrected beliefs detected by the surprise drain.
    # turn_event_ids and detector_output hold json.dumps() TEXT. This module
    # only ever writes 'pending' (mode on) or 'shadow' (mode shadow, terminal:
    # never distilled, never replayed). The distiller sets rejected / written
    # / merged; written and merged carry entry_id.
    "CREATE TABLE IF NOT EXISTS surprise_candidates ("
    "id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
    f"shape INTEGER NOT NULL {_SURPRISE_SHAPE_CHECK}, "
    "session_id TEXT NOT NULL, "
    "project TEXT NOT NULL DEFAULT '', "
    "turn_event_ids TEXT NOT NULL DEFAULT '[]', "
    "detector_model TEXT NOT NULL, "
    "detector_output TEXT NOT NULL DEFAULT '{}', "
    "status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', "
    "'shadow', 'rejected', 'written', 'merged')), "
    "entry_id TEXT, "
    "created_at TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_surprise_candidates_status"
    " ON surprise_candidates(status, id)",
    "CREATE INDEX IF NOT EXISTS idx_surprise_candidates_session"
    " ON surprise_candidates(session_id)",
    # surprise_detections: one row per detection decision (negatives too),
    # mirroring listener_decisions / gate_decisions. candidate_id is non-NULL
    # iff outcome = 'candidate'.
    "CREATE TABLE IF NOT EXISTS surprise_detections ("
    "id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
    "event_id TEXT NOT NULL, "
    "session_id TEXT NOT NULL, "
    "project TEXT NOT NULL DEFAULT '', "
    f"shape INTEGER NOT NULL {_SURPRISE_SHAPE_CHECK}, "
    "mode TEXT NOT NULL CHECK (mode IN ('shadow', 'on')), "
    "outcome TEXT NOT NULL CHECK (outcome IN ('candidate', 'not_applicable', "
    "'no_llm', 'llm_error', 'unparseable', 'invalid_fields', 'no_surprise', "
    "'low_confidence', 'ungrounded')), "
    "reason TEXT NOT NULL DEFAULT '', "
    "detector_model TEXT NOT NULL DEFAULT '', "
    "detector_version INTEGER NOT NULL, "
    "confidence DOUBLE PRECISION, "
    "candidate_id BIGINT, "
    "details TEXT NOT NULL DEFAULT '{}', "
    "raw_response_excerpt TEXT, "
    "prompt_chars INTEGER, "
    "response_chars INTEGER, "
    "latency_ms INTEGER, "
    "ts TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_surprise_detections_ts ON surprise_detections(ts)",
    "CREATE INDEX IF NOT EXISTS idx_surprise_detections_session"
    " ON surprise_detections(session_id)",
    # surprise_distillations: one row per distill decision on a pending
    # surprise candidate (see surprise_worker.distill_candidates), mirroring
    # surprise_detections. There is no mode column: distillation runs only in
    # KB_SURPRISE_CAPTURE=on, so every row is an 'on' decision. Candidates
    # skipped as no_llm or by the double_distill tripwire get no row.
    "CREATE TABLE IF NOT EXISTS surprise_distillations ("
    "id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
    "candidate_id BIGINT NOT NULL, "
    "session_id TEXT NOT NULL, "
    "project TEXT NOT NULL DEFAULT '', "
    f"shape INTEGER NOT NULL {_SURPRISE_SHAPE_CHECK}, "
    "outcome TEXT NOT NULL CHECK (outcome IN ('written', 'merged', "
    "'same_session', 'covered', 'not_durable', 'redacted', 'gate_induced', "
    "'llm_error', 'unparseable', 'invalid_fields', 'invalid_resolution', "
    "'secret_detected', 'no_project', 'kb_error')), "
    "reason TEXT NOT NULL DEFAULT '', "
    "entry_id TEXT, "
    "matched_entry_id TEXT, "
    "match_kind TEXT NOT NULL DEFAULT '' CHECK (match_kind IN ('', 'exact', "
    "'cosine')), "
    "similarity DOUBLE PRECISION, "
    "near_duplicate_status TEXT NOT NULL DEFAULT '', "
    "near_duplicate_floor DOUBLE PRECISION, "
    "cue_target_class TEXT NOT NULL DEFAULT '', "
    "observed_sessions_before INTEGER, "
    "observed_sessions_after INTEGER, "
    "verdict TEXT, "
    "distiller_model TEXT NOT NULL DEFAULT '', "
    "distiller_version INTEGER NOT NULL, "
    "raw_response_excerpt TEXT, "
    "prompt_chars INTEGER, "
    "response_chars INTEGER, "
    "latency_ms INTEGER, "
    "ts TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_surprise_distillations_candidate"
    " ON surprise_distillations(candidate_id)",
    "CREATE INDEX IF NOT EXISTS idx_surprise_distillations_ts"
    " ON surprise_distillations(ts)",
    # surprise_dry_runs: one row per shadow candidate, recording what the
    # mode-'on' distill path WOULD have done (see
    # surprise_worker.dry_run_candidates). No KB write ever backs a row. No
    # CHECK on would_outcome: the on-path outcome names with written/merged
    # replaced by would_write/would_merge. payload holds json.dumps() TEXT.
    # mode is the candidate's last turn event's interactive|headless ('' when
    # unknown). The unique candidate_id index enforces one dry run each.
    "CREATE TABLE IF NOT EXISTS surprise_dry_runs ("
    "id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
    "candidate_id BIGINT NOT NULL, "
    "session_id TEXT NOT NULL, "
    "project TEXT NOT NULL DEFAULT '', "
    "shape INTEGER NOT NULL, "
    "distiller_model TEXT NOT NULL DEFAULT '', "
    "distiller_version INTEGER NOT NULL, "
    "would_outcome TEXT NOT NULL, "
    "reason TEXT NOT NULL DEFAULT '', "
    "payload TEXT NOT NULL DEFAULT '{}', "
    "mode TEXT NOT NULL DEFAULT '', "
    "created_at TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_surprise_dry_runs_created_at"
    " ON surprise_dry_runs(created_at)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_surprise_dry_runs_candidate"
    " ON surprise_dry_runs(candidate_id)",
    # Widen an ALREADY-DEPLOYED Postgres shape CHECK on the three surprise
    # tables to SURPRISE_SHAPES, mirroring the listener_decisions_reason_check
    # and gate_decisions_decision_check DROP+ADD above: Postgres autogenerates
    # the column-CHECK name <table>_<column>_check, and DROP+ADD is idempotent
    # on every init_db(). SQLite skips these and rebuilds the tables instead
    # (_rebuild_sqlite_table via _SQLITE_CHECK_REBUILDS).
    *(
        stmt
        for t in _SURPRISE_SHAPE_TABLES
        for stmt in (
            f"ALTER TABLE {t} DROP CONSTRAINT IF EXISTS {t}_shape_check",
            f"ALTER TABLE {t} ADD CONSTRAINT {t}_shape_check {_SURPRISE_SHAPE_CHECK}",
        )
    ),
]


# Mirrors kb_service.main._DEFAULT_KB_DB_PATH / personal_kb.config.get_db_path.
_DEFAULT_KB_DB_PATH = "~/.local/share/personal_kb/knowledge.db"
SERVICE_DB_FILENAME = "service.db"

_ADD_COLUMN_RE = re.compile(
    r"^\s*ALTER TABLE (\w+) ADD COLUMN IF NOT EXISTS (\w+) (.*)$", re.DOTALL
)
_PG_IDENTITY = "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY"
_SQLITE_IDENTITY = "INTEGER PRIMARY KEY AUTOINCREMENT"
# Present in the stored SQLite gate_decisions DDL once its decision CHECK
# carries the failure-context members. Any future CHECK widening must point
# this at the new last member, or existing SQLite tables are never rebuilt.
_GATE_DECISIONS_CHECK_MARKER = "'overridden'"
# (table, marker): an existing SQLite table whose stored DDL lacks its marker
# carries an older, narrower CHECK and is rebuilt by _rebuild_sqlite_table.
_SQLITE_CHECK_REBUILDS: tuple[tuple[str, str], ...] = (
    ("gate_decisions", _GATE_DECISIONS_CHECK_MARKER),
    ("surprise_candidates", _SURPRISE_SHAPE_CHECK),
    ("surprise_detections", _SURPRISE_SHAPE_CHECK),
    ("surprise_distillations", _SURPRISE_SHAPE_CHECK),
)


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


def check_database_config() -> None:
    """Fail closed when ``KB_DB_PATH`` is set alongside a Postgres URL.

    A Postgres URL silently wins over ``KB_DB_PATH``, so a throwaway-SQLite
    intent could write to a live KB. Empty values count as unset. Only the
    variable names are reported, never their values.

    Raises:
        RuntimeError: If ``KB_DB_PATH`` and ``KB_DATABASE_URL`` and/or
            ``KB_SERVICE_DATABASE_URL`` are all non-empty.
    """
    if not os.environ.get("KB_DB_PATH", "").strip():
        return
    conflicts = [
        name
        for name in ("KB_DATABASE_URL", "KB_SERVICE_DATABASE_URL")
        if os.environ.get(name, "").strip()
    ]
    if conflicts:
        names = " and ".join(conflicts)
        raise RuntimeError(
            f"ambiguous database config: KB_DB_PATH is set together with {names}; "
            f"unset {names} to use the SQLite file, or unset KB_DB_PATH to use Postgres"
        )


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
        check_database_config()
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
    * ``DROP/ADD CONSTRAINT`` (unsupported in SQLite) are skipped: a fresh
      ``CREATE TABLE`` already carries the full CHECK. A pre-existing table
      listed in :data:`_SQLITE_CHECK_REBUILDS` with an older, narrower CHECK
      IS migrated: :func:`_rebuild_sqlite_table` rebuilds it last.
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
        for table, marker in _SQLITE_CHECK_REBUILDS:
            await _rebuild_sqlite_table(conn, table, marker)


async def _rebuild_sqlite_table(
    conn: SqliteConnection, table: str, marker: str
) -> None:
    """Rebuild an old SQLite *table* whose stored DDL lacks *marker*.

    SQLite cannot alter a CHECK, so the table is renamed, recreated from the
    current DDL, refilled and dropped, then its indexes are recreated — all in
    one transaction. Uses only *conn* (never the pool: ``SqlitePool.acquire``
    holds a non-reentrant lock). A no-op when the table is absent or its
    stored DDL already carries *marker*.
    """
    row = await conn.fetchrow(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = $1", table
    )
    if row is None or row["sql"] is None or marker in row["sql"]:
        return
    create_prefix = f"CREATE TABLE IF NOT EXISTS {table} ("
    create = next(s for s in _SCHEMA_STATEMENTS if s.startswith(create_prefix))
    indexes = [
        s
        for s in _SCHEMA_STATEMENTS
        if s.startswith("CREATE ") and " INDEX " in s and f" ON {table}(" in s
    ]
    old = f"{table}_pre_rebuild"
    async with conn.transaction():
        await conn.execute(f"ALTER TABLE {table} RENAME TO {old}")
        await conn.execute(create.replace(_PG_IDENTITY, _SQLITE_IDENTITY))
        info = await conn.fetch(f"PRAGMA table_info({old})")
        cols = ", ".join(str(c["name"]) for c in info)
        await conn.execute(
            f"INSERT INTO {table} ({cols})"  # noqa: S608 - names from PRAGMA
            f" SELECT {cols} FROM {old}"
        )
        n = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
        await conn.execute(f"DROP TABLE {old}")
        for stmt in indexes:
            await conn.execute(stmt)
    logger.warning("service_db check_rebuild table=%s rows=%d", table, n)


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
