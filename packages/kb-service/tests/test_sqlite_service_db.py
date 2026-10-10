"""Local-mode SQLite service DB: unset KB_SERVICE_DATABASE_URL + KB_AUTH_MODE=none.

These tests use the REAL ``kb_service.database`` (no fake pool): ``get_db()``
opens ``service.db`` next to a temp ``KB_DB_PATH``. Every test resets the
module-level pool so nothing leaks between tests or into the rest of the suite.
"""

import json
import sqlite3
import typing
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kb_service.database as database
from kb_service import attribution, chat_history, models
from kb_service.db_sqlite import SqlitePool, translate_sql
from kb_service.main import app
from kb_service.models import User
from kb_service.routes import listener_routes

# Env vars that would steer the app away from the hermetic local SQLite path.
_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
)


@pytest.fixture
def local_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Local no-auth mode with the data DB and service DB in *tmp_path*."""
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    # No Ollama / LLM in the sandbox: point at a closed port and fail fast.
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setattr(database, "_pool", None)
    yield tmp_path / "service.db"


@pytest.fixture
async def pool(local_env: Path) -> AsyncIterator[SqlitePool]:
    """An initialised SQLite service DB, closed afterwards."""
    await database.init_db()
    db = await database.get_db()
    assert isinstance(db, SqlitePool)
    yield db
    await database.close_db()


def _local_user() -> User:
    from kb_service.auth import _synthetic_user

    return _synthetic_user()


# ─── SQL translation ────────────────────────────────────────────────────────


def test_translate_sql_numbers_placeholders_and_drops_for_update() -> None:
    sql = "SELECT * FROM invites WHERE token = $1 AND x = $12 FOR UPDATE"
    assert translate_sql(sql) == "SELECT * FROM invites WHERE token = ?1 AND x = ?12"


# ─── get_db selection / fail-closed ─────────────────────────────────────────


async def test_service_db_lives_next_to_kb_db_path(local_env: Path) -> None:
    assert database.sqlite_service_db_path() == local_env
    await database.init_db()
    try:
        assert local_env.is_file()
        db = await database.get_db()
        rows = await db.fetch("SELECT name FROM sqlite_master WHERE type = 'table'")
        names = {r["name"] for r in rows}
        assert {
            "users",
            "api_keys",
            "invites",
            "password_resets",
            "app_config",
            "chats",
            "chat_messages",
            "whisper_telemetry",
            "listener_decisions",
            "failure_events",
            "gate_decisions",
            "turn_events",
            "surprise_candidates",
            "surprise_detections",
            "surprise_distillations",
        } <= names
        mode = await db.fetchval("PRAGMA journal_mode")
        assert mode == "wal"
    finally:
        await database.close_db()


@pytest.mark.parametrize("auth_mode", [None, "jwt"])
async def test_unset_service_url_fails_closed_outside_no_auth(
    local_env: Path, monkeypatch: pytest.MonkeyPatch, auth_mode: str | None
) -> None:
    if auth_mode is None:
        monkeypatch.delenv("KB_AUTH_MODE", raising=False)
    else:
        monkeypatch.setenv("KB_AUTH_MODE", auth_mode)
    with pytest.raises(RuntimeError, match="KB_SERVICE_DATABASE_URL"):
        await database.init_db()
    assert database._pool is None
    assert not local_env.exists()


def test_app_startup_fails_closed_in_jwt_mode(
    local_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_AUTH_MODE", "jwt")
    with (
        pytest.raises(RuntimeError, match="KB_SERVICE_DATABASE_URL"),
        TestClient(app),
    ):
        pass
    assert not local_env.exists()


# ─── migrations ─────────────────────────────────────────────────────────────


async def test_init_db_is_idempotent_across_restarts(local_env: Path) -> None:
    for _ in range(2):
        await database.init_db()
        await database.close_db()
    await database.init_db()
    try:
        db = await database.get_db()
        cols = {
            r["name"] for r in await db.fetch("PRAGMA table_info(listener_decisions)")
        }
        assert {
            "vote_shape",
            "candidate_signal",
            "candidate_ids",
            "whispered_ids",
            "n_retrieved",
            "n_after_a",
            "n_after_b",
            "retrieval_path",
        } <= cols
        wt_cols = {
            r["name"] for r in await db.fetch("PRAGMA table_info(whisper_telemetry)")
        }
        assert {"emit_count", "last_emitted_ts"} <= wt_cols
    finally:
        await database.close_db()


async def test_failure_events_schema_is_idempotent(local_env: Path) -> None:
    await database.init_db()
    await database.init_db()
    try:
        db = await database.get_db()
        tables = await db.fetchval(
            "SELECT COUNT(*) FROM sqlite_master"
            " WHERE type = 'table' AND name = 'failure_events'"
        )
        indexes = await db.fetchval(
            "SELECT COUNT(*) FROM sqlite_master"
            " WHERE type = 'index' AND name LIKE 'idx_failure_events_%'"
        )
        assert tables == 1
        assert indexes == 3
    finally:
        await database.close_db()


async def test_gate_decisions_schema_is_idempotent(local_env: Path) -> None:
    await database.init_db()
    await database.init_db()
    try:
        db = await database.get_db()
        tables = await db.fetchval(
            "SELECT COUNT(*) FROM sqlite_master"
            " WHERE type = 'table' AND name = 'gate_decisions'"
        )
        indexes = await db.fetchval(
            "SELECT COUNT(*) FROM sqlite_master"
            " WHERE type = 'index' AND name LIKE 'idx_gate_decisions_%'"
        )
        assert tables == 1
        assert indexes == 3
    finally:
        await database.close_db()


_GATE_INSERT = (
    "INSERT INTO gate_decisions (decision_id, session_id, harness, mode, tool,"
    " decision, ts, received_ts) VALUES ($1, 's1', 'claude-code', 'interactive',"
    " 'Bash', $2, '2026-10-07T12:00:00+00:00', '2026-10-07T12:00:00+00:00')"
)


async def test_gate_decisions_old_check_is_rebuilt(local_env: Path) -> None:
    """An existing SQLite gate_decisions with the 7-member CHECK is widened."""
    create = next(
        s
        for s in database._SCHEMA_STATEMENTS
        if s.startswith("CREATE TABLE IF NOT EXISTS gate_decisions (")
    )
    old_ddl = create.replace(
        "'summary', 'failure_context', 'failure_context_repeat',"
        " 'failure_context_error', 'rearmed', 'overridden'",
        "'summary'",
    ).replace(database._PG_IDENTITY, database._SQLITE_IDENTITY)
    assert "'failure_context" not in old_ddl
    assert "'rearmed'" not in old_ddl
    indexes = [
        s
        for s in database._SCHEMA_STATEMENTS
        if s.startswith("CREATE INDEX IF NOT EXISTS idx_gate_decisions_")
    ]
    assert len(indexes) == 3
    raw = await SqlitePool.open(local_env)
    try:
        await raw.execute(old_ddl)
        for stmt in indexes:
            await raw.execute(stmt)
        await raw.execute(_GATE_INSERT, "d-armed", "armed")
        armed_id = await raw.fetchval(
            "SELECT id FROM gate_decisions WHERE decision_id = 'd-armed'"
        )
        with pytest.raises(sqlite3.IntegrityError):
            await raw.execute(_GATE_INSERT, "d-fc0", "failure_context")
    finally:
        await raw.close()

    await database.init_db()
    try:
        db = await database.get_db()
        assert (
            await db.fetchval(
                "SELECT id FROM gate_decisions WHERE decision_id = 'd-armed'"
            )
            == armed_id
        )
        await db.execute(_GATE_INSERT, "d-fc1", "failure_context")
        await db.execute(_GATE_INSERT, "d-re1", "rearmed")
        await db.execute(_GATE_INSERT, "d-ov1", "overridden")
        sql = await db.fetchval(
            "SELECT sql FROM sqlite_master"
            " WHERE type = 'table' AND name = 'gate_decisions'"
        )
        assert database._GATE_DECISIONS_CHECK_MARKER in sql
        assert (
            await db.fetchval(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index'"
                " AND name LIKE 'idx_gate_decisions_%' AND tbl_name = 'gate_decisions'"
            )
            == 3
        )
        assert (
            await db.fetchval(
                "SELECT COUNT(*) FROM sqlite_master"
                " WHERE name = 'gate_decisions_pre_rebuild'"
            )
            == 0
        )
        count = await db.fetchval("SELECT COUNT(*) FROM gate_decisions")
    finally:
        await database.close_db()

    await database.init_db()
    try:
        db = await database.get_db()
        assert await db.fetchval("SELECT COUNT(*) FROM gate_decisions") == count
        assert (
            await db.fetchval(
                "SELECT sql FROM sqlite_master"
                " WHERE type = 'table' AND name = 'gate_decisions'"
            )
            == sql
        )
    finally:
        await database.close_db()


async def test_add_column_migration_upgrades_an_older_table(local_env: Path) -> None:
    """A pre-existing table missing the ALTERed columns is upgraded in place."""
    raw = await SqlitePool.open(local_env)
    await raw.execute(
        "CREATE TABLE whisper_telemetry (session_id TEXT NOT NULL, host TEXT NOT"
        " NULL, surface TEXT NOT NULL, map_id TEXT NOT NULL, source_kb TEXT NOT"
        " NULL, cwd_project TEXT, trigger_context TEXT NOT NULL DEFAULT '{}',"
        " emitted_ts TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0,"
        " consumed_ts TEXT, build_engine TEXT, flushed_at TEXT NOT NULL,"
        " PRIMARY KEY (session_id, surface, map_id))"
    )
    await raw.execute(
        "INSERT INTO whisper_telemetry (session_id, host, surface, map_id,"
        " source_kb, emitted_ts, flushed_at) VALUES ($1, $2, $3, $4, $5, $6, $7)",
        "s",
        "h",
        "roster",
        "kb-1",
        "personal",
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:00+00:00",
    )
    await raw.close()

    await database.init_db()
    try:
        db = await database.get_db()
        row = await db.fetchrow("SELECT * FROM whisper_telemetry")
        assert row is not None
        assert row["emit_count"] == 1
        assert row["last_emitted_ts"] == "2026-01-01T00:00:00+00:00"
    finally:
        await database.close_db()


# ─── get_db() callers on SQLite ─────────────────────────────────────────────


async def test_attribution_settings_round_trip(pool: SqlitePool) -> None:
    assert await attribution.get_setting("team") is None
    await attribution.set_setting("team", "platform")
    await attribution.set_setting("team", "  core  ")
    assert await attribution.get_setting("team") == "  core  "
    attr = await attribution.resolve_attribution(_local_user())
    assert attr.contributor == "local@localhost"
    assert attr.team == "core"
    await attribution.delete_setting("team")
    assert await attribution.get_setting("team") is None
    assert await attribution.is_machine_principal(_local_user()) is False


async def test_listener_decision_insert(pool: SqlitePool) -> None:
    await listener_routes._record_listener_decision(
        session_id="sess",
        cwd_project="proj",
        source_kb="personal",
        candidates_considered=2,
        reason="whispered",
        candidate_signal="lexical",
        candidate_ids=["kb-1", "kb-2"],
        whispered_ids=["kb-1"],
        n_retrieved=2,
        n_after_a=2,
        n_after_b=1,
    )
    row = await pool.fetchrow("SELECT * FROM listener_decisions")
    assert row is not None
    assert row["decision"] == "whisper"
    assert row["candidate_signal"] == "lexical"
    assert json.loads(row["candidate_ids"]) == ["kb-1", "kb-2"]
    assert row["retrieval_path"] == "primary"


async def test_chat_history_on_sqlite(pool: SqlitePool) -> None:
    await pool.execute(
        "INSERT INTO users (id, email, hashed_password, is_admin, created_at)"
        " VALUES ($1, $2, $3, $4, $5)",
        "u1",
        "u1@example.com",
        "",
        0,
        "2026-01-01T00:00:00+00:00",
    )
    await chat_history.create_chat("c1", "u1", "Title")
    await chat_history.save_message("c1", "user", "hi")
    await chat_history.save_messages_bulk(
        "c1", [{"role": "assistant", "content": "yo"}]
    )
    assert await chat_history.chat_exists("c1", "u1")
    assert [c["id"] for c in await chat_history.list_chats("u1")] == ["c1"]
    assert await chat_history.get_messages("c1") == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    assert await chat_history.delete_chat("c1", "u1") is True
    assert await chat_history.delete_chat("c1", "u1") is False


async def test_transaction_rolls_back_on_error(pool: SqlitePool) -> None:
    with pytest.raises(ValueError, match="boom"):
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO app_config (key, value, updated_at) VALUES ($1, $2, $3)",
                "k",
                "v",
                "now",
            )
            raise ValueError("boom")
    assert await pool.fetchrow("SELECT * FROM app_config WHERE key = $1", "k") is None

    async with pool.acquire() as conn, conn.transaction():
        status = await conn.execute(
            "INSERT INTO app_config (key, value, updated_at) VALUES ($1, $2, $3)",
            "k",
            "v",
            "now",
        )
    assert status == "INSERT 0 1"
    assert (
        await pool.fetchval("SELECT value FROM app_config WHERE key = $1", "k") == "v"
    )
    assert await pool.execute("UPDATE app_config SET value = $1", "w") == "UPDATE 1"


# ─── full app, real lifespan ────────────────────────────────────────────────


@pytest.fixture
def local_client(local_env: Path) -> Iterator[TestClient]:
    """The real FastAPI app with a real SQLite data DB and service DB."""
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_local_mode_writes_succeed(local_client: TestClient, local_env: Path) -> None:
    assert local_env.is_file()

    created = local_client.post(
        "/api/kb/store",
        json={
            "short_title": "Local write",
            "long_title": "A write in local no-auth mode",
            "knowledge_details": "Writes work against a SQLite service DB.",
            "project_ref": "personal-kb",
        },
    )
    assert created.status_code == 200, created.text
    entry = created.json()["entry"]
    assert created.json()["action"] == "created"
    assert entry["contributor"] == "local@localhost"

    updated = local_client.post(
        "/api/kb/store",
        json={
            "update_entry_id": entry["id"],
            "knowledge_details": "Updated in local mode.",
            "change_reason": "test",
        },
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["action"] == "updated"

    fb = local_client.post(
        "/api/kb/feedback", json={"feedback_type": "friction", "detail": "x"}
    )
    assert fb.status_code == 200, fb.text
    assert fb.json() == {"status": "recorded", "feedback_type": "friction"}


def _telemetry_row(emitted_ts: str, consumed: bool = False) -> dict[str, object]:
    return {
        "session_id": "sess",
        "host": "laptop",
        "surface": "roster",
        "map_id": "kb-1",
        "source_kb": "personal",
        "cwd_project": "personal-kb",
        "trigger_context": {"k": "v"},
        "emitted_ts": emitted_ts,
        "consumed": consumed,
    }


def test_local_mode_telemetry_flush_upserts(local_client: TestClient) -> None:
    first = "2026-10-06T20:00:00.123456+00:00"
    later = "2026-10-06T21:00:00+00:00"
    for rows in (
        [_telemetry_row(first)],
        [_telemetry_row(first, consumed=True)],  # re-delivery: no new emission
        [_telemetry_row(later, consumed=True)],  # genuine re-emission
    ):
        resp = local_client.post("/api/kb/telemetry/whispers", json={"rows": rows})
        assert resp.status_code == 200, resp.text

    import sqlite3

    conn = sqlite3.connect(database.sqlite_service_db_path())
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM whisper_telemetry").fetchone()
    finally:
        conn.close()
    assert row["emit_count"] == 2
    assert row["emitted_ts"] == first
    assert row["last_emitted_ts"] == later
    assert row["consumed"] == 1


async def test_turn_events_schema_is_idempotent(local_env: Path) -> None:
    await database.init_db()
    await database.init_db()
    try:
        db = await database.get_db()
        tables = await db.fetchval(
            "SELECT COUNT(*) FROM sqlite_master"
            " WHERE type = 'table' AND name = 'turn_events'"
        )
        indexes = await db.fetchval(
            "SELECT COUNT(*) FROM sqlite_master"
            " WHERE type = 'index' AND name LIKE 'idx_turn_events_%'"
        )
        assert tables == 1
        assert indexes == 2
    finally:
        await database.close_db()


async def test_surprise_schema_is_idempotent(local_env: Path) -> None:
    await database.init_db()
    await database.init_db()
    try:
        db = await database.get_db()
        for table in ("surprise_candidates", "surprise_detections"):
            count = await db.fetchval(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = $1",
                table,
            )
            assert count == 1
            indexes = await db.fetchval(
                "SELECT COUNT(*) FROM sqlite_master"
                " WHERE type = 'index' AND name LIKE $1",
                f"idx_{table}_%",
            )
            assert indexes == 2
        cols = await db.fetch("PRAGMA table_info(turn_events)")
        assert "detected_at" not in {c["name"] for c in cols}
    finally:
        await database.close_db()


async def test_surprise_distillations_schema_is_idempotent(local_env: Path) -> None:
    await database.init_db()
    await database.init_db()
    try:
        db = await database.get_db()
        tables = await db.fetchval(
            "SELECT COUNT(*) FROM sqlite_master"
            " WHERE type = 'table' AND name = 'surprise_distillations'"
        )
        indexes = await db.fetchval(
            "SELECT COUNT(*) FROM sqlite_master"
            " WHERE type = 'index' AND name LIKE 'idx_surprise_distillations_%'"
        )
        assert tables == 1
        assert indexes == 2
        cols = await db.fetch("PRAGMA table_info(surprise_distillations)")
        names = {c["name"] for c in cols}
        assert "mode" not in names
        assert len(names) == 24  # id + 23 data columns
    finally:
        await database.close_db()


# --- surprise shape CHECK ------------------------------------------------------

_SHAPE_INSERT = {
    "surprise_candidates": (
        "INSERT INTO surprise_candidates (shape, session_id, detector_model,"
        " created_at) VALUES ($1, 's1', 'm', 't')"
    ),
    "surprise_detections": (
        "INSERT INTO surprise_detections (event_id, session_id, shape, mode,"
        " outcome, detector_version, ts) VALUES ('s1:0', 's1', $1, 'shadow',"
        " 'no_surprise', 1, 't')"
    ),
    "surprise_distillations": (
        "INSERT INTO surprise_distillations (candidate_id, session_id, shape,"
        " outcome, distiller_version, ts) VALUES (1, 's1', $1, 'written', 1, 't')"
    ),
}
_SHAPE_TABLES = list(_SHAPE_INSERT)


def _create(table: str) -> str:
    prefix = f"CREATE TABLE IF NOT EXISTS {table} ("
    return next(s for s in database._SCHEMA_STATEMENTS if s.startswith(prefix))


def _index_stmts(table: str) -> list[str]:
    return [
        s
        for s in database._SCHEMA_STATEMENTS
        if s.startswith("CREATE ") and " INDEX " in s and f" ON {table}(" in s
    ]


def _drop(table: str) -> str:
    return f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {table}_shape_check"


def _add(table: str) -> str:
    return (
        f"ALTER TABLE {table} ADD CONSTRAINT {table}_shape_check"
        " CHECK (shape IN (1, 2, 3, 4, 5))"
    )


def test_surprise_shape_constants() -> None:
    assert database.SURPRISE_SHAPES == (1, 2, 3, 4, 5)
    assert database._SURPRISE_SHAPE_CHECK == "CHECK (shape IN (1, 2, 3, 4, 5))"
    assert typing.get_args(models.SurpriseShape) == database.SURPRISE_SHAPES
    for table in _SHAPE_TABLES:
        assert "shape INTEGER NOT NULL CHECK (shape IN (1, 2, 3, 4, 5)), " in _create(
            table
        )
    assert "CHECK (shape" not in _create("surprise_dry_runs")


def test_surprise_shape_postgres_statement_order() -> None:
    stmts = database._SCHEMA_STATEMENTS
    for table in _SHAPE_TABLES:
        assert (
            stmts.index(_create(table))
            < stmts.index(_drop(table))
            < stmts.index(_add(table))
        )


async def test_postgres_init_runs_shape_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[str] = []

    class _Conn:
        async def execute(self, sql: str, *args: Any) -> str:
            executed.append(sql)
            return "OK"

    class _Pool:
        @asynccontextmanager
        async def acquire(self) -> AsyncIterator[_Conn]:
            yield _Conn()

    monkeypatch.setattr(database, "_pool", _Pool())
    await database.init_db()
    monkeypatch.setattr(database, "_pool", None)
    expected = [stmt for t in _SHAPE_TABLES for stmt in (_drop(t), _add(t))]
    assert [s for s in executed if s in expected] == expected


@pytest.mark.parametrize("table", _SHAPE_TABLES)
async def test_surprise_shape_check_is_rebuilt(
    local_env: Path, caplog: pytest.LogCaptureFixture, table: str
) -> None:
    old_ddl = (
        _create(table)
        .replace(database._SURPRISE_SHAPE_CHECK, "CHECK (shape IN (1, 2, 3))")
        .replace(database._PG_IDENTITY, database._SQLITE_IDENTITY)
    )
    assert database._SURPRISE_SHAPE_CHECK not in old_ddl
    insert = _SHAPE_INSERT[table]
    raw = await SqlitePool.open(local_env)
    try:
        await raw.execute(old_ddl)
        for stmt in _index_stmts(table):
            await raw.execute(stmt)
        await raw.execute(insert, 3)
        old_id = await raw.fetchval(f"SELECT id FROM {table}")  # noqa: S608
        with pytest.raises(sqlite3.IntegrityError):
            await raw.execute(insert, 4)
    finally:
        await raw.close()

    await database.init_db()
    try:
        db = await database.get_db()
        assert (
            await db.fetchval(f"SELECT id FROM {table} WHERE shape = 3")  # noqa: S608
            == old_id
        )
        await db.execute(insert, 5)
        with pytest.raises(sqlite3.IntegrityError):
            await db.execute(insert, 6)
        sql = await db.fetchval(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = $1", table
        )
        assert database._SURPRISE_SHAPE_CHECK in sql
        assert (
            await db.fetchval(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index'"
                " AND name LIKE $1 AND tbl_name = $2",
                f"idx_{table}_%",
                table,
            )
            == 2
        )
        assert (
            await db.fetchval(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = $1",
                f"{table}_pre_rebuild",
            )
            == 0
        )
        assert f"service_db check_rebuild table={table} rows=1" in caplog.text
    finally:
        await database.close_db()

    caplog.clear()
    await database.init_db()
    try:
        db = await database.get_db()
        assert (
            await db.fetchval(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = $1",
                table,
            )
            == sql
        )
        assert await db.fetchval(f"SELECT COUNT(*) FROM {table}") == 2  # noqa: S608
        assert "check_rebuild" not in caplog.text
    finally:
        await database.close_db()


async def test_fresh_db_surprise_shape_check(local_env: Path) -> None:
    await database.init_db()
    try:
        db = await database.get_db()
        for table, insert in _SHAPE_INSERT.items():
            await db.execute(insert, 5)
            with pytest.raises(sqlite3.IntegrityError):
                await db.execute(insert, 6)
            assert await db.fetchval(f"SELECT COUNT(*) FROM {table}") == 1  # noqa: S608
    finally:
        await database.close_db()


@pytest.mark.parametrize("table", _SHAPE_TABLES)
async def test_surprise_shape_check_rebuilt_from_shape4(
    local_env: Path, table: str
) -> None:
    """A 043c8dd3-era (1..4) CHECK is widened to admit the write-policy shape 5."""
    old_ddl = (
        _create(table)
        .replace(database._SURPRISE_SHAPE_CHECK, "CHECK (shape IN (1, 2, 3, 4))")
        .replace(database._PG_IDENTITY, database._SQLITE_IDENTITY)
    )
    insert = _SHAPE_INSERT[table]
    raw = await SqlitePool.open(local_env)
    try:
        await raw.execute(old_ddl)
        await raw.execute(insert, 4)
        with pytest.raises(sqlite3.IntegrityError):
            await raw.execute(insert, 5)
    finally:
        await raw.close()
    await database.init_db()
    try:
        db = await database.get_db()
        assert await db.fetchval(f"SELECT COUNT(*) FROM {table} WHERE shape = 4") == 1  # noqa: S608
        await db.execute(insert, 5)
        with pytest.raises(sqlite3.IntegrityError):
            await db.execute(insert, 6)
    finally:
        await database.close_db()


# --- api_keys.surface (write policy) -------------------------------------------

_SURFACE_STMT = (
    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS surface TEXT CHECK (surface IS NULL"
    " OR surface IN ('interactive', 'headless', 'autonomous'))"
)
_OLD_API_KEYS_DDL = (
    "CREATE TABLE IF NOT EXISTS api_keys (id TEXT PRIMARY KEY, user_id TEXT NOT"
    " NULL REFERENCES users(id), key_hash TEXT UNIQUE NOT NULL, name TEXT NOT NULL"
    " DEFAULT '', created_at TEXT NOT NULL)"
)


def test_api_keys_surface_statement_position() -> None:
    stmts = database._SCHEMA_STATEMENTS
    idx = stmts.index(
        "CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)"
    )
    assert stmts[idx + 1] == _SURFACE_STMT


async def test_api_keys_surface_migration(local_env: Path) -> None:
    raw = await SqlitePool.open(local_env)
    try:
        await raw.execute(
            "CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,"
            " hashed_password TEXT NOT NULL, is_admin INTEGER NOT NULL DEFAULT 0,"
            " created_at TEXT NOT NULL)"
        )
        await raw.execute(_OLD_API_KEYS_DDL)
        await raw.execute(
            "INSERT INTO users (id, email, hashed_password, created_at)"
            " VALUES ('u1', 'a@b', 'x', 't')"
        )
        await raw.execute(
            "INSERT INTO api_keys (id, user_id, key_hash, name, created_at)"
            " VALUES ('k1', 'u1', 'h1', 'n', 't')"
        )
    finally:
        await raw.close()

    await database.init_db()
    try:
        db = await database.get_db()
        cols = await db.fetch("PRAGMA table_info(api_keys)")
        assert "surface" in {c["name"] for c in cols}
        assert await db.fetchval("SELECT surface FROM api_keys WHERE id = 'k1'") is None
        with pytest.raises(sqlite3.IntegrityError):
            await db.execute("UPDATE api_keys SET surface = 'bogus' WHERE id = 'k1'")
        await db.execute("UPDATE api_keys SET surface = 'headless' WHERE id = 'k1'")
        assert (
            await db.fetchval("SELECT surface FROM api_keys WHERE id = 'k1'")
            == "headless"
        )
    finally:
        await database.close_db()

    await database.init_db()
    try:
        db = await database.get_db()
        assert len(await db.fetch("PRAGMA table_info(api_keys)")) == len(cols)
    finally:
        await database.close_db()
