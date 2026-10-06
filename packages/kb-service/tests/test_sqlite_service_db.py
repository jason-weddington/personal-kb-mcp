"""Local-mode SQLite service DB: unset KB_SERVICE_DATABASE_URL + KB_AUTH_MODE=none.

These tests use the REAL ``kb_service.database`` (no fake pool): ``get_db()``
opens ``service.db`` next to a temp ``KB_DB_PATH``. Every test resets the
module-level pool so nothing leaks between tests or into the rest of the suite.
"""

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import kb_service.database as database
from kb_service import attribution, chat_history
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
