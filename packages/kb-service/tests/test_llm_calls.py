"""Tests for the llm_calls ledger: env switch, DDL, recorder, drain, prune, wiring.

DB tests run against the local SQLite service DB (KB_AUTH_MODE=none, a tmp
KB_DB_PATH). No network.
"""

import ast
import asyncio
import dataclasses
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from kb_core.llm.provider import LLMCallRecord

import kb_service.database as database
from kb_service import llm_calls, surprise_worker
from kb_service.config import build_anthropic_config, get_llm_diagnostics
from kb_service.db_sqlite import SqlitePool
from kb_service.llm_calls import (
    candidate_scope,
    current_candidate_id,
    drain_pending_llm_calls,
    insert_llm_call,
    prune_llm_calls,
    record_llm_call,
)

_SRC = Path(__file__).resolve().parent.parent / "src" / "kb_service"


@pytest.fixture(autouse=True)
def _fresh_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_calls, "_PENDING", set())


@pytest.fixture
def local_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.delenv("KB_SERVICE_DATABASE_URL", raising=False)
    monkeypatch.delenv("KB_DATABASE_URL", raising=False)
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setattr(database, "_pool", None)
    return tmp_path / "service.db"


@pytest.fixture
async def pool(local_env: Path) -> AsyncIterator[SqlitePool]:
    await database.init_db()
    db = await database.get_db()
    assert isinstance(db, SqlitePool)
    yield db
    await database.close_db()


def _record(**overrides: Any) -> LLMCallRecord:
    fields: dict[str, Any] = {
        "ts": "2026-10-11T12:00:00+00:00",
        "purpose": "planner",
        "model": "claude-haiku-4-5",
        "outcome": "ok",
        "error_type": "NoError",
        "response_id": "msg_1",
        "stop_reason": "end_turn",
        "input_tokens": 11,
        "output_tokens": 12,
        "cache_creation_input_tokens": 13,
        "cache_creation_1h_input_tokens": 14,
        "cache_read_input_tokens": 15,
        "diagnostics_state": "sent",
        "previous_message_id": "msg_0",
        "cache_miss_reason": "system_changed",
        "cache_missed_input_tokens": 16,
        "diagnostics_raw": '{"cache_miss_reason":null}',
        "system_sha256": "abcdef0123456789",
        "latency_ms": 17,
        "thinking": "disabled",
        "effort": "low",
    }
    fields.update(overrides)
    return LLMCallRecord(**fields)


# --- AC-14: env switch ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("", True),
        ("1", True),
        ("0", False),
        ("false", False),
        ("OFF", False),
    ],
)
def test_get_llm_diagnostics(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
) -> None:
    if value is None:
        monkeypatch.delenv("KB_LLM_DIAGNOSTICS", raising=False)
    else:
        monkeypatch.setenv("KB_LLM_DIAGNOSTICS", value)
    assert get_llm_diagnostics() is expected
    assert build_anthropic_config().diagnostics is expected


def test_no_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_LLM_DIAGNOSTICS", " no ")
    assert get_llm_diagnostics() is False


# --- AC-15: DDL ----------------------------------------------------------------


async def test_llm_calls_ddl_idempotent(local_env: Path) -> None:
    await database.init_db()
    await database.init_db()
    pool = await database.get_db()
    try:
        names = {
            r["name"]
            for r in await pool.fetch(
                "SELECT name FROM sqlite_master WHERE name LIKE '%llm_calls%'"
            )
        }
        assert {"llm_calls", "idx_llm_calls_ts", "idx_llm_calls_purpose_ts"} <= names
        cols = [r["name"] for r in await pool.fetch("PRAGMA table_info(llm_calls)")]
        # SQLite fresh installs get candidate_id from the CREATE TABLE and the
        # older text columns via ALTER, so the physical order is not the record
        # field order; the names are the invariant.
        assert set(cols) == {"id"} | {f.name for f in dataclasses.fields(LLMCallRecord)}
    finally:
        await database.close_db()


# --- AC-16: recorder module ---------------------------------------------------


def test_insert_sql_columns_in_field_order() -> None:
    names = [f.name for f in dataclasses.fields(LLMCallRecord)]
    expected = (
        f"INSERT INTO llm_calls ({', '.join(names)}) "  # noqa: S608 - fixed columns
        f"VALUES ({', '.join(f'${i}' for i in range(1, 23))})"
    )
    assert expected == llm_calls._INSERT_SQL


async def test_insert_round_trip(pool: SqlitePool) -> None:
    record = _record()
    await insert_llm_call(pool, record)
    row = await pool.fetchrow("SELECT * FROM llm_calls")
    assert row is not None
    assert row["id"] == 1
    for name, value in dataclasses.asdict(record).items():
        assert row[name] == value, name


async def test_record_is_non_blocking(
    pool: SqlitePool, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = asyncio.Event()
    calls: list[LLMCallRecord] = []

    async def _slow_insert(p: Any, record: LLMCallRecord) -> None:
        await gate.wait()
        calls.append(record)

    monkeypatch.setattr(llm_calls, "insert_llm_call", _slow_insert)
    record_llm_call(_record())
    assert len(llm_calls._PENDING) == 1
    assert calls == []
    gate.set()
    await drain_pending_llm_calls()
    assert len(calls) == 1


async def test_insert_failure_logs_and_clears(
    pool: SqlitePool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _boom(p: Any, record: LLMCallRecord) -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr(llm_calls, "insert_llm_call", _boom)
    with caplog.at_level(logging.WARNING, logger="kb_service.llm_calls"):
        record_llm_call(_record())
        await drain_pending_llm_calls()
        await asyncio.sleep(0)
    assert not llm_calls._PENDING
    msgs = [r.getMessage() for r in caplog.records if "insert failed" in r.getMessage()]
    assert len(msgs) == 1
    assert "purpose=planner" in msgs[0]
    assert "model=claude-haiku-4-5" in msgs[0]
    assert "response_id=msg_1" in msgs[0]


async def test_no_pool_writes_nothing(local_env: Path) -> None:
    record_llm_call(_record())
    await drain_pending_llm_calls()
    assert database.current_pool() is None
    assert not database.sqlite_service_db_path().exists()


async def test_drain_with_nothing_pending() -> None:
    assert await drain_pending_llm_calls() is None


class _ForeignTask:
    def get_loop(self) -> object:
        return object()


async def test_drain_discards_foreign_loop_task() -> None:
    stub: Any = _ForeignTask()
    llm_calls._PENDING.add(stub)
    await drain_pending_llm_calls()
    assert not llm_calls._PENDING


async def test_drain_timeout_cancels(
    pool: SqlitePool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _never(p: Any, record: LLMCallRecord) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(llm_calls, "insert_llm_call", _never)
    with caplog.at_level(logging.WARNING, logger="kb_service.llm_calls"):
        record_llm_call(_record())
        await drain_pending_llm_calls(timeout=0.01)
        await asyncio.sleep(0)
    msgs = [r.getMessage() for r in caplog.records]
    assert "llm_calls drain timed out pending=1" in msgs
    assert not llm_calls._PENDING
    assert not [m for m in msgs if "insert failed" in m]


# --- candidate attribution: contextvar and scope ------------------------------


async def test_candidate_scope_sets_and_resets() -> None:
    with candidate_scope(7):
        assert current_candidate_id.get() == 7
    assert current_candidate_id.get() is None


async def test_candidate_scope_resets_on_exception() -> None:
    with pytest.raises(RuntimeError), candidate_scope(3):
        raise RuntimeError("boom")
    assert current_candidate_id.get() is None


async def test_candidate_scope_nesting_restores_outer() -> None:
    with candidate_scope(1):
        with candidate_scope(2):
            assert current_candidate_id.get() == 2
        assert current_candidate_id.get() == 1
    assert current_candidate_id.get() is None


async def test_record_llm_call_fills_candidate_id(pool: SqlitePool) -> None:
    inside = _record()
    outside = _record()
    explicit = _record(candidate_id=5)
    with candidate_scope(9):
        record_llm_call(inside)
    record_llm_call(outside)
    with candidate_scope(6):
        record_llm_call(explicit)
    await drain_pending_llm_calls()
    rows = await pool.fetch("SELECT candidate_id FROM llm_calls ORDER BY id")
    assert [r["candidate_id"] for r in rows] == [9, None, 5]


async def test_prune(pool: SqlitePool) -> None:
    now = datetime(2026, 10, 11, tzinfo=UTC)
    await insert_llm_call(
        pool, _record(ts=(now - timedelta(days=100)).isoformat(timespec="seconds"))
    )
    await insert_llm_call(
        pool, _record(ts=(now - timedelta(days=10)).isoformat(timespec="seconds"))
    )
    assert await prune_llm_calls(pool, now=now) == 1
    rows = await pool.fetch("SELECT ts FROM llm_calls")
    assert [r["ts"] for r in rows] == ["2026-10-01T00:00:00+00:00"]


# --- AC-17: wiring -----------------------------------------------------------


@pytest.mark.parametrize(
    ("getter", "cache"),
    [
        ("get_detector_llm", "_DETECTOR_LLM_CACHE"),
        ("get_distiller_llm", "_DISTILLER_LLM_CACHE"),
        ("get_critic_llm", "_CRITIC_LLM_CACHE"),
    ],
)
def test_surprise_clients_carry_recorder(
    monkeypatch: pytest.MonkeyPatch, getter: str, cache: str
) -> None:
    monkeypatch.setattr(surprise_worker, cache, {})
    client = getattr(surprise_worker, getter)()
    assert client is not None
    assert client._recorder is record_llm_call


def _anthropic_calls() -> list[tuple[Path, ast.Call]]:
    out: list[tuple[Path, ast.Call]] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (isinstance(func, ast.Name) and func.id == "AnthropicLLMClient") or (
                isinstance(func, ast.Attribute) and func.attr == "AnthropicLLMClient"
            ):
                out.append((path, node))
    return out


def test_every_service_anthropic_client_has_a_recorder() -> None:
    calls = _anthropic_calls()
    assert calls, "expected AnthropicLLMClient constructions in kb_service"
    missing = [
        f"{path.relative_to(_SRC)}:{node.lineno}"
        for path, node in calls
        if not any(kw.arg == "recorder" for kw in node.keywords)
    ]
    assert missing == []


def _bedrock_calls() -> list[tuple[Path, ast.Call]]:
    out: list[tuple[Path, ast.Call]] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (isinstance(func, ast.Name) and func.id == "BedrockLLMClient") or (
                isinstance(func, ast.Attribute) and func.attr == "BedrockLLMClient"
            ):
                out.append((path, node))
    return out


def test_every_service_bedrock_client_has_a_recorder() -> None:
    calls = _bedrock_calls()
    assert calls, "expected BedrockLLMClient constructions in kb_service"
    missing = [
        f"{path.relative_to(_SRC)}:{node.lineno}"
        for path, node in calls
        if not any(kw.arg == "recorder" for kw in node.keywords)
    ]
    assert missing == []
