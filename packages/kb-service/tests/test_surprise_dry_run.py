"""Tests for shadow-mode dry runs and GET /api/kb/surprise/candidates.

Dry runs run on a real SQLite KB plus a real SQLite service DB with a
scripted FakeLLM (fixtures shared with ``test_surprise_distill_worker``);
the endpoint is driven through the real app in local (no-auth) mode. The
Postgres DDL path is exercised with a statement-recording fake pool. No
network.
"""

import json
import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kb_service.database as database
from kb_service import surprise_worker
from kb_service.db_sqlite import SqlitePool
from kb_service.main import app
from kb_service.surprise import DistillResult, SurpriseCandidate
from kb_service.surprise_distill import (
    SHAPE_DESCRIPTIONS,
    SURPRISE_CRITIC_VERSION,
    SURPRISE_DISTILLER_SYSTEM,
    SURPRISE_DISTILLER_VERSION,
)
from kb_service.surprise_worker import (
    candidate_from_row,
    distill_candidates,
    drain_once,
    dry_run_candidates,
)
from tests.conftest import CRITIC_ACCEPT, S4_OUT, FakeCritic, FakeLLM
from tests.test_surprise_distill_worker import (
    _HERMETIC_ENV,
    _SCRUB_ENV,
    D,
    _cand_row,
    _lessons,
    _make_kb,
    _raise_detector,
    _w,
)


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _HERMETIC_ENV:
        monkeypatch.delenv(var, raising=False)


def _local_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setattr(database, "_pool", None)


@pytest.fixture
def local_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _local_env(tmp_path, monkeypatch)
    return tmp_path / "service.db"


@pytest.fixture
async def pool(local_env: Path) -> AsyncIterator[SqlitePool]:
    await database.init_db()
    db = await database.get_db()
    assert isinstance(db, SqlitePool)
    yield db
    await database.close_db()


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[Any]:
    k = await _make_kb(tmp_path / "kb.db", embedder=None)
    try:
        yield k
    finally:
        await k.close()


@pytest.fixture(autouse=True)
def critic(monkeypatch: pytest.MonkeyPatch) -> FakeCritic:
    fake = FakeCritic()
    monkeypatch.setattr(surprise_worker, "get_critic_llm", lambda: fake)
    return fake


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr(surprise_worker, "get_distiller_llm", lambda: fake)
    monkeypatch.setattr(surprise_worker, "get_detector_llm", _raise_detector)
    return fake


_INSERT_SQL = (
    "INSERT INTO surprise_candidates (shape, session_id, project, turn_event_ids,"
    " detector_model, detector_output, status, entry_id, created_at)"
    " VALUES ($1, $2, $3, $4, $5, $6, $7, NULL, $8) RETURNING id, shape,"
    " session_id, project, turn_event_ids, detector_model, detector_output,"
    " status, entry_id, created_at"
)

_TURN_SQL = (
    "INSERT INTO turn_events (event_id, session_id, harness, mode, host,"
    " project, turn_index, ts, items, capture_mode, received_ts)"
    " VALUES ($1, $2, 'claude-code', $3, $4, 'p', $5, $6, $7, 'shadow', $6)"
)

_NOT_DURABLE = json.dumps({"durable": False, "why": "a one-off typo"})

_RAW_ITEM_TEXT = "sk-live-raw-turn-item-never-returned"


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def _now_iso(**delta: float) -> str:
    return _iso(datetime.now(UTC) - timedelta(**delta))


def _s1_output(**kw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "wrong_belief": "git push origin main",
        "corrected_fact": "git push origin HEAD:main",
        "evidence_excerpt": "rejected",
        "confidence": 1.0,
    }
    out.update(kw)
    return out


async def _cand(
    pool: SqlitePool,
    session: str,
    *,
    status: str = "shadow",
    shape: int = 1,
    project: str = "p",
    created_at: str | None = None,
    output: dict[str, Any] | None = None,
    turn_event_ids: list[str] | None = None,
) -> SurpriseCandidate:
    row = await pool.fetchrow(
        _INSERT_SQL,
        shape,
        session,
        project,
        json.dumps(turn_event_ids or [f"{session}:0"]),
        "rule:shape1" if shape == 1 else "FakeLLM",
        json.dumps(output or _s1_output()),
        status,
        created_at or _now_iso(),
    )
    assert row is not None
    return candidate_from_row(row)


async def _turn(
    pool: SqlitePool, session: str, turn: int = 0, mode: str = "headless"
) -> None:
    await pool.execute(
        _TURN_SQL,
        f"{session}:{turn}",
        session,
        mode,
        "host-a",
        turn,
        _now_iso(),
        json.dumps([{"kind": "assistant_text", "text": _RAW_ITEM_TEXT}]),
    )


async def _dry_runs(pool: SqlitePool) -> list[dict[str, Any]]:
    return await pool.fetch("SELECT * FROM surprise_dry_runs ORDER BY id")


async def _distillations(pool: SqlitePool) -> list[dict[str, Any]]:
    return await pool.fetch("SELECT * FROM surprise_distillations ORDER BY id")


# --- dry_run_candidates -------------------------------------------------------


async def test_shadow_would_write_records_and_writes_nothing(
    pool: SqlitePool, kb: Any, llm: FakeLLM
) -> None:
    await _turn(pool, "s1")
    c = await _cand(pool, "s1")
    llm.enqueue(D)
    assert await dry_run_candidates(pool, kb, [c]) == 1
    assert [call[1] for call in llm.generate_calls] == [SURPRISE_DISTILLER_SYSTEM]
    rows = await _dry_runs(pool)
    assert len(rows) == 1
    r = rows[0]
    assert (r["candidate_id"], r["session_id"], r["project"], r["shape"]) == (
        c.id,
        "s1",
        "p",
        1,
    )
    assert r["would_outcome"] == "would_write"
    assert r["mode"] == "headless"
    assert r["distiller_model"] == "FakeLLM"
    assert r["distiller_version"] == SURPRISE_DISTILLER_VERSION
    payload = json.loads(r["payload"])
    assert payload == {
        "short_title": "Push to HEAD:main",
        "long_title": "Push the current branch with git push origin HEAD:main",
        "corrected_fact": "Push with git push origin HEAD:main",
        "lesson": "git push origin main is rejected as non-fast-forward here.",
        "lesson_class": "none",
        "wrong_belief": "git push origin main",
        "cue": {
            "tool": "Bash",
            "target_class": "git push",
            "args_prefix": "origin main",
        },
        "matched_entry_id": None,
        "similarity": None,
        "critic_version": SURPRISE_CRITIC_VERSION,
        "critic": json.loads(CRITIC_ACCEPT),
    }
    assert await _lessons(kb) == 0
    assert await _cand_row(pool, c.id) == {"status": "shadow", "entry_id": None}
    assert await _distillations(pool) == []


async def test_shadow_not_durable(pool: SqlitePool, kb: Any, llm: FakeLLM) -> None:
    c = await _cand(pool, "s1")
    llm.enqueue(_NOT_DURABLE)
    assert await dry_run_candidates(pool, kb, [c]) == 1
    (r,) = await _dry_runs(pool)
    assert (r["would_outcome"], r["reason"]) == ("not_durable", "a one-off typo")
    assert r["mode"] == ""  # no turn_events row for s1:0
    payload = json.loads(r["payload"])
    assert payload["short_title"] == payload["lesson"] == ""
    assert payload["wrong_belief"] == "git push origin main"
    assert payload["lesson_class"] is None
    assert await _lessons(kb) == 0


async def test_shadow_would_merge_from_another_session(
    pool: SqlitePool, kb: Any, llm: FakeLLM
) -> None:
    w = await _w(kb)
    c = await _cand(pool, "s2")
    assert await dry_run_candidates(pool, kb, [c]) == 1
    assert llm.generate_calls == []  # exact match: no distiller call
    (r,) = await _dry_runs(pool)
    assert r["would_outcome"] == "would_merge"
    assert r["distiller_model"] == ""
    assert json.loads(r["payload"])["matched_entry_id"] == w
    entry = await kb.get(w)
    assert entry.version == 1
    assert entry.hints["resolution"].get("observed_sessions", 1) == 1
    assert await _cand_row(pool, c.id) == {"status": "shadow", "entry_id": None}


async def test_shadow_same_session_and_no_project(
    pool: SqlitePool, kb: Any, llm: FakeLLM
) -> None:
    await _w(kb)  # provenance event id old:0 -> session "old"
    same = await _cand(pool, "old")
    nop = await _cand(pool, "s3", project="")
    assert await dry_run_candidates(pool, kb, [same, nop]) == 2
    rows = await _dry_runs(pool)
    assert [r["would_outcome"] for r in rows] == ["same_session", "no_project"]


async def test_redry_run_adds_nothing(pool: SqlitePool, kb: Any, llm: FakeLLM) -> None:
    c = await _cand(pool, "s1")
    llm.enqueue(D)
    llm.enqueue(D)
    assert await dry_run_candidates(pool, kb, [c]) == 1
    assert await dry_run_candidates(pool, kb, [c]) == 0
    assert len(llm.generate_calls) == 1
    assert len(await _dry_runs(pool)) == 1


async def test_pending_candidate_is_not_dry_run(
    pool: SqlitePool, kb: Any, llm: FakeLLM
) -> None:
    c = await _cand(pool, "s1", status="pending")
    llm.enqueue(D)
    assert await dry_run_candidates(pool, kb, [c]) == 0
    assert llm.generate_calls == []
    assert await _dry_runs(pool) == []


async def test_no_distiller_llm_leaves_it_for_later(
    pool: SqlitePool, kb: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(surprise_worker, "get_distiller_llm", lambda: None)
    c = await _cand(pool, "s1")
    assert await dry_run_candidates(pool, kb, [c]) == 0
    assert await _dry_runs(pool) == []


async def test_mode_on_writes_and_records_no_dry_run(
    pool: SqlitePool, kb: Any, llm: FakeLLM
) -> None:
    c = await _cand(pool, "s1", status="pending")
    llm.enqueue(D)
    res = await distill_candidates(pool, kb, [c], "on")
    assert len(res.entries_written) == 1
    assert await _lessons(kb) == 1
    assert [r["outcome"] for r in await _distillations(pool)] == ["written"]
    assert await _dry_runs(pool) == []


async def test_mode_on_merge_is_unchanged(
    pool: SqlitePool, kb: Any, llm: FakeLLM
) -> None:
    w = await _w(kb)
    c = await _cand(pool, "s2", status="pending")
    res = await distill_candidates(pool, kb, [c], "on")
    assert res == DistillResult([], [w])
    assert (await kb.get(w)).hints["resolution"]["observed_sessions"] == 2
    assert await _cand_row(pool, c.id) == {"status": "merged", "entry_id": w}
    assert await _dry_runs(pool) == []


# --- drain_once wiring --------------------------------------------------------


async def test_drain_shadow_dry_runs_once(
    pool: SqlitePool, kb: Any, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(surprise_worker, "get_detector_llm", lambda: None)
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    fresh = await _cand(pool, "s1")
    old = await _cand(pool, "s0", created_at=_now_iso(hours=48))
    llm.enqueue(D)
    res = await drain_once(pool, kb)
    assert res.entries_written == res.entries_merged == []
    rows = await _dry_runs(pool)
    assert [r["candidate_id"] for r in rows] == [fresh.id]
    assert old.id not in [r["candidate_id"] for r in rows]
    await drain_once(pool, kb)
    assert len(await _dry_runs(pool)) == 1
    assert len(llm.generate_calls) == 1
    assert await _lessons(kb) == 0


@pytest.mark.parametrize("mode", ["off", "on"])
async def test_drain_other_modes_record_no_dry_run(
    pool: SqlitePool,
    kb: Any,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    monkeypatch.setattr(surprise_worker, "get_detector_llm", lambda: None)
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", mode)
    await _cand(pool, "s1")
    llm.enqueue(D)
    await drain_once(pool, kb)
    assert await _dry_runs(pool) == []
    assert llm.generate_calls == []


# --- schema -------------------------------------------------------------------

_DRY_RUN_PREFIX = "CREATE TABLE IF NOT EXISTS surprise_dry_runs ("


def _dry_run_ddl() -> list[str]:
    return [
        s
        for s in database._SCHEMA_STATEMENTS
        if s.startswith(_DRY_RUN_PREFIX) or "ON surprise_dry_runs(" in s
    ]


def test_dry_run_ddl_is_postgres_shaped() -> None:
    create, *indexes = _dry_run_ddl()
    assert database._PG_IDENTITY in create
    assert "CHECK" not in create
    assert len(indexes) == 2
    assert any("(created_at)" in s for s in indexes)


async def test_postgres_init_runs_dry_run_ddl_verbatim(
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
    for stmt in _dry_run_ddl():
        assert stmt in executed
    monkeypatch.setattr(database, "_pool", None)


async def test_sqlite_dry_run_schema(local_env: Path) -> None:
    await database.init_db()
    await database.init_db()
    try:
        db = await database.get_db()
        cols = await db.fetch("PRAGMA table_info(surprise_dry_runs)")
        assert [c["name"] for c in cols] == [
            "id",
            "candidate_id",
            "session_id",
            "project",
            "shape",
            "distiller_model",
            "distiller_version",
            "would_outcome",
            "reason",
            "payload",
            "mode",
            "created_at",
        ]
        indexes = await db.fetchval(
            "SELECT COUNT(*) FROM sqlite_master"
            " WHERE type = 'index' AND name LIKE 'idx_surprise_dry_runs_%'"
        )
        assert indexes == 2
    finally:
        await database.close_db()


# --- GET /api/kb/surprise/candidates -----------------------------------------


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    _local_env(tmp_path, monkeypatch)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _exec(sql: str, *args: Any) -> int:
    conn = sqlite3.connect(database.sqlite_service_db_path())
    try:
        cur = conn.execute(sql, args)
        conn.commit()
        return int(cur.lastrowid or 0)
    finally:
        conn.close()


def _seed_candidate(
    session: str,
    *,
    status: str = "shadow",
    shape: int = 1,
    project: str = "p",
    created_at: str,
    event_ids: list[str] | None = None,
) -> int:
    return _exec(
        "INSERT INTO surprise_candidates (shape, session_id, project,"
        " turn_event_ids, detector_model, detector_output, status, entry_id,"
        " created_at) VALUES (?, ?, ?, ?, 'm', ?, ?, ?, ?)",
        shape,
        session,
        project,
        json.dumps(event_ids if event_ids is not None else [f"{session}:0"]),
        json.dumps({"wrong_belief": f"wb {session}"}),
        status,
        "kb-00001" if status == "written" else None,
        created_at,
    )


def _seed_turn(session: str, mode: str, host: str) -> None:
    _exec(
        "INSERT INTO turn_events (event_id, session_id, harness, mode, host,"
        " project, turn_index, ts, items, capture_mode, received_ts)"
        " VALUES (?, ?, 'claude-code', ?, ?, 'p', 0, ?, ?, 'shadow', ?)",
        f"{session}:0",
        session,
        mode,
        host,
        _now_iso(),
        json.dumps([{"kind": "assistant_text", "text": _RAW_ITEM_TEXT}]),
        _now_iso(),
    )


def _seed_dry_run(candidate_id: int, session: str) -> None:
    _exec(
        "INSERT INTO surprise_dry_runs (candidate_id, session_id, project, shape,"
        " distiller_model, distiller_version, would_outcome, reason, payload,"
        " mode, created_at) VALUES (?, ?, 'p', 1, 'FakeLLM', 4, 'would_write',"
        " '', ?, 'headless', ?)",
        candidate_id,
        session,
        json.dumps({"short_title": "t", "cue": None}),
        _now_iso(),
    )


def _get(client: TestClient, **params: Any) -> list[dict[str, Any]]:
    resp = client.get("/api/kb/surprise/candidates", params=params)
    assert resp.status_code == 200, resp.text
    cands: list[dict[str, Any]] = resp.json()["candidates"]
    return cands


def test_candidates_endpoint(client: TestClient) -> None:
    _seed_turn("a", "headless", "host-a")
    a = _seed_candidate("a", created_at=_now_iso(hours=2))
    _seed_dry_run(a, "a")
    b = _seed_candidate(
        "b", status="written", shape=2, project="q", created_at=_now_iso(hours=1)
    )
    old = _seed_candidate("c", created_at=_now_iso(hours=30))
    empty = _seed_candidate("d", created_at=_now_iso(hours=3), event_ids=[])

    cands = _get(client)
    assert [c["id"] for c in cands] == [b, a, empty]  # newest first, 24h default
    by_id = {c["id"]: c for c in cands}
    ca = by_id[a]
    assert (ca["mode"], ca["host"]) == ("headless", "host-a")
    assert ca["dry_run"] == {
        "would_outcome": "would_write",
        "reason": "",
        "payload": {"short_title": "t", "cue": None},
        "distiller_model": "FakeLLM",
        "distiller_version": 4,
    }
    assert ca["turn_event_ids"] == ["a:0"]
    assert ca["detector_output"] == {"wrong_belief": "wb a"}
    assert ca["status"] == "shadow" and ca["entry_id"] is None
    cb = by_id[b]
    assert cb["dry_run"] is None
    assert (cb["mode"], cb["host"]) == (None, None)  # turn row pruned/absent
    assert cb["entry_id"] == "kb-00001"
    assert by_id[empty]["mode"] is None
    assert _RAW_ITEM_TEXT not in json.dumps(cands)

    assert [c["id"] for c in _get(client, since=_now_iso(hours=48))] == [
        b,
        a,
        empty,
        old,
    ]
    assert [c["id"] for c in _get(client, since=_now_iso(minutes=90))] == [b]
    assert [c["id"] for c in _get(client, status="written")] == [b]
    assert [c["id"] for c in _get(client, shape=2)] == [b]
    assert [c["id"] for c in _get(client, project="q")] == [b]
    assert [c["id"] for c in _get(client, project="p", shape=1)] == [a, empty]
    assert [c["id"] for c in _get(client, limit=1)] == [b]


@pytest.mark.parametrize(
    "params",
    [{"limit": 1001}, {"limit": 0}, {"shape": 6}, {"status": "bogus"}],
)
def test_candidates_endpoint_rejects_bad_params(
    client: TestClient, params: dict[str, Any]
) -> None:
    resp = client.get("/api/kb/surprise/candidates", params=params)
    assert resp.status_code == 422


# --- critic pass in the dry run ---------------------------------------------------


async def test_shadow_critic_rejection_records_critic_rejected(
    pool: SqlitePool, kb: Any, llm: FakeLLM, critic: FakeCritic
) -> None:
    reply = json.loads(CRITIC_ACCEPT)
    reply.update(durable=False, reason="the user said this will change")
    critic.enqueue(json.dumps(reply))
    await _turn(pool, "s1")
    c = await _cand(pool, "s1")
    llm.enqueue(D)
    assert await dry_run_candidates(pool, kb, [c]) == 1
    (r,) = await _dry_runs(pool)
    assert r["would_outcome"] == "critic_rejected"
    assert r["reason"] == "critic: the user said this will change"
    payload = json.loads(r["payload"])
    assert payload["critic_version"] == SURPRISE_CRITIC_VERSION
    assert payload["critic"]["durable"] is False
    assert payload["short_title"] == "Push to HEAD:main"
    assert payload["lesson_class"] == "none"
    assert await _distillations(pool) == []
    assert await _lessons(kb) == 0
    assert len(critic.generate_calls) == 1


async def test_shadow_unparseable_critic_records_unparseable(
    pool: SqlitePool, kb: Any, llm: FakeLLM, critic: FakeCritic
) -> None:
    critic.enqueue("nope")
    await _turn(pool, "s1")
    c = await _cand(pool, "s1")
    llm.enqueue(D)
    assert await dry_run_candidates(pool, kb, [c]) == 1
    (r,) = await _dry_runs(pool)
    assert (r["would_outcome"], r["reason"]) == ("unparseable", "critic: unparseable")


async def test_shadow_shape4_would_write(
    pool: SqlitePool, kb: Any, llm: FakeLLM, critic: FakeCritic
) -> None:
    await _turn(pool, "s4")
    c = await _cand(pool, "s4", shape=4, output=S4_OUT)
    llm.enqueue(D)
    assert await dry_run_candidates(pool, kb, [c]) == 1
    (r,) = await _dry_runs(pool)
    assert (r["shape"], r["would_outcome"], r["mode"]) == (4, "would_write", "headless")
    payload = json.loads(r["payload"])
    assert payload["cue"] is None
    assert payload["critic_version"] == SURPRISE_CRITIC_VERSION
    assert llm.generate_calls[0][0].split("\n")[0] == SHAPE_DESCRIPTIONS[4]
    assert "[tool_result]" not in critic.generate_calls[0][0]
    assert await _lessons(kb) == 0


def test_candidates_endpoint_shape4(client: TestClient) -> None:
    e = _seed_candidate("e", shape=4, created_at=_now_iso(hours=1))
    assert [(c["id"], c["shape"]) for c in _get(client, shape=4)] == [(e, 4)]
