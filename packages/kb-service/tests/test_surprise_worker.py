"""Tests for kb_service.surprise_worker: helpers, worker loop, drain scenarios.

The drain scenarios run the real app in local no-auth mode (SQLite service
DB in tmp_path), seed digests through POST /api/kb/turn and drive
POST /api/kb/surprise/drain with a scripted FakeLLM. No network.
"""

import asyncio
import json
import logging
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kb_service.database as database
from kb_service import surprise_worker
from kb_service.main import app
from kb_service.surprise import (
    SURPRISE_DETECTOR_SYSTEM,
    SURPRISE_DETECTOR_VERSION,
    DistillResult,
    SurpriseCandidate,
)
from tests.conftest import FakeLLM

LOGGER = "kb_service.surprise_worker"

_EMPTY = {
    "digests_processed": 0,
    "candidates": [],
    "entries_written": [],
    "entries_merged": [],
}

_SHAPE2_TRUE = (
    '{"surprise": true, "wrong_belief": "port 8080 is free",'
    ' "corrected_fact": "port 8080 is taken by caddy",'
    ' "evidence_excerpt": "8080 is taken by caddy", "confidence": 0.9}'
)
SCRIPT_B = ['{"surprise": false}', _SHAPE2_TRUE]


# --- helpers (AC-12) -------------------------------------------------------------


@pytest.fixture
def fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(surprise_worker, "_DETECTOR_LLM_CACHE", {})


def test_get_detector_llm_default(fresh_cache: None) -> None:
    llm = surprise_worker.get_detector_llm()
    assert type(llm).__name__ == "AnthropicLLMClient"
    assert llm._config.model == "claude-sonnet-5-5"  # type: ignore[union-attr]


def test_get_detector_llm_override_is_cached(
    fresh_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_DETECTOR_MODEL", "test-model-x")
    llm = surprise_worker.get_detector_llm()
    assert llm._config.model == "test-model-x"  # type: ignore[union-attr]
    assert surprise_worker.get_detector_llm() is llm
    assert surprise_worker.detector_model_name(llm) == "test-model-x"


def test_get_detector_llm_ignores_query_provider(
    fresh_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    llm = surprise_worker.get_detector_llm()
    assert type(llm).__name__ == "AnthropicLLMClient"
    assert llm._config.model == "claude-sonnet-5-5"  # type: ignore[union-attr]


def test_get_detector_llm_import_error(
    fresh_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "kb_core.llm.anthropic", None)
    assert surprise_worker.get_detector_llm() is None
    assert surprise_worker._DETECTOR_LLM_CACHE == {}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 0.7), ("0.5", 0.5), (" 0.9 ", 0.9), ("0", 0.0), ("1", 1.0)],
)
def test_detector_min_confidence_valid(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    raw: str | None,
    expected: float,
) -> None:
    if raw is not None:
        monkeypatch.setenv("KB_SURPRISE_MIN_CONFIDENCE", raw)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert surprise_worker.detector_min_confidence() == expected
    assert not caplog.records


@pytest.mark.parametrize("raw", ["bogus", "nan", "1.5", "-0.1"])
def test_detector_min_confidence_invalid(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
) -> None:
    monkeypatch.setenv("KB_SURPRISE_MIN_CONFIDENCE", raw)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert surprise_worker.detector_min_confidence() == 0.7
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "bad_min_confidence" in warnings[0].getMessage()


def test_detector_model_name() -> None:
    llm = SimpleNamespace(_config=SimpleNamespace(model="claude-sonnet-4-6"))
    assert surprise_worker.detector_model_name(llm) == "claude-sonnet-4-6"
    assert surprise_worker.detector_model_name(FakeLLM()) == "FakeLLM"


def _row(**kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event_id": "s:0",
        "session_id": "s",
        "project": None,
        "turn_index": 0,
        "user_prompt": "hi",
        "items": [],
        "final_message": None,
        "truncated": 0,
        "ts": "2026-10-09T00:00:00+00:00",
    }
    row.update(kw)
    return row


def test_digest_from_row(caplog: pytest.LogCaptureFixture) -> None:
    d = surprise_worker.digest_from_row(
        _row(items=json.dumps([{"kind": "assistant_text", "text": "x"}, 3]))
    )
    assert d.items == [{"kind": "assistant_text", "text": "x"}]
    assert d.project == ""
    assert d.truncated is False
    dt = datetime(2026, 10, 9, tzinfo=UTC)
    assert surprise_worker.digest_from_row(_row(ts=dt)).ts == dt.isoformat()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        bad = surprise_worker.digest_from_row(_row(items="not json"))
        assert surprise_worker.digest_from_row(_row(items='{"a": 1}')).items == []
    assert bad.items == []
    assert any("bad_items" in r.getMessage() for r in caplog.records)


def test_candidate_from_row() -> None:
    row = {
        "id": 3,
        "shape": 2,
        "session_id": "s",
        "project": "p",
        "turn_event_ids": '["s:0", "s:1"]',
        "detector_model": "m",
        "detector_output": '{"confidence": 1.0}',
        "status": "pending",
        "entry_id": None,
        "created_at": "t",
    }
    cand = surprise_worker.candidate_from_row(row)
    assert cand.turn_event_ids == ["s:0", "s:1"]
    assert cand.detector_output == {"confidence": 1.0}
    already = surprise_worker.candidate_from_row(
        {**row, "turn_event_ids": ["a"], "detector_output": {}}
    )
    assert already.turn_event_ids == ["a"]


async def test_detect_digest_reads_env_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I5's positional three-argument call reads KB_SURPRISE_MIN_CONFIDENCE."""
    monkeypatch.setenv("KB_SURPRISE_MIN_CONFIDENCE", "0.95")
    prev = surprise_worker.digest_from_row(
        _row(final_message="port 8080 is free", user_prompt="go")
    )
    cur = surprise_worker.digest_from_row(
        _row(
            event_id="s:1",
            turn_index=1,
            user_prompt="no, 8080 is taken by caddy, use 8081",
        )
    )
    llm = FakeLLM()
    llm.enqueue(_SHAPE2_TRUE)
    records = await surprise_worker.detect_digest(llm, cur, [prev, cur])
    shape2 = next(r for r in records if r.shape == 2)
    assert shape2.outcome == "low_confidence"
    assert shape2.details == {"min_confidence": 0.95}


# --- background worker (AC-18) -----------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "url", "expected"),
    [
        ("off", "postgresql://x", False),
        ("shadow", None, False),
        ("shadow", "", False),
        ("shadow", "postgresql://x", True),
        ("on", "postgresql://x", True),
    ],
)
def test_should_start_matrix(mode: Any, url: str | None, expected: bool) -> None:
    assert surprise_worker.should_start_surprise_worker(mode, url) is expected


class _DrainCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []
        self.raise_first = False


@pytest.fixture
def drain_counter(monkeypatch: pytest.MonkeyPatch) -> _DrainCounter:
    counter = _DrainCounter()
    sentinel = object()

    async def _get_db() -> Any:
        return sentinel

    async def _drain(pool: Any, kb: Any) -> None:
        assert pool is sentinel
        counter.calls.append((pool, kb))
        if len(counter.calls) == 1 and counter.raise_first:
            raise RuntimeError("first drain fails")

    monkeypatch.setattr(surprise_worker, "get_db", _get_db)
    monkeypatch.setattr(surprise_worker, "drain_once", _drain)
    return counter


async def _wait_for(pred: Any) -> None:
    for _ in range(500):
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never met")


async def test_worker_runs_and_stops(drain_counter: _DrainCounter) -> None:
    worker = surprise_worker.SurpriseCaptureWorker(
        "kb", asyncio.Lock(), poll_interval_seconds=0.01
    )
    await worker.start()
    task = worker._task
    await worker.start()
    assert worker._task is task
    assert worker.running
    await _wait_for(lambda: len(drain_counter.calls) >= 1)
    assert drain_counter.calls[0][1] == "kb"
    await worker.stop()
    assert not worker.running


async def test_worker_stop_never_started() -> None:
    worker = surprise_worker.SurpriseCaptureWorker("kb", asyncio.Lock())
    await worker.stop()
    assert not worker.running


async def test_worker_survives_failed_drain(drain_counter: _DrainCounter) -> None:
    drain_counter.raise_first = True
    worker = surprise_worker.SurpriseCaptureWorker(
        "kb", asyncio.Lock(), poll_interval_seconds=0.01
    )
    await worker.start()
    await _wait_for(lambda: len(drain_counter.calls) >= 2)
    assert worker.running
    await worker.stop()


# --- drain scenarios (AC-21) ---------------------------------------------------------

_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
)


class _Ctx:
    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.distill_calls: list[tuple[list[int], str]] = []
        self.distill_result = DistillResult()
        self.dry_run_calls: list[list[int]] = []


@contextmanager
def _local_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setattr(database, "_pool", None)
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch) -> _Ctx:
    c = _Ctx(FakeLLM())
    monkeypatch.setattr(surprise_worker, "get_detector_llm", lambda: c.llm)

    async def _distill(
        pool: Any, kb: Any, candidates: list[SurpriseCandidate], mode: str
    ) -> DistillResult:
        c.distill_calls.append(([cand.id for cand in candidates], mode))
        return c.distill_result

    monkeypatch.setattr(surprise_worker, "distill_candidates", _distill)

    async def _dry_run(pool: Any, kb: Any, candidates: list[SurpriseCandidate]) -> int:
        c.dry_run_calls.append([cand.id for cand in candidates])
        return 0

    monkeypatch.setattr(surprise_worker, "dry_run_candidates", _dry_run)
    return c


@pytest.fixture
def local_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ctx: _Ctx
) -> Iterator[TestClient]:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    with _local_client(tmp_path, monkeypatch) as client:
        yield client


def _query(sql: str, *args: Any) -> list[sqlite3.Row]:
    conn = sqlite3.connect(database.sqlite_service_db_path())
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def _exec(sql: str, *args: Any) -> None:
    conn = sqlite3.connect(database.sqlite_service_db_path())
    try:
        conn.execute(sql, args)
        conn.commit()
    finally:
        conn.close()


def _seed(
    client: TestClient,
    session: str,
    turn: int,
    *,
    user_prompt: str | None = None,
    final_message: str | None = None,
    items: list[dict[str, Any]] | None = None,
    harness: str = "claude-code",
    mode: str = "interactive",
) -> None:
    body = {
        "event_id": f"{session}:{turn}",
        "session_id": session,
        "harness": harness,
        "mode": mode,
        "engine": None,
        "host": "h",
        "hook_version": "1.3.0",
        "project": "p",
        "turn_index": turn,
        "ts": "2026-10-09T00:00:00+00:00",
        "user_prompt": user_prompt,
        "items": items or [],
        "final_message": final_message,
        "truncated": False,
    }
    resp = client.post("/api/kb/turn", json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["reason"] == "recorded"


def _seed_f(client: TestClient) -> None:
    _seed(
        client,
        "s1",
        0,
        user_prompt="use port 8080",
        final_message="Push failed; port 8080 is free.",
        items=[
            {"kind": "assistant_text", "text": "port 8080 is free"},
            {
                "kind": "tool_call",
                "tool_use_id": "t1",
                "tool": "Bash",
                "target": "git push origin main",
                "target_class": "git push",
            },
            {
                "kind": "tool_result",
                "tool_use_id": "t1",
                "is_error": True,
                "excerpt": "rejected",
            },
        ],
    )
    _seed(
        client,
        "s1",
        1,
        user_prompt="no, 8080 is taken by caddy, use 8081",
        final_message=None,
        items=[
            {
                "kind": "tool_call",
                "tool_use_id": "t2",
                "tool": "Bash",
                "target": "git push origin HEAD:main",
                "target_class": "git push",
            },
            {
                "kind": "tool_result",
                "tool_use_id": "t2",
                "is_error": False,
                "excerpt": "ok",
            },
        ],
    )


def _drain(client: TestClient) -> dict[str, Any]:
    resp = client.post("/api/kb/surprise/drain")
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


def _enqueue(llm: FakeLLM, script: list[str]) -> None:
    for item in script:
        llm.enqueue(item)


def _detections(event_id: str | None = None) -> list[sqlite3.Row]:
    if event_id is None:
        return _query("SELECT * FROM surprise_detections ORDER BY id")
    return _query(
        "SELECT * FROM surprise_detections WHERE event_id = ? ORDER BY id", event_id
    )


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER and r.levelno == level
    ]


def _info_line(caplog: pytest.LogCaptureFixture) -> str:
    lines = [m for m in _messages(caplog, logging.INFO) if "surprise_drain mode=" in m]
    assert len(lines) == 1, lines
    return lines[0]


def test_drain_mode_off(
    local_client: TestClient,
    ctx: _Ctx,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _seed_f(local_client)
    assert len(_query("SELECT 1 FROM turn_events WHERE processed_at IS NULL")) == 2
    prune_calls: list[int] = []
    real_prune = surprise_worker.prune_turn_events

    async def _spy(pool: Any) -> int:
        prune_calls.append(1)
        return await real_prune(pool)

    monkeypatch.setattr(surprise_worker, "prune_turn_events", _spy)
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "off")
    caplog.set_level(logging.INFO, logger=LOGGER)
    assert _drain(local_client) == _EMPTY
    assert prune_calls == []
    assert ctx.llm.generate_calls == []
    assert len(_query("SELECT 1 FROM turn_events WHERE processed_at IS NULL")) == 2
    assert _detections() == []
    assert [r for r in caplog.records if r.name == LOGGER] == []


def _assert_shadow_pass(body: dict[str, Any], ctx: _Ctx) -> None:
    assert body["digests_processed"] == 2
    calls = ctx.llm.generate_calls
    assert len(calls) == 2
    assert "port 8080 is free" in calls[0][0]
    assert "no, 8080 is taken by caddy" in calls[1][0]
    assert all(system == SURPRISE_DETECTOR_SYSTEM for _, system in calls)
    cands = body["candidates"]
    assert [c["shape"] for c in cands] == [1, 2]
    assert [c["id"] for c in cands] == sorted(c["id"] for c in cands)
    assert all(c["status"] == "shadow" for c in cands)
    assert cands[0]["detector_model"] == "rule:shape1"
    assert cands[0]["turn_event_ids"] == ["s1:0", "s1:1"]
    assert cands[0]["detector_output"] == {
        "wrong_belief": "git push origin main",
        "corrected_fact": "git push origin HEAD:main",
        "evidence_excerpt": "rejected",
        "confidence": 1.0,
    }
    assert cands[1]["detector_model"] == "FakeLLM"
    assert cands[1]["turn_event_ids"] == ["s1:0", "s1:1"]
    assert cands[1]["detector_output"] == {
        "wrong_belief": "port 8080 is free",
        "corrected_fact": "port 8080 is taken by caddy",
        "evidence_excerpt": "8080 is taken by caddy",
        "confidence": 0.9,
    }
    assert body["entries_written"] == body["entries_merged"] == []


def test_drain_shadow_then_idempotent(
    local_client: TestClient, ctx: _Ctx, caplog: pytest.LogCaptureFixture
) -> None:
    _seed_f(local_client)
    _enqueue(ctx.llm, SCRIPT_B)
    caplog.set_level(logging.INFO, logger=LOGGER)
    body = _drain(local_client)
    _assert_shadow_pass(body, ctx)
    assert ctx.distill_calls == []
    assert _query("SELECT 1 FROM turn_events WHERE processed_at IS NULL") == []
    rows = _detections()
    assert [(r["event_id"], r["shape"], r["outcome"], r["reason"]) for r in rows] == [
        ("s1:0", 1, "no_surprise", ""),
        ("s1:0", 2, "not_applicable", "no_prev"),
        ("s1:0", 3, "no_surprise", ""),
        ("s1:1", 1, "candidate", ""),
        ("s1:1", 2, "candidate", ""),
        ("s1:1", 3, "not_applicable", "no_text_after_result"),
    ]
    assert all(r["mode"] == "shadow" for r in rows)
    assert all(r["detector_version"] == SURPRISE_DETECTOR_VERSION for r in rows)
    model_rows = [rows[2], rows[4]]
    for r in model_rows:
        assert r["prompt_chars"] > 0
        assert json.loads(r["details"])["min_confidence"] in (0.5, 0.7)
    assert (rows[3]["candidate_id"] is not None) and (rows[4]["candidate_id"])
    assert rows[0]["candidate_id"] is None
    line = _info_line(caplog)
    assert "llm_calls=2" in line
    assert "min_confidence_s2=0.50 min_confidence_s3=0.70" in line
    assert "distill_input=0" in line
    assert "new_candidates=2" in line
    assert "dry_run_input=2" in line
    assert ctx.dry_run_calls == [sorted(r["candidate_id"] for r in rows[3:5])]

    # (c) a second shadow drain is a no-op.
    body2 = _drain(local_client)
    assert body2["digests_processed"] == 0
    assert body2["candidates"] == []
    assert len(ctx.llm.generate_calls) == 2
    assert len(_query("SELECT 1 FROM surprise_candidates")) == 2
    assert len(_detections()) == 6
    assert ctx.distill_calls == []


def test_drain_on_distills(
    local_client: TestClient,
    ctx: _Ctx,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _seed_f(local_client)
    _enqueue(ctx.llm, SCRIPT_B)
    ctx.distill_result = DistillResult(["kb-00001"], ["kb-00002"])
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "on")
    caplog.set_level(logging.INFO, logger=LOGGER)
    body = _drain(local_client)
    ids = [r["id"] for r in _query("SELECT id FROM surprise_candidates ORDER BY id")]
    assert len(ids) == 2
    assert ctx.distill_calls == [(ids, "on")]
    statuses = _query("SELECT status FROM surprise_candidates")
    assert [r["status"] for r in statuses] == ["pending", "pending"]
    assert [c["id"] for c in body["candidates"]] == ids
    assert body["entries_written"] == ["kb-00001"]
    assert body["entries_merged"] == ["kb-00002"]
    assert "distill_input=2" in _info_line(caplog)
    assert ctx.dry_run_calls == []
    assert all(r["mode"] == "on" for r in _detections())


def test_drain_shadow_candidates_never_replayed(
    local_client: TestClient, ctx: _Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_f(local_client)
    _enqueue(ctx.llm, SCRIPT_B)
    _drain(local_client)
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "on")
    body = _drain(local_client)
    assert body["digests_processed"] == 0
    assert body["candidates"] == []
    assert ctx.distill_calls == []
    statuses = _query("SELECT status FROM surprise_candidates")
    assert [r["status"] for r in statuses] == ["shadow", "shadow"]


def test_drain_mode_is_read_at_processing_time(
    local_client: TestClient, ctx: _Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "on")
    _seed_f(local_client)
    assert {r["capture_mode"] for r in _query("SELECT * FROM turn_events")} == {"on"}
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    _enqueue(ctx.llm, SCRIPT_B)
    body = _drain(local_client)
    assert [c["status"] for c in body["candidates"]] == ["shadow", "shadow"]
    assert ctx.distill_calls == []


def test_drain_llm_returns_none(
    local_client: TestClient, ctx: _Ctx, caplog: pytest.LogCaptureFixture
) -> None:
    _seed_f(local_client)
    caplog.set_level(logging.INFO, logger=LOGGER)
    body = _drain(local_client)
    assert [c["shape"] for c in body["candidates"]] == [1]
    assert _query("SELECT 1 FROM turn_events WHERE processed_at IS NULL") == []
    model_rows = [
        r
        for r in _detections()
        if r["outcome"] not in ("not_applicable",) and r["shape"] in (2, 3)
    ]
    assert [(r["outcome"], r["reason"]) for r in model_rows] == [
        ("llm_error", "none"),
        ("llm_error", "none"),
    ]
    warnings = _messages(caplog, logging.WARNING)
    assert sum("surprise_drain detector unavailable" in m for m in warnings) == 1
    assert sum("detector_failed" in m for m in warnings) == 2


class _RaisingLLM(FakeLLM):
    async def generate(self, prompt: Any, *, system: Any = None) -> str | None:
        raise RuntimeError("down")


def test_drain_llm_raises(local_client: TestClient, ctx: _Ctx) -> None:
    ctx.llm = _RaisingLLM()
    _seed_f(local_client)
    _drain(local_client)
    model_rows = [
        r
        for r in _detections()
        if r["outcome"] != "not_applicable" and r["shape"] in (2, 3)
    ]
    assert [(r["outcome"], r["reason"]) for r in model_rows] == [
        ("llm_error", "exception"),
        ("llm_error", "exception"),
    ]
    assert all(r["detector_model"] == "_RaisingLLM" for r in model_rows)


def test_drain_without_llm(
    local_client: TestClient, ctx: _Ctx, caplog: pytest.LogCaptureFixture
) -> None:
    ctx.llm = None
    _seed_f(local_client)
    caplog.set_level(logging.INFO, logger=LOGGER)
    body = _drain(local_client)
    assert [c["shape"] for c in body["candidates"]] == [1]
    no_llm = [
        (r["event_id"], r["shape"], r["detector_model"])
        for r in _detections()
        if r["outcome"] == "no_llm"
    ]
    assert no_llm == [("s1:0", 3, ""), ("s1:1", 2, "")]
    warnings = _messages(caplog, logging.WARNING)
    assert sum("surprise_drain no detector LLM" in m for m in warnings) == 1
    assert "no_llm=2" in _info_line(caplog)

    caplog.clear()
    _drain(local_client)
    assert _messages(caplog, logging.WARNING) == []


_S3_ITEMS = [
    {"kind": "assistant_text", "text": "the config lives in /etc/foo.conf"},
    {
        "kind": "tool_call",
        "tool_use_id": "c1",
        "tool": "Bash",
        "target": "cat /etc/foo.conf",
    },
    {
        "kind": "tool_result",
        "tool_use_id": "c1",
        "is_error": True,
        "excerpt": "No such file: /etc/foo.conf; config is at /etc/foo/main.conf",
    },
    {
        "kind": "assistant_text",
        "text": "/etc/foo.conf does not exist; the config is at /etc/foo/main.conf.",
    },
]


def _s3_verdict(evidence: str, confidence: float) -> str:
    return json.dumps(
        {
            "surprise": True,
            "wrong_belief": "config is in /etc/foo.conf",
            "corrected_fact": "config is at /etc/foo/main.conf",
            "evidence_excerpt": evidence,
            "confidence": confidence,
        }
    )


@pytest.mark.parametrize(
    ("response", "outcome", "warnings"),
    [
        (_s3_verdict("config is at /etc/foo/main.conf", 0.69), "low_confidence", 0),
        (_s3_verdict("config moved somewhere else", 0.9), "ungrounded", 0),
        ('{"surprise": false}', "no_surprise", 0),
        ("not json", "unparseable", 1),
        (_s3_verdict("No such FILE:  /etc/foo.conf", 0.9), "candidate", 0),
    ],
)
def test_drain_shape3_outcomes(
    local_client: TestClient,
    ctx: _Ctx,
    caplog: pytest.LogCaptureFixture,
    response: str,
    outcome: str,
    warnings: int,
) -> None:
    _seed(local_client, "s3", 0, user_prompt="where is it", items=_S3_ITEMS)
    ctx.llm.enqueue(response)
    caplog.set_level(logging.INFO, logger=LOGGER)
    body = _drain(local_client)
    rows = [r for r in _detections("s3:0") if r["shape"] == 3]
    assert len(rows) == 1
    row = rows[0]
    assert row["outcome"] == outcome
    assert row["raw_response_excerpt"] == response
    if outcome == "low_confidence":
        assert row["confidence"] == 0.69
        assert row["candidate_id"] is None
    if outcome == "ungrounded":
        assert "config moved somewhere else" in row["raw_response_excerpt"]
    if outcome == "candidate":
        assert row["candidate_id"] == body["candidates"][0]["id"]
        assert body["candidates"][0]["turn_event_ids"] == ["s3:0"]
        assert "shape3=1" in _info_line(caplog)
    warns = _messages(caplog, logging.WARNING)
    assert len(warns) == warnings
    if warnings:
        assert "detector_failed" in warns[0]


def test_drain_shape3_assistant_text_evidence_ungrounded(
    local_client: TestClient, ctx: _Ctx
) -> None:
    _seed(local_client, "s3", 0, user_prompt="where is it", items=_S3_ITEMS)
    ctx.llm.enqueue(_s3_verdict("/etc/foo.conf does not exist", 0.9))
    _drain(local_client)
    rows = [r for r in _detections("s3:0") if r["shape"] == 3]
    assert len(rows) == 1
    assert rows[0]["outcome"] == "ungrounded"
    assert rows[0]["candidate_id"] is None
    assert json.loads(rows[0]["details"])["evidence_in_reasoning"] is False


_BELIEF = "I believed the config lives in /etc/foo.conf all along"


def test_drain_shape3_reasoning_evidence_ungrounded(
    local_client: TestClient, ctx: _Ctx
) -> None:
    _seed(
        local_client,
        "s3",
        0,
        user_prompt="where is it",
        harness="talos",
        mode="headless",
        items=[
            _S3_ITEMS[0],
            {"kind": "reasoning", "text": _BELIEF, "truncated": False},
            *_S3_ITEMS[1:],
        ],
    )
    ctx.llm.enqueue(_s3_verdict(_BELIEF, 0.9))
    _drain(local_client)
    rows = [r for r in _detections("s3:0") if r["shape"] == 3]
    assert len(rows) == 1
    assert rows[0]["outcome"] == "ungrounded"
    assert rows[0]["candidate_id"] is None
    assert rows[0]["detector_version"] == 3
    details = json.loads(rows[0]["details"])
    assert details["evidence_in_reasoning"] is True
    assert details["reasoning_items"] == 1
    assert details["reasoning_chars"] == len(_BELIEF) == 54
    assert details["reasoning_after_result"] is False
    assert f"[reasoning] {_BELIEF}" in ctx.llm.generate_calls[0][0]


_CALL_RESULT = [_S3_ITEMS[1], _S3_ITEMS[2]]


async def test_detect_digest_shape3_not_applicable_reasoning_details() -> None:
    cur = surprise_worker.digest_from_row(
        _row(
            items=[
                *_CALL_RESULT,
                {"kind": "reasoning", "text": "Root cause: x", "truncated": False},
            ]
        )
    )
    llm = FakeLLM()
    records = await surprise_worker.detect_digest(llm, cur, [cur])
    r3 = next(r for r in records if r.shape == 3)
    assert r3.outcome == "not_applicable"
    assert r3.reason == "no_text_after_result"
    assert r3.details == {
        "reasoning_items": 1,
        "reasoning_chars": 13,
        "reasoning_after_result": True,
    }
    assert llm.generate_calls == []


async def test_detect_digest_shape3_no_llm_reasoning_details() -> None:
    cur = surprise_worker.digest_from_row(
        _row(
            items=[
                *_CALL_RESULT,
                {"kind": "reasoning", "text": "x", "truncated": False},
                {"kind": "assistant_text", "text": "Root cause: y"},
            ]
        )
    )
    records = await surprise_worker.detect_digest(None, cur, [cur])
    r3 = next(r for r in records if r.shape == 3)
    assert r3.outcome == "no_llm"
    assert r3.details == {
        "reasoning_items": 1,
        "reasoning_chars": 1,
        "reasoning_after_result": True,
    }


def test_drain_shape3_details_scope(local_client: TestClient, ctx: _Ctx) -> None:
    _seed(local_client, "s3", 0, user_prompt="where is it", items=_S3_ITEMS)
    _seed(
        local_client,
        "s5",
        0,
        user_prompt="where is it",
        items=_S3_ITEMS[:3],
        final_message="The config is at /etc/foo/main.conf.",
    )
    ctx.llm.enqueue('{"surprise": false}')
    ctx.llm.enqueue('{"surprise": false}')
    _drain(local_client)
    for ev, scope, rendered in (
        ("s3:0", "text_after_result", False),
        ("s5:0", "final_message_only", True),
    ):
        rows = [r for r in _detections(ev) if r["shape"] == 3]
        assert len(rows) == 1
        details = json.loads(rows[0]["details"])
        assert details["shape3_scope"] == scope
        assert details["final_rendered"] is rendered
        assert details["min_confidence"] == 0.7
        assert details["reasoning_items"] == 0
        assert details["reasoning_chars"] == 0
        assert details["reasoning_after_result"] is False
        assert "evidence_in_reasoning" not in details


def test_drain_claim_race(
    local_client: TestClient,
    ctx: _Ctx,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _seed_f(local_client)
    _enqueue(ctx.llm, SCRIPT_B)
    real = surprise_worker.detect_digest

    async def _racing(llm: Any, cur: Any, session: Any, **kw: Any) -> Any:
        if cur.event_id == "s1:1":
            await surprise_worker.mark_turn_digests_processed(
                await database.get_db(), ["s1:1"], "x"
            )
        return await real(llm, cur, session, **kw)

    monkeypatch.setattr(surprise_worker, "detect_digest", _racing)
    caplog.set_level(logging.INFO, logger=LOGGER)
    body = _drain(local_client)
    assert body["digests_processed"] == 1
    assert _detections("s1:1") == []
    assert (
        _query("SELECT 1 FROM surprise_candidates WHERE turn_event_ids LIKE '%s1:1%'")
        == []
    )
    warnings = _messages(caplog, logging.WARNING)
    assert sum("tripwire=double_detect" in m for m in warnings) == 1
    assert len(_detections("s1:0")) == 3
    assert "double_detect=1" in _info_line(caplog)


def test_drain_turn_gap_and_out_of_order(local_client: TestClient) -> None:
    _seed(local_client, "s4", 0, user_prompt="a", final_message="b")
    _seed(local_client, "s4", 2, user_prompt="c")
    _drain(local_client)
    rows = _detections("s4:2")
    assert [r["reason"] for r in rows if r["shape"] == 2] == ["turn_gap"]
    assert all(json.loads(r["details"])["turn_gap"] is True for r in rows)
    assert all(
        json.loads(r["details"])["turn_gap"] is False for r in _detections("s4:0")
    )

    _seed(local_client, "s5", 1, user_prompt="a")
    _drain(local_client)
    _seed(local_client, "s5", 0, user_prompt="b")
    _drain(local_client)
    rows = _detections("s5:0")
    assert rows
    assert all(json.loads(r["details"])["out_of_order"] is True for r in rows)


def test_worker_not_started_on_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ctx: _Ctx
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    with _local_client(tmp_path, monkeypatch) as client:
        assert app.state.surprise_worker is None
        body = _drain(client)
        assert set(body) == set(_EMPTY)


def test_drain_prunes_first(
    local_client: TestClient,
    ctx: _Ctx,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _seed_f(local_client)
    old = (datetime.now(UTC) - timedelta(days=31)).isoformat(timespec="seconds")
    _exec("UPDATE turn_events SET received_ts = ? WHERE event_id = 's1:0'", old)
    order: list[str] = []
    real_prune = surprise_worker.prune_turn_events
    real_list = surprise_worker.list_pending_turn_digests

    async def _prune(pool: Any) -> int:
        order.append("prune")
        return await real_prune(pool)

    async def _list(pool: Any) -> Any:
        order.append("list_pending")
        return await real_list(pool)

    monkeypatch.setattr(surprise_worker, "prune_turn_events", _prune)
    monkeypatch.setattr(surprise_worker, "list_pending_turn_digests", _list)
    _enqueue(ctx.llm, SCRIPT_B)
    caplog.set_level(logging.INFO, logger=LOGGER)
    body = _drain(local_client)
    assert order[:2] == ["prune", "list_pending"]
    assert _query("SELECT 1 FROM turn_events WHERE event_id = 's1:0'") == []
    assert body["digests_processed"] == 1
    assert _detections("s1:0") == []
    assert "pruned=1" in _info_line(caplog)


def test_drain_ignores_listener_switch(
    local_client: TestClient, ctx: _Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_LISTENER_ENABLED", "FALSE")
    monkeypatch.setenv("PERSONAL_KB_LISTENER", "0")
    _seed_f(local_client)
    _enqueue(ctx.llm, SCRIPT_B)
    body = _drain(local_client)
    assert body["digests_processed"] == 2
    assert len(_query("SELECT 1 FROM surprise_candidates")) == 2


def test_drain_min_confidence_override(
    local_client: TestClient,
    ctx: _Ctx,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_MIN_CONFIDENCE", "0.95")
    _seed_f(local_client)
    _enqueue(ctx.llm, SCRIPT_B)
    caplog.set_level(logging.INFO, logger=LOGGER)
    body = _drain(local_client)
    row = next(r for r in _detections("s1:1") if r["shape"] == 2)
    assert row["outcome"] == "low_confidence"
    assert row["confidence"] == 0.9
    assert row["candidate_id"] is None
    assert json.loads(row["details"])["min_confidence"] == 0.95
    assert [c["shape"] for c in body["candidates"]] == [1]
    assert "min_confidence_s2=0.95 min_confidence_s3=0.95" in _info_line(caplog)


def test_drain_bad_min_confidence_falls_back(
    local_client: TestClient,
    ctx: _Ctx,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_MIN_CONFIDENCE", "bogus")
    _seed_f(local_client)
    _enqueue(ctx.llm, SCRIPT_B)
    caplog.set_level(logging.INFO, logger=LOGGER)
    body = _drain(local_client)
    assert [c["shape"] for c in body["candidates"]] == [1, 2]
    warnings = _messages(caplog, logging.WARNING)
    assert sum("bad_min_confidence" in m for m in warnings) == 1


def test_drain_lost_after_claim(
    local_client: TestClient, ctx: _Ctx, caplog: pytest.LogCaptureFixture
) -> None:
    _seed_f(local_client)
    _exec("DROP TABLE surprise_detections")
    _enqueue(ctx.llm, SCRIPT_B)
    caplog.set_level(logging.INFO, logger=LOGGER)
    with pytest.raises(sqlite3.OperationalError):
        local_client.post("/api/kb/surprise/drain")
    errors = _messages(caplog, logging.ERROR)
    assert any("surprise_drain lost_after_claim event_id=s1:0" in m for m in errors)
    row = _query("SELECT processed_at FROM turn_events WHERE event_id = 's1:0'")
    assert row[0]["processed_at"] is not None
    assert _query("SELECT 1 FROM surprise_candidates") == []


async def test_distill_candidates_hook_is_noop() -> None:
    result = await surprise_worker.distill_candidates(object(), object(), [], "on")
    assert result == DistillResult()


# --- per-shape confidence floors -----------------------------------------------


def test_floor_defaults_per_shape() -> None:
    assert surprise_worker.detector_min_confidence_for(2) == 0.5
    assert surprise_worker.detector_min_confidence_for(3) == 0.7
    assert surprise_worker.detector_min_confidence() == 0.7


def test_floor_resolution_order(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("KB_SURPRISE_MIN_CONFIDENCE", "0.8")
    assert surprise_worker.detector_min_confidence_for(2) == 0.8
    assert surprise_worker.detector_min_confidence_for(3) == 0.8
    monkeypatch.setenv("KB_SURPRISE_MIN_CONFIDENCE_SHAPE2", "0.3")
    assert surprise_worker.detector_min_confidence_for(2) == 0.3
    assert surprise_worker.detector_min_confidence_for(3) == 0.8
    monkeypatch.setenv("KB_SURPRISE_MIN_CONFIDENCE_SHAPE3", "0.6")
    assert surprise_worker.detector_min_confidence_for(3) == 0.6
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        monkeypatch.setenv("KB_SURPRISE_MIN_CONFIDENCE_SHAPE2", "bogus")
        assert surprise_worker.detector_min_confidence_for(2) == 0.8
        monkeypatch.delenv("KB_SURPRISE_MIN_CONFIDENCE")
        assert surprise_worker.detector_min_confidence_for(2) == 0.5
    assert any("bad_min_confidence" in r.getMessage() for r in caplog.records)


async def test_detect_digest_shape_floors_under_defaults() -> None:
    """Confidence 0.6 is a candidate for shape 2 but low_confidence for shape 3."""
    s2 = json.dumps(
        {
            "surprise": True,
            "wrong_belief": "port 8080 is free",
            "corrected_fact": "port 8080 is taken by caddy",
            "evidence_excerpt": "8080 is taken by caddy",
            "confidence": 0.6,
        }
    )
    prev = surprise_worker.digest_from_row(
        _row(final_message="port 8080 is free", user_prompt="go")
    )
    cur = surprise_worker.digest_from_row(
        _row(
            event_id="s:1",
            turn_index=1,
            user_prompt="no, 8080 is taken by caddy, use 8081",
            items=_S3_ITEMS,
        )
    )
    llm = FakeLLM()
    llm.enqueue(s2)
    llm.enqueue(_s3_verdict("config is at /etc/foo/main.conf", 0.6))
    records = await surprise_worker.detect_digest(llm, cur, [prev, cur])
    by_shape = {r.shape: r for r in records}
    assert by_shape[2].outcome == "candidate"
    assert by_shape[2].details["min_confidence"] == 0.5
    assert by_shape[3].outcome == "low_confidence"
    assert by_shape[3].details["min_confidence"] == 0.7
