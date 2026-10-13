"""Tests for kb_service.surprise_worker: helpers, worker loop, drain scenarios.

The drain scenarios run the real app in local no-auth mode (SQLite service
DB in tmp_path), seed digests through POST /api/kb/turn and drive
POST /api/kb/surprise/drain with a scripted FakeLLM. No network.
"""

import asyncio
import contextlib
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
from tests.conftest import S4_OUT, FakeLLM

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


def test_capture_clients_ignore_thinking_env(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """KB_ANTHROPIC_THINKING never reaches the detector/distiller/critic configs.

    Unique model env values force real builds (the caches return the cached
    client before any config build otherwise); all three caches are cleared
    because the fresh_cache fixture only clears the detector's.
    """
    from kb_core.config import AnthropicProviderConfig

    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "adaptive")
    monkeypatch.setenv("KB_SURPRISE_DETECTOR_MODEL", "claude-sonnet-5-5-det-probe")
    monkeypatch.setenv("KB_SURPRISE_DISTILL_MODEL", "claude-sonnet-5-5-dis-probe")
    monkeypatch.setenv("KB_SURPRISE_CRITIC_MODEL", "claude-sonnet-5-5-cri-probe")
    monkeypatch.setattr(surprise_worker, "_DETECTOR_LLM_CACHE", {})
    monkeypatch.setattr(surprise_worker, "_DISTILLER_LLM_CACHE", {})
    monkeypatch.setattr(surprise_worker, "_CRITIC_LLM_CACHE", {})
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        detector = surprise_worker.get_detector_llm()
        distiller = surprise_worker.get_distiller_llm()
        critic = surprise_worker.get_critic_llm()
    for client, model in (
        (detector, "claude-sonnet-5-5-det-probe"),
        (distiller, "claude-sonnet-5-5-dis-probe"),
        (critic, "claude-sonnet-5-5-cri-probe"),
    ):
        assert isinstance(client._config, AnthropicProviderConfig)
        assert client._config.model == model
        assert client._config.thinking is None
    assert not any(
        "surprise capture client built with thinking" in r.getMessage()
        for r in caplog.records
    )


def test_capture_client_tripwire_warns_on_injected_thinking(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A capture client that somehow carries thinking trips the warning."""
    from kb_core.config import AnthropicProviderConfig

    def _injected(*, model: str | None = None, **_: object) -> AnthropicProviderConfig:
        return AnthropicProviderConfig(model=model or "probe", thinking="adaptive")

    monkeypatch.setattr(surprise_worker, "_DETECTOR_LLM_CACHE", {})
    monkeypatch.setattr(surprise_worker, "build_anthropic_config", _injected)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        llm = surprise_worker.get_detector_llm()
    assert llm is not None
    assert llm._config.thinking == "adaptive"
    trips = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "thinking=adaptive" in r.getMessage()
    ]
    assert len(trips) == 1
    assert "expected None" in trips[0].getMessage()


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


def _listener_row(session_id: str) -> sqlite3.Row:
    rows = _query("SELECT * FROM listener_decisions WHERE session_id = ?", session_id)
    assert len(rows) == 1, rows
    return rows[0]


def _insert_listener_vote_log(
    session_id: str, decided_ts: str, *, vote_prompt: str = "gate prompt"
) -> None:
    """One voted listener_decisions row carrying vote text."""
    _exec(
        "INSERT INTO listener_decisions (session_id, source_kb, decided_ts,"
        " candidates_considered, decision, reason, vote_shape, candidate_signal,"
        " candidate_ids, whispered_ids, n_retrieved, n_after_a, n_after_b,"
        " retrieval_path, vote_model, vote_prompt, vote_raw, vote_truncated)"
        " VALUES (?, 'kb', ?, 1, 'whisper', 'whispered',"
        ' \'[["kb-00001"], ["kb-00001"], ["kb-00001"]]\', \'fallback\','
        " '[\"kb-00001\"]', '[\"kb-00001\"]', 1, 1, 1, 'fallback-direct',"
        ' \'claude-sonnet-4-6\', ?, \'["kb-00001", "kb-00001", "kb-00001"]\', 1)',
        session_id,
        decided_ts,
        vote_prompt,
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
    old = (datetime.now(UTC) - timedelta(days=91)).isoformat(timespec="seconds")
    recent = (datetime.now(UTC) - timedelta(days=1)).isoformat(timespec="seconds")
    _insert_listener_vote_log("s-old", old, vote_prompt="old prompt")
    _insert_listener_vote_log("s-new", recent, vote_prompt="new prompt")
    prune_calls: list[int] = []
    real_prune = surprise_worker.prune_turn_events

    async def _spy(pool: Any) -> int:
        prune_calls.append(1)
        return await real_prune(pool)

    monkeypatch.setattr(surprise_worker, "prune_turn_events", _spy)
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "off")
    caplog.set_level(logging.INFO, logger=LOGGER)
    assert _drain(local_client) == _EMPTY
    # Capture-off drains still touch nothing capture-shaped ...
    assert prune_calls == []
    assert ctx.llm.generate_calls == []
    assert len(_query("SELECT 1 FROM turn_events WHERE processed_at IS NULL")) == 2
    assert _detections() == []
    # ... but the listener vote-text retention NULLing runs anyway (the
    # listener writes vote text whatever the capture mode), so an old row's
    # text goes and a recent row's stays.
    assert "listener vote text pruned=1" in _messages(caplog, logging.INFO)
    aged, fresh = _listener_row("s-old"), _listener_row("s-new")
    assert aged["vote_prompt"] is None
    assert aged["vote_raw"] is None
    assert aged["vote_truncated"] == 0
    assert aged["vote_model"] == "claude-sonnet-4-6"
    assert fresh["vote_prompt"] == "new prompt"
    assert fresh["vote_raw"] == '["kb-00001", "kb-00001", "kb-00001"]'
    assert fresh["vote_truncated"] == 1
    # No OTHER worker log line fired: the drain really was a no-op apart from
    # the vote-text prune.
    assert [
        m
        for m in _messages(caplog, logging.INFO)
        if not m.startswith("listener vote text pruned=")
    ] == []


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


def test_detector_calls_carry_shape_purposes(
    local_client: TestClient, ctx: _Ctx
) -> None:
    """A digest where both model shapes run tags each call with its shape purpose."""
    _seed(local_client, "s1", 0, user_prompt="hi", final_message="port 8080 is free")
    _seed(
        local_client,
        "s1",
        1,
        user_prompt="no, 8080 is taken, use 8081",
        final_message="Pushed on 8081.",
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
    _drain(local_client)
    assert ctx.llm.purposes == ["surprise_detector_shape2", "surprise_detector_shape3"]


class _RaisingLLM(FakeLLM):
    async def generate(
        self, prompt: Any, *, purpose: str, system: Any = None
    ) -> str | None:
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
    assert rows[0]["detector_version"] == SURPRISE_DETECTOR_VERSION
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
        # No drain loop is scheduled on SQLite, but the listener vote-text
        # retention still runs: the lifespan starts its own prune loop.
        assert app.state.listener_vote_prune_task is not None
        assert not app.state.listener_vote_prune_task.done()
        body = _drain(client)
        assert set(body) == set(_EMPTY)
    # Shutdown cancelled the loop task.
    assert app.state.listener_vote_prune_task is None


async def test_prune_listener_vote_text_loop_prunes_each_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The startup-scheduled loop prunes once per interval, best-effort."""
    sentinel = object()
    calls: list[object] = []

    async def _spy(pool: Any) -> int:
        calls.append(pool)
        return 1

    async def _get_db() -> Any:
        return sentinel

    monkeypatch.setattr(surprise_worker, "prune_listener_vote_text", _spy)
    monkeypatch.setattr(surprise_worker, "get_db", _get_db)
    task = asyncio.create_task(
        surprise_worker.prune_listener_vote_text_loop(poll_interval_seconds=0.001)
    )
    try:
        await _wait_for(lambda: len(calls) >= 2)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert calls
    assert all(pool is sentinel for pool in calls)
    assert len(calls) >= 2


def test_drain_prunes_first(
    local_client: TestClient,
    ctx: _Ctx,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _seed_f(local_client)
    old = (datetime.now(UTC) - timedelta(days=91)).isoformat(timespec="seconds")
    recent = (datetime.now(UTC) - timedelta(days=1)).isoformat(timespec="seconds")
    _exec("UPDATE turn_events SET received_ts = ? WHERE event_id = 's1:0'", old)
    _insert_listener_vote_log("s-old", old, vote_prompt="old prompt")
    _insert_listener_vote_log("s-new", recent, vote_prompt="new prompt")
    order: list[str] = []
    real_prune = surprise_worker.prune_turn_events
    real_list = surprise_worker.list_pending_turn_digests
    real_vote_prune = surprise_worker.prune_listener_vote_text

    async def _prune(pool: Any) -> int:
        order.append("prune")
        return await real_prune(pool)

    async def _list(pool: Any) -> Any:
        order.append("list_pending")
        return await real_list(pool)

    async def _vote_prune(pool: Any) -> int:
        order.append("vote_text_prune")
        return await real_vote_prune(pool)

    monkeypatch.setattr(surprise_worker, "prune_turn_events", _prune)
    monkeypatch.setattr(surprise_worker, "list_pending_turn_digests", _list)
    monkeypatch.setattr(surprise_worker, "prune_listener_vote_text", _vote_prune)
    _enqueue(ctx.llm, SCRIPT_B)
    caplog.set_level(logging.INFO, logger=LOGGER)
    body = _drain(local_client)
    # The listener vote-text prune runs first — before the mode check and
    # before the turn_events prune, so it never depends on capture mode.
    assert order[:3] == ["vote_text_prune", "prune", "list_pending"]
    assert _query("SELECT 1 FROM turn_events WHERE event_id = 's1:0'") == []
    assert body["digests_processed"] == 1
    assert _detections("s1:0") == []
    assert "pruned=1" in _info_line(caplog)
    assert "listener vote text pruned=1" in _messages(caplog, logging.INFO)
    # Only the OLD row's vote text is NULLed; vote_model stays on both.
    aged, fresh = _listener_row("s-old"), _listener_row("s-new")
    assert aged["vote_prompt"] is None
    assert aged["vote_raw"] is None
    assert aged["vote_truncated"] == 0
    assert aged["vote_model"] == "claude-sonnet-4-6"
    assert fresh["vote_prompt"] == "new prompt"
    assert fresh["vote_raw"] == '["kb-00001", "kb-00001", "kb-00001"]'
    assert fresh["vote_truncated"] == 1


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


# --- shape 4 -----------------------------------------------------------------

_S4_ITEMS: list[dict[str, Any]] = [
    {
        "kind": "tool_call",
        "tool_use_id": "c1",
        "tool": "run_checks",
        "target": "",
        "target_class": "",
    },
    {
        "kind": "tool_result",
        "tool_use_id": "c1",
        "is_error": True,
        "excerpt": "ruff format --check: would reformat src/a.py",
    },
    {
        "kind": "tool_call",
        "tool_use_id": "c2",
        "tool": "bash",
        "target": "uv run ruff format src/a.py",
        "target_class": "",
    },
    {
        "kind": "tool_result",
        "tool_use_id": "c2",
        "is_error": False,
        "excerpt": "1 file reformatted",
    },
    {
        "kind": "tool_call",
        "tool_use_id": "c3",
        "tool": "run_checks",
        "target": "",
        "target_class": "",
    },
    {
        "kind": "tool_result",
        "tool_use_id": "c3",
        "is_error": False,
        "excerpt": "all checks passed",
    },
    {
        "kind": "harness_correction",
        "trigger": "gate_red",
        "detail": "ruff format --check: would reformat src/a.py",
        "resolved_by": ["c2", "c3"],
        "resolved": True,
    },
    {
        "kind": "harness_correction",
        "trigger": "nudge",
        "detail": "call finish when the work is done",
        "resolved_by": [],
        "resolved": False,
    },
]


def test_drain_shape4_talos_digest(
    local_client: TestClient, ctx: _Ctx, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    _seed(
        local_client,
        "t4",
        0,
        user_prompt="fix it",
        harness="talos",
        mode="headless",
        items=_S4_ITEMS,
    )
    body = _drain(local_client)
    assert ctx.llm.generate_calls == []
    stored = _query("SELECT items FROM turn_events WHERE event_id = 't4:0'")
    assert json.loads(stored[0]["items"])[2]["tool"] == "Bash"
    rows = _detections()
    assert [(r["event_id"], r["shape"], r["outcome"], r["reason"]) for r in rows] == [
        ("t4:0", 1, "no_surprise", ""),
        ("t4:0", 2, "not_applicable", "no_prev"),
        ("t4:0", 3, "not_applicable", "no_text_after_result"),
        ("t4:0", 4, "candidate", ""),
        ("t4:0", 4, "not_applicable", "unresolved"),
    ]
    cand_row, unresolved = rows[3], rows[4]
    for r in (cand_row, unresolved):
        assert r["detector_model"] == "rule:shape4"
        assert r["detector_version"] == SURPRISE_DETECTOR_VERSION
    flags = {"truncated": False, "out_of_order": False, "turn_gap": False}
    assert cand_row["confidence"] == 1.0
    assert json.loads(cand_row["details"]) == {
        "trigger": "gate_red",
        "resolved": True,
        "resolved_by_count": 2,
        "resolved_by_matched": 2,
        "resolved_by_errored": 0,
        "last_resolving_error": False,
        "resolved_by_tools": ["Bash", "run_checks"],
        **flags,
    }
    assert unresolved["candidate_id"] is None
    assert unresolved["confidence"] is None
    assert json.loads(unresolved["details"]) == {
        "trigger": "nudge",
        "resolved": False,
        "resolved_by_count": 0,
        "resolved_by_matched": 0,
        "resolved_by_errored": 0,
        "last_resolving_error": None,
        "resolved_by_tools": [],
        **flags,
    }
    (cand,) = body["candidates"]
    assert cand["shape"] == 4
    assert cand["status"] == "shadow"
    assert cand["detector_model"] == "rule:shape4"
    assert cand["turn_event_ids"] == ["t4:0"]
    assert cand["detector_output"] == S4_OUT
    assert ctx.dry_run_calls == [[cand["id"]]]
    line = _info_line(caplog)
    assert "shape4=1 shape4_records=2 shape4_unmatched=0 shape4_partial=0" in line
    resp = local_client.get("/api/kb/surprise/candidates", params={"shape": 4})
    assert resp.status_code == 200, resp.text
    (audit,) = resp.json()["candidates"]
    assert audit["shape"] == 4
    assert audit["mode"] == "headless"


def test_drain_shape4_window_and_counters(
    local_client: TestClient, ctx: _Ctx, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    _seed(
        local_client,
        "t5",
        0,
        items=[
            {
                "kind": "harness_correction",
                "trigger": "claim_rejected",
                "detail": "criterion 2 is done",
                "resolved_by": ["c9"],
                "resolved": True,
            }
        ],
    )
    _seed(
        local_client,
        "t5",
        1,
        items=[
            {
                "kind": "tool_call",
                "tool_use_id": "c9",
                "tool": "Edit",
                "target": "src/a.py",
                "target_class": "ext:py",
            },
            {
                "kind": "tool_result",
                "tool_use_id": "c9",
                "is_error": False,
                "excerpt": "",
            },
        ],
    )
    _seed(
        local_client,
        "t6",
        0,
        items=[
            {
                "kind": "tool_call",
                "tool_use_id": "c7",
                "tool": "Edit",
                "target": "src/a.py",
                "target_class": "ext:py",
            },
            {
                "kind": "tool_result",
                "tool_use_id": "c7",
                "is_error": False,
                "excerpt": "",
            },
            {
                "kind": "harness_correction",
                "trigger": "gate_red",
                "detail": "FAILED tests/test_a.py::test_x",
                "resolved_by": ["c7", "zz"],
                "resolved": True,
            },
        ],
    )
    _drain(local_client)
    assert ctx.llm.generate_calls == []
    assert [
        (r["event_id"], r["outcome"], r["reason"])
        for r in _detections()
        if r["shape"] == 4
    ] == [
        ("t5:0", "ungrounded", "resolved_by_unmatched"),
        ("t6:0", "candidate", ""),
    ]
    line = _info_line(caplog)
    assert "shape4=1 shape4_records=2 shape4_unmatched=1 shape4_partial=1" in line
    assert (
        "surprise_drain shape4_unmatched event_id=t5:0 trigger=claim_rejected"
        " resolved_by=1 turn_gap=False"
    ) in _messages(caplog, logging.WARNING)


async def test_detect_digest_shape4_last() -> None:
    hc = {
        "kind": "harness_correction",
        "trigger": "gate_red",
        "detail": "x",
        "resolved_by": ["c2"],
        "resolved": True,
    }
    cur = surprise_worker.digest_from_row(_row(items=[*_S4_ITEMS[2:4], hc]))
    records = await surprise_worker.detect_digest(None, cur, [cur])
    assert [r.shape for r in records] == [1, 2, 3, 4]
    assert records[-1].outcome == "candidate"
    cur = surprise_worker.digest_from_row(_row(items=_S4_ITEMS[2:4]))
    records = await surprise_worker.detect_digest(None, cur, [cur])
    assert [r.shape for r in records] == [1, 2, 3]


# --- subagent digests are never distilled (capture for observation only) ------


def _seed_agent(
    client: TestClient,
    session: str,
    turn: int,
    agent: str = "a1",
    **kw: Any,
) -> None:
    body: dict[str, Any] = {
        "event_id": f"{session}:agent:{agent}",
        "session_id": session,
        "harness": "claude-code",
        "mode": "interactive",
        "engine": None,
        "host": "h",
        "hook_version": "1.3.0",
        "project": "p",
        "turn_index": turn,
        "agent_id": agent,
        "agent_type": "general-purpose",
        "ts": "2026-10-09T00:00:00+00:00",
        "user_prompt": "subagent prompt",
        "items": [],
        "final_message": "subagent done",
        "truncated": False,
    }
    body.update(kw)
    resp = client.post("/api/kb/turn", json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["reason"] == "recorded"


def test_drain_agent_rows_claimed_and_skipped(
    local_client: TestClient,
    ctx: _Ctx,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pending agent digest leaves the queue without any detector work."""
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "on")
    caplog.set_level(logging.INFO, logger=LOGGER)
    _seed_f(local_client)  # s1:0 and s1:1 pending
    _seed_agent(local_client, "s1", 1, agent="a1")  # parent session, turn 1
    _seed_agent(local_client, "s2", 0, agent="a2")
    _enqueue(ctx.llm, SCRIPT_B)
    body = _drain(local_client)
    assert body["digests_processed"] == 2  # only the non-agent rows
    assert "agent_skipped=2" in _info_line(caplog)
    rows = _query("SELECT processed_at FROM turn_events WHERE agent_id IS NOT NULL")
    assert [r["processed_at"] is not None for r in rows] == [True, True]
    # Zero detector or LLM work is attributable to either agent row: the two
    # parent digests account for every call and every candidate.
    assert len(ctx.llm.generate_calls) == 2
    assert _detections("s1:agent:a1") == []
    assert _detections("s2:agent:a2") == []
    for event_id in ("s1:agent:a1", "s2:agent:a2"):
        assert (
            _query(
                "SELECT 1 FROM surprise_candidates"
                " WHERE turn_event_ids LIKE '%' || ? || '%'",
                event_id,
            )
            == []
        )
    cands = body["candidates"]
    assert "s1:agent:a1" not in {i for c in cands for i in c["turn_event_ids"]}
    prompts = " ".join(str(p) for p, _ in ctx.llm.generate_calls)
    assert "subagent prompt" not in prompts
    assert "subagent done" not in prompts


def test_drain_agent_row_never_in_session_context(
    local_client: TestClient, ctx: _Ctx, caplog: pytest.LogCaptureFixture
) -> None:
    """An agent digest sharing the parent's session never feeds the detectors."""
    caplog.set_level(logging.INFO, logger=LOGGER)
    _seed(local_client, "s6", 2, user_prompt="a", final_message="b")
    _seed_agent(local_client, "s6", 1, agent="a1")
    _enqueue(ctx.llm, SCRIPT_B)
    body = _drain(local_client)
    assert body["digests_processed"] == 1
    # The shape-2 prev pick for s6:2 must not land on the agent digest at
    # turn_index 1: with no parent digest at turn 1 the shape is not
    # applicable, and the agent's prompt never reaches any LLM call.
    shape2 = [r for r in _detections("s6:2") if r["shape"] == 2]
    assert [r["outcome"] for r in shape2] == ["not_applicable"]
    prompts = " ".join(str(p) for p, _ in ctx.llm.generate_calls)
    assert "subagent prompt" not in prompts
    assert "subagent done" not in prompts
    # The turn_gap flag is computed over the agent-free list too.
    assert all(
        json.loads(r["details"])["turn_gap"] is True for r in _detections("s6:2")
    )
