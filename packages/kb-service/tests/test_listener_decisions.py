"""Hermetic tests for the ``listener_decisions`` best-effort telemetry sink.

Companion to ``test_listener_routes.py`` (which locks in the HTTP response
shape per branch). This file locks in the SEPARATE, additive
``listener_decisions`` row written on every branch of
``POST /api/kb/listener`` — including every decline path, which previously
left no durable trace at all (GTD 65b308de).

Test matrix:
  (a) one row per branch: kill-switch, no-candidates, rule-a, rule-b, no-llm,
      whispered, vote-split, vote-none — decision + reason + candidates_considered
      match the branch.
  (b) exactly one INSERT per listener request (never more, never fewer).
  (c) a raising DB layer still yields a normal 200 response (best-effort,
      never fails or slows the caller).
  (d) primary (non-fallback) path keeps the granular reason and records
      retrieval_path="primary".
  (e) THE REASON MASK IS GONE (GTD 268e2af3): `reason` ALWAYS stays the
      granular per-branch value, even when the fallback direct-map-search
      path (GTD bf40d4f1) ran — the fallback is now recorded separately in
      the `retrieval_path` column ("fallback-direct" / "primary"), so a
      fallback-path decline's real decline cause is never destroyed.
  (f) candidate_ids / whispered_ids / per-stage counters: candidate_ids is
      the PRE-RULE-A retrieved pool (non-empty even for a rule-a/rule-b
      decline — the entire point of the column); whispered_ids is only
      non-empty on the whisper path; n_retrieved/n_after_a/n_after_b track
      attrition through the two filter stages.

``_make_map_result`` seeds ``entry_type=MENTAL_MAP`` results, which the
PRIMARY detail-match retrieval stage always filters out (see GTD bf40d4f1) —
so every case in matrix (a) exercises the FALLBACK path by construction. A
dedicated primary-path (no fallback) case (d) proves retrieval_path flips to
"primary" when detail-matching itself resolves the candidate.
"""

import json
import logging
import types
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.models.search import SearchResult

import kb_service.routes.listener_routes as listener_routes
import kb_service.turn_digest as turn_digest
from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, FakeLLM, fake_user

# ─── recording fake pool ─────────────────────────────────────────────────────


class _RecordingDecisionPool:
    """Minimal pool recording ``execute()`` calls, or raising if configured."""

    def __init__(self, raise_on_execute: Exception | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._raise_on_execute = raise_on_execute

    async def execute(self, sql: str, *args: Any) -> str:
        if self._raise_on_execute is not None:
            raise self._raise_on_execute
        self.calls.append((sql, args))
        return "OK"


@pytest.fixture
def decision_pool() -> _RecordingDecisionPool:
    return _RecordingDecisionPool()


@pytest.fixture
def _patch_decision_pool(
    monkeypatch: pytest.MonkeyPatch, decision_pool: _RecordingDecisionPool
) -> None:
    async def _fake_get_db() -> _RecordingDecisionPool:
        return decision_pool

    monkeypatch.setattr(listener_routes, "get_db", _fake_get_db)


def _make_map_result(
    entry_id: str,
    *,
    short_title: str = "Test Map",
    knowledge_details: str = "Details about this map.",
    project_ref: str | None = None,
    hints: dict | None = None,
) -> SearchResult:
    entry = KnowledgeEntry(
        id=entry_id,
        short_title=short_title,
        long_title=short_title + " (long title)",
        knowledge_details=knowledge_details,
        entry_type=EntryType.MENTAL_MAP,
        project_ref=project_ref,
        hints=hints if hints is not None else {},
    )
    return SearchResult(
        entry=entry,
        score=0.9,
        effective_confidence=0.9,
        staleness_warning=None,
        match_source="hybrid",
    )


def _decision_args(pool: _RecordingDecisionPool) -> dict[str, Any]:
    """Unpack the single recorded INSERT's positional args into a dict."""
    assert len(pool.calls) == 1, (
        f"expected exactly 1 decision write, got {len(pool.calls)}"
    )
    sql, args = pool.calls[0]
    assert "INSERT INTO listener_decisions" in sql
    assert len(args) == 20, f"expected 20 INSERT args, got {len(args)}"
    (
        session_id,
        cwd_project,
        source_kb,
        decided_ts,
        candidates_considered,
        decision,
        reason,
        vote_shape,
        candidate_signal,
        candidate_ids,
        whispered_ids,
        n_retrieved,
        n_after_a,
        n_after_b,
        retrieval_path,
        text_source,
        vote_model,
        vote_prompt,
        vote_raw,
        vote_truncated,
    ) = args
    return {
        "session_id": session_id,
        "cwd_project": cwd_project,
        "source_kb": source_kb,
        "decided_ts": decided_ts,
        "candidates_considered": candidates_considered,
        "decision": decision,
        "reason": reason,
        "vote_shape": vote_shape,
        "candidate_signal": candidate_signal,
        "candidate_ids": json.loads(candidate_ids),
        "whispered_ids": json.loads(whispered_ids),
        "n_retrieved": n_retrieved,
        "n_after_a": n_after_a,
        "n_after_b": n_after_b,
        "retrieval_path": retrieval_path,
        "text_source": text_source,
        "vote_model": vote_model,
        "vote_prompt": vote_prompt,
        "vote_raw": vote_raw,
        "vote_truncated": vote_truncated,
    }


def _assert_no_vote_log(row: dict[str, Any]) -> None:
    """Every branch that never reached the LLM gate stores NULLs, not ''."""
    assert row["vote_model"] is None
    assert row["vote_prompt"] is None
    assert row["vote_raw"] is None
    assert row["vote_truncated"] == 0


# ─── (a) one row per branch ──────────────────────────────────────────────────


@pytest.mark.usefixtures("_patch_decision_pool")
def test_decision_kill_switch(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    monkeypatch.setenv("KB_LISTENER_ENABLED", "FALSE")
    app.dependency_overrides[get_current_user] = fake_user

    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "session_id": "sess-1", "cwd_project": "proj-a"},
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["decision"] == "declined"
    assert row["reason"] == "kill-switch"
    _assert_no_vote_log(row)
    assert row["candidates_considered"] == 0
    assert row["candidate_signal"] == ""
    assert row["session_id"] == "sess-1"
    assert row["cwd_project"] == "proj-a"
    # Kill switch fires before retrieval ever runs.
    assert row["candidate_ids"] == []
    assert row["whispered_ids"] == []
    assert row["n_retrieved"] == 0
    assert row["n_after_a"] == 0
    assert row["n_after_b"] == 0


@pytest.mark.usefixtures("_patch_decision_pool")
@pytest.mark.parametrize(
    ("extra", "expected"),
    [({"text_source": "last_assistant_message"}, "last_assistant_message"), ({}, None)],
)
def test_decision_text_source_round_trip(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
    extra: dict[str, str],
    expected: str | None,
) -> None:
    monkeypatch.setenv("KB_LISTENER_ENABLED", "FALSE")
    app.dependency_overrides[get_current_user] = fake_user

    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "s", **extra}
    )
    assert resp.status_code == 200
    assert _decision_args(decision_pool)["text_source"] == expected


@pytest.mark.usefixtures("_patch_decision_pool")
def test_decision_no_candidates_from_retrieval(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = []
    fake_kb.filtered_count = 0

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-2"}
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["decision"] == "declined"
    # GTD 268e2af3: reason stays granular even on the fallback path; the
    # fallback itself is recorded separately in retrieval_path.
    assert row["reason"] == "no-candidates"
    _assert_no_vote_log(row)
    assert row["retrieval_path"] == "fallback-direct"
    assert row["candidates_considered"] == 0
    assert row["candidate_signal"] == ""  # decline branches never attribute a signal
    assert row["candidate_ids"] == []
    assert row["n_retrieved"] == 0
    assert row["n_after_a"] == 0
    assert row["n_after_b"] == 0


@pytest.mark.usefixtures("_patch_decision_pool")
def test_decision_rule_a(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [_make_map_result("kb-00001", project_ref="my-project")]
    fake_kb.filtered_count = 1

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "cwd_project": "my-project", "session_id": "sess-3"},
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["decision"] == "declined"
    # THE load-bearing assertion (GTD 268e2af3): a rule-a decline still
    _assert_no_vote_log(row)
    # records the FULL pre-filter candidate pool, non-empty.
    assert row["reason"] == "rule-a"
    assert row["retrieval_path"] == "fallback-direct"
    assert row["candidates_considered"] == 0
    assert row["candidate_signal"] == ""
    assert row["candidate_ids"] == ["kb-00001"]
    assert row["n_retrieved"] == 1
    assert row["n_after_a"] == 0
    assert row["n_after_b"] == 0


@pytest.mark.usefixtures("_patch_decision_pool")
def test_decision_rule_b(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [
        _make_map_result("kb-00001", hints={"operated_via": "mcp:agent-gtd"}),
    ]
    fake_kb.filtered_count = 1

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "operating": ["mcp:agent-gtd"], "session_id": "sess-4"},
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["decision"] == "declined"
    assert row["reason"] == "rule-b"
    _assert_no_vote_log(row)
    assert row["retrieval_path"] == "fallback-direct"
    assert row["candidates_considered"] == 0
    assert row["candidate_signal"] == ""
    assert row["candidate_ids"] == ["kb-00001"]
    assert row["n_retrieved"] == 1
    assert row["n_after_a"] == 1
    assert row["n_after_b"] == 0


@pytest.mark.usefixtures("_patch_decision_pool")
def test_decision_no_llm(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [_make_map_result("kb-00001", short_title="Test Map")]
    fake_kb.filtered_count = 1
    fake_kb.synthesis_llm = None

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-5"}
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["decision"] == "declined"
    assert row["reason"] == "no-llm"
    _assert_no_vote_log(row)
    assert row["retrieval_path"] == "fallback-direct"
    assert row["candidates_considered"] == 1
    assert row["candidate_signal"] == ""
    assert row["candidate_ids"] == ["kb-00001"]
    assert row["n_retrieved"] == 1
    assert row["n_after_a"] == 1
    assert row["n_after_b"] == 1


@pytest.mark.usefixtures("_patch_decision_pool")
def test_decision_whispered(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_kb.results = [_make_map_result("kb-00001", short_title="Home Network Map")]
    fake_kb.filtered_count = 1
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-6"}
    )
    assert resp.status_code == 200
    assert resp.json()["pointer"] is not None

    row = _decision_args(decision_pool)
    assert row["decision"] == "whisper"
    assert row["reason"] == "whispered"
    assert row["retrieval_path"] == "fallback-direct"
    assert row["candidates_considered"] == 1
    assert row["vote_shape"] == '[["kb-00001"], ["kb-00001"], ["kb-00001"]]'
    # Sole candidate came from the detail-match retrieval's fallback leg (no
    # maps_projects/maps_rows seeded in this fixture -> the lexical path
    # never fires here).
    assert row["candidate_signal"] == "fallback"
    assert row["candidate_ids"] == ["kb-00001"]
    assert row["whispered_ids"] == ["kb-00001"]
    assert row["n_retrieved"] == 1
    assert row["n_after_a"] == 1
    assert row["n_after_b"] == 1


@pytest.mark.usefixtures("_patch_decision_pool")
def test_decision_vote_majority_whispers(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    """GTD 66ea1fe4 reframe: a 2/3 majority (previously a non-unanimous
    decline) now whispers — ``decision`` flips to "whisper" accordingly."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [
        _make_map_result("kb-00001", short_title="Map One"),
        _make_map_result("kb-00002", short_title="Map Two"),
    ]
    fake_kb.filtered_count = 2

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00002")
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-7"}
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["decision"] == "whisper"
    assert row["reason"] == "whispered"
    assert row["retrieval_path"] == "fallback-direct"
    assert row["candidates_considered"] == 2
    assert row["vote_shape"] == '[["kb-00001"], ["kb-00001"], ["kb-00002"]]'
    assert row["candidate_signal"] == "fallback"
    assert row["candidate_ids"] == ["kb-00001", "kb-00002"]
    assert row["whispered_ids"] == ["kb-00001"]
    assert row["n_retrieved"] == 2
    assert row["n_after_a"] == 2
    assert row["n_after_b"] == 2


@pytest.mark.usefixtures("_patch_decision_pool")
def test_decision_no_majority_declines(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    """A genuine 3-way split (no id reaches the 2/3 majority bar) still declines."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [
        _make_map_result("kb-00001", short_title="Map One"),
        _make_map_result("kb-00002", short_title="Map Two"),
        _make_map_result("kb-00003", short_title="Map Three"),
    ]
    fake_kb.filtered_count = 3

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00002")
    fake_llm.enqueue("kb-00003")
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-7b"}
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["decision"] == "declined"
    assert row["reason"] == "vote-split"
    assert row["retrieval_path"] == "fallback-direct"
    assert row["candidates_considered"] == 3
    assert row["vote_shape"] == '[["kb-00001"], ["kb-00002"], ["kb-00003"]]'
    assert row["candidate_signal"] == ""
    assert row["candidate_ids"] == ["kb-00001", "kb-00002", "kb-00003"]
    assert row["whispered_ids"] == []
    assert row["n_retrieved"] == 3
    assert row["n_after_a"] == 3
    assert row["n_after_b"] == 3


@pytest.mark.usefixtures("_patch_decision_pool")
def test_decision_vote_none(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [_make_map_result("kb-00001", short_title="Test Map")]
    fake_kb.filtered_count = 1

    fake_llm = FakeLLM()
    fake_llm.enqueue("NONE")
    fake_llm.enqueue("NONE")
    fake_llm.enqueue("NONE")
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-8"}
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["decision"] == "declined"
    assert row["reason"] == "vote-none"
    assert row["retrieval_path"] == "fallback-direct"
    assert row["candidates_considered"] == 1
    assert row["vote_shape"] == "[[], [], []]"
    assert row["candidate_signal"] == ""
    assert row["candidate_ids"] == ["kb-00001"]
    assert row["whispered_ids"] == []
    assert row["n_retrieved"] == 1
    assert row["n_after_a"] == 1
    assert row["n_after_b"] == 1


# ─── (b) exactly one INSERT per request ─────────────────────────────────────


@pytest.mark.usefixtures("_patch_decision_pool")
def test_exactly_one_decision_row_per_request(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    """Two sequential listener requests write exactly two decision rows total."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "FALSE")
    app.dependency_overrides[get_current_user] = fake_user

    client.post("/api/kb/listener", json={"text": "hello", "session_id": "sess-9"})
    client.post("/api/kb/listener", json={"text": "world", "session_id": "sess-10"})

    assert len(decision_pool.calls) == 2


# ─── (c) best-effort: a raising DB layer never breaks the response ──────────


def test_decision_write_failure_does_not_break_response(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising DB layer must still yield a normal 200 response.

    The pointer/null response is returned regardless of whether the
    best-effort ``listener_decisions`` write succeeds.
    """
    raising_pool = _RecordingDecisionPool(raise_on_execute=RuntimeError("db is down"))

    async def _fake_get_db() -> _RecordingDecisionPool:
        return raising_pool

    monkeypatch.setattr(listener_routes, "get_db", _fake_get_db)
    monkeypatch.setenv("KB_LISTENER_ENABLED", "FALSE")
    app.dependency_overrides[get_current_user] = fake_user
    caplog.set_level(logging.WARNING, logger="kb_service.routes.listener_routes")

    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-11"}
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "pointer": None,
        "pointers": [],
        "reason": "kill-switch: KB_LISTENER_ENABLED!=TRUE",
    }
    # The pool recorded no successful call (it raised), proving the write was
    # attempted and swallowed rather than skipped.
    assert raising_pool.calls == []
    # Exactly one INSERT attempt, surfaced at WARNING (with the traceback),
    # never DEBUG: a silent sink is indistinguishable from a working one.
    warnings = [
        r
        for r in caplog.records
        if r.getMessage().startswith("listener decision write failed")
    ]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert warnings[0].exc_info is not None
    assert "db is down" in warnings[0].getMessage()


def test_decision_write_failure_with_candidates_and_llm_path(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same as above, but on the unanimous-whisper path (post-LLM-gate write)."""
    raising_pool = _RecordingDecisionPool(raise_on_execute=RuntimeError("db is down"))

    async def _fake_get_db() -> _RecordingDecisionPool:
        return raising_pool

    monkeypatch.setattr(listener_routes, "get_db", _fake_get_db)
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_kb.results = [_make_map_result("kb-00001", short_title="Home Network Map")]
    fake_kb.filtered_count = 1
    fake_kb.synthesis_llm = fake_llm
    app.dependency_overrides[get_current_user] = fake_user
    caplog.set_level(logging.WARNING, logger="kb_service.routes.listener_routes")

    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-12"}
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "pointer": {"id": "kb-00001", "short_title": "Home Network Map"},
        "pointers": [{"id": "kb-00001", "short_title": "Home Network Map"}],
        "reason": "matched kb-00001 (unanimous 3/3)",
    }
    assert raising_pool.calls == []
    assert [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("listener decision write failed")
    ]


# ─── (d) primary (non-fallback) path: retrieval_path == "primary" ──────────


@pytest.mark.usefixtures("_patch_decision_pool")
def test_decision_primary_path_rule_a_keeps_granular_reason(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    """When detail-matching itself resolves a candidate (no fallback), the
    stored reason is the granular per-branch value AND retrieval_path
    records "primary" — "fallback-direct" is reserved for retrieval_path on
    decisions reached via the fallback path (see matrix (e) below for that).
    """
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [
        SearchResult(
            entry=KnowledgeEntry(
                id="kb-00099",
                short_title="Detail entry",
                long_title="Detail entry (long)",
                knowledge_details="rsync camera-profiles-data raw-pairs a7r6",
                entry_type=EntryType.FACTUAL_REFERENCE,
            ),
            score=1.0 / 61,
            effective_confidence=0.9,
            staleness_warning=None,
            match_source="fts",
        )
    ]
    fake_kb.filtered_count = 0
    fake_kb.db.rows_for["graph_edges"] = [("kb-00099", "kb-00001")]
    fake_kb.entries["kb-00001"] = KnowledgeEntry(
        id="kb-00001",
        short_title="My Project Map",
        long_title="My Project Map (long)",
        knowledge_details="Orientation map.",
        entry_type=EntryType.MENTAL_MAP,
        project_ref="my-project",
        is_active=True,
    )

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "cwd_project": "my-project", "session_id": "sess-13"},
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["decision"] == "declined"
    assert row["reason"] == "rule-a"
    assert row["retrieval_path"] == "primary"
    assert row["candidates_considered"] == 0
    assert row["candidate_signal"] == ""
    # PRE-RULE-A capture: candidate_ids is non-empty even though the
    # candidate was dropped by rule A before the LLM ever ran.
    assert row["candidate_ids"] == ["kb-00001"]
    assert row["n_retrieved"] == 1
    assert row["n_after_a"] == 0
    assert row["n_after_b"] == 0


# ─── (e) fallback-path decline records BOTH reason and retrieval_path ──────


@pytest.mark.usefixtures("_patch_decision_pool")
def test_decision_fallback_path_records_both_reason_and_retrieval_path(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    """A fallback-path decline records its REAL granular reason ("rule-a")
    AND separately flags that the fallback retrieval path ran
    (retrieval_path="fallback-direct") — the reason mask (GTD 268e2af3) is
    gone; neither field is sacrificed for the other.
    """
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    # A MENTAL_MAP result is excluded by the primary detail-match stage
    # (GTD bf40d4f1), so this always falls back to direct mental_map search.
    fake_kb.results = [_make_map_result("kb-00001", project_ref="my-project")]
    fake_kb.filtered_count = 1

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "cwd_project": "my-project", "session_id": "sess-14"},
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["reason"] == "rule-a"
    assert row["retrieval_path"] == "fallback-direct"
    assert row["candidate_ids"] == ["kb-00001"]


# ─── endpoint mounted (sanity, mirrors test_listener_routes.py) ─────────────


def test_listener_endpoint_still_mounted() -> None:
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/api/kb/listener" in paths


# ─── vote log (owner decision 2026-10-13, hypothesis H-020) ──────────────────

# Assembled from fragments (mirroring kb-core's test_redact_secrets) so no
# source line in THIS file matches a scanner rule.
_ANT_TOKEN = "sk-ant-api03-" + "abcdefghijklmnopqrstuvwxyz0123456789"
_AWS_KEY_ID = "AKIA" + "IOSFODNN7EXAMPLE"


def _voted_llm(*votes: str | None, model: str | None = "claude-sonnet-4-6") -> FakeLLM:
    """A FakeLLM with three enqueued votes and an optional model config."""
    fake_llm = FakeLLM()
    if model is not None:
        fake_llm._config = types.SimpleNamespace(model=model)
    for vote in votes:
        fake_llm.enqueue(vote)
    return fake_llm


def _two_candidate_kb(fake_kb: FakeKnowledgeBase, fake_llm: FakeLLM) -> None:
    fake_kb.results = [
        _make_map_result("kb-00001", short_title="Map One"),
        _make_map_result("kb-00002", short_title="Map Two"),
    ]
    fake_kb.filtered_count = 2
    fake_kb.synthesis_llm = fake_llm


@pytest.mark.usefixtures("_patch_decision_pool")
def test_vote_log_stored_on_voted_decision(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    """A voted decision stores the model, the prompt and the 3 raw answers."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_llm = _voted_llm("kb-00001", "kb-00002", "kb-00001")
    _two_candidate_kb(fake_kb, fake_llm)
    app.dependency_overrides[get_current_user] = fake_user

    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-v1"}
    )
    assert resp.status_code == 200
    assert resp.json()["pointers"] == [{"id": "kb-00001", "short_title": "Map One"}]

    row = _decision_args(decision_pool)
    assert row["reason"] == "whispered"
    # The model id comes from the client's config, verbatim.
    assert row["vote_model"] == "claude-sonnet-4-6"
    # The exact prompt string all three listener_vote calls received.
    assert len(fake_llm.generate_calls) == 3
    assert fake_llm.purposes == ["listener_vote"] * 3
    assert row["vote_prompt"] == fake_llm.generate_calls[0][0]
    # Three raw answers in issue order.
    assert json.loads(row["vote_raw"]) == ["kb-00001", "kb-00002", "kb-00001"]
    assert row["vote_truncated"] == 0


@pytest.mark.usefixtures("_patch_decision_pool")
def test_vote_log_model_none_for_config_less_client(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    """A client whose config cannot be read stores NULL, never a class name."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_llm = _voted_llm("kb-00001", "kb-00001", "kb-00001", model=None)
    _two_candidate_kb(fake_kb, fake_llm)
    app.dependency_overrides[get_current_user] = fake_user

    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-v2"}
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["reason"] == "whispered"
    assert row["vote_model"] is None
    assert row["vote_prompt"] == fake_llm.generate_calls[0][0]
    assert json.loads(row["vote_raw"]) == ["kb-00001", "kb-00001", "kb-00001"]


@pytest.mark.usefixtures("_patch_decision_pool")
def test_vote_log_redacts_secrets_in_prompt_and_answers(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    """A secret in the judged text or in an answer is stored only redacted."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_llm = _voted_llm(_ANT_TOKEN, "kb-00001", "kb-00001")
    _two_candidate_kb(fake_kb, fake_llm)
    app.dependency_overrides[get_current_user] = fake_user

    resp = client.post(
        "/api/kb/listener",
        json={"text": f"AWS_ACCESS_KEY_ID={_AWS_KEY_ID}", "session_id": "sess-v3"},
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["reason"] == "whispered"
    # Redacted, never raw — in the prompt...
    assert _AWS_KEY_ID not in row["vote_prompt"]
    assert "[REDACTED:" in row["vote_prompt"]
    # ...and in each raw answer.
    raw = json.loads(row["vote_raw"])
    assert len(raw) == 3
    assert _ANT_TOKEN not in raw[0]
    assert raw[0].startswith("[REDACTED:")
    assert raw[1] == "kb-00001"
    assert raw[2] == "kb-00001"


@pytest.mark.usefixtures("_patch_decision_pool")
def test_vote_log_caps_prompt_after_redaction(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Redaction runs BEFORE the cap: a 17000-char prompt stores exactly 16000
    redacted chars, and the re-truncation is visible in vote_truncated + INFO."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_llm = _voted_llm("kb-00001", "kb-00001", "kb-00001")
    _two_candidate_kb(fake_kb, fake_llm)
    app.dependency_overrides[get_current_user] = fake_user

    text = _ANT_TOKEN + "\n" + "x" * 17000  # secret in the FIRST 500 chars
    caplog.set_level(logging.INFO, logger="kb_service.routes.listener_routes")
    resp = client.post("/api/kb/listener", json={"text": text, "session_id": "sess-v4"})
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert len(row["vote_prompt"]) == 16000
    assert "[REDACTED:Anthropic API Key]" in row["vote_prompt"]
    assert _ANT_TOKEN not in row["vote_prompt"]
    assert row["vote_truncated"] == 1
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("listener vote_log retruncated")
    ]
    assert len(lines) == 1
    assert "session_id=sess-v4" in lines[0]
    assert "fields=1" in lines[0]


@pytest.mark.usefixtures("_patch_decision_pool")
def test_vote_log_caps_secret_free_prompt_at_16000(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    """A secret-free over-long prompt stores exactly its first 16000 chars."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_llm = _voted_llm("kb-00001", "kb-00001", "kb-00001")
    _two_candidate_kb(fake_kb, fake_llm)
    app.dependency_overrides[get_current_user] = fake_user

    resp = client.post(
        "/api/kb/listener", json={"text": "y" * 17000, "session_id": "sess-v5"}
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["vote_prompt"] == fake_llm.generate_calls[0][0][:16000]
    assert row["vote_truncated"] == 1


@pytest.mark.usefixtures("_patch_decision_pool")
def test_vote_log_caps_each_raw_vote_at_2000(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    """A 2500-char answer stores exactly 2000 chars and flips vote_truncated."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_llm = _voted_llm("kb-00001", "z" * 2500, "kb-00001")
    _two_candidate_kb(fake_kb, fake_llm)
    app.dependency_overrides[get_current_user] = fake_user

    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-v6"}
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    raw = json.loads(row["vote_raw"])
    assert len(raw) == 3
    assert raw[0] == "kb-00001"
    assert raw[1] == "z" * 2000
    assert raw[2] == "kb-00001"
    assert row["vote_truncated"] == 1


@pytest.mark.usefixtures("_patch_decision_pool")
def test_vote_log_preserves_none_vote_slot(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    """A None answer stores json null in ITS slot on a genuine vote-split row.

    Two DISTINCT ids are required: ('kb-00001', 'kb-00001', None) would reach
    the whispered branch instead (2/3 majority), not a split.
    """
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_llm = _voted_llm("kb-00001", "kb-00002", None)
    _two_candidate_kb(fake_kb, fake_llm)
    app.dependency_overrides[get_current_user] = fake_user

    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-v7"}
    )
    assert resp.status_code == 200

    row = _decision_args(decision_pool)
    assert row["decision"] == "declined"
    assert row["reason"] == "vote-split"
    assert row["vote_shape"] == '[["kb-00001"], ["kb-00002"], []]'
    raw = json.loads(row["vote_raw"])
    assert len(raw) == 3
    assert raw[0] == "kb-00001"
    assert raw[1] == "kb-00002"
    assert raw[2] is None


@pytest.mark.usefixtures("_patch_decision_pool")
def test_vote_log_redaction_unavailable_stores_no_text(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Redaction down: no vote text is stored (never raw), the decision stands.

    The route reuses turn_digest's redactor, so turning its
    ``redact_secrets`` into a None-returning stub (the production ImportError
    shape) is enough to take the whole vote log down.
    """
    monkeypatch.setattr(turn_digest, "redact_secrets", lambda _c: None)
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_llm = _voted_llm("kb-00001", "kb-00001", "kb-00001")
    _two_candidate_kb(fake_kb, fake_llm)
    app.dependency_overrides[get_current_user] = fake_user

    caplog.set_level(logging.WARNING, logger="kb_service.routes.listener_routes")
    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-v8"}
    )
    assert resp.status_code == 200
    # The decision itself is untouched by the redaction failure.
    assert resp.json()["pointers"] == [{"id": "kb-00001", "short_title": "Map One"}]

    row = _decision_args(decision_pool)
    assert row["reason"] == "whispered"
    # No redactor output from this request is ever stored...
    assert row["vote_prompt"] is None
    assert row["vote_raw"] is None
    assert row["vote_truncated"] == 0
    # ...but the model is config metadata, so the row stays distinguishable
    # from a model-unknown row.
    assert row["vote_model"] == "claude-sonnet-4-6"
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("listener vote_log redaction-unavailable")
    ]
    assert len(lines) == 1
    assert "session_id=sess-v8" in lines[0]
