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
  (d) fallback-direct: any whisper/decline reached via the fallback
      direct-map-search path (GTD bf40d4f1) is recorded with
      reason="fallback-direct" instead of the granular branch reason, while
      `decision` (whisper/declined) stays correct.

``_make_map_result`` seeds ``entry_type=MENTAL_MAP`` results, which the
PRIMARY detail-match retrieval stage always filters out (see GTD bf40d4f1) —
so every case in matrix (a) exercises the FALLBACK path by construction, and
the stored `reason` is therefore "fallback-direct" rather than the granular
per-branch value the pre-fix code recorded. A dedicated primary-path (no
fallback) case proves the granular reason is preserved when detail-matching
itself resolves the candidate.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.models.search import SearchResult

import kb_service.routes.listener_routes as listener_routes
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
    }


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
    assert row["candidates_considered"] == 0
    assert row["candidate_signal"] == ""
    assert row["session_id"] == "sess-1"
    assert row["cwd_project"] == "proj-a"


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
    assert row["reason"] == "fallback-direct"  # was "no-candidates" pre-fallback
    assert row["candidates_considered"] == 0
    assert row["candidate_signal"] == ""  # decline branches never attribute a signal


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
    assert row["reason"] == "fallback-direct"  # was "rule-a" pre-fallback
    assert row["candidates_considered"] == 0
    assert row["candidate_signal"] == ""


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
    assert row["reason"] == "fallback-direct"  # was "rule-b" pre-fallback
    assert row["candidates_considered"] == 0
    assert row["candidate_signal"] == ""


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
    assert row["reason"] == "fallback-direct"  # was "no-llm" pre-fallback
    assert row["candidates_considered"] == 1
    assert row["candidate_signal"] == ""


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
    # was "whispered" pre-fallback; `decision` still correctly says "whisper"
    assert row["reason"] == "fallback-direct"
    assert row["candidates_considered"] == 1
    assert row["vote_shape"] == '[["kb-00001"], ["kb-00001"], ["kb-00001"]]'
    # Sole candidate came from the detail-match retrieval's fallback leg (no
    # maps_projects/maps_rows seeded in this fixture -> the lexical path
    # never fires here).
    assert row["candidate_signal"] == "fallback"


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
    assert row["reason"] == "fallback-direct"  # was "vote-split" pre-fallback
    assert row["candidates_considered"] == 2
    assert row["vote_shape"] == '[["kb-00001"], ["kb-00001"], ["kb-00002"]]'
    assert row["candidate_signal"] == "fallback"


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
    assert row["reason"] == "fallback-direct"  # was "vote-split" pre-fallback
    assert row["candidates_considered"] == 3
    assert row["vote_shape"] == '[["kb-00001"], ["kb-00002"], ["kb-00003"]]'
    assert row["candidate_signal"] == ""


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
    assert row["reason"] == "fallback-direct"  # was "vote-none" pre-fallback
    assert row["candidates_considered"] == 1
    assert row["vote_shape"] == "[[], [], []]"
    assert row["candidate_signal"] == ""


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


def test_decision_write_failure_with_candidates_and_llm_path(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
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
    resp = client.post(
        "/api/kb/listener", json={"text": "hello", "session_id": "sess-12"}
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "pointer": {"id": "kb-00001", "short_title": "Home Network Map"},
        "pointers": [{"id": "kb-00001", "short_title": "Home Network Map"}],
        "reason": "matched kb-00001 (unanimous 3/3)",
    }


# ─── (d) primary (non-fallback) path keeps the granular reason ─────────────


@pytest.mark.usefixtures("_patch_decision_pool")
def test_decision_primary_path_rule_a_keeps_granular_reason(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
    decision_pool: _RecordingDecisionPool,
) -> None:
    """When detail-matching itself resolves a candidate (no fallback), the
    stored reason stays the granular per-branch value — "fallback-direct" is
    reserved for decisions reached via the fallback path.
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
    assert row["candidates_considered"] == 0
    assert row["candidate_signal"] == ""


# ─── endpoint mounted (sanity, mirrors test_listener_routes.py) ─────────────


def test_listener_endpoint_still_mounted() -> None:
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/api/kb/listener" in paths
