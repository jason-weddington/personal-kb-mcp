"""Hermetic tests for POST /api/kb/listener (listener gate).

Test matrix (cases a-k):
  (a) 401 without Bearer
  (b) 422 on empty text
  (c) kill switch -> {pointer: null}; asserts zero search and zero LLM calls
  (d) rule A drops the cwd-project candidate
  (e) rule B drops the operated candidate
  (f) unanimous 3-vote pick -> pointer; asserts search shape + prompt content
  (g) split vote (2-1) -> null
  (h) any None vote -> null
  (i) non-candidate id vote -> null
  (j) zero candidates after filters -> null; asserts search once, zero LLM calls
  (k) synthesis_llm is None with non-empty candidates -> null; zero LLM calls

Candidates are seeded via ``fake_kb.results``; the LLM is injected via
``fake_kb.synthesis_llm = fake_llm``.  All tests are hermetic — no live
Postgres, Ollama, or network.
"""

import pytest
from fastapi.testclient import TestClient
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.models.search import SearchResult

from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, FakeLLM, fake_user

# ─── helpers ─────────────────────────────────────────────────────────────────


def _make_map_result(
    entry_id: str,
    *,
    short_title: str = "Test Map",
    knowledge_details: str = "Details about this map.",
    project_ref: str | None = None,
    hints: dict | None = None,
) -> SearchResult:
    """Build a SearchResult with EntryType.MENTAL_MAP for seeding fake_kb."""
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


# ─── (a) 401 without Bearer ──────────────────────────────────────────────────


def test_listener_requires_auth(client: TestClient) -> None:
    """POST /api/kb/listener without a Bearer token returns 401."""
    resp = client.post("/api/kb/listener", json={"text": "hello"})
    assert resp.status_code == 401


# ─── (b) 422 on empty text ───────────────────────────────────────────────────


def test_listener_empty_text_422(client: TestClient) -> None:
    """POST /api/kb/listener with text='' returns 422 (min_length=1)."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/listener", json={"text": ""})
    assert resp.status_code == 422


# ─── (c) kill switch ─────────────────────────────────────────────────────────


def test_listener_kill_switch(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KB_LISTENER_ENABLED=FALSE -> {pointer: null} with zero search + LLM calls.

    Seeded with both non-empty results AND a fake LLM so that a regression
    cannot pass by tripping the None short-circuit instead of the kill switch.
    """
    fake_llm = FakeLLM()
    fake_kb.results = [_make_map_result("kb-00001", short_title="Home Network Map")]
    fake_kb.filtered_count = 1
    fake_kb.synthesis_llm = fake_llm

    monkeypatch.setenv("KB_LISTENER_ENABLED", "FALSE")
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/listener", json={"text": "which machine runs traefik"})

    assert resp.status_code == 200
    assert resp.json() == {"pointer": None}
    assert len(fake_kb.search_calls) == 0
    assert len(fake_llm.generate_calls) == 0


# ─── (d) rule A: cross-project filter ───────────────────────────────────────


def test_listener_rule_a_drops_cwd_project(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule A: candidate with project_ref == cwd_project is dropped."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [
        _make_map_result("kb-00001", project_ref="my-project"),
    ]
    fake_kb.filtered_count = 1
    # synthesis_llm stays None to guarantee short-circuit after rule A

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "cwd_project": "my-project"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"pointer": None}
    # Search ran once; rule A filtered the only candidate; LLM not reached
    assert len(fake_kb.search_calls) == 1


def test_listener_rule_a_keeps_other_project(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule A does NOT drop a candidate whose project_ref differs from cwd_project."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_kb.results = [
        _make_map_result(
            "kb-00001", short_title="Other Map", project_ref="other-project"
        ),
    ]
    fake_kb.filtered_count = 1
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "cwd_project": "my-project"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"pointer": {"id": "kb-00001", "short_title": "Other Map"}}


# ─── (e) rule B: operating context filter ───────────────────────────────────


def test_listener_rule_b_drops_operated_candidate(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule B: candidate with operated_via hint in body.operating is dropped."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [
        _make_map_result(
            "kb-00001",
            hints={"operated_via": "mcp:agent-gtd"},
        ),
    ]
    fake_kb.filtered_count = 1
    # synthesis_llm stays None; rule B empties candidates first

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "operating": ["mcp:agent-gtd"]},
    )

    assert resp.status_code == 200
    assert resp.json() == {"pointer": None}
    assert len(fake_kb.search_calls) == 1


def test_listener_rule_b_keeps_unmatched_hint(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule B does NOT drop a candidate whose operated_via is not in operating."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_kb.results = [
        _make_map_result(
            "kb-00001",
            short_title="Personal KB Map",
            hints={"operated_via": "mcp:personal-kb"},
        ),
    ]
    fake_kb.filtered_count = 1
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "operating": ["mcp:agent-gtd"]},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "pointer": {"id": "kb-00001", "short_title": "Personal KB Map"}
    }


def test_listener_rule_b_no_hint_never_dropped(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule B: maps with no operated_via hint are never dropped."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_kb.results = [
        _make_map_result("kb-00001", short_title="No Hint Map", hints={})
    ]
    fake_kb.filtered_count = 1
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "operating": ["mcp:agent-gtd", "mcp:personal-kb"]},
    )

    assert resp.status_code == 200
    assert resp.json() == {"pointer": {"id": "kb-00001", "short_title": "No Hint Map"}}


# ─── (f) unanimous 3-vote pick ───────────────────────────────────────────────


def test_listener_unanimous_pick_with_assertions(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unanimous 3-vote pick returns the pointer; asserts search shape + prompt.

    Uses two candidates:
    - kb-00001: short details (survives rule A — different project_ref)
    - kb-00002: long details (> 400 chars) to verify truncation to 400

    All 3 votes return "kb-00001"; expects pointer = {id: kb-00001, ...}.
    """
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")

    long_details = "x" * 400 + "_TRUNCATED"  # 410 chars; only first 400 in prompt

    fake_kb.results = [
        _make_map_result(
            "kb-00001",
            short_title="Home Network Map",
            knowledge_details="Short details about the home network.",
            project_ref="other-project",
        ),
        _make_map_result(
            "kb-00002",
            short_title="Long Details Map",
            knowledge_details=long_details,
            project_ref="yet-another-project",
        ),
    ]
    fake_kb.filtered_count = 2

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    body = {
        "text": "which machine runs traefik",
        "cwd_project": "my-project",
        "source_label": "home",
    }
    resp = client.post("/api/kb/listener", json=body)

    assert resp.status_code == 200
    assert resp.json() == {
        "pointer": {"id": "kb-00001", "short_title": "Home Network Map"}
    }

    # ── search shape ──────────────────────────────────────────────────────────
    assert len(fake_kb.search_calls) == 1
    q, _ = fake_kb.search_calls[0]
    assert q.query == "which machine runs traefik"
    assert q.entry_type == EntryType.MENTAL_MAP
    assert q.limit == 5

    # ── LLM call count ────────────────────────────────────────────────────────
    assert len(fake_llm.generate_calls) == 3

    # ── prompt content ────────────────────────────────────────────────────────
    prompt, _ = fake_llm.generate_calls[0]

    assert prompt.startswith(
        "You are a strict relevance gate for a knowledge-surfacing system."
    )
    assert 'working in the project "home"' in prompt
    assert "### kb-00001: Home Network Map" in prompt
    assert "### kb-00002: Long Details Map" in prompt

    # knowledge_details truncated to 400 chars (long_details is 410 chars)
    assert "x" * 400 in prompt
    assert "_TRUNCATED" not in prompt

    assert prompt.endswith("When in doubt: NONE. No other text.")

    # all three calls receive the same prompt
    for call_prompt, _ in fake_llm.generate_calls:
        assert call_prompt == prompt


# ─── (g) split vote (2-1) -> null ────────────────────────────────────────────


def test_listener_split_vote_returns_null(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2-1 split vote (non-unanimous) produces {pointer: null}."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [
        _make_map_result("kb-00001", short_title="Map One"),
        _make_map_result("kb-00002", short_title="Map Two"),
    ]
    fake_kb.filtered_count = 2

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")  # vote 1
    fake_llm.enqueue("kb-00001")  # vote 2
    fake_llm.enqueue("kb-00002")  # vote 3 — splits
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/listener", json={"text": "hello"})

    assert resp.status_code == 200
    assert resp.json() == {"pointer": None}
    assert len(fake_llm.generate_calls) == 3


# ─── (h) any None vote -> null ───────────────────────────────────────────────


def test_listener_none_vote_returns_null(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any None vote (provider failure or NONE response) produces {pointer: null}."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [_make_map_result("kb-00001", short_title="Test Map")]
    fake_kb.filtered_count = 1

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue(None)  # one None vote
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/listener", json={"text": "hello"})

    assert resp.status_code == 200
    assert resp.json() == {"pointer": None}


def test_listener_none_text_vote_returns_null(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 'NONE' text response (parsed to None vote) produces {pointer: null}."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [_make_map_result("kb-00001", short_title="Test Map")]
    fake_kb.filtered_count = 1

    fake_llm = FakeLLM()
    fake_llm.enqueue("NONE")
    fake_llm.enqueue("NONE")
    fake_llm.enqueue("NONE")
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/listener", json={"text": "hello"})

    assert resp.status_code == 200
    assert resp.json() == {"pointer": None}


# ─── (i) non-candidate id vote -> null ──────────────────────────────────────


def test_listener_non_candidate_id_vote_returns_null(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vote naming a kb-id NOT in the candidate set yields {pointer: null}.

    Enforces the retrieve-and-cite invariant: the response pointer id can only
    be an id from the post-filter candidate set.
    """
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [_make_map_result("kb-00001", short_title="Real Candidate")]
    fake_kb.filtered_count = 1

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-99999")  # not in candidates
    fake_llm.enqueue("kb-99999")
    fake_llm.enqueue("kb-99999")
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/listener", json={"text": "hello"})

    assert resp.status_code == 200
    assert resp.json() == {"pointer": None}


# ─── (j) zero candidates after filters -> null ──────────────────────────────


def test_listener_zero_candidates_short_circuits(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero candidates after rules A+B: search called once, zero LLM calls.

    Seeded with a fake_llm to rule out the synthesis_llm-None short-circuit
    as the reason for null; the candidate is filtered away by rule A.
    """
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [
        _make_map_result("kb-00001", project_ref="my-project"),
    ]
    fake_kb.filtered_count = 1

    fake_llm = FakeLLM()
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "cwd_project": "my-project"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"pointer": None}
    assert len(fake_kb.search_calls) == 1
    assert len(fake_llm.generate_calls) == 0


# ─── (k) synthesis_llm is None -> null ──────────────────────────────────────


def test_listener_no_synthesis_llm_returns_null(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """synthesis_llm is None with non-empty candidates: search once, zero LLM calls."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [_make_map_result("kb-00001", short_title="Test Map")]
    fake_kb.filtered_count = 1
    fake_kb.synthesis_llm = None  # explicitly None

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/listener", json={"text": "hello"})

    assert resp.status_code == 200
    assert resp.json() == {"pointer": None}
    assert len(fake_kb.search_calls) == 1


# ─── source_label fallback ───────────────────────────────────────────────────


def test_listener_source_label_fallback_to_cwd_project(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When source_label is omitted, cwd_project is used as the prompt source."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [_make_map_result("kb-00001", project_ref="other")]
    fake_kb.filtered_count = 1

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "cwd_project": "grit-mile"},
    )

    assert resp.status_code == 200
    prompt, _ = fake_llm.generate_calls[0]
    assert 'working in the project "grit-mile"' in prompt


def test_listener_source_label_fallback_to_unknown(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both source_label and cwd_project are None, prompt source is 'unknown'."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [_make_map_result("kb-00001")]
    fake_kb.filtered_count = 1

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/listener", json={"text": "hello"})

    assert resp.status_code == 200
    prompt, _ = fake_llm.generate_calls[0]
    assert 'working in the project "unknown"' in prompt


# ─── endpoint is mounted ─────────────────────────────────────────────────────


def test_listener_endpoint_mounted() -> None:
    """POST /api/kb/listener is registered in the FastAPI app."""
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/api/kb/listener" in paths
