"""Hermetic tests for POST /api/kb/listener (listener gate).

Test matrix (cases a-k):
  (a) 401 without Bearer
  (b) 422 on empty text
  (c) kill switch -> {pointer: null}; asserts zero search and zero LLM calls
  (d) rule A drops the cwd-project candidate
  (e) rule B drops the operated candidate
  (f) unanimous 3-vote pick -> pointer; asserts search shape + prompt content
  (g) 2/3 majority vote (GTD 66ea1fe4 reframe: majority, not unanimity, is
      now enough) -> whisper; a genuine 3-way split (no id reaches 2/3) -> null
  (h) a None/abstain vote alongside a 2/3 majority still whispers; all-None
      -> null
  (i) non-candidate id vote -> null
  (j) zero candidates after filters -> null; asserts search once, zero LLM calls
  (k) synthesis_llm is None with non-empty candidates -> null; zero LLM calls
  (l) plural pointers: two majority-voted candidates, both meeting the
      second-slot evidence bar -> both returned, in evidence order
  (m) earned second slot: a weakly-supported second candidate reaching
      majority is DROPPED (no independent evidence) -> only the strong one
      returned

Candidates in cases (d)-(k) are seeded via ``fake_kb.results`` using
``_make_map_result`` (``entry_type=MENTAL_MAP``). Since GTD bf40d4f1, the
PRIMARY retrieval stage searches the non-map corpus first and filters out
``entry_type=MENTAL_MAP`` hits client-side — so a ``mental_map``-only
``fake_kb.results`` always yields zero detail hits, which triggers the
FALLBACK direct-map search (mirroring the exact pre-fix retrieval). That
fallback search reuses ``fake_kb.results`` verbatim, so these cases still
exercise Rule A/B/vote exactly as before; the only change is that
``fake_kb.search_calls`` now has TWO entries (detail search, then fallback
map search) instead of one. The detail-match PRIMARY path itself (no
fallback) is covered by ``tests/test_listener_retrieval.py``, which drives
``_retrieve_candidate_maps`` directly.

The LLM is injected via ``fake_kb.synthesis_llm = fake_llm``.  All tests are
hermetic — no live Postgres, Ollama, or network.
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
    body = resp.json()
    assert body == {
        "pointer": None,
        "pointers": [],
        "reason": "kill-switch: KB_LISTENER_ENABLED!=TRUE",
    }
    # The reason key is present and a str (pinned per AC).
    assert "reason" in body
    assert isinstance(body["reason"], str)
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
    assert resp.json() == {
        "pointer": None,
        "pointers": [],
        "reason": "no-injection: 1 candidate(s), all dropped by rule-A (cwd-project)",
    }
    # Detail search finds zero non-map hits (the seeded result is a map) ->
    # falls back to a direct map search; rule A then filters the only
    # candidate; LLM not reached.
    assert len(fake_kb.search_calls) == 2


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
    assert resp.json() == {
        "pointer": {"id": "kb-00001", "short_title": "Other Map"},
        "pointers": [{"id": "kb-00001", "short_title": "Other Map"}],
        "reason": "matched kb-00001 (unanimous 3/3)",
    }


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
    rule_b_reason = (
        "no-injection: 1 candidate(s), all dropped by rule-B (operating-manifest)"
    )
    assert resp.json() == {"pointer": None, "pointers": [], "reason": rule_b_reason}
    # Falls back to the direct map search (see module docstring); two calls.
    assert len(fake_kb.search_calls) == 2


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
        "pointer": {"id": "kb-00001", "short_title": "Personal KB Map"},
        "pointers": [{"id": "kb-00001", "short_title": "Personal KB Map"}],
        "reason": "matched kb-00001 (unanimous 3/3)",
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
    assert resp.json() == {
        "pointer": {"id": "kb-00001", "short_title": "No Hint Map"},
        "pointers": [{"id": "kb-00001", "short_title": "No Hint Map"}],
        "reason": "matched kb-00001 (unanimous 3/3)",
    }


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
        "pointer": {"id": "kb-00001", "short_title": "Home Network Map"},
        "pointers": [{"id": "kb-00001", "short_title": "Home Network Map"}],
        "reason": "matched kb-00001 (unanimous 3/3)",
    }

    # ── search shape ──────────────────────────────────────────────────────────
    # fake_kb.results seeds MENTAL_MAP entries, so the primary detail search
    # (call 0) filters them all out and the route falls back (call 1) to the
    # direct map search — see module docstring.
    assert len(fake_kb.search_calls) == 2
    detail_q, _ = fake_kb.search_calls[0]
    assert detail_q.query == "which machine runs traefik"
    assert detail_q.entry_type is None
    assert detail_q.limit == 50

    fallback_q, _ = fake_kb.search_calls[1]
    assert fallback_q.query == "which machine runs traefik"
    assert fallback_q.entry_type == EntryType.MENTAL_MAP
    assert fallback_q.limit == 5

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

    assert prompt.endswith(
        "When in doubt about any one candidate: leave it out. No other text."
    )

    # all three calls receive the same prompt
    for call_prompt, _ in fake_llm.generate_calls:
        assert call_prompt == prompt


# ─── (g) 2/3 majority whispers; a genuine 3-way split does not ─────────────


def test_listener_majority_vote_whispers(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GTD 66ea1fe4 reframe: a 2/3 majority (no longer full unanimity) whispers."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [
        _make_map_result("kb-00001", short_title="Map One"),
        _make_map_result("kb-00002", short_title="Map Two"),
    ]
    fake_kb.filtered_count = 2

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")  # vote 1
    fake_llm.enqueue("kb-00001")  # vote 2
    fake_llm.enqueue(
        "kb-00002"
    )  # vote 3 — different candidate, still a 2/3 majority for kb-00001
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/listener", json={"text": "hello"})

    assert resp.status_code == 200
    assert resp.json() == {
        "pointer": {"id": "kb-00001", "short_title": "Map One"},
        "pointers": [{"id": "kb-00001", "short_title": "Map One"}],
        "reason": "matched kb-00001 (majority 2/3)",
    }
    assert len(fake_llm.generate_calls) == 3


def test_listener_no_majority_three_way_split_returns_null(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three distinct single-id votes: no id reaches the 2/3 majority bar."""
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
    resp = client.post("/api/kb/listener", json={"text": "hello"})

    assert resp.status_code == 200
    assert resp.json() == {
        "pointer": None,
        "pointers": [],
        "reason": "no-injection: no candidate reached majority (best: kb-00001 1/3)",
    }
    assert len(fake_llm.generate_calls) == 3


# ─── (h) a None/abstain vote alongside a majority still whispers ───────────


def test_listener_partial_abstain_still_reaches_majority(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One voter abstaining (None) doesn't block the other two from a majority."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [_make_map_result("kb-00001", short_title="Test Map")]
    fake_kb.filtered_count = 1

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue(None)  # one abstaining vote
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/listener", json={"text": "hello"})

    assert resp.status_code == 200
    assert resp.json() == {
        "pointer": {"id": "kb-00001", "short_title": "Test Map"},
        "pointers": [{"id": "kb-00001", "short_title": "Test Map"}],
        "reason": "matched kb-00001 (majority 2/3)",
    }


def test_listener_all_none_votes_returns_null(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three voters abstaining (None) produces {pointer: null}."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [_make_map_result("kb-00001", short_title="Test Map")]
    fake_kb.filtered_count = 1

    fake_llm = FakeLLM()
    fake_llm.enqueue(None)
    fake_llm.enqueue(None)
    fake_llm.enqueue(None)
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/listener", json={"text": "hello"})

    assert resp.status_code == 200
    assert resp.json() == {
        "pointer": None,
        "pointers": [],
        "reason": "no-injection: no candidate received a vote (0/3)",
    }


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
    assert resp.json() == {
        "pointer": None,
        "pointers": [],
        "reason": "no-injection: no candidate received a vote (0/3)",
    }


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
    assert resp.json() == {
        "pointer": None,
        "pointers": [],
        "reason": "no-injection: no candidate received a vote (0/3)",
    }


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
    assert resp.json() == {
        "pointer": None,
        "pointers": [],
        "reason": "no-injection: 1 candidate(s), all dropped by rule-A (cwd-project)",
    }
    assert len(fake_kb.search_calls) == 2
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
    assert resp.json() == {
        "pointer": None,
        "pointers": [],
        "reason": "no-injection: LLM unavailable",
    }
    assert len(fake_kb.search_calls) == 2


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


# ─── detail-match primary path (GTD bf40d4f1), end-to-end via HTTP ─────────
#
# Aggregation/dedupe/cap/ordering/fallback mechanics are unit-tested directly
# against `_retrieve_candidate_maps` in tests/test_listener_retrieval.py.
# These end-to-end cases prove the primary path is wired into the full route
# (Rule A/B, the vote, and the response shape) without going through the
# fallback branch.


def _make_detail_result(entry_id: str, **kwargs: object) -> SearchResult:
    """A non-map SearchResult (chunky detail entry) for seeding fake_kb.results."""
    defaults: dict[str, object] = {
        "id": entry_id,
        "short_title": f"Detail {entry_id}",
        "long_title": f"Detail {entry_id} (long)",
        "knowledge_details": "rsync camera-profiles-data raw-pairs a7r6 dispatch-host-a",
        "entry_type": EntryType.FACTUAL_REFERENCE,
    }
    defaults.update(kwargs)
    entry = KnowledgeEntry(**defaults)
    return SearchResult(
        entry=entry,
        score=1.0 / 61,
        effective_confidence=0.9,
        staleness_warning=None,
        match_source="fts",
    )


def test_listener_primary_path_unanimous_pick_end_to_end(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detail hit resolves to its owning map via the primary path (no fallback);
    Rule A/B and the unanimous vote apply to the resolved map exactly as today.
    """
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [_make_detail_result("kb-00099", project_ref="camera-profiles")]
    fake_kb.filtered_count = 0
    fake_kb.db.rows_for["graph_edges"] = [("kb-00099", "kb-00001")]
    fake_kb.entries["kb-00001"] = KnowledgeEntry(
        id="kb-00001",
        short_title="Camera Profiles Map",
        long_title="Camera Profiles Map (long title)",
        knowledge_details="Orientation map for camera-profiles.",
        entry_type=EntryType.MENTAL_MAP,
        project_ref="camera-profiles",
        is_active=True,
    )

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00001")
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={
            "text": "rsync the a7r6 raw-pairs to dispatch-host-a",
            "cwd_project": "grit-mile",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "pointer": {"id": "kb-00001", "short_title": "Camera Profiles Map"},
        "pointers": [{"id": "kb-00001", "short_title": "Camera Profiles Map"}],
        "reason": "matched kb-00001 (unanimous 3/3)",
    }
    # No fallback needed — exactly one search call (the detail search).
    assert len(fake_kb.search_calls) == 1
    detail_q, _ = fake_kb.search_calls[0]
    assert detail_q.entry_type is None
    assert detail_q.limit == 50


def test_listener_primary_path_rule_a_drops_resolved_map(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule A drops a primary-path-resolved candidate whose project_ref matches."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [_make_detail_result("kb-00099")]
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
        json={"text": "hello", "cwd_project": "my-project"},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "pointer": None,
        "pointers": [],
        "reason": "no-injection: 1 candidate(s), all dropped by rule-A (cwd-project)",
    }
    # Primary path resolved exactly one candidate -> no fallback search.
    assert len(fake_kb.search_calls) == 1


# ─── (l)/(m) plural pointers (GTD 66ea1fe4) ─────────────────────────────────


def test_listener_plural_pointers_both_earn_their_slot(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two majority-voted maps, both independently supported -> both returned,
    in evidence order (kb-00001 at best-rank 1, kb-00002 earning its slot via
    2 distinct detail hits)."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [
        _make_detail_result("kb-00099", project_ref="camera-profiles"),  # rank 1
        _make_detail_result("kb-00098", project_ref="grit-mile"),  # rank 2
        _make_detail_result("kb-00097", project_ref="grit-mile"),  # rank 3
    ]
    fake_kb.filtered_count = 0
    fake_kb.db.rows_for["graph_edges"] = [
        ("kb-00099", "kb-00001"),
        ("kb-00098", "kb-00002"),
        ("kb-00097", "kb-00002"),  # kb-00002's 2nd distinct detail hit
    ]
    fake_kb.entries["kb-00001"] = KnowledgeEntry(
        id="kb-00001",
        short_title="Camera Profiles Map",
        long_title="Camera Profiles Map (long)",
        knowledge_details="Orientation map for camera-profiles.",
        entry_type=EntryType.MENTAL_MAP,
        project_ref="camera-profiles",
        is_active=True,
    )
    fake_kb.entries["kb-00002"] = KnowledgeEntry(
        id="kb-00002",
        short_title="Grit Mile Map",
        long_title="Grit Mile Map (long)",
        knowledge_details="Orientation map for grit-mile.",
        entry_type=EntryType.MENTAL_MAP,
        project_ref="grit-mile",
        is_active=True,
    )

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001, kb-00002")
    fake_llm.enqueue("kb-00001, kb-00002")
    fake_llm.enqueue("kb-00001, kb-00002")
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "cwd_project": "other-project"},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "pointer": {"id": "kb-00001", "short_title": "Camera Profiles Map"},
        "pointers": [
            {"id": "kb-00001", "short_title": "Camera Profiles Map"},
            {"id": "kb-00002", "short_title": "Grit Mile Map"},
        ],
        "reason": "matched kb-00001 (unanimous 3/3), kb-00002 (unanimous 3/3)",
    }


def test_listener_weak_second_candidate_does_not_ride_along(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second majority-voted map with only ONE detail hit ranked past the
    top-3 does NOT earn a slot — proves the second slot is earned, not filled.
    """
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [
        _make_detail_result("kb-00099", project_ref="camera-profiles"),  # rank 1
        _make_detail_result("kb-00098", project_ref="noise-a"),  # rank 2, no edge
        _make_detail_result("kb-00097", project_ref="noise-b"),  # rank 3, no edge
        _make_detail_result("kb-00096", project_ref="grit-mile"),  # rank 4
    ]
    fake_kb.filtered_count = 0
    fake_kb.db.rows_for["graph_edges"] = [
        ("kb-00099", "kb-00001"),
        ("kb-00096", "kb-00002"),  # kb-00002's ONLY hit, at rank 4 (> top-3)
    ]
    fake_kb.entries["kb-00001"] = KnowledgeEntry(
        id="kb-00001",
        short_title="Camera Profiles Map",
        long_title="Camera Profiles Map (long)",
        knowledge_details="Orientation map for camera-profiles.",
        entry_type=EntryType.MENTAL_MAP,
        project_ref="camera-profiles",
        is_active=True,
    )
    fake_kb.entries["kb-00002"] = KnowledgeEntry(
        id="kb-00002",
        short_title="Grit Mile Map",
        long_title="Grit Mile Map (long)",
        knowledge_details="Orientation map for grit-mile.",
        entry_type=EntryType.MENTAL_MAP,
        project_ref="grit-mile",
        is_active=True,
    )

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001, kb-00002")
    fake_llm.enqueue("kb-00001, kb-00002")
    fake_llm.enqueue("kb-00001, kb-00002")
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/listener",
        json={"text": "hello", "cwd_project": "other-project"},
    )

    assert resp.status_code == 200
    # Both candidates reached a 3/3 majority, but kb-00002 has hit_count=1
    # and best_rank=4 — it clears neither second-slot evidence bar, so only
    # the strongly-evidenced kb-00001 is emitted.
    assert resp.json() == {
        "pointer": {"id": "kb-00001", "short_title": "Camera Profiles Map"},
        "pointers": [{"id": "kb-00001", "short_title": "Camera Profiles Map"}],
        "reason": "matched kb-00001 (unanimous 3/3)",
    }


def test_listener_caps_at_two_even_with_three_way_majority(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three candidates can each independently reach a 2/3 majority (voters
    overlapping pairwise: {A,B}, {B,C}, {C,A} — each id appears in exactly 2
    of the 3 sets). The hard MAX_POINTERS_PER_RESPONSE=2 cap still applies:
    only the top-2 by evidence rank are emitted, never three."""
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb.results = [
        _make_map_result("kb-00001", short_title="Map One"),
        _make_map_result("kb-00002", short_title="Map Two"),
        _make_map_result("kb-00003", short_title="Map Three"),
    ]
    fake_kb.filtered_count = 3

    fake_llm = FakeLLM()
    fake_llm.enqueue("kb-00001, kb-00002")
    fake_llm.enqueue("kb-00002, kb-00003")
    fake_llm.enqueue("kb-00003, kb-00001")
    fake_kb.synthesis_llm = fake_llm

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/listener", json={"text": "hello"})

    assert resp.status_code == 200
    assert resp.json() == {
        "pointer": {"id": "kb-00001", "short_title": "Map One"},
        "pointers": [
            {"id": "kb-00001", "short_title": "Map One"},
            {"id": "kb-00002", "short_title": "Map Two"},
        ],
        "reason": "matched kb-00001 (majority 2/3), kb-00002 (majority 2/3)",
    }


# ─── endpoint is mounted ─────────────────────────────────────────────────────


def test_listener_endpoint_mounted() -> None:
    """POST /api/kb/listener is registered in the FastAPI app."""
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/api/kb/listener" in paths
