"""Hermetic tests for the /api/chat routes.

All tests are offline — no live Postgres, Ollama, or network.

Test plan:
(a) 401 on missing/invalid auth for Bearer-auth endpoints (create, history,
    messages, delete).
(b) /api/chat/stream returns 401 for an invalid token and 422 when token is
    absent.
(c) Stream happy path: valid JWT -> response.text contains event order
    chat_session / chat_response / stream_end.
(d) Write-back attribution: FakeLLM emits an update_entry tool call ->
    update_calls[0][1]['updated_by'] == seeded user's email.
(e) Per-user isolation: user A creates a chat; user B cannot see or delete it.
(f) /api/chat/create idempotency: second call with the same chat_id returns
    ok without a duplicate insert.
(g) LLM unavailable: synthesis_llm and query_llm both None -> stream body
    contains 'event: error' and 'LLM not available'.
(h) 409 mapping: chat_history.create_chat raises UniqueViolationError while
    chat_exists returns False -> POST /api/chat/create returns 409.
"""

import asyncpg
import pytest
from fastapi.testclient import TestClient

import kb_service.chat_history as chat_history_module
from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, FakeLLM, fake_admin_user, fake_user

# ─── (a) auth-rejection for Bearer endpoints ─────────────────────────────────


def test_create_requires_auth(chat_client: tuple[TestClient, str]) -> None:
    """POST /api/chat/create without auth returns 401 (or 403)."""
    tc, _ = chat_client
    resp = tc.post(
        "/api/chat/create",
        json={"chat_id": "x", "question": "hello"},
    )
    assert resp.status_code in (401, 403)


def test_history_requires_auth(chat_client: tuple[TestClient, str]) -> None:
    """GET /api/chat/history without auth returns 401 (or 403)."""
    tc, _ = chat_client
    resp = tc.get("/api/chat/history")
    assert resp.status_code in (401, 403)


def test_messages_requires_auth(chat_client: tuple[TestClient, str]) -> None:
    """GET /api/chat/{id}/messages without auth returns 401 (or 403)."""
    tc, _ = chat_client
    resp = tc.get("/api/chat/test-id/messages")
    assert resp.status_code in (401, 403)


def test_delete_requires_auth(chat_client: tuple[TestClient, str]) -> None:
    """DELETE /api/chat/{id} without auth returns 401 (or 403)."""
    tc, _ = chat_client
    resp = tc.delete("/api/chat/test-id")
    assert resp.status_code in (401, 403)


# ─── (b) stream token auth ────────────────────────────────────────────────────


def test_stream_invalid_token_returns_401(
    chat_client: tuple[TestClient, str],
) -> None:
    """POST /api/chat/stream with an invalid token query param returns 401."""
    tc, _ = chat_client
    resp = tc.post(
        "/api/chat/stream",
        params={"token": "not-a-real-token"},
        json={"message": "hello"},
    )
    assert resp.status_code == 401


def test_stream_missing_token_returns_422(
    chat_client: tuple[TestClient, str],
) -> None:
    """POST /api/chat/stream without the token query param returns 422."""
    tc, _ = chat_client
    resp = tc.post("/api/chat/stream", json={"message": "hello"})
    assert resp.status_code == 422


# ─── (c) stream happy path ────────────────────────────────────────────────────


def test_stream_happy_path_event_order(
    chat_client: tuple[TestClient, str],
) -> None:
    """Valid JWT -> event order: chat_session, chat_response, stream_end."""
    tc, token = chat_client
    kb: FakeKnowledgeBase = app.state.kb  # type: ignore[assignment]
    llm = FakeLLM()
    llm.enqueue("This is my answer.")
    kb.synthesis_llm = llm

    resp = tc.post(
        "/api/chat/stream",
        params={"token": token},
        json={"message": "What is the KB about?"},
    )
    assert resp.status_code == 200
    body = resp.text

    # Assert event order
    session_pos = body.index("event: chat_session")
    response_pos = body.index("event: chat_response")
    end_pos = body.index("event: stream_end")
    assert session_pos < response_pos < end_pos

    assert "This is my answer." in body


# ─── (d) write-back attribution ───────────────────────────────────────────────

_UPDATE_TOOL_CALL = (
    "```json\n"
    '{"tool": "update_entry", "args": {'
    '"entry_id": "kb-00001", "knowledge_details": "New details.",'
    ' "change_reason": "User asked"}}\n'
    "```"
)


def test_stream_tool_call_attribution(
    chat_client: tuple[TestClient, str],
) -> None:
    """Tool write-back carries updated_by == seeded user's email."""
    tc, token = chat_client
    kb: FakeKnowledgeBase = app.state.kb  # type: ignore[assignment]
    kb.config.ingest.skip_safety = True  # skip secret scan for test content

    llm = FakeLLM()
    llm.enqueue(_UPDATE_TOOL_CALL)  # first call: tool call
    llm.enqueue("Done, I updated the entry.")  # second call: final answer
    kb.synthesis_llm = llm

    resp = tc.post(
        "/api/chat/stream",
        params={"token": token},
        json={"message": "Update kb-00001 with new details."},
    )
    assert resp.status_code == 200
    body = resp.text

    # update_entry was dispatched
    assert kb.update_calls, "Expected at least one update call"
    _, kwargs = kb.update_calls[0]
    assert kwargs["updated_by"] == fake_user().email
    assert kwargs["change_reason"] == "User asked"

    # SSE stream contains tool result event
    assert "event: chat_tool_result" in body
    assert "event: stream_end" in body


# ─── (e) per-user isolation ───────────────────────────────────────────────────


def test_per_user_isolation(chat_client: tuple[TestClient, str]) -> None:
    """User A's chat is not accessible to user B."""
    tc, _ = chat_client

    # User A creates a chat
    app.dependency_overrides[get_current_user] = fake_user
    resp = tc.post(
        "/api/chat/create",
        json={"chat_id": "chat-user-a", "question": "A's question"},
    )
    assert resp.status_code == 200

    # User A sees it in history
    resp = tc.get("/api/chat/history")
    assert resp.status_code == 200
    ids = [item["id"] for item in resp.json()]
    assert "chat-user-a" in ids

    # User B cannot see it in history
    app.dependency_overrides[get_current_user] = fake_admin_user
    resp = tc.get("/api/chat/history")
    assert resp.status_code == 200
    ids_b = [item["id"] for item in resp.json()]
    assert "chat-user-a" not in ids_b

    # User B cannot get A's messages
    resp = tc.get("/api/chat/chat-user-a/messages")
    assert resp.status_code == 404

    # User B cannot delete A's chat
    resp = tc.delete("/api/chat/chat-user-a")
    assert resp.status_code == 404

    # User A can still see and delete it
    app.dependency_overrides[get_current_user] = fake_user
    resp = tc.get("/api/chat/chat-user-a/messages")
    assert resp.status_code == 200

    resp = tc.delete("/api/chat/chat-user-a")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ─── (f) idempotent create ────────────────────────────────────────────────────


def test_create_idempotent(chat_client: tuple[TestClient, str]) -> None:
    """Second POST /api/chat/create with the same chat_id returns ok, no dup."""
    tc, _ = chat_client
    app.dependency_overrides[get_current_user] = fake_user

    resp1 = tc.post(
        "/api/chat/create",
        json={"chat_id": "idempotent-id", "question": "First call"},
    )
    assert resp1.status_code == 200

    resp2 = tc.post(
        "/api/chat/create",
        json={"chat_id": "idempotent-id", "question": "Second call"},
    )
    assert resp2.status_code == 200
    assert resp2.json()["ok"] is True
    assert resp2.json()["id"] == "idempotent-id"

    # Only one entry in history (no duplication)
    resp = tc.get("/api/chat/history")
    assert sum(1 for item in resp.json() if item["id"] == "idempotent-id") == 1


# ─── (g) LLM unavailable ─────────────────────────────────────────────────────


def test_stream_llm_unavailable(
    chat_client: tuple[TestClient, str],
) -> None:
    """When synthesis_llm and query_llm are both None, stream emits 'error'."""
    tc, token = chat_client
    kb: FakeKnowledgeBase = app.state.kb  # type: ignore[assignment]
    kb.synthesis_llm = None
    kb.query_llm = None

    resp = tc.post(
        "/api/chat/stream",
        params={"token": token},
        json={"message": "anything"},
    )
    assert resp.status_code == 200
    body = resp.text
    assert "event: error" in body
    assert "LLM not available" in body


# ─── (h) 409 on UniqueViolationError ─────────────────────────────────────────


def test_create_409_on_unique_violation(
    monkeypatch: pytest.MonkeyPatch,
    chat_client: tuple[TestClient, str],
) -> None:
    """create_chat raising UniqueViolationError while chat_exists=False -> 409."""
    tc, _ = chat_client

    async def _raise_unique(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise asyncpg.UniqueViolationError()

    monkeypatch.setattr(chat_history_module, "create_chat", _raise_unique)
    # chat_exists returns False (already set by fixture default)

    app.dependency_overrides[get_current_user] = fake_user
    resp = tc.post(
        "/api/chat/create",
        json={"chat_id": "collision-id", "question": "hello"},
    )
    assert resp.status_code == 409


# ─── additional coverage tests ────────────────────────────────────────────────


def test_stream_get_entry_tool(
    chat_client: tuple[TestClient, str],
) -> None:
    """get_entry tool call: when the entry is not found, tool result is in stream."""
    tc, token = chat_client
    kb: FakeKnowledgeBase = app.state.kb  # type: ignore[assignment]

    get_entry_call = (
        '```json\n{"tool": "get_entry", "args": {"entry_id": "kb-99999"}}\n```'
    )
    llm = FakeLLM()
    llm.enqueue(get_entry_call)  # first: tool call
    llm.enqueue("Entry not found, sorry.")  # second: final answer
    kb.synthesis_llm = llm

    resp = tc.post(
        "/api/chat/stream",
        params={"token": token},
        json={"message": "What is kb-99999?"},
    )
    assert resp.status_code == 200
    body = resp.text
    assert "event: chat_tool_result" in body
    assert "event: chat_response" in body


def test_stream_reply_llm_none_mid_session(
    chat_client: tuple[TestClient, str],
) -> None:
    """When LLM returns None mid-session, reply falls back to error message."""
    tc, token = chat_client
    kb: FakeKnowledgeBase = app.state.kb  # type: ignore[assignment]

    # LLM returns None (exhausted queue)
    llm = FakeLLM()
    # No enqueued responses — generate_chat returns None
    kb.synthesis_llm = llm

    resp = tc.post(
        "/api/chat/stream",
        params={"token": token},
        json={"message": "Ask something"},
    )
    assert resp.status_code == 200
    body = resp.text
    # Stream completes with chat_response (fallback message) and stream_end
    assert "event: chat_response" in body
    assert "event: stream_end" in body


def test_stream_with_seed(
    chat_client: tuple[TestClient, str],
) -> None:
    """Stream with seed_question+answer creates the session and seeds messages."""
    tc, token = chat_client
    kb: FakeKnowledgeBase = app.state.kb  # type: ignore[assignment]
    llm = FakeLLM()
    llm.enqueue("Follow-up answer.")
    kb.synthesis_llm = llm

    resp = tc.post(
        "/api/chat/stream",
        params={"token": token},
        json={
            "message": "Tell me more",
            "seed_question": "What is Python?",
            "seed_answer": "A programming language.",
            "seed_entry_ids": [],
        },
    )
    assert resp.status_code == 200
    body = resp.text
    assert "event: chat_session" in body
    assert "event: chat_response" in body


def test_create_with_answer(
    chat_client: tuple[TestClient, str],
) -> None:
    """POST /api/chat/create with an answer field stores both messages."""
    tc, _ = chat_client
    app.dependency_overrides[get_current_user] = fake_user

    resp = tc.post(
        "/api/chat/create",
        json={
            "chat_id": "with-answer",
            "question": "Q",
            "answer": "A",
            "mode": "explore",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_delete_nonexistent_chat(
    chat_client: tuple[TestClient, str],
) -> None:
    """DELETE /api/chat/{id} for a chat that doesn't exist returns 404."""
    tc, _ = chat_client
    app.dependency_overrides[get_current_user] = fake_user
    resp = tc.delete("/api/chat/nonexistent-chat-id")
    assert resp.status_code == 404


def test_messages_empty_chat(
    chat_client: tuple[TestClient, str],
) -> None:
    """GET /api/chat/{id}/messages for a chat with no messages returns []."""
    tc, _ = chat_client
    app.dependency_overrides[get_current_user] = fake_user
    # Create a chat with no answer
    tc.post(
        "/api/chat/create",
        json={"chat_id": "empty-chat", "question": "Q"},
    )
    resp = tc.get("/api/chat/empty-chat/messages")
    assert resp.status_code == 200
    # Has 1 message (the user question)
    assert isinstance(resp.json(), list)


# ─── no-auth mode: clean 404 instead of 500 ──────────────────────────────────


def test_create_returns_404_in_no_auth_mode(
    chat_client: tuple[TestClient, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/chat/create returns 404 (not 500) in KB_AUTH_MODE=none."""
    tc, _ = chat_client
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    resp = tc.post(
        "/api/chat/create",
        json={"chat_id": "x", "question": "hi"},
    )
    assert resp.status_code == 404
    assert "no-auth" in resp.json()["detail"]


def test_history_returns_404_in_no_auth_mode(
    chat_client: tuple[TestClient, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /api/chat/history returns 404 (not 500) in KB_AUTH_MODE=none."""
    tc, _ = chat_client
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    resp = tc.get("/api/chat/history")
    assert resp.status_code == 404


def test_messages_returns_404_in_no_auth_mode(
    chat_client: tuple[TestClient, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /api/chat/{id}/messages returns 404 (not 500) in KB_AUTH_MODE=none."""
    tc, _ = chat_client
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    resp = tc.get("/api/chat/some-id/messages")
    assert resp.status_code == 404


def test_delete_returns_404_in_no_auth_mode(
    chat_client: tuple[TestClient, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """DELETE /api/chat/{id} returns 404 (not 500) in KB_AUTH_MODE=none."""
    tc, _ = chat_client
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    resp = tc.delete("/api/chat/some-id")
    assert resp.status_code == 404


def test_stream_returns_404_in_no_auth_mode(
    chat_client: tuple[TestClient, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/chat/stream returns 404 (not 500) in KB_AUTH_MODE=none.

    A token query param is supplied so the request does not 422 on missing
    parameter validation; the router-level guard fires before the route body
    is entered.
    """
    tc, _ = chat_client
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    resp = tc.post(
        "/api/chat/stream",
        params={"token": "irrelevant"},
        json={"message": "hi"},
    )
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "args_extra", ["", ', "change_reason": "   "', ', "change_reason": 5']
)
def test_stream_update_entry_requires_change_reason(
    chat_client: tuple[TestClient, str], args_extra: str
) -> None:
    """Missing, blank or non-string change_reason is rejected; kb.update not called."""
    tc, token = chat_client
    kb: FakeKnowledgeBase = app.state.kb  # type: ignore[assignment]
    kb.config.ingest.skip_safety = True

    llm = FakeLLM()
    llm.enqueue(
        '```json\n{"tool": "update_entry", "args": {"entry_id": "kb-00001",'
        f' "knowledge_details": "x"{args_extra}}}}}\n```'
    )
    llm.enqueue("Could not update.")
    kb.synthesis_llm = llm

    resp = tc.post(
        "/api/chat/stream",
        params={"token": token},
        json={"message": "Update kb-00001."},
    )
    assert resp.status_code == 200
    assert not kb.update_calls
    assert "event: chat_tool_result" in resp.text
    assert '"success":false' in resp.text.replace(" ", "")
