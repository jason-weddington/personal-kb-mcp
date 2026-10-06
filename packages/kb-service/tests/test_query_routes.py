"""Tests for POST /api/kb/ask and POST /api/kb/summarize endpoints."""

from fastapi.testclient import TestClient

from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, fake_user


def test_ask_requires_auth(client: TestClient) -> None:
    """POST /api/kb/ask without auth is rejected (401 or 403).

    Mirrors the hedge in test_app.py::test_search_requires_auth — FastAPI
    0.136.3 HTTPBearer returns 401, but the hedge guards against version drift
    (see kb-01745).
    """
    resp = client.post("/api/kb/ask", json={"question": "anything"})
    assert resp.status_code in (401, 403)


def test_ask_authed_happy_path(client: TestClient) -> None:
    """Authed ask returns 200 with the pinned fake literals."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/ask", json={"question": "what is X?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_turns_used"] == 3
    assert body["entries"][0]["entry"]["id"] == "kb-00001"
    assert body["entries"][0]["context"] == "fake ask context"


def test_ask_kwarg_forwarding(client: TestClient) -> None:
    """Ask forwards body fields and leaves agentic/max_tool_calls at None."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/ask",
        json={
            "question": "my question",
            "scope": "project:myproject",
            "limit": 15,
            "include_graph_context": False,
        },
    )
    assert resp.status_code == 200
    kb: FakeKnowledgeBase = app.state.kb
    question, kwargs = kb.ask_calls[-1]
    assert question == "my question"
    assert kwargs["scope"] == "project:myproject"
    assert kwargs["limit"] == 15
    assert kwargs["include_graph_context"] is False
    assert kwargs["agentic"] is None
    assert kwargs["max_tool_calls"] is None


def test_summarize_requires_auth(client: TestClient) -> None:
    """POST /api/kb/summarize without auth is rejected (401 or 403)."""
    resp = client.post("/api/kb/summarize", json={"question": "anything"})
    assert resp.status_code in (401, 403)


def test_summarize_authed_happy_path(client: TestClient) -> None:
    """Authed summarize returns 200 with the pinned fake answer."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/summarize", json={"question": "summarize X"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "fake synthesized answer"


def test_summarize_kwarg_forwarding(client: TestClient) -> None:
    """Summarize forwards question/scope/limit from body."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post(
        "/api/kb/summarize",
        json={"question": "my question", "scope": "tag:mytag", "limit": 30},
    )
    assert resp.status_code == 200
    kb: FakeKnowledgeBase = app.state.kb
    question, kwargs = kb.summarize_calls[-1]
    assert question == "my question"
    assert kwargs["scope"] == "tag:mytag"
    assert kwargs["limit"] == 30


def test_routers_mounted() -> None:
    """Both /api/kb/ask and /api/kb/summarize appear in the app route paths."""
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/api/kb/ask" in paths
    assert "/api/kb/summarize" in paths


def test_ask_limit_validation(client: TestClient) -> None:
    """Ask rejects limit=51 and limit=0 with 422 (Pydantic ge=1/le=50)."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/ask", json={"question": "x", "limit": 51})
    assert resp.status_code == 422
    resp = client.post("/api/kb/ask", json={"question": "x", "limit": 0})
    assert resp.status_code == 422
