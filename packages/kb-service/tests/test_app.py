"""Tests for the KB service shell: health, auth gate, and search serialization."""

import pytest
from fastapi.testclient import TestClient

from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, fake_user


def test_health(client: TestClient) -> None:
    """GET /api/health returns 200 with the ok status body."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_requires_auth(client: TestClient) -> None:
    """POST /api/kb/search with no Authorization header is rejected.

    FastAPI 0.136+ HTTPBearer(auto_error=True) returns 401 for missing
    credentials (older 0.115-era versions returned 403). Either way the
    endpoint is gated; we assert the gate fires with an unauthorized status.
    """
    resp = client.post("/api/kb/search", json={"query": "anything"})
    assert resp.status_code in (401, 403)


def test_search_bad_token(client: TestClient) -> None:
    """POST /api/kb/search with a malformed bearer token is rejected (401)."""
    resp = client.post(
        "/api/kb/search",
        json={"query": "anything"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert resp.status_code == 401


def test_search_empty_results(client: TestClient) -> None:
    """An authed search with an empty mock returns an empty, well-shaped body."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/search", json={"query": "anything"})
    assert resp.status_code == 200
    assert resp.json() == {"results": [], "filtered_count": 0}


@pytest.mark.parametrize("fake_kb", ["one_result"], indirect=True)
def test_search_one_result(client: TestClient) -> None:
    """An authed search returning one result serializes len==1, filtered_count==1."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/search", json={"query": "anything"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 1
    assert body["filtered_count"] == 1


def test_search_passes_contributor_telemetry(client: TestClient) -> None:
    """The user's email is forwarded as the telemetry contributor kwarg."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/search", json={"query": "x"})
    assert resp.status_code == 200
    kb: FakeKnowledgeBase = app.state.kb
    assert kb.search_calls
    _query, contributor = kb.search_calls[-1]
    assert contributor == "tester@example.com"


def test_routers_mounted() -> None:
    """The auth, admin, and kb routers are all mounted on the app."""
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/api/auth/login" in paths
    assert "/api/admin/invites" in paths
    assert "/api/kb/search" in paths
    assert "/api/health" in paths
