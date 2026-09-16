"""Hermetic tests for GET /api/kb/embedding-queue (admin-only, GTD 735a7e1d)."""

import pytest
from fastapi.testclient import TestClient

from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, fake_admin_user, fake_user

# ---------------------------------------------------------------------------
# 401 / 403 gate
# ---------------------------------------------------------------------------


def test_embedding_queue_requires_auth_401(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No Authorization header -> 401, NOT 403 (kb-01745: HTTPBearer 0.136 semantics).

    ``auth._synthetic_user()`` is is_admin=True and short-circuits at
    auth.py:164 when KB_AUTH_MODE=none, which would make this assertion
    vacuous — delenv first (idiom copied from tests/test_app.py:116).
    """
    monkeypatch.delenv("KB_AUTH_MODE", raising=False)
    resp = client.get("/api/kb/embedding-queue")
    assert resp.status_code == 401


def test_embedding_queue_non_admin_403(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-admin authenticated user gets 403 from require_admin."""
    monkeypatch.delenv("KB_AUTH_MODE", raising=False)
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.get("/api/kb/embedding-queue")
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# 200 — admin, every canned stat echoed field-for-field
# ---------------------------------------------------------------------------


def test_embedding_queue_admin_200_echoes_every_stat(client: TestClient) -> None:
    app.dependency_overrides[get_current_user] = fake_admin_user
    resp = client.get("/api/kb/embedding-queue")
    assert resp.status_code == 200

    kb: FakeKnowledgeBase = app.state.kb
    # Mirrors the canned dict FakeKnowledgeBase.embedding_queue_stats() returns
    # (tests/conftest.py) — field-for-field parity, not a subset check.
    assert resp.json() == {
        "pending": 3,
        "exhausted": 1,
        "oldest_pending_age_seconds": 42.5,
        "next_due_at": "2026-09-16T12:00:00+00:00",
        "vectorless_unqueued": 0,
        "worker_enabled": True,
        "worker_running": kb.embedding_worker_running,
    }


def test_embedding_queue_worker_enabled_false_when_env_set(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """worker_enabled reflects KB_EMBED_WORKER_ENABLED=FALSE."""
    monkeypatch.setenv("KB_EMBED_WORKER_ENABLED", "FALSE")
    app.dependency_overrides[get_current_user] = fake_admin_user
    resp = client.get("/api/kb/embedding-queue")
    assert resp.status_code == 200
    assert resp.json()["worker_enabled"] is False
