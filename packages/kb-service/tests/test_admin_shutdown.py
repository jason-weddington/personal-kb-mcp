"""Tests for the health handshake fields and the loopback-only shutdown endpoint."""

import sys
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from kb_service.main import app
from kb_service.routes import admin_routes


def test_health_reports_version_and_install_id(client: TestClient) -> None:
    """Health keeps status=ok and adds version + install_id (sys.prefix)."""
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["install_id"] == sys.prefix
    assert body["version"]


@pytest.fixture
def shutdown_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the real SIGTERM with a recorder."""
    calls: list[int] = []
    monkeypatch.setattr(admin_routes, "_request_shutdown", lambda: calls.append(1))
    return calls


@pytest.fixture
def loopback_client(client: TestClient) -> Iterator[TestClient]:
    """A client whose requests appear to come from 127.0.0.1."""
    with TestClient(app, client=("127.0.0.1", 50000)) as c:
        yield c


def test_shutdown_404_when_auth_mode_not_none(
    monkeypatch: pytest.MonkeyPatch,
    loopback_client: TestClient,
    shutdown_calls: list[int],
) -> None:
    monkeypatch.setenv("KB_AUTH_MODE", "jwt")
    assert loopback_client.post("/api/admin/shutdown").status_code == 404
    assert shutdown_calls == []


def test_shutdown_403_when_not_loopback(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, shutdown_calls: list[int]
) -> None:
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    with TestClient(app, client=("10.1.2.3", 50000)) as remote:
        assert remote.post("/api/admin/shutdown").status_code == 403
    assert shutdown_calls == []


def test_shutdown_accepted_on_loopback_no_auth(
    monkeypatch: pytest.MonkeyPatch,
    loopback_client: TestClient,
    shutdown_calls: list[int],
) -> None:
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setattr(
        admin_routes, "_request_shutdown", lambda: shutdown_calls.append(1)
    )
    resp = loopback_client.post("/api/admin/shutdown")
    assert resp.status_code == 202
