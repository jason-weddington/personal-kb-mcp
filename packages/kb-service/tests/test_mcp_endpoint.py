"""The /mcp endpoint: mounting, lifespan, auth, methods, role gating (B14)."""

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kb_service.mcp_server.backend as backend_module
from kb_service.main import app
from kb_service.mcp_server.observability import MCP_ENDPOINT_MARKER
from tests.conftest import FakeKnowledgeBase, install_mcp_fakes

MCP_HEADERS = {"Accept": "application/json, text/event-stream"}
USER = {**MCP_HEADERS, "Authorization": "Bearer kb_test_user"}
ADMIN = {**MCP_HEADERS, "Authorization": "Bearer kb_test_admin"}


def rpc(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        body["params"] = params
    return body


def list_names(client: TestClient, headers: dict[str, str]) -> list[str]:
    resp = client.post("/mcp", json=rpc("tools/list"), headers=headers)
    assert resp.status_code == 200, resp.text
    return [t["name"] for t in resp.json()["result"]["tools"]]


# ─── (a) lifespan re-entry ───────────────────────────────────────────────────


def test_two_sequential_lifespans_each_serve_tools_list(
    monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    install_mcp_fakes(monkeypatch, fake_kb)
    for _ in range(2):
        with TestClient(app) as client:
            assert "kb_store" in list_names(client, USER)


# ─── (b) mounting ahead of the SPA ───────────────────────────────────────────


def test_post_exact_mcp_path_is_json_not_redirect(mcp_client: TestClient) -> None:
    assert any(getattr(r, "path", None) == "/{full_path:path}" for r in app.routes), (
        "SPA catch-all expected to be mounted (packaged static UI)"
    )
    resp = mcp_client.post(
        "/mcp", json=rpc("tools/list"), headers=USER, follow_redirects=False
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")


def test_mcp_route_precedes_spa_catch_all() -> None:
    paths = [getattr(r, "path", None) for r in app.routes]
    assert paths.index("/mcp") < paths.index("/{full_path:path}")


# ─── (c)-(e) auth ────────────────────────────────────────────────────────────


def test_missing_authorization_is_401_with_challenge(mcp_client: TestClient) -> None:
    resp = mcp_client.post("/mcp", json=rpc("tools/list"), headers=MCP_HEADERS)
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Not authenticated"}
    assert resp.headers["www-authenticate"] == "Bearer"


def test_unknown_key_is_401_invalid_api_key(mcp_client: TestClient) -> None:
    resp = mcp_client.post(
        "/mcp",
        json=rpc("tools/list"),
        headers={**MCP_HEADERS, "Authorization": "Bearer kb_unknown"},
    )
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Invalid API key"}
    assert resp.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("value", ["Basic abc", "Bearer "])
def test_non_bearer_or_empty_bearer_is_not_authenticated(
    mcp_client: TestClient, value: str
) -> None:
    resp = mcp_client.post(
        "/mcp",
        json=rpc("tools/list"),
        headers={**MCP_HEADERS, "Authorization": value},
    )
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Not authenticated"}


def test_jwt_bearer_is_accepted(mcp_client: TestClient) -> None:
    from kb_service.auth import create_token
    from tests.conftest import fake_user

    token = create_token(fake_user().id)
    headers = {**MCP_HEADERS, "Authorization": f"Bearer {token}"}
    assert "kb_store" in list_names(mcp_client, headers)


# ─── (f) no-auth mode ────────────────────────────────────────────────────────


def test_no_auth_mode_needs_no_header(
    mcp_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    assert "kb_store" in list_names(mcp_client, MCP_HEADERS)


# ─── (g) GET / DELETE ────────────────────────────────────────────────────────


def test_authenticated_get_is_405_not_html(mcp_client: TestClient) -> None:
    resp = mcp_client.get(
        "/mcp",
        headers={"Accept": "text/event-stream", "Authorization": "Bearer kb_test_user"},
    )
    assert resp.status_code == 405
    assert not resp.headers.get("content-type", "").startswith("text/html")


def test_unauthenticated_get_is_401(mcp_client: TestClient) -> None:
    resp = mcp_client.get("/mcp", headers={"Accept": "text/event-stream"})
    assert resp.status_code == 401


def test_authenticated_delete_is_405(mcp_client: TestClient) -> None:
    resp = mcp_client.delete("/mcp", headers=USER)
    assert resp.status_code == 405


# ─── (h) no OAuth discovery ──────────────────────────────────────────────────


def test_well_known_oauth_resource_is_404(mcp_client: TestClient) -> None:
    resp = mcp_client.get("/.well-known/oauth-protected-resource")
    assert resp.status_code == 404


# ─── (i) not started ─────────────────────────────────────────────────────────


def test_post_without_lifespan_is_503(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setattr(app.state, "mcp_http_app", None, raising=False)
    caplog.set_level(logging.WARNING, logger="kb_service")
    client = TestClient(app)
    resp = client.post("/mcp", json=rpc("tools/list"), headers=MCP_HEADERS)
    assert resp.status_code == 503
    assert resp.json() == {"detail": "MCP endpoint not started"}
    assert any(
        r.levelno == logging.WARNING
        and r.getMessage() == f"{MCP_ENDPOINT_MARKER} not_started method=POST"
        for r in caplog.records
    )


# ─── (j) principal reaches the backend ───────────────────────────────────────


def test_api_key_principal_reaches_in_process_backend(
    mcp_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real_init = backend_module.InProcessBackend.__init__

    def _recording_init(self: Any, app: Any, principal: Any, raw_headers: Any) -> None:
        seen.append(principal)
        real_init(self, app=app, principal=principal, raw_headers=raw_headers)

    monkeypatch.setattr(backend_module.InProcessBackend, "__init__", _recording_init)
    resp = mcp_client.post(
        "/mcp",
        json=rpc("tools/call", {"name": "kb_list_projects", "arguments": {}}),
        headers=USER,
    )
    assert resp.status_code == 200, resp.text
    assert len(seen) == 1
    assert seen[0].api_key_id == "key-user"
    assert seen[0].auth_method == "api_key"


# ─── (k) role / manager gating ───────────────────────────────────────────────


def test_instance_role_and_manager_gating(
    monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    install_mcp_fakes(monkeypatch, fake_kb)
    monkeypatch.setenv("KB_INSTANCE_ROLE", "team")
    with TestClient(app) as client:
        names = list_names(client, USER)
    assert "team_kb_store" in names
    assert "team_kb_maintain" in names
    assert not any(n.endswith("kb_ingest") for n in names)

    monkeypatch.delenv("KB_INSTANCE_ROLE")
    monkeypatch.delenv("KB_MANAGER")
    with TestClient(app) as client:
        names = list_names(client, USER)
    assert "kb_store" in names
    assert "kb_maintain" not in names
    assert "kb_bulk_update" not in names
    assert "kb_ingest" not in names
    assert "kb_ingest_url" in names


def test_contributor_gating(
    monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    install_mcp_fakes(monkeypatch, fake_kb)
    monkeypatch.delenv("KB_CONTRIBUTOR")
    with TestClient(app) as client:
        names = list_names(client, USER)
    assert "kb_list_projects" in names
    assert "kb_list_contributors" not in names
    assert "kb_list_teams" not in names


# ─── (B17b) protocol-version contract ────────────────────────────────────────

# mcp 1.26 negotiates from mcp.shared.version.SUPPORTED_PROTOCOL_VERSIONS
# (["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]): a supported
# requested version is echoed back. talos offers 2025-11-25.


@pytest.mark.parametrize("version", ["2025-11-25", "2025-03-26"])
def test_initialize_negotiates_requested_protocol_version(
    mcp_client: TestClient, version: str
) -> None:
    resp = mcp_client.post(
        "/mcp",
        json=rpc(
            "initialize",
            {
                "protocolVersion": version,
                "capabilities": {},
                "clientInfo": {"name": "contract-test", "version": "0"},
            },
        ),
        headers=USER,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"]["protocolVersion"] == version
