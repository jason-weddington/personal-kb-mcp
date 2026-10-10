"""Per-surface /mcp tool visibility: headless/autonomous see only the allow-list."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

import kb_service.auth as auth_module
from kb_service.main import app
from kb_service.mcp_server import context
from kb_service.mcp_server.surface_filter import NON_INTERACTIVE_TOOL_BASES
from tests.conftest import FakeKnowledgeBase, fake_admin_user, install_mcp_fakes
from tests.test_mcp_endpoint import MCP_HEADERS, rpc

ALLOWED = {f"kb_{b}" for b in NON_INTERACTIVE_TOOL_BASES}


def _client(monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase) -> TestClient:
    pool = install_mcp_fakes(monkeypatch, fake_kb)
    admin = fake_admin_user()
    for token, surface in (
        ("kb_auto", "autonomous"),
        ("kb_head", "headless"),
        ("kb_inter", "interactive"),
    ):
        pool._api_keys[auth_module.hash_api_key(token)] = {
            "id": f"key-{surface}",
            "user_id": admin.id,
            "surface": surface,
        }
    return TestClient(app)


def _headers(token: str, **extra: str) -> dict[str, str]:
    return {**MCP_HEADERS, "Authorization": f"Bearer {token}", **extra}


def _names(client: TestClient, headers: dict[str, str]) -> set[str]:
    resp = client.post("/mcp", json=rpc("tools/list"), headers=headers)
    assert resp.status_code == 200, resp.text
    return {t["name"] for t in resp.json()["result"]["tools"]}


def test_allow_list_is_exact() -> None:
    assert {
        "search",
        "get",
        "ask",
        "summarize",
        "preflight",
        "explore",
        "map_eligibility",
        "list_projects",
        "list_contributors",
        "list_teams",
        "store",
        "store_batch",
        "feedback",
    } == NON_INTERACTIVE_TOOL_BASES


def test_autonomous_key_sees_only_allow_list(
    monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    with _client(monkeypatch, fake_kb) as client:
        full = _names(client, _headers("kb_inter"))
        assert {"kb_ingest_url", "kb_maintain", "kb_bulk_update"} <= full
        assert {"kb_map_eligibility_override"} <= full
        assert _names(client, _headers("kb_auto")) == ALLOWED
        assert full > ALLOWED
    app.dependency_overrides.clear()


def test_header_downgrade_filters_interactive_key(
    monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    with _client(monkeypatch, fake_kb) as client:
        assert _names(client, _headers("kb_head")) == ALLOWED
        assert (
            _names(client, _headers("kb_inter", **{"X-KB-Mode": "headless"})) == ALLOWED
        )
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("kb_ingest_url", {"url": "https://example.com", "content": "x"}),
        ("kb_maintain", {"action": "reconcile_supersession"}),
    ],
)
def test_hidden_tool_call_errors_without_running(
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
    tool: str,
    arguments: dict[str, Any],
) -> None:
    built: list[Any] = []
    real = context.backend_for_request

    def _spy() -> Any:
        built.append(1)
        return real()

    monkeypatch.setattr(context, "backend_for_request", _spy)
    with _client(monkeypatch, fake_kb) as client:
        resp = client.post(
            "/mcp",
            json=rpc("tools/call", {"name": tool, "arguments": arguments}),
            headers=_headers("kb_auto"),
        )
        assert resp.status_code == 200, resp.text
        result = resp.json()["result"]
        assert result["isError"] is True
        assert result["content"][0]["text"] == (
            f"write policy: {tool} is not available from the autonomous surface;"
            " use an interactive session."
        )
    app.dependency_overrides.clear()
    assert built == []


def test_allowed_tool_runs_for_autonomous(
    monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    with _client(monkeypatch, fake_kb) as client:
        resp = client.post(
            "/mcp",
            json=rpc("tools/call", {"name": "kb_list_projects", "arguments": {}}),
            headers=_headers("kb_auto"),
        )
        assert resp.status_code == 200, resp.text
        assert not resp.json()["result"].get("isError")
    app.dependency_overrides.clear()


def test_missing_principal_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from kb_service.mcp_server import surface_filter

    class _State:
        pass

    class _Req:
        state = _State()

    monkeypatch.setattr(surface_filter, "get_http_request", lambda: _Req())
    assert asyncio.run(surface_filter._effective_surface()) == "headless"
