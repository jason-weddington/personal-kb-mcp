"""mcp-call / mcp-auth / mcp-endpoint log records (B14d)."""

import logging
from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from kb_service.main import app
from kb_service.mcp_server.observability import (
    MCP_AUTH_MARKER,
    MCP_CALL_MARKER,
    MCP_ENDPOINT_MARKER,
    record_backend_status,
)
from kb_service.routes import kb_read_routes
from tests.conftest import FakeKnowledgeBase, install_mcp_fakes
from tests.test_mcp_endpoint import MCP_HEADERS, USER, rpc
from tests.test_mcp_tools import STORE_ARGS, call


@pytest.fixture
def logged_client(
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
    caplog: pytest.LogCaptureFixture,
) -> Iterator[TestClient]:
    """``mcp_client`` equivalent with INFO capture on before the lifespan."""
    caplog.set_level(logging.INFO, logger="kb_service")
    install_mcp_fakes(monkeypatch, fake_kb)
    with TestClient(app) as client:
        caplog.clear()
        yield client


def _messages(caplog: pytest.LogCaptureFixture, marker: str) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith(marker + " ")
    ]


def test_successful_store_logs_one_ok_call(
    logged_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    out = call(logged_client, "kb_store", STORE_ARGS, headers=USER)
    assert out["content"][0]["text"].startswith("Created")
    records = _messages(caplog, MCP_CALL_MARKER)
    assert len(records) == 1
    msg = records[0]
    assert "tool=kb_store outcome=ok worst_status=200 " in msg
    assert "key_id=key-user auth=api_key" in msg
    assert "user_id=00000000-0000-0000-0000-000000000001" in msg
    assert "ua='testclient'" in msg


def test_admin_denial_logs_tool_error(
    logged_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    call(
        logged_client,
        "kb_maintain",
        {"action": "reactivate", "entry_id": "kb-00001"},
        headers=USER,
    )
    records = _messages(caplog, MCP_CALL_MARKER)
    assert len(records) == 1
    assert "outcome=tool_error worst_status=403" in records[0]


def test_uncaught_backend_error_logs_exception(
    logged_client: TestClient,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _down(**_kw: object) -> object:
        raise HTTPException(503, "down")

    monkeypatch.setattr(kb_read_routes, "get_entries", _down)
    result = call(logged_client, "kb_get", {"entry_id": "kb-00001"}, headers=USER)
    assert result["isError"] is True
    records = _messages(caplog, MCP_CALL_MARKER)
    assert len(records) == 1
    assert "tool=kb_get outcome=exception worst_status=503" in records[0]


def test_tool_without_backend_call_logs_worst_status_none(
    logged_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    call(logged_client, "kb_explore", headers=USER)
    records = _messages(caplog, MCP_CALL_MARKER)
    assert len(records) == 1
    assert "outcome=ok worst_status=none" in records[0]


def test_auth_denials_log_reason_without_secrets(
    logged_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    call(logged_client, "kb_list_projects", headers=USER)
    resp = logged_client.post("/mcp", json=rpc("tools/list"), headers=MCP_HEADERS)
    assert resp.status_code == 401
    resp = logged_client.post(
        "/mcp",
        json=rpc("tools/list"),
        headers={**MCP_HEADERS, "Authorization": "Bearer kb_unknown"},
    )
    assert resp.status_code == 401
    denials = _messages(caplog, MCP_AUTH_MARKER)
    assert len(denials) == 2
    assert "denied status=401 reason=missing key_fp=none" in denials[0]
    assert "denied status=401 reason=invalid_key key_fp=" in denials[1]
    assert "key_fp=none" not in denials[1]
    assert "kb_test_user" not in caplog.text
    assert "kb_unknown" not in caplog.text


def test_endpoint_started_record_lists_tools(
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="kb_service")
    install_mcp_fakes(monkeypatch, fake_kb)
    with TestClient(app):
        pass
    started = _messages(caplog, MCP_ENDPOINT_MARKER)
    assert len(started) == 1
    assert started[0].startswith(f"{MCP_ENDPOINT_MARKER} started path=/mcp prefix=kb_")
    names = started[0].split("names=", 1)[1].split(",")
    assert "kb_store" in names
    assert "kb_ingest" not in names


def test_record_backend_status_outside_a_call_is_a_no_op() -> None:
    record_backend_status(500)  # no enclosing tool call: must not raise
