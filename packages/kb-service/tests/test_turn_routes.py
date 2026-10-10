"""POST /api/kb/turn + GET /api/kb/turn/heartbeat on the SQLite service DB."""

import json
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kb_service.database as database
from kb_service import turn_digest
from kb_service.main import app
from kb_service.routes import turn_routes

_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
)
_AWS_LINE = 'export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"'

ITEMS: list[dict[str, Any]] = [
    {"kind": "assistant_text", "text": "checking"},
    {
        "kind": "tool_call",
        "tool_use_id": "toolu_1",
        "tool": "Bash",
        "target": "ls /tmp",
        "target_class": "ls",
    },
    {
        "kind": "tool_result",
        "tool_use_id": "toolu_1",
        "is_error": False,
        "excerpt": "a\nb",
    },
]


@pytest.fixture
def local_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("KB_SURPRISE_CAPTURE", raising=False)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setattr(database, "_pool", None)
    monkeypatch.setattr(turn_routes, "_OUTCOMES", {})
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _rows(sql: str = "SELECT * FROM turn_events ORDER BY id") -> list[sqlite3.Row]:
    conn = sqlite3.connect(database.sqlite_service_db_path())
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _digest(session: str = "sess-1", turn: int = 0, **kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "event_id": f"{session}:{turn}",
        "session_id": session,
        "turn_index": turn,
        "host": "h1",
        "hook_version": "1.0.0",
        "project": "personal-kb",
        "ts": "2026-10-09T12:00:00+00:00",
        "user_prompt": "do it",
        "items": ITEMS,
        "final_message": "done",
    }
    body.update(kw)
    return body


def _post(client: TestClient, body: dict[str, Any]) -> Any:
    return client.post("/api/kb/turn", json=body)


def test_capture_off(local_client: TestClient) -> None:
    resp = _post(local_client, _digest())
    assert resp.status_code == 200
    assert resp.json() == {"recorded": False, "reason": "capture-off", "redactions": []}
    assert _rows("SELECT COUNT(*) FROM turn_events")[0][0] == 0


def test_shadow_records(
    local_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    caplog.set_level(logging.INFO)
    resp = _post(local_client, _digest())
    assert resp.json()["reason"] == "recorded"
    (row,) = _rows()
    assert row["capture_mode"] == "shadow"
    assert row["processed_at"] is None
    assert row["anomaly"] is None
    assert json.loads(row["items"]) == ITEMS
    for frag in (
        "reason=recorded",
        "capture_mode=shadow",
        "tool_calls=1",
        "tool_results=1",
        "bytes=",
        "harness=claude-code",
        "reasoning=0",
    ):
        assert frag in caplog.text


def test_on_mode(local_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "on")
    _post(local_client, _digest())
    assert _rows()[0]["capture_mode"] == "on"


def test_duplicate_and_mismatch(
    local_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    assert _post(local_client, _digest()).json()["reason"] == "recorded"
    assert _post(local_client, _digest()).json()["reason"] == "duplicate"
    assert len(_rows()) == 1
    other = _digest(items=ITEMS[:1])
    assert _post(local_client, other).json()["reason"] == "duplicate-mismatch"
    assert json.loads(_rows()[0]["items"]) == ITEMS
    assert "duplicate_mismatch" in caplog.text


def test_validation_failures(
    local_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    assert (
        _post(local_client, _digest(event_id="sess-1:5", turn_index=4)).status_code
        == 422
    )
    assert (
        _post(
            local_client, _digest(items=[{"kind": "thinking", "text": "x"}])
        ).status_code
        == 422
    )
    assert _post(local_client, _digest(items=[ITEMS[0]] * 201)).status_code == 422
    assert _post(local_client, _digest(user_prompt="a" * 4001)).status_code == 422
    big = [{**ITEMS[2], "excerpt": "e" * 1501}]
    assert _post(local_client, _digest(items=big)).status_code == 422
    assert _rows("SELECT COUNT(*) FROM turn_events")[0][0] == 0
    caplog.clear()
    prompt = 'password = "hunter2hunter2"'
    resp = _post(local_client, _digest(items=[ITEMS[0]] * 201, user_prompt=prompt))
    assert resp.status_code == 422
    assert "reason=invalid" in caplog.text
    assert "too_long" in caplog.text
    assert "hunter2hunter2" not in caplog.text
    hb = local_client.get("/api/kb/turn/heartbeat").json()
    assert hb["route_outcomes"]["invalid"] == 6


@pytest.mark.parametrize("env", ["shadow", None])
def test_too_large(
    local_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    env: str | None,
) -> None:
    if env:
        monkeypatch.setenv("KB_SURPRISE_CAPTURE", env)
    items = [{"kind": "assistant_text", "text": "t" * 2000}] * 40
    raw = json.dumps(_digest(items=items)).encode()
    resp = local_client.post(
        "/api/kb/turn", content=raw, headers={"content-type": "application/json"}
    )
    assert resp.status_code == 413
    assert _rows("SELECT COUNT(*) FROM turn_events")[0][0] == 0
    assert "reason=too-large" in caplog.text
    assert f"bytes={len(raw)}" in caplog.text


def test_redaction(local_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    items = [{**ITEMS[2], "excerpt": _AWS_LINE}]
    resp = _post(local_client, _digest(items=items))
    assert resp.json()["redactions"] == ["Secret Keyword", "AWS Access Key"]
    (row,) = _rows()
    assert "wJalrXUtnFEMI" not in row["items"]
    assert "[REDACTED:Secret Keyword]" in row["items"]
    assert row["redactions"] == '["Secret Keyword", "AWS Access Key"]'


def test_private_key_prompt(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    pem = (
        "-----BEGIN RSA PRIV"
        + "ATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIV"
        + "ATE KEY-----"
    )
    _post(local_client, _digest(user_prompt=pem))
    assert _rows()[0]["user_prompt"] == "[REDACTED:Private Key]"


def test_redaction_unavailable(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    monkeypatch.setattr(turn_digest, "redact_secrets", lambda _c: None)
    resp = _post(local_client, _digest())
    assert resp.json()["reason"] == "redaction-unavailable"
    assert _rows() == []


def test_write_failed(
    local_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    caplog.set_level(logging.WARNING)

    async def boom() -> Any:
        raise RuntimeError("db down")

    monkeypatch.setattr(turn_routes, "get_db", boom)
    resp = _post(local_client, _digest())
    assert resp.status_code == 200
    assert resp.json() == {
        "recorded": False,
        "reason": "write-failed",
        "redactions": [],
    }
    assert "turn_event write_failed" in caplog.text
    assert "harness=claude-code" in caplog.text


def test_invalid_mode_value_is_off(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "yes")
    assert _post(local_client, _digest()).json()["reason"] == "capture-off"


def test_requires_auth(client: TestClient) -> None:
    assert client.post("/api/kb/turn", json=_digest()).status_code == 401
    assert client.get("/api/kb/turn/heartbeat").status_code == 401


def test_heartbeat(local_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    for session, turn in (("g", 0), ("g", 2), ("h", 0)):
        assert (
            _post(local_client, _digest(session, turn)).json()["reason"] == "recorded"
        )
    hb = local_client.get("/api/kb/turn/heartbeat").json()
    assert hb["sessions_with_gaps"] == 1
    assert hb["pending"] == 3
    assert hb["oldest_pending_received_ts"]
    (row,) = hb["rows"]
    assert row["count"] == 3
    assert hb["route_outcomes"]["recorded"] == 3


def test_anomaly_stored_and_logged(
    local_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    items = [{**ITEMS[1], "target_class": ""}, ITEMS[2]]
    _post(local_client, _digest(items=items))
    assert _rows()[0]["anomaly"] == "empty_bash_target_class"
    assert "anomaly=empty_bash_target_class" in caplog.text


def test_ignores_listener_switches(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    monkeypatch.setenv("KB_LISTENER_ENABLED", "FALSE")
    monkeypatch.setenv("PERSONAL_KB_LISTENER", "0")
    assert _post(local_client, _digest()).json()["reason"] == "recorded"
    assert len(_rows()) == 1


def _reasoning(text: str, **kw: Any) -> dict[str, Any]:
    return {"kind": "reasoning", "text": text, "truncated": False, **kw}


def test_reasoning_item_recorded(
    local_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    caplog.set_level(logging.INFO)
    resp = _post(
        local_client,
        _digest(harness="talos", mode="headless", items=[*ITEMS, _reasoning("r")]),
    )
    assert resp.json()["reason"] == "recorded"
    (row,) = _rows()
    assert json.loads(row["items"])[3] == _reasoning("r")
    assert row["harness"] == "talos"
    assert "harness=talos" in caplog.text
    assert "reasoning=1" in caplog.text


def test_reasoning_text_over_cap_422(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    resp = _post(local_client, _digest(items=[_reasoning("r" * 2001)]))
    assert resp.status_code == 422
    assert _rows("SELECT COUNT(*) FROM turn_events")[0][0] == 0


def test_reasoning_missing_truncated_422(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    resp = _post(local_client, _digest(items=[{"kind": "reasoning", "text": "r"}]))
    assert resp.status_code == 422
    assert _rows("SELECT COUNT(*) FROM turn_events")[0][0] == 0


def test_reasoning_redacted(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    resp = _post(local_client, _digest(items=[_reasoning(_AWS_LINE)]))
    assert resp.json()["reason"] == "recorded"
    assert resp.json()["redactions"] == ["Secret Keyword", "AWS Access Key"]
    (row,) = _rows()
    assert "wJalrXUtnFEMI" not in row["items"]
    assert "[REDACTED:Secret Keyword]" in row["items"]


def test_too_large_reasoning(
    local_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    raw = json.dumps(
        _digest(harness="talos", mode="headless", items=[_reasoning("r" * 2000)] * 33)
    ).encode()
    resp = local_client.post(
        "/api/kb/turn", content=raw, headers={"content-type": "application/json"}
    )
    assert resp.status_code == 413
    assert _rows("SELECT COUNT(*) FROM turn_events")[0][0] == 0
    assert "reason=too-large" in caplog.text
    assert "harness=talos" in caplog.text
    assert "reasoning=33" in caplog.text
    assert f"bytes={len(raw)}" in caplog.text
