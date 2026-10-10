"""POST /api/kb/turn + GET /api/kb/turn/heartbeat on the SQLite service DB."""

import asyncio
import json
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kb_service.database as database
from kb_service import surprise, surprise_worker, turn_digest
from kb_service.db_sqlite import SqlitePool
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
    monkeypatch.setattr(turn_routes, "_UNMAPPED", {})
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


def _call_item(i: int, tool: str, target: str, cls: str = "") -> dict[str, Any]:
    return {
        "kind": "tool_call",
        "tool_use_id": f"toolu_{i}",
        "tool": tool,
        "target": target,
        "target_class": cls,
    }


def _result_item(i: int, is_error: bool = False, excerpt: str = "") -> dict[str, Any]:
    return {
        "kind": "tool_result",
        "tool_use_id": f"toolu_{i}",
        "is_error": is_error,
        "excerpt": excerpt,
    }


TALOS_ITEMS: list[dict[str, Any]] = [
    {"kind": "assistant_text", "text": "pushing"},
    _call_item(1, "bash", "git push github main"),
    _result_item(1, True, "remote: Permission denied"),
    _call_item(2, "bash", "git push origin main"),
    _result_item(2),
    _call_item(3, "edit_file", "src/kb_service/models.py"),
    _result_item(3),
    _call_item(4, "run_checks", ""),
    _result_item(4),
]

TALOS_EXPECTED: list[dict[str, Any]] = [
    {"kind": "assistant_text", "text": "pushing"},
    _call_item(1, "Bash", "git push github main", "git push"),
    _result_item(1, True, "remote: Permission denied"),
    _call_item(2, "Bash", "git push origin main", "git push"),
    _result_item(2),
    _call_item(3, "Edit", "src/kb_service/models.py", "ext:py"),
    _result_item(3),
    _call_item(4, "run_checks", ""),
    _result_item(4),
]


def _stored(session: str) -> list[Any]:
    async def run() -> list[Any]:
        pool = await SqlitePool.open(database.sqlite_service_db_path())
        try:
            return await turn_digest.get_session_turn_digests(pool, session)
        finally:
            await pool.close()

    return asyncio.run(run())


def _talos(session: str = "talos-1", **kw: Any) -> dict[str, Any]:
    return _digest(session=session, harness="talos", mode="headless", **kw)


def test_talos_digest_stored_canonical(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    resp = _post(local_client, _talos(items=TALOS_ITEMS))
    assert resp.json() == {"recorded": True, "reason": "recorded", "redactions": []}
    (row,) = _rows()
    assert row["harness"] == "talos"
    assert row["event_id"] == "talos-1:0"
    assert row["anomaly"] is None
    assert json.loads(row["items"]) == TALOS_EXPECTED


def test_talos_redaction_after_normalization(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    items = [_call_item(1, "bash", _AWS_LINE), _result_item(1)]
    resp = _post(local_client, _talos("talos-red", items=items))
    assert resp.json()["redactions"] == ["Secret Keyword", "AWS Access Key"]
    (row,) = _rows()
    assert "wJalrXUtnFEMI" not in row["items"]
    assert json.loads(row["items"])[0] == _call_item(
        1, "Bash", "[REDACTED:Secret Keyword]", "export"
    )


def test_talos_duplicate_after_normalization(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    assert _post(local_client, _talos(items=TALOS_ITEMS)).json()["reason"] == "recorded"
    assert (
        _post(local_client, _talos(items=TALOS_ITEMS)).json()["reason"] == "duplicate"
    )
    assert _rows("SELECT COUNT(*) FROM turn_events")[0][0] == 1


def test_claude_code_digest_untouched_and_extra_key_dropped(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    call = _call_item(1, "bash", "git push github main", "x")
    items = [{**call, "native_tool": "spoof"}, _result_item(1)]
    assert _post(local_client, _digest(items=items)).json()["reason"] == "recorded"
    (row,) = _rows()
    assert json.loads(row["items"])[0] == call


def test_talos_empty_bash_target_anomaly(
    local_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    items = [
        {**_call_item(1, "bash", ""), "args": {"command": "git push github main"}},
        _result_item(1),
    ]
    _post(local_client, _talos("talos-empty", items=items))
    (row,) = _rows()
    assert row["anomaly"] == "empty_bash_target_class"
    assert "anomaly=empty_bash_target_class" in caplog.text
    assert json.loads(row["items"])[0] == _call_item(1, "Bash", "", "")


def test_empty_tool_target_anomaly_route(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    talos = [_call_item(1, "edit_file", ""), _result_item(1)]
    _post(local_client, _talos("t1", items=talos))
    cc = [_call_item(1, "Edit", ""), _result_item(1)]
    _post(local_client, _digest(session="c1", items=cc))
    rows = {r["session_id"]: r["anomaly"] for r in _rows()}
    assert rows == {"t1": "empty_tool_target", "c1": None}


def test_shape1_fires_on_talos_digest(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    _post(local_client, _talos(items=TALOS_ITEMS))
    stored = _stored("talos-1")
    assert len(stored) == 1
    digest = surprise_worker.digest_from_row(stored[0].model_dump())
    result = surprise.detect_shape1([digest], digest.event_id)
    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert c.shape == 1
    assert c.turn_event_ids == ["talos-1:0"]
    assert c.detector_output == {
        "wrong_belief": "git push github main",
        "corrected_fact": "git push origin main",
        "evidence_excerpt": "remote: Permission denied",
        "confidence": 1.0,
    }
    assert result.stats["bash_calls"] == 2
    assert result.stats["failures"] == 1
    assert result.stats["pairs"] == 1

    _post(
        local_client,
        _digest(
            session="cc-1", harness="claude-code", mode="headless", items=TALOS_ITEMS
        ),
    )
    cc = _stored("cc-1")
    digest = surprise_worker.digest_from_row(cc[0].model_dump())
    result = surprise.detect_shape1([digest], digest.event_id)
    assert result.candidates == []
    assert result.stats["bash_calls"] == 0


def test_unmapped_tools_logged_and_counted(
    local_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    items = [
        _call_item(1, "web_fetch", "https://x"),
        _result_item(1),
        _call_item(2, "web_fetch", "https://y"),
        _result_item(2),
    ]
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    _post(local_client, _talos(items=items))
    assert "harness=talos" in caplog.text
    assert "unmapped_tools=['web_fetch']" in caplog.text
    hb = local_client.get("/api/kb/turn/heartbeat").json()
    assert hb["unmapped_tools"] == {"talos:web_fetch": 1}
    caplog.clear()
    _post(local_client, _digest(session="cc"))
    assert "harness=claude-code" in caplog.text
    assert "unmapped_tools=[]" in caplog.text


def test_unmapped_tools_none_when_capture_off(
    local_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    items = [_call_item(1, "web_fetch", "https://x"), _result_item(1)]
    _post(local_client, _talos(items=items))
    assert "harness=talos" in caplog.text
    assert "unmapped_tools=None" in caplog.text
    assert local_client.get("/api/kb/turn/heartbeat").json()["unmapped_tools"] == {}


def test_invalid_talos_digest_logs_harness(
    local_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    items = [_call_item(1, "bash", "x" * 501), _result_item(1)]
    assert _post(local_client, _talos(items=items)).status_code == 422
    assert "reason=invalid" in caplog.text
    assert "harness=talos" in caplog.text
    caplog.clear()
    resp = local_client.post(
        "/api/kb/turn", content=b"[]", headers={"content-type": "application/json"}
    )
    assert resp.status_code == 422
    assert "harness=None" in caplog.text


def test_too_large_talos_logs_harness(
    local_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    items = [{"kind": "assistant_text", "text": "t" * 2000}] * 40
    assert _post(local_client, _talos(items=items)).status_code == 413
    assert "reason=too-large" in caplog.text
    assert "harness=talos" in caplog.text
