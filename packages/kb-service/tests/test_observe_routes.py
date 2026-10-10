"""GET /api/kb/observe/*: admin-only paged raw-signal feeds (real SQLite)."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kb_service.database as database
from kb_service import models
from kb_service.auth import get_current_user
from kb_service.main import app
from kb_service.routes import event_routes, turn_routes
from tests.conftest import fake_user

_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
)
_PATHS = ("turn-events", "failure-events", "surprise-detections")
_TABLES = {
    "turn-events": ("turn_events", "received_ts", models.ObserveTurnEventRow),
    "failure-events": ("failure_events", "received_ts", models.ObserveFailureEventRow),
    "surprise-detections": (
        "surprise_detections",
        "ts",
        models.ObserveSurpriseDetectionRow,
    ),
}
_ITEMS = [{"kind": "assistant_text", "text": "hi"}]


@pytest.fixture
def local_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setattr(database, "_pool", None)
    monkeypatch.setattr(turn_routes, "_OUTCOMES", {})
    monkeypatch.setattr(turn_routes, "_UNMAPPED", {})
    monkeypatch.setattr(event_routes, "_OUTCOMES", {})
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _sql(sql: str, *args: Any) -> None:
    conn = sqlite3.connect(database.sqlite_service_db_path())
    try:
        conn.execute(sql, args)
        conn.commit()
    finally:
        conn.close()


def _post_turn(client: TestClient, n: int) -> None:
    resp = client.post(
        "/api/kb/turn",
        json={
            "event_id": f"s:{n}",
            "session_id": "s",
            "harness": "claude-code",
            "mode": "interactive",
            "turn_index": n,
            "ts": "2026-10-09T12:00:00+00:00",
            "items": _ITEMS,
        },
    )
    assert resp.status_code == 200 and resp.json()["recorded"], resp.text


def _post_failure(client: TestClient, n: int) -> None:
    resp = client.post(
        "/api/kb/event",
        json={
            "type": "post_tool",
            "event_id": f"cc:s:{n}",
            "session_id": "s",
            "harness": "claude-code",
            "mode": "interactive",
            "cwd": "/home/j/x",
            "ts": "2026-10-09T12:00:00+00:00",
            "tool_name": "Bash",
            "tool_input": {"command": "false"},
            "tool_use_id": f"t{n}",
            "error": "Exit code 1\nboom",
            "is_error": True,
            "is_interrupt": False,
        },
    )
    assert resp.status_code == 200 and resp.json()["recorded"], resp.text


def _insert_detection(n: int, ts: str = "2026-10-09T12:00:00+00:00") -> None:
    _sql(
        "INSERT INTO surprise_detections (event_id, session_id, project, shape,"
        " mode, outcome, detector_version, details, ts)"
        " VALUES (?, 's', 'p', 2, 'shadow', 'no_surprise', 1, '{\"k\": 1}', ?)",
        f"s:{n}",
        ts,
    )


def _seed(client: TestClient, n: int = 3) -> None:
    for i in range(n):
        _post_turn(client, i)
        _post_failure(client, i)
        _insert_detection(i)


def test_paging_limit_2(local_client: TestClient) -> None:
    _seed(local_client)
    for path in _PATHS:
        first = local_client.get(f"/api/kb/observe/{path}", params={"limit": 2}).json()
        assert len(first["rows"]) == 2
        assert first["next_after_id"] == first["rows"][1]["id"]
        second = local_client.get(
            f"/api/kb/observe/{path}",
            params={"limit": 2, "after_id": first["next_after_id"]},
        ).json()
        assert len(second["rows"]) == 1
        assert second["next_after_id"] is None
        assert second["rows"][0]["id"] > first["rows"][1]["id"]


def test_since_until_window(local_client: TestClient) -> None:
    _seed(local_client)
    _sql(
        "UPDATE turn_events SET received_ts = '2026-01-01T00:00:00+00:00' WHERE id = 1"
    )
    _sql(
        "UPDATE failure_events SET received_ts = '2026-01-01T00:00:00+00:00'"
        " WHERE id = 1"
    )
    _sql("UPDATE surprise_detections SET ts = '2026-01-01T00:00:00+00:00' WHERE id = 1")
    _sql(
        "UPDATE turn_events SET received_ts = '2026-03-01T00:00:00+00:00' WHERE id = 3"
    )
    _sql(
        "UPDATE failure_events SET received_ts = '2026-03-01T00:00:00+00:00'"
        " WHERE id = 3"
    )
    _sql("UPDATE surprise_detections SET ts = '2026-03-01T00:00:00+00:00' WHERE id = 3")
    for path in _PATHS:
        _, col, _ = _TABLES[path]
        _sql(
            f"UPDATE {_TABLES[path][0]} SET {col} = '2026-02-01T00:00:00+00:00'"  # noqa: S608
            " WHERE id = 2"
        )
        resp = local_client.get(
            f"/api/kb/observe/{path}",
            params={"since": "2026-02-01T00:00:00Z", "until": "2026-03-01T00:00:00Z"},
        )
        assert [r["id"] for r in resp.json()["rows"]] == [2]


def test_turn_items_parsed(local_client: TestClient) -> None:
    _post_turn(local_client, 0)
    row = local_client.get("/api/kb/observe/turn-events").json()["rows"][0]
    assert row["items"] == _ITEMS
    assert isinstance(row["redactions"], list)


def test_surprise_details_parsed(local_client: TestClient) -> None:
    _insert_detection(0)
    row = local_client.get("/api/kb/observe/surprise-detections").json()["rows"][0]
    assert row["details"] == {"k": 1}


@pytest.mark.parametrize("path", _PATHS)
def test_auth(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    monkeypatch.delenv("KB_AUTH_MODE", raising=False)
    assert local_client.get(f"/api/kb/observe/{path}").status_code == 401
    app.dependency_overrides[get_current_user] = fake_user
    assert local_client.get(f"/api/kb/observe/{path}").status_code == 403


@pytest.mark.parametrize("path", _PATHS)
def test_malformed_params_422(local_client: TestClient, path: str) -> None:
    for params in (
        {"since": "yesterday"},
        {"until": "nope"},
        {"limit": 0},
        {"limit": 501},
        {"after_id": -1},
    ):
        resp = local_client.get(f"/api/kb/observe/{path}", params=params)
        assert resp.status_code == 422, params


def test_observe_logs_one_info_line(
    local_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="kb_service.routes.observe_routes")
    local_client.get("/api/kb/observe/failure-events", params={"limit": 5})
    lines = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("observe ")
    ]
    assert len(lines) == 1
    assert "table=failure_events" in lines[0] and "rows=0" in lines[0]


@pytest.mark.parametrize("path", _PATHS)
def test_row_model_matches_table_columns(local_client: TestClient, path: str) -> None:
    table, _, model = _TABLES[path]
    conn = sqlite3.connect(database.sqlite_service_db_path())
    try:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()
    assert set(model.model_fields) == cols
