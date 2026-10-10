"""POST /api/kb/event + GET /api/kb/event/heartbeat on the SQLite service DB."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.cues import build_cue

import kb_service.database as database
from kb_service.main import app
from kb_service.routes import event_routes

_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
)


@pytest.fixture
def local_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """The real app in local no-auth mode, service DB in *tmp_path*."""
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setattr(database, "_pool", None)
    monkeypatch.setattr(event_routes, "_OUTCOMES", {})
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _rows(sql: str = "SELECT * FROM failure_events ORDER BY id") -> list[sqlite3.Row]:
    conn = sqlite3.connect(database.sqlite_service_db_path())
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _event(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": "post_tool",
        "event_id": "cc:sess-1:toolu_1",
        "session_id": "sess-1",
        "harness": "claude-code",
        "mode": "interactive",
        "host": "h1",
        "hook_version": "1.0.0",
        "cwd": "/home/j/personal_kb",
        "project": None,
        "ts": "2026-10-07T12:00:00+00:00",
        "tool_name": "Bash",
        "tool_input": {"command": "cd /x && git push origin main"},
        "tool_use_id": "toolu_1",
        "error": "Exit code 1\nerror: failed to push some refs to /srv/git/x.git",
        "is_error": True,
        "is_interrupt": False,
        "duration_ms": 12,
    }
    body.update(overrides)
    return body


def test_recorded_post_tool_failure(local_client: TestClient) -> None:
    body = _event(mode="headless", engine="claude-code")
    resp = local_client.post("/api/kb/event", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    expected = build_cue(
        body["tool_name"],
        body["tool_input"],
        body["error"],
        body["project"],
        body["cwd"],
    )
    assert data == {
        "recorded": True,
        "cue_key": expected.cue_key,
        "normalizer_version": expected.normalizer_version,
        "reason": "recorded",
    }
    [row] = _rows()
    assert row["cue_key"] == expected.cue_key
    assert row["mode"] == "headless"
    assert row["engine"] == "claude-code"
    assert row["project"] == "personal-kb"
    assert row["project_source"] == "cwd_basename"
    assert row["host_class"] == "linux"
    assert row["target_class"] == "git push"
    assert row["target"] == "cd /x && git push origin main"
    assert row["normalized_error"] == expected.normalized_error
    assert row["error_rule"] == "keyword"
    assert row["anomaly"] is None
    assert row["raw_error_excerpt"] == body["error"]
    assert row["is_interrupt"] == 0
    assert row["ts"] == "2026-10-07T12:00:00+00:00"


def test_duplicate_event_id(local_client: TestClient) -> None:
    first = local_client.post("/api/kb/event", json=_event())
    second = local_client.post("/api/kb/event", json=_event())
    assert first.json()["reason"] == "recorded"
    assert second.status_code == 200
    assert second.json()["reason"] == "duplicate"
    assert second.json()["recorded"] is False
    assert second.json()["cue_key"] == first.json()["cue_key"]
    assert len(_rows()) == 1


def test_not_failure(local_client: TestClient) -> None:
    resp = local_client.post("/api/kb/event", json=_event(is_error=False))
    assert resp.json() == {
        "recorded": False,
        "cue_key": None,
        "normalizer_version": None,
        "reason": "not-failure",
    }
    assert _rows() == []


def test_unsupported_type(local_client: TestClient) -> None:
    resp = local_client.post("/api/kb/event", json=_event(type="turn_end"))
    assert resp.status_code == 200
    assert resp.json()["reason"] == "unsupported-type"
    assert _rows() == []


@pytest.mark.parametrize(
    "overrides",
    [{"error": "   "}, {"error": None}, {"tool_name": None}, {"tool_name": " "}],
)
def test_missing_fields(local_client: TestClient, overrides: dict[str, Any]) -> None:
    resp = local_client.post("/api/kb/event", json=_event(**overrides))
    assert resp.status_code == 200
    assert resp.json()["reason"] == "missing-fields"
    assert _rows() == []


def test_bogus_type_422(local_client: TestClient) -> None:
    resp = local_client.post("/api/kb/event", json=_event(type="bogus"))
    assert resp.status_code == 422


def test_write_failed(
    local_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenPool:
        async def execute(self, sql: str, *args: Any) -> str:
            raise RuntimeError("db down")

    async def _get_db() -> _BrokenPool:
        return _BrokenPool()

    monkeypatch.setattr(event_routes, "get_db", _get_db)
    resp = local_client.post("/api/kb/event", json=_event())
    assert resp.status_code == 200
    assert resp.json() == {
        "recorded": False,
        "cue_key": None,
        "normalizer_version": None,
        "reason": "write-failed",
    }


@pytest.mark.parametrize(
    ("raw", "stored"),
    [
        ("2026-10-07T12:00:00.123Z", "2026-10-07T12:00:00+00:00"),
        ("2026-10-07T14:00:00+02:00", "2026-10-07T12:00:00+00:00"),
        ("2026-10-07T12:00:00", "2026-10-07T12:00:00+00:00"),
    ],
)
def test_ts_normalized(local_client: TestClient, raw: str, stored: str) -> None:
    resp = local_client.post("/api/kb/event", json=_event(ts=raw))
    assert resp.json()["reason"] == "recorded"
    [row] = _rows()
    assert row["ts"] == stored


@pytest.mark.parametrize("raw", [None, "not-a-timestamp"])
def test_ts_falls_back_to_received(local_client: TestClient, raw: str | None) -> None:
    local_client.post("/api/kb/event", json=_event(ts=raw))
    [row] = _rows()
    assert row["ts"] == row["received_ts"]


def test_empty_error_anomaly(local_client: TestClient) -> None:
    resp = local_client.post("/api/kb/event", json=_event(error="\x1b[0m"))
    assert resp.json()["recorded"] is True
    [row] = _rows()
    assert row["anomaly"] == "empty_error"
    assert row["normalized_error"] == ""


def test_empty_bash_target_anomaly(local_client: TestClient) -> None:
    resp = local_client.post("/api/kb/event", json=_event(tool_input={}))
    assert resp.json()["recorded"] is True
    [row] = _rows()
    assert row["anomaly"] == "empty_bash_target"


def test_explicit_project_and_interrupt(local_client: TestClient) -> None:
    local_client.post(
        "/api/kb/event",
        json=_event(project="personal-kb", is_interrupt=True, cwd="/Users/j/elsewhere"),
    )
    [row] = _rows()
    assert row["project"] == "personal-kb"
    assert row["project_source"] == "kb_project"
    assert row["host_class"] == "darwin"
    assert row["is_interrupt"] == 1


def test_unauthenticated_rejected_like_listener(client: TestClient) -> None:
    listener = client.post("/api/kb/listener", json={"text": "hello"})
    event = client.post("/api/kb/event", json=_event())
    heartbeat = client.get("/api/kb/event/heartbeat")
    assert listener.status_code == 401
    assert event.status_code == listener.status_code
    assert heartbeat.status_code == listener.status_code


# ─── heartbeat ──────────────────────────────────────────────────────────────


def test_heartbeat_groups_and_window(local_client: TestClient) -> None:
    posts = [
        ("e1", "interactive", "h1"),
        ("e2", "headless", "h1"),
        ("e3", "headless", "h1"),
        ("e4", "headless", "h2"),
        ("e5", "headless", "h2"),
    ]
    for event_id, mode, host in posts:
        resp = local_client.post(
            "/api/kb/event", json=_event(event_id=event_id, mode=mode, host=host)
        )
        assert resp.json()["reason"] == "recorded"
    local_client.post("/api/kb/event", json=_event(event_id="e1"))  # duplicate
    local_client.post("/api/kb/event", json=_event(event_id="e6", type="turn_end"))
    local_client.post(
        "/api/kb/event", json=_event(event_id="e7", error="\x1b[0m", host="h2")
    )

    # Push one h2 row outside the 24h window.
    conn = sqlite3.connect(database.sqlite_service_db_path())
    try:
        conn.execute(
            "UPDATE failure_events SET received_ts = '2000-01-01T00:00:00+00:00'"
            " WHERE event_id = 'e5'"
        )
        conn.commit()
    finally:
        conn.close()

    resp = local_client.get("/api/kb/event/heartbeat")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    groups = {(r["harness"], r["mode"], r["host"]): r for r in data["rows"]}
    assert set(groups) == {
        ("claude-code", "interactive", "h1"),
        ("claude-code", "interactive", "h2"),
        ("claude-code", "headless", "h1"),
        ("claude-code", "headless", "h2"),
    }
    assert groups[("claude-code", "interactive", "h1")]["count"] == 1
    assert groups[("claude-code", "headless", "h1")]["count"] == 2
    assert groups[("claude-code", "headless", "h2")]["count"] == 1
    assert groups[("claude-code", "interactive", "h2")]["anomalies"] == 1
    assert groups[("claude-code", "headless", "h1")]["anomalies"] == 0
    assert groups[("claude-code", "headless", "h1")]["hook_version"] == "1.0.0"
    assert data["route_outcomes"] == {
        "recorded": 6,
        "duplicate": 1,
        "unsupported-type": 1,
    }


def test_heartbeat_three_groups(local_client: TestClient) -> None:
    for event_id, mode, host in [
        ("a", "interactive", "h1"),
        ("b", "headless", "h1"),
        ("c", "headless", "h2"),
        ("d", "headless", "h2"),
    ]:
        local_client.post(
            "/api/kb/event", json=_event(event_id=event_id, mode=mode, host=host)
        )
    data = local_client.get("/api/kb/event/heartbeat", params={"hours": 720}).json()
    counts = [(r["mode"], r["host"], r["count"]) for r in data["rows"]]
    assert counts == [
        ("headless", "h1", 1),
        ("headless", "h2", 2),
        ("interactive", "h1", 1),
    ]


@pytest.mark.parametrize("hours", [0, 721])
def test_heartbeat_hours_bounds(local_client: TestClient, hours: int) -> None:
    resp = local_client.get("/api/kb/event/heartbeat", params={"hours": hours})
    assert resp.status_code == 422


def test_event_endpoints_mounted() -> None:
    paths = {getattr(r, "path", None) for r in app.routes}
    assert {"/api/kb/event", "/api/kb/event/heartbeat"} <= paths


def test_talos_tool_name_canonicalized(local_client: TestClient) -> None:
    body = _event(event_id="talos:sess-1:toolu_1", harness="talos", tool_name="bash")
    resp = local_client.post("/api/kb/event", json=body)
    assert resp.json()["reason"] == "recorded"
    expected = build_cue(
        "Bash", body["tool_input"], body["error"], body["project"], body["cwd"]
    )
    assert resp.json()["cue_key"] == expected.cue_key
    [row] = _rows()
    assert row["harness"] == "talos"
    assert row["tool"] == "Bash"
    assert row["target"] == "cd /x && git push origin main"
    assert row["target_class"] == "git push"
    assert row["anomaly"] is None


def test_claude_code_lowercase_tool_untouched(local_client: TestClient) -> None:
    body = _event(event_id="cc:sess-1:toolu_9", tool_name="bash")
    assert local_client.post("/api/kb/event", json=body).json()["reason"] == "recorded"
    [row] = _rows()
    assert (row["tool"], row["target"], row["target_class"]) == ("bash", "", "")
    assert row["anomaly"] is None
