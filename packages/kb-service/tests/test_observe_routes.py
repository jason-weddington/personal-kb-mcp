"""GET /api/kb/observe/*: admin-only paged raw-signal feeds (real SQLite)."""

import sqlite3
import types
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
from tests.conftest import FakeKnowledgeBase, FakeLLM, fake_user
from tests.test_listener_decisions import _make_map_result

_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
)
_PATHS = (
    "turn-events",
    "failure-events",
    "surprise-detections",
    "llm-calls",
    "listener-decisions",
)
_TABLES = {
    "turn-events": ("turn_events", "received_ts", models.ObserveTurnEventRow),
    "failure-events": ("failure_events", "received_ts", models.ObserveFailureEventRow),
    "surprise-detections": (
        "surprise_detections",
        "ts",
        models.ObserveSurpriseDetectionRow,
    ),
    "llm-calls": ("llm_calls", "ts", models.ObserveLlmCallRow),
    "listener-decisions": (
        "listener_decisions",
        "decided_ts",
        models.ObserveListenerDecisionRow,
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


def _insert_llm_call(n: int, ts: str = "2026-10-09T12:00:00+00:00") -> None:
    _sql(
        "INSERT INTO llm_calls (ts, purpose, model, outcome, diagnostics_state,"
        " latency_ms) VALUES (?, 'planner', 'claude-haiku-4-5', 'ok', 'sent', ?)",
        ts,
        n,
    )


def _insert_listener_decision(n: int, ts: str = "2026-10-09T12:00:00+00:00") -> None:
    _sql(
        "INSERT INTO listener_decisions (session_id, cwd_project, source_kb,"
        " decided_ts, candidates_considered, decision, reason, vote_shape,"
        " candidate_signal, candidate_ids, whispered_ids, n_retrieved, n_after_a,"
        " n_after_b, retrieval_path, text_source, vote_model, vote_prompt,"
        " vote_raw, vote_truncated)"
        " VALUES (?, 'p', 'kb', ?, 2, 'whisper', 'whispered',"
        ' \'[["kb-00001"], ["kb-00001"], ["kb-00001"]]\', \'fallback\','
        ' \'["kb-00001", "kb-00002"]\', \'["kb-00001"]\', 2, 2, 2,'
        " 'fallback-direct', 'last_assistant_message', 'claude-sonnet-4-6',"
        ' \'gate prompt text\', \'["kb-00001", "kb-00001", "kb-00001"]\', 0)',
        f"s:{n}",
        ts,
    )


def _seed(client: TestClient, n: int = 3) -> None:
    for i in range(n):
        _post_turn(client, i)
        _post_failure(client, i)
        _insert_detection(i)
        _insert_llm_call(i)
        _insert_listener_decision(i)


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
    _sql("UPDATE llm_calls SET ts = '2026-01-01T00:00:00+00:00' WHERE id = 1")
    _sql(
        "UPDATE listener_decisions SET decided_ts = '2026-01-01T00:00:00+00:00'"
        " WHERE id = 1"
    )
    _sql(
        "UPDATE turn_events SET received_ts = '2026-03-01T00:00:00+00:00' WHERE id = 3"
    )
    _sql(
        "UPDATE failure_events SET received_ts = '2026-03-01T00:00:00+00:00'"
        " WHERE id = 3"
    )
    _sql("UPDATE surprise_detections SET ts = '2026-03-01T00:00:00+00:00' WHERE id = 3")
    _sql("UPDATE llm_calls SET ts = '2026-03-01T00:00:00+00:00' WHERE id = 3")
    _sql(
        "UPDATE listener_decisions SET decided_ts = '2026-03-01T00:00:00+00:00'"
        " WHERE id = 3"
    )
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


def test_listener_decisions_vote_log_round_trip(local_client: TestClient) -> None:
    """The vote-log columns round-trip: parsed lists, preserved null slots."""
    _insert_listener_decision(0)
    # A second, NON-voted row: all three text columns are SQL NULL.
    _sql(
        "INSERT INTO listener_decisions (session_id, source_kb, decided_ts,"
        " candidates_considered, decision, reason)"
        " VALUES ('s:none', 'kb', '2026-10-09T12:00:01+00:00', 0, 'declined',"
        " 'no-candidates')"
    )
    rows = local_client.get(
        "/api/kb/observe/listener-decisions", params={"limit": 10}
    ).json()["rows"]
    assert [r["session_id"] for r in rows] == ["s:0", "s:none"]
    voted, plain = rows
    assert voted["vote_model"] == "claude-sonnet-4-6"
    assert voted["vote_prompt"] == "gate prompt text"
    assert voted["vote_raw"] == ["kb-00001", "kb-00001", "kb-00001"]
    assert voted["vote_truncated"] == 0
    assert voted["candidate_ids"] == ["kb-00001", "kb-00002"]
    assert voted["whispered_ids"] == ["kb-00001"]
    # SQL NULL parses to [] (never None) on the wire, matching candidate_ids.
    assert plain["vote_model"] is None
    assert plain["vote_prompt"] is None
    assert plain["vote_raw"] == []
    assert plain["vote_truncated"] == 0


def test_listener_decisions_null_vote_slot_round_trip(
    local_client: TestClient,
) -> None:
    """A json null vote slot survives as None inside the parsed list."""
    _sql(
        "INSERT INTO listener_decisions (session_id, source_kb, decided_ts,"
        " candidates_considered, decision, reason, vote_raw)"
        " VALUES ('s:split', 'kb', '2026-10-09T12:00:02+00:00', 2, 'declined',"
        " 'vote-split', '[\"kb-00001\", \"kb-00002\", null]')"
    )
    row = local_client.get("/api/kb/observe/listener-decisions").json()["rows"][0]
    assert row["vote_raw"] == ["kb-00001", "kb-00002", None]


def test_listener_decision_row_from_real_listener_post(
    local_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real POST /api/kb/listener row fetched back through the endpoint.

    End-to-end shape of the vote log (owner decision 2026-10-13): the model
    from the client's config, the redacted prompt the votes saw, and the
    3-element raw list with the None slot preserved (hypothesis H-020 fixture).
    """
    monkeypatch.setenv("KB_LISTENER_ENABLED", "TRUE")
    fake_kb = FakeKnowledgeBase(
        results=[
            _make_map_result("kb-00001", short_title="Map One"),
            _make_map_result("kb-00002", short_title="Map Two"),
        ],
        filtered_count=2,
    )
    fake_llm = FakeLLM()
    fake_llm._config = types.SimpleNamespace(model="claude-sonnet-4-6")
    fake_llm.enqueue("kb-00001")
    fake_llm.enqueue("kb-00002")
    fake_llm.enqueue(None)
    fake_kb.synthesis_llm = fake_llm
    # A secret in the judged text proves redaction survives the whole round
    # trip (assembled from fragments so no source line matches a rule).
    secret = "sk-ant-api03-" + "abcdefghijklmnopqrstuvwxyz0123456789"
    text = f"AWS_ACCESS_KEY_ID={'AKIA' + 'IOSFODNN7EXAMPLE'}\n{secret}"

    real_kb = app.state.kb
    app.state.kb = fake_kb
    try:
        resp = local_client.post(
            "/api/kb/listener", json={"text": text, "session_id": "s-real"}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["pointers"] == []
        row = local_client.get("/api/kb/observe/listener-decisions").json()["rows"][0]
    finally:
        app.state.kb = real_kb

    assert row["session_id"] == "s-real"
    assert row["decision"] == "declined"
    assert row["reason"] == "vote-split"
    assert row["vote_shape"] == '[["kb-00001"], ["kb-00002"], []]'
    assert row["vote_model"] == "claude-sonnet-4-6"
    # Redacted, never raw — and not the unredacted prompt the votes saw.
    assert row["vote_prompt"] != fake_llm.generate_calls[0][0]
    assert secret not in row["vote_prompt"]
    assert "AKIAIOSFODNN7EXAMPLE" not in row["vote_prompt"]
    assert "[REDACTED:" in row["vote_prompt"]
    assert row["vote_raw"] == ["kb-00001", "kb-00002", None]
    assert row["vote_truncated"] == 0


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


def test_observe_rows_expose_agent_columns(local_client: TestClient) -> None:
    """Turn and failure rows expose the subagent columns, null or not."""
    resp = local_client.post(
        "/api/kb/turn",
        json={
            "event_id": "s:agent:a1",
            "session_id": "s",
            "harness": "claude-code",
            "mode": "interactive",
            "turn_index": 1,
            "agent_id": "a1",
            "agent_type": "general-purpose",
            "ts": "2026-10-09T12:00:00+00:00",
            "items": _ITEMS,
        },
    )
    assert resp.status_code == 200 and resp.json()["recorded"], resp.text
    _post_failure(local_client, 0)
    failure = local_client.post(
        "/api/kb/event",
        json={
            "type": "post_tool",
            "event_id": "cc:s:agentfail",
            "session_id": "s",
            "harness": "claude-code",
            "mode": "interactive",
            "cwd": "/home/j/x",
            "ts": "2026-10-09T12:00:00+00:00",
            "tool_name": "Bash",
            "tool_input": {"command": "false"},
            "tool_use_id": "t-agent",
            "agent_id": "a1",
            "agent_type": "general-purpose",
            "error": "Exit code 1\nboom",
            "is_error": True,
            "is_interrupt": False,
        },
    )
    assert failure.status_code == 200 and failure.json()["recorded"], failure.text
    _post_turn(local_client, 2)  # legacy row: no agent keys
    turns = local_client.get("/api/kb/observe/turn-events", params={"limit": 10}).json()
    by_event = {r["event_id"]: r for r in turns["rows"]}
    assert by_event["s:agent:a1"]["agent_id"] == "a1"
    assert by_event["s:agent:a1"]["agent_type"] == "general-purpose"
    assert by_event["s:2"]["agent_id"] is None
    assert by_event["s:2"]["agent_type"] is None
    fails = local_client.get(
        "/api/kb/observe/failure-events", params={"limit": 10}
    ).json()
    by_event = {r["event_id"]: r for r in fails["rows"]}
    assert by_event["cc:s:0"]["tool_use_id"] == "t0"
    assert by_event["cc:s:0"]["agent_id"] is None
    assert by_event["cc:s:0"]["agent_type"] is None
    assert by_event["cc:s:agentfail"]["tool_use_id"] == "t-agent"
    assert by_event["cc:s:agentfail"]["agent_id"] == "a1"
    assert by_event["cc:s:agentfail"]["agent_type"] == "general-purpose"
