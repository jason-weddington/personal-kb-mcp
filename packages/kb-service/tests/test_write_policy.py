"""Write policy: surfaces, routing of non-interactive stores, rejections, logs.

Covers ``kb_service.write_policy`` and its call sites: the pure surface
helpers, the default surface, ``resolve_write_context``, the queued create
and batch over REST and /mcp, the 403 rejections, key minting, the shape-5
candidate pipeline end to end for capture modes on / shadow / off, the
decision and startup logs, and the route classification guard.
"""

import contextlib
import inspect
import json
import logging
import re
import sqlite3
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from kb_core import Attribution, create_sqlite
from kb_core.models.entry import EntryType
from kb_core.near_duplicates import NearDuplicateCandidate, NearDuplicateCheck
from starlette.requests import Request

import kb_service.attribution as attribution_module
import kb_service.auth as auth_module
import kb_service.database as database
from kb_service import surprise_worker, write_policy
from kb_service.auth import AuthPrincipal, create_token, hash_api_key
from kb_service.db_sqlite import SqlitePool
from kb_service.main import app
from kb_service.models import User
from kb_service.models_kb import StoreRequest
from kb_service.resolution_hint import ResolutionHintError
from kb_service.store_distill import STORE_CANDIDATE_SHAPE
from kb_service.surprise import SurpriseCandidate
from kb_service.surprise_worker import (
    _decide_store_one,
    _Decision,
    candidate_from_row,
    distill_candidates,
)
from kb_service.write_policy import (
    REJECTION_TEMPLATES,
    WRITE_ROUTE_CLASSES,
    WriteContext,
    _queued_surface,
    default_surface,
    downgrade,
    is_coerced,
    parse_surface,
    resolve_write_context,
    sanitize_harness,
    sanitize_store_hints,
)
from tests.conftest import (
    FakeCritic,
    FakeKnowledgeBase,
    FakeLLM,
    install_mcp_fakes,
)

WP_LOGGER = "kb_service.write_policy"
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}
HEADLESS = {"X-KB-Mode": "headless"}

_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
    "KB_SKIP_SAFETY",
    "KB_NEAR_DUPLICATE_FLOOR",
    "KB_SURPRISE_LESSON_TTL_DAYS",
)

DURABLE = json.dumps(
    {
        "durable": True,
        "why": "",
        "short_title": "Run the gate with uv",
        "long_title": "The quality gate runs through uv run with --frozen",
        "lesson_class": "project_tooling",
    }
)
NOT_DURABLE = json.dumps({"durable": False, "why": "progress report"})
CRITIC_REJECT = json.dumps(
    {
        "supported": False,
        "scope_ok": True,
        "durable": True,
        "misleading": False,
        "reason": "hedged",
    }
)


def _store_body(**kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "short_title": "Gate uses uv",
        "long_title": "The quality gate is run with uv run --frozen",
        "knowledge_details": "Run the gate with uv run --frozen pytest.",
        "project_ref": "p",
        "supersedes": "none",
    }
    body.update(kw)
    return body


def _rpc(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def _tool_text(resp: Any) -> str:
    assert resp.status_code == 200, resp.text
    return str(resp.json()["result"]["content"][0]["text"])


def _wp_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("write-policy op=")
    ]


# --- AC1 pure helpers ---------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "  "])
def test_parse_surface_blank(raw: str | None) -> None:
    assert parse_surface(raw) is None
    assert is_coerced(raw) is False


def test_parse_surface_values() -> None:
    assert parse_surface(" Headless ") == "headless"
    assert parse_surface("AUTONOMOUS") == "autonomous"
    assert parse_surface("interactive") == "interactive"
    assert parse_surface("interactve") == "headless"
    assert is_coerced("interactve") is True
    assert is_coerced("headless") is False


def test_downgrade() -> None:
    assert downgrade("interactive", "autonomous") == "autonomous"
    assert downgrade("headless", "interactive") == "headless"
    assert downgrade("autonomous", None) == "autonomous"
    assert downgrade("headless", "headless") == "headless"


def test_sanitize_harness() -> None:
    assert sanitize_harness("talos glm!") == "talosglm"
    assert len(sanitize_harness("x" * 100)) == 64
    assert sanitize_harness(None) == ""
    assert sanitize_harness(" claude-code/1.2:a_b ") == "claude-code/1.2:a_b"


def test_constants() -> None:
    assert write_policy.SURFACES == ("interactive", "headless", "autonomous")
    assert write_policy.SURFACE_TRUST == {
        "interactive": 2,
        "headless": 1,
        "autonomous": 0,
    }
    assert write_policy.WRITE_POLICY_DEFAULT_SURFACE_ENV == (
        "KB_WRITE_POLICY_DEFAULT_SURFACE"
    )
    assert write_policy.MODE_HEADER == "X-KB-Mode"
    assert write_policy.HARNESS_HEADER == "X-KB-Harness"
    assert write_policy.HARNESS_MAX == 64
    assert write_policy.USER_AGENT_MAX == 80
    assert write_policy.WRITE_POLICY_MARKER == "write-policy"
    assert write_policy.WRITE_POLICY_DETAIL_PREFIX == "write policy: "
    assert len(REJECTION_TEMPLATES) == 9


# --- AC2 default surface ------------------------------------------------------


@pytest.fixture
def fresh_warned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(write_policy, "_WARNED_DEFAULT_VALUES", set())


@pytest.mark.usefixtures("fresh_warned")
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "interactive"),
        ("  ", "interactive"),
        (" Headless ", "headless"),
        ("autonomous", "autonomous"),
    ],
)
def test_default_surface(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: str
) -> None:
    if value is None:
        monkeypatch.delenv("KB_WRITE_POLICY_DEFAULT_SURFACE", raising=False)
    else:
        monkeypatch.setenv("KB_WRITE_POLICY_DEFAULT_SURFACE", value)
    assert default_surface() == expected


@pytest.mark.usefixtures("fresh_warned")
def test_default_surface_bad_value_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=WP_LOGGER)
    monkeypatch.setenv("KB_WRITE_POLICY_DEFAULT_SURFACE", "nonsense")
    assert default_surface() == "headless"
    assert default_surface() == "headless"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].getMessage() == (
        "write-policy bad_default_surface value='nonsense' fallback=headless"
    )


# --- AC3 resolve_write_context ------------------------------------------------


class _SurfacePool:
    def __init__(self, surface: str | None) -> None:
        self.surface = surface
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        self.calls.append((sql, args))
        return {"surface": self.surface}


def _user() -> User:
    return User(
        id="u1",
        email="u1@example.com",
        hashed_password="x",
        is_admin=False,
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
    )


def _request(principal: AuthPrincipal | None, headers: dict[str, str]) -> Request:
    state: dict[str, Any] = {}
    if principal is not None:
        state["kb_principal"] = principal
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/kb/store",
            "query_string": b"",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "state": state,
        }
    )


@pytest.mark.usefixtures("fresh_warned")
@pytest.mark.parametrize(
    ("key_surface", "auth", "default", "mode", "surface", "source"),
    [
        (None, "api_key", None, None, "interactive", "default"),
        (None, "api_key", None, "headless", "headless", "header"),
        ("interactive", "api_key", None, "autonomous", "autonomous", "header"),
        ("headless", "api_key", None, "interactive", "headless", "key"),
        ("autonomous", "api_key", None, "headless", "autonomous", "key"),
        (None, "api_key", "headless", "interactive", "headless", "default"),
        ("interactive", "api_key", None, "bogus", "headless", "header"),
        (None, "jwt", "headless", None, "interactive", "jwt"),
        (None, "jwt", "headless", "headless", "headless", "header"),
        (None, "none", "autonomous", None, "autonomous", "default"),
        ("headless", "api_key", "autonomous", None, "headless", "key"),
    ],
)
async def test_resolve_write_context_table(
    monkeypatch: pytest.MonkeyPatch,
    key_surface: str | None,
    auth: str,
    default: str | None,
    mode: str | None,
    surface: str,
    source: str,
) -> None:
    pool = _SurfacePool(key_surface)

    async def _get_db() -> _SurfacePool:
        return pool

    monkeypatch.setattr(database, "get_db", _get_db)
    if default is None:
        monkeypatch.delenv("KB_WRITE_POLICY_DEFAULT_SURFACE", raising=False)
    else:
        monkeypatch.setenv("KB_WRITE_POLICY_DEFAULT_SURFACE", default)
    key_id = "k1" if auth == "api_key" else None
    principal = AuthPrincipal(_user(), key_id, auth)
    headers = {"X-KB-Harness": "talos glm!", "User-Agent": "u" * 100}
    if mode is not None:
        headers["X-KB-Mode"] = mode
    wctx = await resolve_write_context(_request(principal, headers), _user())
    assert (wctx.surface, wctx.source) == (surface, source)
    assert wctx.header_mode_coerced is (mode == "bogus")
    assert wctx.harness == "talosglm"
    assert wctx.user_agent == "u" * 80
    assert wctx.user_id == "u1"
    assert wctx.auth_method == auth
    if key_id is not None:
        assert pool.calls == [("SELECT surface FROM api_keys WHERE id = $1", ("k1",))]
        assert wctx.session_key == "key:k1"
        assert wctx.key_surface == key_surface
    else:
        assert pool.calls == []
        assert wctx.session_key == "user:u1"


@pytest.mark.usefixtures("fresh_warned")
async def test_resolve_write_context_no_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _SurfacePool("autonomous")

    async def _get_db() -> _SurfacePool:
        return pool

    monkeypatch.setattr(database, "get_db", _get_db)
    wctx = await resolve_write_context(_request(None, {}), _user())
    assert (wctx.surface, wctx.source) == ("interactive", "default")
    assert wctx.auth_method == "unknown"
    assert wctx.api_key_id is None
    assert pool.calls == []


async def test_resolve_write_context_unknown_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NonePool:
        async def fetchrow(self, sql: str, *args: Any) -> None:
            return None

    async def _get_db() -> _NonePool:
        return _NonePool()

    monkeypatch.setattr(database, "get_db", _get_db)
    principal = AuthPrincipal(_user(), "gone", "api_key")
    wctx = await resolve_write_context(_request(principal, {}), _user())
    assert wctx.key_surface is None
    assert (wctx.surface, wctx.source) == ("interactive", "default")


def test_write_policy_imports_database_module() -> None:
    assert "from kb_service.database import" not in inspect.getsource(write_policy)


def test_write_context_fields() -> None:
    assert list(WriteContext.__dataclass_fields__) == [
        "surface",
        "source",
        "key_surface",
        "header_mode",
        "header_mode_coerced",
        "harness",
        "api_key_id",
        "auth_method",
        "user_id",
        "session_key",
        "user_agent",
        "engine",
    ]


# --- AC7 / AC9 / AC10 helpers -------------------------------------------------


def test_queued_surface() -> None:
    assert _queued_surface("headless") == "headless"
    assert _queued_surface("autonomous") == "autonomous"
    with pytest.raises(RuntimeError, match="interactive surface"):
        _queued_surface("interactive")


def _wctx(surface: Literal["interactive", "headless", "autonomous"]) -> WriteContext:
    return WriteContext(
        surface=surface,
        source="header",
        key_surface=None,
        header_mode=surface,
        header_mode_coerced=False,
        harness="",
        api_key_id=None,
        auth_method="none",
        user_id="local",
        session_key="user:local",
        user_agent="",
    )


def test_policy_rejection_detail() -> None:
    exc = write_policy.policy_rejection(
        "store_batch", "no_project", _wctx("headless"), index=3
    )
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 403
    assert exc.detail == (
        "write policy: entry 3: a store from a headless surface needs a project_ref."
    )


def test_require_interactive_ingest_reason(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=WP_LOGGER)
    with pytest.raises(HTTPException) as info:
        write_policy.require_interactive("ingest_url", _wctx("autonomous"))
    assert info.value.detail == (
        "write policy: ingesting content requires an interactive surface; this"
        " request is autonomous. Store the entry with kb_store and it is queued for"
        " review."
    )
    write_policy.require_interactive("ingest_url", _wctx("interactive"))
    lines = _wp_lines(caplog)
    assert "op=ingest_url outcome=rejected" in lines[0]
    assert "reason=ingest" in lines[0]
    assert "op=ingest_url outcome=allowed" in lines[1]


def test_sanitize_store_hints() -> None:
    assert sanitize_store_hints(None, EntryType.FACTUAL_REFERENCE) is None
    assert (
        sanitize_store_hints(
            {"supersedes": ["kb-00001"], "write_policy": {}},
            EntryType.FACTUAL_REFERENCE,
        )
        is None
    )
    out = sanitize_store_hints(
        {
            "surprise_capture": {"shape": 2},
            "person": "jason",
            "resolution": {
                "corrected_fact": "use uv",
                "observed_sessions": 5,
                "scope": "global",
                "provenance": {
                    "capture": "deliberate",
                    "grounding": "observed",
                    "event_id": "s:1",
                },
            },
        },
        EntryType.LESSON_LEARNED,
    )
    assert out is not None
    assert "surprise_capture" not in out
    assert out["person"] == "jason"
    assert out["resolution"] == {
        "corrected_fact": "use uv",
        "observed_sessions": 1,
        "scope": "project",
        "provenance": {"capture": "autonomous", "grounding": "asserted"},
    }
    with pytest.raises(ResolutionHintError):
        sanitize_store_hints({"resolution": "nope"}, EntryType.LESSON_LEARNED)


def test_sanitize_store_hints_tripwire(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _bad(h: dict[str, Any], **kw: Any) -> dict[str, Any]:
        return {**h, "surprise_capture": {"shape": 1}}

    monkeypatch.setattr(write_policy, "validate_and_stamp_resolution", _bad)
    with pytest.raises(RuntimeError, match="routed resolution was not sanitized"):
        sanitize_store_hints({"tags": 1}, EntryType.LESSON_LEARNED)
    assert "write-policy tripwire=resolution_not_sanitized" in caplog.text


# --- shared app fixtures ------------------------------------------------------


def _base_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setenv("KB_LISTENER_ENABLED", "FALSE")
    monkeypatch.setattr(database, "_pool", None)
    monkeypatch.setattr(write_policy, "_WARNED_DEFAULT_VALUES", set())


WpClient = Callable[
    [str | None, Literal["off", "shadow", "on"]],
    contextlib.AbstractContextManager[tuple[TestClient, FakeLLM, FakeCritic]],
]


@pytest.fixture
def wp_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WpClient:
    """Factory: a real no-auth app on SQLite with the given default and mode."""

    @contextlib.contextmanager
    def _make(
        default_surface: str | None, capture: Literal["off", "shadow", "on"]
    ) -> Iterator[tuple[TestClient, FakeLLM, FakeCritic]]:
        _base_env(monkeypatch, tmp_path)
        monkeypatch.setenv("KB_AUTH_MODE", "none")
        monkeypatch.setenv("KB_SURPRISE_CAPTURE", capture)
        if default_surface is None:
            monkeypatch.delenv("KB_WRITE_POLICY_DEFAULT_SURFACE", raising=False)
        else:
            monkeypatch.setenv("KB_WRITE_POLICY_DEFAULT_SURFACE", default_surface)
        llm = FakeLLM()
        critic = FakeCritic()
        monkeypatch.setattr(surprise_worker, "get_distiller_llm", lambda: llm)
        monkeypatch.setattr(surprise_worker, "get_detector_llm", lambda: llm)
        monkeypatch.setattr(surprise_worker, "get_critic_llm", lambda: critic)
        with TestClient(app) as client:
            yield client, llm, critic
        app.dependency_overrides.clear()

    return _make


def _service_rows(tmp_path: Path, sql: str, *args: Any) -> list[sqlite3.Row]:
    conn = sqlite3.connect(tmp_path / "service.db")
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(sql, args).fetchall())
    finally:
        conn.close()


def _candidates(tmp_path: Path) -> list[sqlite3.Row]:
    return _service_rows(tmp_path, "SELECT * FROM surprise_candidates ORDER BY id")


def _get_entry(client: TestClient, entry_id: str) -> dict[str, Any]:
    resp = client.post("/api/kb/get", json={"ids": [entry_id]})
    assert resp.status_code == 200, resp.text
    entry: dict[str, Any] = resp.json()["results"][0]["entry"]
    return entry


# --- AC18 capture modes end to end --------------------------------------------


def test_capture_on_writes_after_distill(wp_client: WpClient, tmp_path: Path) -> None:
    with wp_client("headless", "on") as (client, llm, critic):
        resp = client.post("/api/kb/store", json=_store_body(tags=["gate"]))
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "status": "queued",
            "candidate_id": 1,
            "surface": "headless",
            "capture_mode": "on",
        }
        assert (
            client.post("/api/kb/get", json={"ids": ["kb-00001"]}).json()["results"][0][
                "found"
            ]
            is False
        )
        row = _candidates(tmp_path)[0]
        assert (row["shape"], row["status"], row["detector_model"]) == (
            5,
            "pending",
            "write-policy",
        )
        assert row["session_id"] == "user:local"
        assert row["turn_event_ids"] == "[]"
        out = json.loads(row["detector_output"])
        assert set(out) == {
            "kind",
            "op",
            "batch_index",
            "surface",
            "source",
            "harness",
            "engine",
            "api_key_id",
            "user_id",
            "auth_method",
            "contributor",
            "team",
            "capture_mode",
            "request",
        }
        assert (out["kind"], out["op"], out["batch_index"]) == ("store", "store", None)
        assert (out["source"], out["auth_method"]) == ("default", "none")
        assert set(out["request"]) == {
            "short_title",
            "long_title",
            "knowledge_details",
            "entry_type",
            "project_ref",
            "source_context",
            "confidence_level",
            "tags",
            "hints",
            "sensitivity",
            "ttl",
        }
        assert out["request"]["entry_type"] == "factual_reference"

        llm.enqueue(DURABLE)
        drain = client.post("/api/kb/surprise/drain")
        assert drain.status_code == 200, drain.text
        body = drain.json()
        assert len(body["entries_written"]) == 1
        cand = next(c for c in body["candidates"] if c["id"] == 1)
        assert (cand["shape"], cand["status"]) == (5, "written")
        assert len(critic.generate_calls) == 1
        assert critic.generate_calls[0][1] == write_policy_critic_system()

        entry = _get_entry(client, body["entries_written"][0])
        assert entry["tags"][:1] == ["gate"]
        assert {
            "write-policy",
            "surface:headless",
            "lesson-class:project_tooling",
        } <= set(entry["tags"])
        assert entry["hints"]["write_policy"]["candidate_id"] == 1
        assert entry["hints"]["write_policy"]["engine"] == ""
        assert entry["contributor"] == "local@localhost"
        assert entry["confidence_level"] == pytest.approx(0.7)
        expires = datetime.fromisoformat(entry["expires_at"])
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        target = datetime.now(UTC) + timedelta(days=30)
        assert abs((expires - target).total_seconds()) < 60
        dist = _service_rows(tmp_path, "SELECT * FROM surprise_distillations")
        assert len(dist) == 1
        assert (dist[0]["shape"], dist[0]["outcome"], dist[0]["distiller_version"]) == (
            5,
            "written",
            1,
        )
        assert json.loads(dist[0]["verdict"])["critic_version"] == 1


def write_policy_critic_system() -> str:
    from kb_service.store_distill import STORE_CRITIC_SYSTEM

    return STORE_CRITIC_SYSTEM


def test_capture_on_not_durable_and_critic_reject(
    wp_client: WpClient, tmp_path: Path
) -> None:
    with wp_client("headless", "on") as (client, llm, critic):
        client.post("/api/kb/store", json=_store_body())
        llm.enqueue(NOT_DURABLE)
        client.post("/api/kb/surprise/drain")
        client.post("/api/kb/store", json=_store_body(short_title="Second"))
        llm.enqueue(DURABLE)
        critic.enqueue(CRITIC_REJECT)
        client.post("/api/kb/surprise/drain")
    rows = _candidates(tmp_path)
    assert [r["status"] for r in rows] == ["rejected", "rejected"]
    dist = _service_rows(
        tmp_path, "SELECT outcome, reason FROM surprise_distillations ORDER BY id"
    )
    assert (dist[0]["outcome"], dist[0]["reason"]) == ("not_durable", "progress report")
    assert dist[1]["outcome"] == "not_durable"
    assert dist[1]["reason"].startswith("critic: ")


def test_capture_shadow_dry_run(wp_client: WpClient, tmp_path: Path) -> None:
    with wp_client("headless", "shadow") as (client, llm, _critic):
        resp = client.post("/api/kb/store", json=_store_body())
        assert resp.json()["capture_mode"] == "shadow"
        assert _candidates(tmp_path)[0]["status"] == "shadow"
        llm.enqueue(DURABLE)
        drain = client.post("/api/kb/surprise/drain")
        assert drain.status_code == 200, drain.text
        assert drain.json()["entries_written"] == []
        listed = client.get("/api/kb/surprise/candidates", params={"shape": 5})
        assert listed.status_code == 200, listed.text
        cands = listed.json()["candidates"]
        assert len(cands) == 1
        assert cands[0]["detector_output"]["kind"] == "store"
        assert cands[0]["mode"] is None
        assert cands[0]["dry_run"]["would_outcome"] == "would_write"
    runs = _service_rows(tmp_path, "SELECT * FROM surprise_dry_runs")
    assert len(runs) == 1
    assert (runs[0]["would_outcome"], runs[0]["distiller_version"]) == (
        "would_write",
        1,
    )
    assert json.loads(runs[0]["payload"])["critic_version"] == 1
    assert _candidates(tmp_path)[0]["status"] == "shadow"
    assert (
        _service_rows(tmp_path, "SELECT COUNT(*) AS n FROM surprise_distillations")[0][
            "n"
        ]
        == 0
    )


def test_capture_off_records_for_audit(
    wp_client: WpClient, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="kb_service")
    with wp_client("headless", "off") as (client, llm, _critic):
        resp = client.post("/api/kb/store", json=_store_body())
        assert resp.json()["capture_mode"] == "off"
        drain = client.post("/api/kb/surprise/drain")
        assert drain.json()["digests_processed"] == 0
        assert drain.json()["candidates"] == []
    assert _candidates(tmp_path)[0]["status"] == "shadow"
    assert llm.generate_calls == []
    assert "write-policy queue_unattended reason=capture_off" in caplog.text


# --- AC18(f) worker-level cosine ----------------------------------------------


class _ConstEmbedder:
    async def embed(self, text: str) -> list[float] | None:
        return [1.0] + [0.0] * 1023


@pytest.fixture
def local_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    yield tmp_path / "service.db"


@pytest.fixture
async def pool(local_env: Path) -> AsyncIterator[SqlitePool]:
    await database.init_db()
    db = await database.get_db()
    assert isinstance(db, SqlitePool)
    yield db
    await database.close_db()


@pytest.fixture
async def kb_cos(tmp_path: Path) -> AsyncIterator[Any]:
    k = await create_sqlite(
        tmp_path / "kb_cos.db",
        extraction_llm=None,
        query_llm=None,
        synthesis_llm=None,
        embedder=_ConstEmbedder(),
    )
    try:
        yield k
    finally:
        await k.close()


def _output(**request: Any) -> dict[str, Any]:
    req = {
        "short_title": "Gate uses uv",
        "long_title": "The gate runs with uv",
        "knowledge_details": "Run the gate with uv run --frozen pytest.",
        "entry_type": "factual_reference",
        "project_ref": "p",
        "source_context": None,
        "confidence_level": None,
        "tags": None,
        "hints": None,
        "sensitivity": None,
        "ttl": None,
    }
    req.update(request)
    return {
        "kind": "store",
        "op": "store",
        "batch_index": None,
        "surface": "headless",
        "source": "header",
        "harness": "talos",
        "api_key_id": None,
        "user_id": "local",
        "auth_method": "none",
        "contributor": "c",
        "team": None,
        "capture_mode": "on",
        "request": req,
    }


async def _insert_store_candidate(
    pool: SqlitePool, output: dict[str, Any], project: str = "p"
) -> SurpriseCandidate:
    row = await pool.fetchrow(
        write_policy._INSERT_STORE_CANDIDATE_SQL,
        STORE_CANDIDATE_SHAPE,
        "user:local",
        project,
        "write-policy",
        json.dumps(output),
        "pending",
        "2026-10-10T00:00:00+00:00",
    )
    full = await pool.fetchrow(
        "SELECT id, shape, session_id, project, turn_event_ids, detector_model,"
        " detector_output, status, entry_id, created_at FROM surprise_candidates"
        " WHERE id = $1",
        row["id"],
    )
    return candidate_from_row(full)


async def test_worker_cosine_covered(
    pool: SqlitePool, kb_cos: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = FakeLLM()
    critic = FakeCritic()
    monkeypatch.setattr(surprise_worker, "get_distiller_llm", lambda: llm)
    monkeypatch.setattr(surprise_worker, "get_critic_llm", lambda: critic)
    await kb_cos.store(
        short_title="Gate facts",
        long_title="How the gate runs",
        knowledge_details="uv run",
        project_ref="p",
        enrich=False,
    )
    cand = await _insert_store_candidate(pool, _output())
    llm.enqueue(DURABLE)
    await distill_candidates(pool, kb_cos, [cand], "on")
    status = await pool.fetchval(
        "SELECT status FROM surprise_candidates WHERE id = $1", cand.id
    )
    assert status == "rejected"
    row = await pool.fetchrow("SELECT * FROM surprise_distillations")
    assert (row["outcome"], row["reason"], row["match_kind"]) == (
        "covered",
        "near_duplicate",
        "cosine",
    )
    assert row["near_duplicate_status"] == "checked"
    assert row["distiller_version"] == 1


# --- AC17 W1-W9 branches ------------------------------------------------------


class _Ingest:
    def __init__(self, skip: bool) -> None:
        self.skip_safety = skip


class _Cfg:
    def __init__(self, skip: bool) -> None:
        self.ingest = _Ingest(skip)


class _StubKb:
    def __init__(self, status: str = "checked", skip: bool = False) -> None:
        self.config = _Cfg(skip)
        self.status = status

    async def find_near_duplicates(self, **kw: Any) -> NearDuplicateCheck:
        return NearDuplicateCheck(
            status=self.status, candidates=(), top_similarity=None
        )  # type: ignore[arg-type]


def _cand(output: dict[str, Any], project: str = "p") -> SurpriseCandidate:
    return SurpriseCandidate(
        id=9,
        shape=5,
        session_id="user:local",
        project=project,
        turn_event_ids=[],
        detector_model="write-policy",
        detector_output=output,
        status="pending",
        entry_id=None,
        created_at="t",
    )


class _BoomLLM:
    model = "boom"

    async def generate(self, prompt: Any, *, system: Any = None) -> str | None:
        raise RuntimeError("down")


async def _decide(
    c: SurpriseCandidate,
    kb: Any,
    llm: Any,
    critic: Any,
) -> tuple[bool, _Decision, Any]:
    from collections import Counter

    d = _Decision()
    counts: Counter[str] = Counter()
    ok = await _decide_store_one(c, kb, llm, 0.88, d, counts, critic=critic)
    return ok, d, counts


async def test_store_decide_branches() -> None:
    kb = _StubKb()
    llm, critic = FakeLLM(), FakeCritic()
    ok, d, _ = await _decide(_cand({"request": "x"}), kb, llm, critic)
    assert (ok, d.outcome, d.reason) == (True, "invalid_fields", "request")
    ok, d, _ = await _decide(_cand(_output(short_title="  ")), kb, llm, critic)
    assert (d.outcome, d.reason) == ("invalid_fields", "request")
    ok, d, _ = await _decide(_cand(_output(), project=" "), kb, llm, critic)
    assert d.outcome == "no_project"
    ok, d, _ = await _decide(_cand(_output()), kb, None, critic)
    assert ok is False
    ok, d, _ = await _decide(_cand(_output()), kb, _BoomLLM(), critic)
    assert (d.outcome, d.reason) == ("llm_error", "exception")
    llm.enqueue(None)
    ok, d, _ = await _decide(_cand(_output()), kb, llm, critic)
    assert (d.outcome, d.reason) == ("llm_error", "none")
    llm.enqueue("not json")
    ok, d, _ = await _decide(_cand(_output()), kb, llm, critic)
    assert d.outcome == "unparseable"
    llm.enqueue(NOT_DURABLE)
    ok, d, _ = await _decide(_cand(_output()), kb, llm, critic)
    assert (d.outcome, d.reason) == ("not_durable", "progress report")
    llm.enqueue(DURABLE)
    ok, d, _ = await _decide(
        _cand(_output(knowledge_details="key [REDACTED:aws]")), kb, llm, critic
    )
    assert (d.outcome, d.reason) == ("redacted", "verdict")
    llm.enqueue(DURABLE)
    ok, d, _ = await _decide(
        _cand(_output(knowledge_details="curl https://user:s3cretpass@example.com/x")),
        kb,
        llm,
        critic,
    )
    assert d.outcome == "secret_detected"
    llm.enqueue(DURABLE)
    critic.enqueue("garbage")
    ok, d, _ = await _decide(_cand(_output()), kb, llm, critic)
    assert (d.outcome, d.reason) == ("unparseable", "critic: unparseable")
    llm.enqueue(DURABLE)
    ok, d, _ = await _decide(_cand(_output()), kb, llm, _BoomLLM())
    assert (d.outcome, d.reason) == ("llm_error", "critic: exception")
    llm.enqueue(DURABLE)
    ok, d, counts = await _decide(
        _cand(_output()), _StubKb(status="embedder_unavailable"), llm, critic
    )
    assert d.outcome == "would_write"
    assert counts["near_dup_unavailable"] == 1
    assert counts["store_llm_calls"] == 1
    assert counts["store_critic_calls"] == 1
    assert d.store_kwargs["tags"] == [
        "write-policy",
        "surface:headless",
        "lesson-class:project_tooling",
    ]


async def test_store_decide_cosine_candidate() -> None:
    class _DupKb(_StubKb):
        async def find_near_duplicates(self, **kw: Any) -> NearDuplicateCheck:
            top = NearDuplicateCandidate(
                id="kb-00007",
                short_title="t",
                entry_type="factual_reference",
                similarity=0.93,
                updated_at=None,
            )
            return NearDuplicateCheck(
                status="checked", candidates=(top,), top_similarity=0.93
            )

    llm, critic = FakeLLM(), FakeCritic()
    llm.enqueue(DURABLE)
    _ok, d, counts = await _decide(_cand(_output()), _DupKb(skip=True), llm, critic)
    assert (d.outcome, d.reason, d.matched_id, d.match_kind) == (
        "covered",
        "near_duplicate",
        "kb-00007",
        "cosine",
    )
    assert d.similarity == pytest.approx(0.93)
    assert counts["cosine_matches"] == 1


# --- AC19 routes --------------------------------------------------------------


def test_interactive_default_creates(wp_client: WpClient, tmp_path: Path) -> None:
    with wp_client(None, "shadow") as (client, _llm, _critic):
        resp = client.post("/api/kb/store", json=_store_body())
        assert resp.status_code == 200, resp.text
        assert resp.json()["action"] == "created"
        batch = client.post(
            "/api/kb/store_batch",
            json={"entries": [_store_body(short_title="Batch one")]},
        )
        assert batch.status_code == 200, batch.text
        assert len(batch.json()["created"]) == 1
    assert _candidates(tmp_path) == []


def test_header_routing_and_logs(
    wp_client: WpClient, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="kb_service")
    with wp_client(None, "shadow") as (client, _llm, _critic):
        resp = client.post(
            "/api/kb/store",
            json=_store_body(),
            headers={"X-KB-Mode": "headless", "X-KB-Harness": "talos"},
        )
        assert resp.json()["status"] == "queued"
        bogus = client.post(
            "/api/kb/store",
            json=_store_body(),
            headers={"X-KB-Mode": "bogus", "X-KB-Harness": "talos glm!"},
        )
        assert bogus.json()["surface"] == "headless"
        created = client.post("/api/kb/store", json=_store_body(short_title="Mine"))
        assert created.json()["action"] == "created"
    lines = _wp_lines(caplog)
    queued = [ln for ln in lines if "outcome=queued" in ln]
    assert "source=header" in queued[0]
    assert "harness='talos'" in queued[0]
    assert "key_id=-" in queued[0]
    assert "candidate_ids=1" in queued[0]
    assert "capture_mode=shadow" in queued[0]
    assert "header_mode_coerced=1" in queued[1]
    assert "harness='talosglm'" in queued[1]
    allowed = [ln for ln in lines if "op=store outcome=allowed" in ln]
    assert "surface=interactive source=default" in allowed[0]
    assert json.loads(_candidates(tmp_path)[0]["detector_output"])["harness"] == "talos"


def test_header_cannot_upgrade(wp_client: WpClient) -> None:
    with wp_client("headless", "shadow") as (client, _llm, _critic):
        resp = client.post(
            "/api/kb/store", json=_store_body(), headers={"X-KB-Mode": "interactive"}
        )
        assert resp.json()["status"] == "queued"
        assert resp.json()["surface"] == "headless"


def _detail(reason: str, surface: str = "headless", index: int | None = None) -> str:
    return (
        "write policy: "
        + (f"entry {index}: " if index is not None else "")
        + REJECTION_TEMPLATES[reason].format(surface=surface)
    )


def test_rejections(
    wp_client: WpClient, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="kb_service")
    with wp_client(None, "shadow") as (client, _llm, _critic):
        cases: list[tuple[Any, str]] = [
            (
                client.post(
                    "/api/kb/store",
                    json={"update_entry_id": "kb-99999", "knowledge_details": "x"},
                    headers=HEADLESS,
                ),
                _detail("update"),
            ),
            (
                client.post(
                    "/api/kb/entries/kb-99999/deactivate",
                    json={"change_reason": "x"},
                    headers=HEADLESS,
                ),
                _detail("deactivate"),
            ),
            (
                client.post("/api/kb/entries/kb-99999/reactivate", headers=HEADLESS),
                _detail("reactivate"),
            ),
            (
                client.post(
                    "/api/kb/bulk_update",
                    json={
                        "filters": {"project_ref": "p"},
                        "updates": {"team": "t"},
                        "dry_run": False,
                    },
                    headers=HEADLESS,
                ),
                _detail("bulk_update"),
            ),
            (
                client.post(
                    "/api/kb/ingest/text",
                    json={"content": "x", "source_name": "a.md", "dry_run": False},
                    headers=HEADLESS,
                ),
                _detail("ingest"),
            ),
            (
                client.post(
                    "/api/kb/ingest/url",
                    json={"url": "https://example.com/a", "content": "x"},
                    headers=HEADLESS,
                ),
                _detail("ingest"),
            ),
            (
                client.post(
                    "/api/kb/ingest/file",
                    files={"file": ("a.md", b"x")},
                    data={"dry_run": "false"},
                    headers=HEADLESS,
                ),
                _detail("ingest"),
            ),
            (
                client.post(
                    "/api/kb/store",
                    json=_store_body(supersedes=["kb-00001"]),
                    headers=HEADLESS,
                ),
                _detail("supersedes"),
            ),
            (
                client.post(
                    "/api/kb/store",
                    json=_store_body(hints={"supersedes": "kb-00001"}),
                    headers=HEADLESS,
                ),
                _detail("supersedes"),
            ),
            (
                client.post(
                    "/api/kb/store",
                    json=_store_body(distinct_from=["kb-00001"]),
                    headers=HEADLESS,
                ),
                _detail("distinct_from"),
            ),
            (
                client.post(
                    "/api/kb/store",
                    json=_store_body(entry_type="mental_map"),
                    headers=HEADLESS,
                ),
                _detail("mental_map"),
            ),
            (
                client.post(
                    "/api/kb/store",
                    json=_store_body(project_ref="  "),
                    headers=HEADLESS,
                ),
                _detail("no_project"),
            ),
            (
                client.post(
                    "/api/kb/store_batch",
                    json={
                        "entries": [
                            _store_body(),
                            _store_body(entry_type="mental_map"),
                        ]
                    },
                    headers=HEADLESS,
                ),
                _detail("mental_map", index=1),
            ),
        ]
        for resp, detail in cases:
            assert resp.status_code == 403, resp.text
            assert resp.json()["detail"] == detail
        for i, kw in enumerate(
            [
                {"project_ref": None},
                {"supersedes": ["kb-00001"]},
                {"distinct_from": ["kb-00001"]},
            ]
        ):
            resp = client.post(
                "/api/kb/store_batch",
                json={"entries": [_store_body(**kw)]},
                headers=HEADLESS,
            )
            assert resp.status_code == 403, (i, resp.text)
            assert resp.json()["detail"].startswith("write policy: entry 0: ")

        dry = client.post(
            "/api/kb/bulk_update",
            json={"filters": {"project_ref": "p"}, "updates": {"team": "t"}},
            headers=HEADLESS,
        )
        assert dry.status_code == 200, dry.text
        dry_ingest = client.post(
            "/api/kb/ingest/text",
            json={"content": "hello", "source_name": "a.md", "dry_run": True},
            headers=HEADLESS,
        )
        assert dry_ingest.status_code == 200, dry_ingest.text
    assert _candidates(tmp_path) == []
    rejected = [ln for ln in _wp_lines(caplog) if "outcome=rejected" in ln]
    assert any("op=store " in ln and "reason=mental_map" in ln for ln in rejected)
    assert len(rejected) == 16


def test_queued_validation_422s_log_nothing(
    wp_client: WpClient, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="kb_service")
    leaky = "curl https://user:s3cretpass@example.com/x"
    with wp_client(None, "shadow") as (client, _llm, _critic):
        resp = client.post(
            "/api/kb/store",
            json=_store_body(knowledge_details=leaky),
            headers=HEADLESS,
        )
        assert resp.status_code == 422, resp.text
        assert _wp_lines(caplog) == []
        for kw, prefix in [
            ({"ttl": "bogus"}, "entry 0: "),
            ({"knowledge_details": leaky}, "entry 0: Secret scan detected"),
            ({"hints": {"resolution": "nope"}}, "entry 0: "),
        ]:
            resp = client.post(
                "/api/kb/store_batch",
                json={"entries": [_store_body(**kw)]},
                headers=HEADLESS,
            )
            assert resp.status_code == 422, resp.text
            assert resp.json()["detail"].startswith(prefix)
        resp = client.post(
            "/api/kb/store",
            json=_store_body(hints={"resolution": "nope"}),
            headers=HEADLESS,
        )
        assert resp.status_code == 422, resp.text
    assert _wp_lines(caplog) == []
    assert _candidates(tmp_path) == []


def test_routed_hints_are_sanitized(wp_client: WpClient, tmp_path: Path) -> None:
    hints = {
        "surprise_capture": {"shape": 2},
        "resolution": {
            "corrected_fact": "use uv",
            "provenance": {"grounding": "observed", "event_id": "s:1"},
        },
    }
    with wp_client(None, "shadow") as (client, _llm, _critic):
        resp = client.post(
            "/api/kb/store",
            json=_store_body(entry_type="lesson_learned", hints=hints),
            headers=HEADLESS,
        )
        assert resp.status_code == 200, resp.text
    out = json.loads(_candidates(tmp_path)[0]["detector_output"])
    routed = out["request"]["hints"]
    assert "surprise_capture" not in routed
    assert routed["resolution"]["provenance"] == {
        "capture": "autonomous",
        "grounding": "asserted",
    }


def test_sanitize_tripwire_answers_500(
    wp_client: WpClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _observed(h: dict[str, Any], **kw: Any) -> dict[str, Any]:
        res = {**h["resolution"], "provenance": {"grounding": "observed"}}
        return {**h, "resolution": res}

    monkeypatch.setattr(write_policy, "validate_and_stamp_resolution", _observed)
    with (
        wp_client(None, "shadow") as (client, _llm, _critic),
        TestClient(app, raise_server_exceptions=False) as raw,
    ):
        del client
        resp = raw.post(
            "/api/kb/store",
            json=_store_body(hints={"resolution": {"corrected_fact": "x"}}),
            headers=HEADLESS,
        )
        assert resp.status_code == 500
    assert _candidates(tmp_path) == []


def test_batch_queues_in_order(wp_client: WpClient, tmp_path: Path) -> None:
    with wp_client(None, "on") as (client, _llm, _critic):
        resp = client.post(
            "/api/kb/store_batch",
            json={
                "entries": [
                    _store_body(short_title="First"),
                    _store_body(short_title="Second", ttl="7d"),
                ]
            },
            headers={"X-KB-Mode": "autonomous"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "status": "queued",
            "requested": 2,
            "candidate_ids": [1, 2],
            "surface": "autonomous",
            "capture_mode": "on",
        }
    rows = _candidates(tmp_path)
    assert [r["status"] for r in rows] == ["pending", "pending"]
    outs = [json.loads(r["detector_output"]) for r in rows]
    assert [o["request"]["short_title"] for o in outs] == ["First", "Second"]
    assert [(o["op"], o["batch_index"]) for o in outs] == [
        ("store_batch", 0),
        ("store_batch", 1),
    ]


def test_mcp_store_routing(
    wp_client: WpClient, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="kb_service")
    args = {
        "short_title": "Gate uses uv",
        "long_title": "The quality gate is run with uv run --frozen",
        "knowledge_details": "Run the gate with uv run --frozen pytest.",
        "project_ref": "p",
        "supersedes": "none",
    }
    with wp_client(None, "shadow") as (client, _llm, _critic):
        resp = client.post(
            "/mcp",
            json=_rpc("kb_store", args),
            headers={**MCP_HEADERS, "X-KB-Mode": "headless", "X-KB-Harness": "talos"},
        )
        assert _tool_text(resp) == (
            "Queued as candidate 1 (write policy: headless surface). Not in the KB:"
            " capture mode is shadow, so it is recorded for audit only."
        )
        plain = client.post(
            "/mcp",
            json=_rpc("kb_store", {**args, "short_title": "Other"}),
            headers=MCP_HEADERS,
        )
        assert re.match(r"^Created kb-\d{5} \(v1\)", _tool_text(plain))
    row = _candidates(tmp_path)[0]
    out = json.loads(row["detector_output"])
    assert row["shape"] == 5
    assert (out["source"], out["harness"]) == ("header", "talos")
    assert any(
        "source=header" in ln and "harness='talos'" in ln for ln in _wp_lines(caplog)
    )


def test_mcp_client_store_looks_up_key_surface(
    monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    pool = install_mcp_fakes(monkeypatch, fake_kb)
    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json=_rpc(
                "kb_store",
                {
                    "short_title": "t",
                    "long_title": "lt",
                    "knowledge_details": "d",
                    "project_ref": "p",
                    "supersedes": "none",
                },
            ),
            headers={**MCP_HEADERS, "Authorization": "Bearer kb_test_user"},
        )
        assert _tool_text(resp).startswith("Created kb-")
    app.dependency_overrides.clear()
    assert ("key-user",) in pool.surface_queries


# --- AC19(f) / AC6 API-key surfaces in jwt mode -------------------------------

_KEYS = {"headless": "kb_head", "interactive": "kb_inter", None: "kb_null"}
_USER_ID = "user-1"


@pytest.fixture
def wp_jwt_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[str | None], contextlib.AbstractContextManager[TestClient]]:
    """Factory: a jwt-mode app whose service DB is a real SQLite file."""

    @contextlib.contextmanager
    def _make(default_surface: str | None) -> Iterator[TestClient]:
        _base_env(monkeypatch, tmp_path)
        monkeypatch.delenv("KB_AUTH_MODE", raising=False)
        monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
        if default_surface is None:
            monkeypatch.delenv("KB_WRITE_POLICY_DEFAULT_SURFACE", raising=False)
        else:
            monkeypatch.setenv("KB_WRITE_POLICY_DEFAULT_SURFACE", default_surface)

        async def _get_db() -> Any:
            if database._pool is None:
                database._pool = await SqlitePool.open(tmp_path / "service.db")
            return database._pool

        monkeypatch.setattr(database, "get_db", _get_db)
        monkeypatch.setattr(auth_module, "get_db", _get_db)
        monkeypatch.setattr(attribution_module, "get_db", _get_db)
        with TestClient(app) as client:
            conn = sqlite3.connect(tmp_path / "service.db")
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO users (id, email, hashed_password,"
                    " is_admin, created_at) VALUES (?, 'u@example.com', 'x', 0, ?)",
                    (_USER_ID, "2020-01-01T00:00:00+00:00"),
                )
                for surface, key in _KEYS.items():
                    conn.execute(
                        "INSERT OR IGNORE INTO api_keys (id, user_id, key_hash, name,"
                        " created_at, surface) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            f"id-{key}",
                            _USER_ID,
                            hash_api_key(key),
                            key,
                            "2020-01-01T00:00:00+00:00",
                            surface,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()
            yield client
        app.dependency_overrides.clear()

    return _make


def _bearer(token: str, **extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", **extra}


def test_key_surfaces(
    wp_jwt_client: Callable[
        [str | None], contextlib.AbstractContextManager[TestClient]
    ],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="kb_service")
    with wp_jwt_client(None) as client:
        head = client.post(
            "/api/kb/store", json=_store_body(), headers=_bearer("kb_head")
        )
        assert head.json()["status"] == "queued", head.text
        auto = client.post(
            "/api/kb/store",
            json=_store_body(),
            headers=_bearer("kb_inter", **{"X-KB-Mode": "autonomous"}),
        )
        assert auto.json()["surface"] == "autonomous"
        null = client.post(
            "/api/kb/store",
            json=_store_body(short_title="N"),
            headers=_bearer("kb_null"),
        )
        assert null.json()["action"] == "created", null.text
    lines = [ln for ln in _wp_lines(caplog) if "outcome=queued" in ln]
    assert "source=key" in lines[0]
    assert "key_id=id-kb_head" in lines[0]
    assert "surface=autonomous source=header" in lines[1]
    assert _candidates(tmp_path)[0]["session_id"] == "key:id-kb_head"


def test_key_default_and_jwt(
    wp_jwt_client: Callable[
        [str | None], contextlib.AbstractContextManager[TestClient]
    ],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="kb_service")
    with wp_jwt_client("headless") as client:
        null = client.post(
            "/api/kb/store", json=_store_body(), headers=_bearer("kb_null")
        )
        assert null.json()["status"] == "queued", null.text
        jwt = client.post(
            "/api/kb/store",
            json=_store_body(short_title="J"),
            headers=_bearer(create_token(_USER_ID)),
        )
        assert jwt.json()["action"] == "created", jwt.text
    lines = _wp_lines(caplog)
    assert any("outcome=queued" in ln and "source=default" in ln for ln in lines)
    assert any("outcome=allowed" in ln and "source=jwt" in ln for ln in lines)


def test_minted_key_inherits_surface(
    wp_jwt_client: Callable[
        [str | None], contextlib.AbstractContextManager[TestClient]
    ],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="kb_service")
    with wp_jwt_client(None) as client:
        for name, headers in [
            ("from-head", _bearer("kb_head")),
            ("from-inter-auto", _bearer("kb_inter", **{"X-KB-Mode": "autonomous"})),
            ("from-jwt", _bearer(create_token(_USER_ID))),
            ("from-null", _bearer("kb_null")),
        ]:
            resp = client.post(
                "/api/auth/api-keys", json={"name": name}, headers=headers
            )
            assert resp.status_code == 201, resp.text
    rows = {
        r["name"]: r["surface"]
        for r in _service_rows(tmp_path, "SELECT name, surface FROM api_keys")
    }
    assert rows["from-head"] == "headless"
    assert rows["from-inter-auto"] == "autonomous"
    assert rows["from-jwt"] is None
    assert rows["from-null"] is None
    mint = [ln for ln in _wp_lines(caplog) if "op=mint_key" in ln]
    assert len(mint) == 4
    assert mint[0].endswith("reason=inherited_surface ua='testclient'")


# --- AC12 startup log ---------------------------------------------------------


def test_startup_logs(wp_client: WpClient, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="kb_service")
    with wp_client(None, "on"):
        pass
    assert (
        "write-policy started default_surface=interactive capture_mode=on"
        " worker=not_started keys=interactive:0,headless:0,autonomous:0,unset:0"
    ) in caplog.text
    assert "queue_unattended" not in caplog.text
    caplog.clear()
    with wp_client("headless", "on"):
        pass
    assert "write-policy queue_unattended reason=worker_not_started" in caplog.text


async def test_log_startup_counts_and_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=WP_LOGGER)

    class _CountPool:
        async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
            return [
                {"surface": "autonomous", "n": 2},
                {"surface": None, "n": 3},
            ]

    async def _get_db() -> _CountPool:
        return _CountPool()

    monkeypatch.setattr(database, "get_db", _get_db)
    await write_policy.log_startup("shadow", worker_started=True)
    assert "keys=interactive:0,headless:0,autonomous:2,unset:3" in caplog.text
    assert "queue_unattended" not in caplog.text

    async def _boom() -> Any:
        raise RuntimeError("no db")

    monkeypatch.setattr(database, "get_db", _boom)
    caplog.clear()
    await write_policy.log_startup("off", worker_started=False)
    assert "keys=unavailable" in caplog.text
    assert "queue_unattended" not in caplog.text


# --- AC13 route classification guard ------------------------------------------


def test_write_route_classification() -> None:
    found: dict[str, APIRoute] = {}
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path.startswith("/api/kb/"):
            for method in route.methods & {"POST", "PUT", "PATCH", "DELETE"}:
                found[f"{method} {route.path}"] = route
    assert set(found) == set(WRITE_ROUTE_CLASSES)
    assert set(WRITE_ROUTE_CLASSES.values()) <= {
        "policed",
        "admin_only",
        "machine_principal_only",
        "pipeline",
        "no_entry_write",
        "unpoliced",
    }
    policed = [k for k, v in WRITE_ROUTE_CLASSES.items() if v == "policed"]
    assert len(policed) == 8
    for key in policed:
        assert "resolve_write_context" in inspect.getsource(found[key].endpoint), key


# --- direct queue helpers -----------------------------------------------------


async def test_queue_store_interactive_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Pool:
        async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
            return {"id": 1}

    async def _get_db() -> _Pool:
        return _Pool()

    monkeypatch.setattr(database, "get_db", _get_db)
    with pytest.raises(RuntimeError, match="interactive surface"):
        await write_policy.queue_store(
            _wctx("interactive"),
            StoreRequest(**_store_body()),
            entry_type=EntryType.FACTUAL_REFERENCE,
            attr=Attribution(),
        )

    class _NonePool:
        async def fetchrow(self, sql: str, *args: Any) -> None:
            return None

    async def _get_none() -> _NonePool:
        return _NonePool()

    monkeypatch.setattr(database, "get_db", _get_none)
    with pytest.raises(RuntimeError, match="returned no id"):
        await write_policy.queue_store(
            _wctx("headless"),
            StoreRequest(**_store_body()),
            entry_type=EntryType.FACTUAL_REFERENCE,
            attr=Attribution(),
        )


# --- follow-ups: engine header, user agent, turn mode clamp -------------------


def test_engine_header_recorded_separately(
    wp_client: WpClient, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="kb_service")
    with wp_client("headless", "shadow") as (client, _llm, _critic):
        new = client.post(
            "/api/kb/store",
            json=_store_body(),
            headers={
                "X-KB-Harness": "claude-code",
                "X-KB-Engine": "claude-code sonnet!",
                "User-Agent": "personal-kb/9.9.9",
            },
        )
        assert new.json()["status"] == "queued"
        old = client.post(
            "/api/kb/store",
            json=_store_body(short_title="Old client"),
            headers={"X-KB-Harness": "claude-code-sonnet"},
        )
        assert old.json()["status"] == "queued"
    queued = [ln for ln in _wp_lines(caplog) if "outcome=queued" in ln]
    assert "harness='claude-code' engine='claude-codesonnet'" in queued[0]
    assert "ua='personal-kb/9.9.9'" in queued[0]
    assert "harness='claude-code-sonnet' engine=''" in queued[1]
    rows = _candidates(tmp_path)
    first = json.loads(rows[0]["detector_output"])
    assert (first["harness"], first["engine"]) == ("claude-code", "claude-codesonnet")
    second = json.loads(rows[1]["detector_output"])
    assert (second["harness"], second["engine"]) == ("claude-code-sonnet", "")


def _turn(mode: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "event_id": "s-1:0",
        "session_id": "s-1",
        "turn_index": 0,
        "project": "p",
        "user_prompt": "hello",
        "items": [],
    }
    if mode is not None:
        body["mode"] = mode
    return body


def _turn_mode(tmp_path: Path) -> str:
    rows = _service_rows(tmp_path, "SELECT mode FROM turn_events")
    assert len(rows) == 1
    return str(rows[0]["mode"])


@pytest.mark.parametrize(
    ("default", "posted", "stored", "clamped"),
    [
        ("headless", "interactive", "headless", True),
        ("headless", None, "headless", True),
        ("autonomous", "interactive", "headless", True),
        ("autonomous", "headless", "headless", False),
        ("interactive", "interactive", "interactive", False),
        ("interactive", "headless", "headless", False),
    ],
)
def test_turn_mode_clamped_to_surface(
    wp_client: WpClient,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    default: str,
    posted: str | None,
    stored: str,
    clamped: bool,
) -> None:
    caplog.set_level(logging.INFO, logger="kb_service")
    with wp_client(default, "shadow") as (client, _llm, _critic):
        resp = client.post("/api/kb/turn", json=_turn(posted))
        assert resp.status_code == 200, resp.text
        assert resp.json()["reason"] == "recorded"
    assert _turn_mode(tmp_path) == stored
    lines = [r.getMessage() for r in caplog.records if "mode_clamped" in r.getMessage()]
    if clamped:
        assert lines == [f"turn_event mode_clamped surface={default} from=interactive"]
    else:
        assert lines == []
