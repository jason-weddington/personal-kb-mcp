"""Read-only mode (``KB_READ_ONLY``): refusals, open reads, coverage guards.

The kb-core data DB is a ``FakeKnowledgeBase`` here, so "the DB is unchanged"
means every recorded write call on the fake stays empty. The surprise-worker
case lives in ``test_surprise_dry_run.py`` (it needs that module's fixtures).
"""

import logging
from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import kb_service.mcp_server.backend as backend_module
from kb_service import write_policy
from kb_service.chat import ChatSession
from kb_service.main import app
from kb_service.mcp_server.backend import WRITE_METHODS
from kb_service.read_only import (
    READ_ONLY_ALLOWLIST,
    READ_ONLY_EXTRA_GUARDED,
    READ_ONLY_GUARDED_ROUTES,
    require_kb_writable,
)
from tests.conftest import FakeKnowledgeBase, make_entry
from tests.test_mcp_endpoint import ADMIN
from tests.test_mcp_tools import call, text

POINTER = "see https://kb.example.com"
BASE = "read-only: this KB is read-only; writes are refused."

_ENTRY = {
    "short_title": "s",
    "long_title": "l",
    "knowledge_details": "d",
    "supersedes": "none",
}

# (method, path as called, kwargs for client.request)
HTTP_WRITE_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("POST", "/api/kb/store", {"json": _ENTRY}),
    ("POST", "/api/kb/store_batch", {"json": {"entries": [_ENTRY]}}),
    ("POST", "/api/kb/entries/kb-00001/deactivate", {"json": {"change_reason": "r"}}),
    ("POST", "/api/kb/entries/kb-00001/reactivate", {"json": {"change_reason": "r"}}),
    (
        "POST",
        "/api/kb/bulk_update",
        {"json": {"filters": {"project_ref": "p"}, "updates": {"tags": ["x"]}}},
    ),
    ("POST", "/api/kb/admin/reconcile-supersession", {}),
    ("POST", "/api/kb/ingest/text", {"json": {"content": "x", "filename": "a.md"}}),
    ("POST", "/api/kb/ingest/url", {"json": {"url": "https://kb.example.com/a"}}),
    (
        "POST",
        "/api/kb/ingest/url",
        {"json": {"url": "https://kb.example.com/a", "dry_run": True}},
    ),
    ("POST", "/api/kb/ingest/file", {"files": {"file": ("a.md", b"# hi")}}),
    ("POST", "/api/kb/map-op", {"json": {}}),
    ("DELETE", "/api/kb/maps/kb-00001", {}),
    (
        "POST",
        "/api/kb/map-eligibility/override",
        {"json": {"project_ref": "p", "eligible": True, "reason": "r"}},
    ),
    ("POST", "/api/kb/map-eligibility/override/clear", {"json": {"project_ref": "p"}}),
    ("POST", "/api/kb/cluster-ledger/decline", {"json": {}}),
    ("POST", "/api/kb/cluster-ledger/match", {"json": {}}),
    ("POST", "/api/kb/surprise/drain", {}),
]

WRITE_TOOL_CASES: list[tuple[str, dict[str, Any]]] = [
    ("kb_store", _ENTRY),
    ("kb_store_batch", {"entries": [_ENTRY]}),
    ("kb_ingest_url", {"url": "https://kb.example.com/a", "content": "# hi"}),
    (
        "kb_map_eligibility_override",
        {"project_ref": "p", "eligible": True, "reason": "r"},
    ),
    ("kb_map_eligibility_override", {"project_ref": "p", "clear": True}),
    (
        "kb_maintain",
        {"action": "deactivate", "entry_id": "kb-00001", "change_reason": "r"},
    ),
    ("kb_maintain", {"action": "reactivate", "entry_id": "kb-00001"}),
    ("kb_maintain", {"action": "reconcile_supersession"}),
    (
        "kb_bulk_update",
        {"filters": {"project_ref": "p"}, "updates": {"tags": ["x"]}},
    ),
]


def _write_footprint(kb: FakeKnowledgeBase) -> list[Any]:
    return [
        kb.store_calls,
        kb.store_batch_calls,
        kb.update_calls,
        kb.deactivate_calls,
        kb.reactivate_calls,
        kb.bulk_update_calls,
        kb.reconcile_calls,
        kb.record_cluster_calls,
        kb.set_map_eligibility_override_calls,
        kb.clear_map_eligibility_override_calls,
        kb.audit_events,
    ]


def _snapshot(kb: FakeKnowledgeBase) -> list[Any]:
    return [list(x) if isinstance(x, list) else x for x in _write_footprint(kb)]


def _assert_untouched(kb: FakeKnowledgeBase, before: list[Any]) -> None:
    assert _snapshot(kb) == before


@pytest.fixture
def ro(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_READ_ONLY", "TRUE")
    monkeypatch.setenv("KB_READ_ONLY_MESSAGE", POINTER)


@pytest.fixture
def authed(client: TestClient) -> TestClient:
    from kb_service.auth import get_current_user
    from tests.conftest import fake_admin_user

    app.dependency_overrides[get_current_user] = fake_admin_user
    return client


# --- (a) every guarded HTTP route refuses ---


@pytest.mark.parametrize(("method", "path", "kw"), HTTP_WRITE_CASES)
def test_http_write_routes_refused(
    authed: TestClient,
    ro: None,
    caplog: pytest.LogCaptureFixture,
    method: str,
    path: str,
    kw: dict[str, Any],
) -> None:
    caplog.set_level(logging.INFO)
    before = _snapshot(app.state.kb)
    resp = authed.request(method, path, **kw)
    assert resp.status_code == 403, resp.text
    detail = resp.json()["detail"]
    assert "read-only" in detail and "https://kb.example.com" in detail
    _assert_untouched(app.state.kb, before)
    refusals = [r for r in caplog.records if "read-only-refused" in r.getMessage()]
    assert len(refusals) == 1
    assert "surface=http" in refusals[0].getMessage()


def test_http_unauthenticated_gets_401_first(client: TestClient, ro: None) -> None:
    assert client.post("/api/kb/store", json=_ENTRY).status_code == 401


def test_http_writes_not_refused_when_off(
    authed: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    for value in (None, "FALSE"):
        if value is None:
            monkeypatch.delenv("KB_READ_ONLY", raising=False)
        else:
            monkeypatch.setenv("KB_READ_ONLY", value)
        resp = authed.post("/api/kb/store", json=_ENTRY)
        assert "read-only" not in resp.text
    assert "read-only-refused" not in caplog.text


def test_message_empty_gives_exact_base_detail(
    authed: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_READ_ONLY", "TRUE")
    monkeypatch.setenv("KB_READ_ONLY_MESSAGE", "   ")
    resp = authed.post("/api/kb/store", json=_ENTRY)
    assert resp.status_code == 403
    assert resp.json()["detail"] == BASE


# --- MCP tools ---


@pytest.mark.parametrize(
    ("tool", "args"),
    WRITE_TOOL_CASES,
    ids=[f"{t}-{i}" for i, (t, _) in enumerate(WRITE_TOOL_CASES)],
)
def test_mcp_write_tools_refused(
    mcp_client: TestClient,
    fake_kb: FakeKnowledgeBase,
    ro: None,
    caplog: pytest.LogCaptureFixture,
    tool: str,
    args: dict[str, Any],
) -> None:
    caplog.set_level(logging.INFO)
    before = _snapshot(fake_kb)
    out = text(call(mcp_client, tool, args, ADMIN))
    assert out.startswith("Error: read-only: "), out
    assert POINTER in out
    _assert_untouched(fake_kb, before)
    refusals = [r for r in caplog.records if "read-only-refused" in r.getMessage()]
    assert len(refusals) == 1
    assert "surface=mcp" in refusals[0].getMessage()


def test_mcp_write_tools_not_refused_when_off(
    mcp_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    for tool, args in WRITE_TOOL_CASES:
        assert "read-only" not in text(call(mcp_client, tool, args, ADMIN))
    assert "read-only-refused" not in caplog.text


def test_mcp_write_tool_cases_cover_write_methods(
    mcp_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: set[str] = set()

    def _wrap(name: str, orig: Any) -> Any:
        async def _inner(*a: Any, **kw: Any) -> Any:
            recorded.add(name)
            return await orig(*a, **kw)

        return _inner

    for name in WRITE_METHODS:
        monkeypatch.setattr(
            backend_module.InProcessBackend,
            name,
            _wrap(name, getattr(backend_module.InProcessBackend, name)),
        )
    for tool, args in WRITE_TOOL_CASES:
        call(mcp_client, tool, args, ADMIN)
    assert recorded == WRITE_METHODS


# --- (d) reads keep working ---


READ_HTTP_CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("POST", "/api/kb/search", {"json": {"query": "x"}}),
    ("POST", "/api/kb/get", {"json": {"entry_ids": ["kb-00001"]}}),
    ("POST", "/api/kb/feedback", {"json": {"feedback_type": "friction"}}),
    ("GET", "/api/kb/projects", {}),
    ("POST", "/api/kb/map-lint", {"json": {}}),
]


@pytest.mark.parametrize(("method", "path", "kw"), READ_HTTP_CASES)
def test_reads_not_refused(
    authed: TestClient, ro: None, method: str, path: str, kw: dict[str, Any]
) -> None:
    resp = authed.request(method, path, **kw)
    assert "read-only" not in resp.text, resp.text
    assert resp.status_code != 403


def test_auth_routes_not_refused(client: TestClient, ro: None) -> None:
    for path, body in (
        ("/api/auth/login", {"email": "a@example.com", "password": "x"}),
        ("/api/auth/api-keys", {"name": "k"}),
    ):
        resp = client.post(path, json=body)
        assert "read-only" not in resp.text


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("kb_search", {"query": "x"}),
        ("kb_get", {"entry_ids": ["kb-00001"]}),
        ("kb_preflight", {"project_ref": "p"}),
        ("kb_list_projects", {}),
        ("kb_explore", {"entry_id": "kb-00001"}),
        ("kb_ask", {"query": "x"}),
        ("kb_summarize", {"query": "x"}),
        ("kb_feedback", {"feedback_type": "friction", "query": "q"}),
    ],
)
def test_mcp_reads_not_refused(
    mcp_client: TestClient, ro: None, tool: str, args: dict[str, Any]
) -> None:
    result = call(mcp_client, tool, args, ADMIN)
    assert "read-only" not in text(result)


def test_get_does_not_touch_accessed_when_read_only(
    authed: TestClient, ro: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kb_service.routes import kb_read_routes

    touched: list[Any] = []

    async def _touch(db: Any, ids: Any) -> None:
        touched.append(ids)

    monkeypatch.setattr(kb_read_routes, "touch_accessed", _touch)
    kb: FakeKnowledgeBase = app.state.kb
    kb.get_entry_result = make_entry()  # type: ignore[attr-defined]
    authed.post("/api/kb/get", json={"entry_ids": ["kb-00001"]})
    assert touched == []


# --- (e) coverage guard ---


def _non_get_routes() -> list[tuple[str, str, APIRoute]]:
    out = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.path.startswith("/api/"):
            continue
        for m in route.methods - {"GET", "HEAD", "OPTIONS"}:
            out.append((m, route.path, route))
    return out


def _carries_guard(route: APIRoute) -> bool:
    stack = list(route.dependant.dependencies)
    while stack:
        dep = stack.pop()
        if dep.call is require_kb_writable:
            return True
        stack.extend(dep.dependencies)
    return False


def test_every_write_route_is_guarded_or_allowlisted() -> None:
    routes = _non_get_routes()
    guarded = {
        (m, p) for m, p, r in routes if p.startswith("/api/kb") and _carries_guard(r)
    }
    assert guarded == READ_ONLY_GUARDED_ROUTES

    policy_guarded = {
        tuple(k.split(" ", 1))
        for k, cls in write_policy.WRITE_ROUTE_CLASSES.items()
        if cls in {"policed", "admin_only", "machine_principal_only", "pipeline"}
    }
    assert policy_guarded | READ_ONLY_EXTRA_GUARDED == READ_ONLY_GUARDED_ROUTES

    kb_routes_ = {(m, p) for m, p, _ in routes if p.startswith("/api/kb")}
    assert kb_routes_ >= READ_ONLY_GUARDED_ROUTES

    others = {(m, p) for m, p, _ in routes if not p.startswith("/api/kb")}
    assert others == READ_ONLY_ALLOWLIST, (
        f"unclassified: {sorted(others - READ_ONLY_ALLOWLIST)};"
        f" stale: {sorted(READ_ONLY_ALLOWLIST - others)}"
    )
    assert len(READ_ONLY_GUARDED_ROUTES) == 16


# --- chat tools ---


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("_tool_update_entry", {"entry_id": "kb-00001", "knowledge_details": "x"}),
        ("_tool_ingest_url", {"url": "https://kb.example.com/a"}),
    ],
)
async def test_chat_write_tools_refused(
    ro: None,
    caplog: pytest.LogCaptureFixture,
    method: str,
    args: dict[str, Any],
) -> None:
    caplog.set_level(logging.INFO)
    kb = FakeKnowledgeBase(results=[], filtered_count=0)
    session = ChatSession.__new__(ChatSession)
    session.kb = kb  # type: ignore[assignment]
    before = _snapshot(kb)
    result = await getattr(session, method)(args)
    assert result.success is False
    assert result.message.startswith("read-only: ")
    _assert_untouched(kb, before)
    assert caplog.text.count("read-only-refused surface=chat") == 1


# --- startup logging ---


def test_startup_lines(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from kb_service.main import _log_read_only_startup

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("KB_READ_ONLY", "TRUE")
    monkeypatch.setenv("KB_READ_ONLY_MESSAGE", POINTER)
    _log_read_only_startup()
    assert (
        "read-only mode=on message_set=true"
        " background_writers=embedding_worker,supersession_reconcile"
    ) in caplog.text
    caplog.clear()
    monkeypatch.setenv("KB_READ_ONLY", "maybe")
    monkeypatch.delenv("KB_READ_ONLY_MESSAGE")
    _log_read_only_startup()
    assert "read-only unrecognised value=maybe" in caplog.text
    assert "read-only mode=off message_set=false" in caplog.text
