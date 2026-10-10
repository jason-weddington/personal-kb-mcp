"""InProcessBackend: handler kwargs, admin gate, drift guards, errors (B14b)."""

import inspect
import logging
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute
from kb_core.models.search import SearchQuery
from starlette.requests import Request

import kb_service.auth as auth_module
import kb_service.mcp_server.context as context_module
from kb_service.auth import AuthPrincipal
from kb_service.main import app
from kb_service.mcp_server.backend import ADMIN_ONLY_METHODS, InProcessBackend
from kb_service.mcp_server.errors import BackendHttpError
from kb_service.mcp_server.observability import (
    MCP_BACKEND_MARKER,
    _backend_statuses,
)
from kb_service.routes import (
    ingest_routes,
    kb_read_routes,
    kb_routes,
    kb_write_routes,
    map_eligibility_routes,
    query_routes,
)
from tests.conftest import fake_admin_user, fake_user

_ENTRY = {
    "short_title": "a",
    "long_title": "b",
    "knowledge_details": "c",
    "supersedes": "none",
}

# method -> (module, handler attr, user kwarg, args, kwargs)
METHODS: dict[str, tuple[ModuleType, str, str, tuple[Any, ...], dict[str, Any]]] = {
    "search": (kb_routes, "search", "user", (SearchQuery(query="x"),), {}),
    "get_entries": (kb_read_routes, "get_entries", "user", (["kb-00001"],), {}),
    "neighbors": (kb_read_routes, "graph_neighbors", "user", ("kb-00001",), {}),
    "store": (
        kb_write_routes,
        "store",
        "user",
        (),
        {
            "short_title": "s",
            "long_title": "l",
            "knowledge_details": "d",
            "supersedes": "none",
        },
    ),
    "deactivate": (
        kb_write_routes,
        "deactivate",
        "user",
        ("kb-00001",),
        {"change_reason": "r"},
    ),
    "reactivate": (kb_write_routes, "reactivate", "user", ("kb-00001",), {}),
    "store_batch": (kb_write_routes, "store_batch", "user", ([_ENTRY],), {}),
    "bulk_update": (
        kb_write_routes,
        "bulk_update",
        "user",
        ({"project_ref": "p"}, {"tags": ["x"]}, True),
        {},
    ),
    "reconcile_supersession": (
        kb_write_routes,
        "reconcile_supersession",
        "user",
        (),
        {},
    ),
    "feedback": (kb_write_routes, "feedback", "user", ("friction",), {}),
    "ask_auto": (query_routes, "ask", "_user", ("q", None, True, 5), {}),
    "summarize": (query_routes, "summarize", "_user", ("q", None, 5), {}),
    "preflight": (kb_read_routes, "preflight", "user", ("p", None), {}),
    "ingest_url": (
        ingest_routes,
        "ingest_url",
        "user",
        ("https://example.com", None, None, True),
        {},
    ),
    "list_projects": (kb_read_routes, "list_projects", "user", (), {}),
    "list_contributors": (kb_read_routes, "list_contributors", "user", (), {}),
    "list_teams": (kb_read_routes, "list_teams", "user", (), {}),
    "map_eligibility": (
        map_eligibility_routes,
        "map_eligibility",
        "_admin",
        (),
        {},
    ),
    "set_map_eligibility_override": (
        map_eligibility_routes,
        "set_override",
        "user",
        ("p",),
        {"eligible": True, "reason": "r"},
    ),
    "clear_map_eligibility_override": (
        map_eligibility_routes,
        "clear_override",
        "user",
        ("p",),
        {},
    ),
}

HEADERS = [(b"x-kb-mode", b"headless"), (b"user-agent", b"pytest")]


def _backend(admin: bool = True) -> tuple[InProcessBackend, AuthPrincipal]:
    user = fake_admin_user() if admin else fake_user()
    principal = AuthPrincipal(user, "key-x", "api_key")
    return InProcessBackend(app=app, principal=principal, raw_headers=HEADERS), (
        principal
    )


def test_method_table_covers_twenty_handlers() -> None:
    assert len(METHODS) == 20
    assert set(METHODS) >= ADMIN_ONLY_METHODS


# ─── (1) kwargs completeness ─────────────────────────────────────────────────


@pytest.mark.parametrize("method", sorted(METHODS))
async def test_handler_receives_every_parameter_explicitly(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, attr, user_kw, args, kwargs = METHODS[method]
    real = getattr(module, attr)
    seen: dict[str, Any] = {}

    async def _spy(**kw: Any) -> Any:
        seen.update(kw)
        raise HTTPException(418, "spy")

    monkeypatch.setattr(module, attr, _spy)
    backend, principal = _backend(admin=True)
    with pytest.raises(BackendHttpError) as exc:
        await getattr(backend, method)(*args, **kwargs)
    assert exc.value.status == 418
    assert exc.value.detail == "spy"
    assert set(seen) == set(inspect.signature(real).parameters)
    assert seen[user_kw] is principal.user
    request = seen["request"]
    assert isinstance(request, Request)
    assert request.state.kb_principal is principal
    assert request.headers.get("x-kb-mode") == "headless"
    assert request.app is app


# ─── (2) admin gate ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", sorted(ADMIN_ONLY_METHODS))
async def test_admin_only_methods_reject_non_admin_before_handler(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module, attr, _user_kw, args, kwargs = METHODS[method]
    called: list[bool] = []

    async def _spy(**kw: Any) -> Any:
        called.append(True)

    monkeypatch.setattr(module, attr, _spy)
    caplog.set_level(logging.WARNING, logger="kb_service")
    backend, _principal = _backend(admin=False)
    with pytest.raises(BackendHttpError) as exc:
        await getattr(backend, method)(*args, **kwargs)
    assert (exc.value.status, exc.value.detail) == (403, "Admin only")
    assert called == []
    assert any(
        f"{MCP_BACKEND_MARKER} op={method} status=403 reason=admin_gate"
        in r.getMessage()
        for r in caplog.records
    )


# ─── (3)/(4) drift guards ────────────────────────────────────────────────────


def _dependency_calls(handler: Callable[..., Any]) -> set[Any]:
    routes = [
        r for r in app.routes if isinstance(r, APIRoute) and r.endpoint is handler
    ]
    assert routes, f"no APIRoute for {handler.__name__}"
    calls: set[Any] = set()
    stack = list(routes[0].dependant.dependencies)
    while stack:
        dep = stack.pop()
        if dep.call is not None:
            calls.add(dep.call)
        stack.extend(dep.dependencies)
    return calls


def test_admin_drift_guard() -> None:
    gated = {
        method
        for method, (module, attr, *_rest) in METHODS.items()
        if auth_module.require_admin in _dependency_calls(getattr(module, attr))
    }
    assert gated == ADMIN_ONLY_METHODS


@pytest.mark.parametrize("method", sorted(METHODS))
def test_dependency_drift_guard(method: str) -> None:
    module, attr, *_rest = METHODS[method]
    calls = _dependency_calls(getattr(module, attr))
    expected = {
        auth_module.get_current_user,
        auth_module.get_current_principal,
        auth_module._bearer,
    }
    if method in ADMIN_ONLY_METHODS:
        expected.add(auth_module.require_admin)
    for call in calls - expected:
        name = getattr(call, "__name__", repr(call))
        pytest.fail(f"InProcessBackend must replicate new dependency {name} for {attr}")
    assert calls == expected


# ─── (5) error mapping and status recording ──────────────────────────────────


async def test_dict_detail_is_json_encoded(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(**kw: Any) -> Any:
        raise HTTPException(409, detail={"a": 1})

    monkeypatch.setattr(kb_write_routes, "store", _boom)
    backend, _ = _backend()
    with pytest.raises(BackendHttpError) as exc:
        await backend.store(short_title="s", supersedes="none")
    assert (exc.value.status, exc.value.detail) == (409, '{"a": 1}')
    assert str(exc.value) == 'KB service returned 409: {"a": 1}'


async def test_request_model_validation_error_is_422() -> None:
    backend, _ = _backend()
    with pytest.raises(BackendHttpError) as exc:
        await backend.feedback("bogus")
    assert exc.value.status == 422
    assert "feedback_type" in exc.value.detail


async def test_unexpected_exception_is_500_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _boom(**kw: Any) -> Any:
        raise RuntimeError("kaput")

    monkeypatch.setattr(kb_read_routes, "list_projects", _boom)
    caplog.set_level(logging.ERROR, logger="kb_service")
    backend, _ = _backend()
    with pytest.raises(BackendHttpError) as exc:
        await backend.list_projects()
    assert (exc.value.status, exc.value.detail) == (500, "Internal Server Error")
    records = [r for r in caplog.records if r.exc_info]
    assert len(records) == 1
    assert (
        f"{MCP_BACKEND_MARKER} op=list_projects status=500" in records[0].getMessage()
    )


async def test_each_call_records_its_status(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _ok(**kw: Any) -> Any:
        return {"items": [{"name": "p", "entry_count": 2}]}

    async def _gone(**kw: Any) -> Any:
        raise HTTPException(404, "nope")

    monkeypatch.setattr(kb_read_routes, "list_projects", _ok)
    monkeypatch.setattr(kb_read_routes, "list_teams", _gone)
    statuses: list[int] = []
    token = _backend_statuses.set(statuses)
    try:
        backend, _ = _backend()
        assert await backend.list_projects() == [("p", 2)]
        with pytest.raises(BackendHttpError):
            await backend.list_teams()
        assert await backend.vector_search_available() is True
    finally:
        _backend_statuses.reset(token)
    assert statuses == [200, 404, 200]


async def test_store_batch_empty_makes_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _spy(**kw: Any) -> Any:
        raise AssertionError("should not be called")

    monkeypatch.setattr(kb_write_routes, "store_batch", _spy)
    backend, _ = _backend()
    assert await backend.store_batch([]) == ([], [], [])


async def test_get_entries_batches_by_twenty(monkeypatch: pytest.MonkeyPatch) -> None:
    batches: list[list[str]] = []

    async def _get(**kw: Any) -> Any:
        ids = list(kw["body"].ids)
        batches.append(ids)
        return {"results": [{"id": i, "found": False} for i in ids]}

    monkeypatch.setattr(kb_read_routes, "get_entries", _get)
    backend, _ = _backend()
    ids = [f"kb-{n:05d}" for n in range(25)]
    out = await backend.get_entries(ids)
    assert [len(b) for b in batches] == [20, 5]
    assert [eid for eid, _e, _r in out] == ids


# ─── (6) fail closed ─────────────────────────────────────────────────────────


def test_backend_for_request_fails_closed_without_principal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    request = Request({"type": "http", "headers": [], "state": {"kb_app": app}})
    monkeypatch.setattr(context_module, "get_http_request", lambda: request)
    caplog.set_level(logging.ERROR, logger="kb_service")
    with pytest.raises(
        RuntimeError, match="MCP tool call reached without an authenticated principal"
    ):
        context_module.backend_for_request()
    assert any(
        "mcp-principal-missing has_principal=False has_app=True" in r.getMessage()
        for r in caplog.records
    )


def test_backend_for_request_fails_closed_outside_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _none() -> Request:
        raise RuntimeError("No active HTTP request found.")

    monkeypatch.setattr(context_module, "get_http_request", _none)
    with pytest.raises(RuntimeError, match="without an authenticated principal"):
        context_module.backend_for_request()


def test_backend_for_request_builds_backend_for_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    principal = AuthPrincipal(fake_user(), None, "jwt")
    request = Request(
        {
            "type": "http",
            "headers": HEADERS,
            "state": {"kb_app": app, "kb_principal": principal},
        }
    )
    monkeypatch.setattr(context_module, "get_http_request", lambda: request)
    backend = context_module.backend_for_request()
    assert isinstance(backend, InProcessBackend)
    assert backend._principal is principal
