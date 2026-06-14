"""Tests for the KB service shell: health, auth gate, and search serialization."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

import kb_service.main as main_module
from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, fake_user


def test_health(client: TestClient) -> None:
    """GET /api/health returns 200 with the ok status body."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_requires_auth(client: TestClient) -> None:
    """POST /api/kb/search with no Authorization header is rejected.

    FastAPI 0.136+ HTTPBearer(auto_error=True) returns 401 for missing
    credentials (older 0.115-era versions returned 403). Either way the
    endpoint is gated; we assert the gate fires with an unauthorized status.
    """
    resp = client.post("/api/kb/search", json={"query": "anything"})
    assert resp.status_code in (401, 403)


def test_search_bad_token(client: TestClient) -> None:
    """POST /api/kb/search with a malformed bearer token is rejected (401)."""
    resp = client.post(
        "/api/kb/search",
        json={"query": "anything"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert resp.status_code == 401


def test_search_empty_results(client: TestClient) -> None:
    """An authed search with an empty mock returns an empty, well-shaped body."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/search", json={"query": "anything"})
    assert resp.status_code == 200
    assert resp.json() == {"results": [], "filtered_count": 0}


@pytest.mark.parametrize("fake_kb", ["one_result"], indirect=True)
def test_search_one_result(client: TestClient) -> None:
    """An authed search returning one result serializes len==1, filtered_count==1."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/search", json={"query": "anything"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 1
    assert body["filtered_count"] == 1


def test_search_passes_contributor_telemetry(client: TestClient) -> None:
    """The user's email is forwarded as the telemetry contributor kwarg."""
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.post("/api/kb/search", json={"query": "x"})
    assert resp.status_code == 200
    kb: FakeKnowledgeBase = app.state.kb
    assert kb.search_calls
    _query, contributor = kb.search_calls[-1]
    assert contributor == "tester@example.com"


def test_routers_mounted() -> None:
    """The auth, admin, and kb routers are all mounted on the app."""
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert "/api/auth/login" in paths
    assert "/api/admin/invites" in paths
    assert "/api/kb/search" in paths
    assert "/api/health" in paths


# --- No-auth (KB_AUTH_MODE=none) single-user mode ---------------------------


def test_no_auth_search_no_header_uses_synthetic_contributor(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In no-auth mode, search succeeds with no Authorization header and
    forwards the synthetic local user's email as the telemetry contributor.

    Crucially, this does NOT override get_current_user — the real dependency
    must return the synthetic user on its own.
    """
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    resp = client.post("/api/kb/search", json={"query": "anything"})
    assert resp.status_code == 200
    kb: FakeKnowledgeBase = app.state.kb
    assert kb.search_calls
    _query, contributor = kb.search_calls[-1]
    assert contributor == "local@localhost"


def test_runtime_reports_none_mode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /api/kb/runtime returns {'auth': 'none'} in no-auth mode."""
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    resp = client.get("/api/kb/runtime")
    assert resp.status_code == 200
    assert resp.json() == {"auth": "none"}


def test_runtime_reports_jwt_mode_by_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /api/kb/runtime returns {'auth': 'jwt'} when KB_AUTH_MODE is unset."""
    monkeypatch.delenv("KB_AUTH_MODE", raising=False)
    resp = client.get("/api/kb/runtime")
    assert resp.status_code == 200
    assert resp.json() == {"auth": "jwt"}


def test_jwt_mode_search_still_requires_auth(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In the default jwt mode, search with no Authorization header still 401s.

    Preserves the gate semantics of test_search_requires_auth.
    """
    monkeypatch.delenv("KB_AUTH_MODE", raising=False)
    resp = client.post("/api/kb/search", json={"query": "anything"})
    assert resp.status_code == 401


def test_invalid_auth_mode_surfaces_value_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid KB_AUTH_MODE surfaces a ValueError listing valid choices."""
    monkeypatch.setenv("KB_AUTH_MODE", "off")
    with pytest.raises(ValueError, match="Choose from: jwt, none"):
        client.get("/api/kb/runtime")


# --- _open_kb branch selection ---------------------------------------------


async def test_open_kb_postgres_branch_when_url_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``KB_DATABASE_URL`` set -> ``create_postgres`` runs, ``create_sqlite`` does NOT.

    This pins the branch logic in ``_open_kb`` so a regression that flips
    the branch (or accidentally calls both factories) trips the test.
    Mirrors the setenv/delenv pattern used in the no-auth tests above.
    """
    postgres_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    sqlite_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def _fake_create_postgres(*args: Any, **kwargs: Any) -> FakeKnowledgeBase:
        postgres_calls.append((args, kwargs))
        return FakeKnowledgeBase(results=[], filtered_count=0)

    async def _fake_create_sqlite(*args: Any, **kwargs: Any) -> FakeKnowledgeBase:
        sqlite_calls.append((args, kwargs))
        return FakeKnowledgeBase(results=[], filtered_count=0)

    monkeypatch.setattr(main_module, "create_postgres", _fake_create_postgres)
    monkeypatch.setattr(main_module, "create_sqlite", _fake_create_sqlite)
    monkeypatch.setenv("KB_DATABASE_URL", "postgresql://x/y")

    kb = await main_module._open_kb()

    assert isinstance(kb, FakeKnowledgeBase)
    assert len(postgres_calls) == 1
    assert postgres_calls[0][0] == ("postgresql://x/y",)
    assert sqlite_calls == []


async def test_open_kb_sqlite_branch_with_default_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``KB_DATABASE_URL`` unset AND ``KB_DB_PATH`` unset -> ``create_sqlite``
    is called with the literal default path ``~/.local/share/personal_kb/
    knowledge.db`` (RAW string — main.py does NOT expanduser; kb-core does).
    """
    postgres_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    sqlite_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def _fake_create_postgres(*args: Any, **kwargs: Any) -> FakeKnowledgeBase:
        postgres_calls.append((args, kwargs))
        return FakeKnowledgeBase(results=[], filtered_count=0)

    async def _fake_create_sqlite(*args: Any, **kwargs: Any) -> FakeKnowledgeBase:
        sqlite_calls.append((args, kwargs))
        return FakeKnowledgeBase(results=[], filtered_count=0)

    monkeypatch.setattr(main_module, "create_postgres", _fake_create_postgres)
    monkeypatch.setattr(main_module, "create_sqlite", _fake_create_sqlite)
    monkeypatch.delenv("KB_DATABASE_URL", raising=False)
    monkeypatch.delenv("KB_DB_PATH", raising=False)

    kb = await main_module._open_kb()

    assert isinstance(kb, FakeKnowledgeBase)
    assert len(sqlite_calls) == 1
    args, kwargs = sqlite_calls[0]
    assert args == ("~/.local/share/personal_kb/knowledge.db",)
    # create_sqlite must NOT receive pool_min/pool_max (its signature has none).
    assert "pool_min" not in kwargs
    assert "pool_max" not in kwargs
    assert postgres_calls == []


async def test_open_kb_sqlite_branch_when_url_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``KB_DATABASE_URL`` set but EMPTY also falls through to SQLite.

    The branch test is ``if database_url:`` — both ``None`` and ``""`` go to
    SQLite, matching the AC pin (`set AND non-empty -> Postgres`).
    """
    postgres_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    sqlite_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def _fake_create_postgres(*args: Any, **kwargs: Any) -> FakeKnowledgeBase:
        postgres_calls.append((args, kwargs))
        return FakeKnowledgeBase(results=[], filtered_count=0)

    async def _fake_create_sqlite(*args: Any, **kwargs: Any) -> FakeKnowledgeBase:
        sqlite_calls.append((args, kwargs))
        return FakeKnowledgeBase(results=[], filtered_count=0)

    monkeypatch.setattr(main_module, "create_postgres", _fake_create_postgres)
    monkeypatch.setattr(main_module, "create_sqlite", _fake_create_sqlite)
    monkeypatch.setenv("KB_DATABASE_URL", "")
    monkeypatch.delenv("KB_DB_PATH", raising=False)

    await main_module._open_kb()

    assert len(sqlite_calls) == 1
    assert sqlite_calls[0][0] == ("~/.local/share/personal_kb/knowledge.db",)
    assert postgres_calls == []


async def test_open_kb_sqlite_uses_kb_db_path_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``KB_DB_PATH`` overrides the default and is forwarded raw."""
    sqlite_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def _fake_create_sqlite(*args: Any, **kwargs: Any) -> FakeKnowledgeBase:
        sqlite_calls.append((args, kwargs))
        return FakeKnowledgeBase(results=[], filtered_count=0)

    monkeypatch.setattr(main_module, "create_sqlite", _fake_create_sqlite)
    monkeypatch.delenv("KB_DATABASE_URL", raising=False)
    monkeypatch.setenv("KB_DB_PATH", "/var/data/custom_kb.db")

    await main_module._open_kb()

    assert len(sqlite_calls) == 1
    assert sqlite_calls[0][0] == ("/var/data/custom_kb.db",)
