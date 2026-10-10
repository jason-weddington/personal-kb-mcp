"""AuthPrincipal resolution: one contract for REST and /mcp (Part A)."""

from typing import Annotated, Any

import pytest
from fastapi import Depends, HTTPException, Request
from fastapi.testclient import TestClient

import kb_service.auth as auth_module
from kb_service.auth import (
    AuthPrincipal,
    create_token,
    get_current_user,
    hash_api_key,
    resolve_principal,
)
from kb_service.main import app
from kb_service.models import User
from tests.conftest import StatefulFakeDbPool, fake_user


def _row(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "hashed_password": user.hashed_password,
        "is_admin": 1 if user.is_admin else 0,
        "created_at": user.created_at.isoformat(),
    }


@pytest.fixture
def pool(monkeypatch: pytest.MonkeyPatch) -> StatefulFakeDbPool:
    user = fake_user()
    shared = StatefulFakeDbPool(
        {},
        users={user.id: _row(user)},
        api_keys={
            hash_api_key("kb_test_user"): {"id": "key-user", "user_id": user.id},
            hash_api_key("kb_orphan"): {"id": "key-orphan", "user_id": "gone"},
        },
    )

    async def _get_db() -> StatefulFakeDbPool:
        return shared

    monkeypatch.setattr(auth_module, "get_db", _get_db)
    return shared


async def test_api_key_resolves_to_api_key_principal(
    pool: StatefulFakeDbPool,
) -> None:
    principal = await resolve_principal("kb_test_user")
    assert isinstance(principal, AuthPrincipal)
    assert principal.user.id == fake_user().id
    assert principal.api_key_id == "key-user"
    assert principal.auth_method == "api_key"


async def test_jwt_resolves_to_jwt_principal(pool: StatefulFakeDbPool) -> None:
    principal = await resolve_principal(create_token(fake_user().id))
    assert principal.user.id == fake_user().id
    assert principal.api_key_id is None
    assert principal.auth_method == "jwt"


async def test_no_auth_mode_resolves_to_local_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    principal = await resolve_principal(None)
    assert principal.auth_method == "none"
    assert principal.user.id == "local"
    assert principal.api_key_id is None


async def test_missing_token_is_401(pool: StatefulFakeDbPool) -> None:
    with pytest.raises(HTTPException) as exc:
        await resolve_principal(None)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Not authenticated"


async def test_unknown_key_is_401_invalid_api_key(pool: StatefulFakeDbPool) -> None:
    with pytest.raises(HTTPException) as exc:
        await resolve_principal("kb_nope")
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid API key"


async def test_key_without_user_is_401_user_not_found(
    pool: StatefulFakeDbPool,
) -> None:
    with pytest.raises(HTTPException) as exc:
        await resolve_principal("kb_orphan")
    assert exc.value.status_code == 401
    assert exc.value.detail == "User not found"


def test_rest_route_sets_request_state_principal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Depends(get_current_user) route sees request.state.kb_principal."""
    user = fake_user()
    shared = StatefulFakeDbPool(
        {},
        users={user.id: _row(user)},
        api_keys={hash_api_key("kb_test_user"): {"id": "key-user", "user_id": user.id}},
    )

    async def _get_db() -> StatefulFakeDbPool:
        return shared

    monkeypatch.setattr(auth_module, "get_db", _get_db)

    seen: list[AuthPrincipal] = []

    async def _probe(
        request: Request, _user: Annotated[User, Depends(get_current_user)]
    ) -> dict[str, str]:
        seen.append(request.state.kb_principal)
        return {"ok": "yes"}

    app.add_api_route("/api/_test_principal_probe", _probe, methods=["GET"])
    # Move ahead of the SPA catch-all (if the UI is mounted).
    app.router.routes.insert(0, app.router.routes.pop())
    try:
        resp = client.get(
            "/api/_test_principal_probe",
            headers={"Authorization": "Bearer kb_test_user"},
        )
    finally:
        app.router.routes[:] = [
            r
            for r in app.router.routes
            if getattr(r, "path", None) != "/api/_test_principal_probe"
        ]
    assert resp.status_code == 200
    assert seen == [AuthPrincipal(user, "key-user", "api_key")]
