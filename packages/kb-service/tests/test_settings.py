"""Hermetic tests for GET/PUT /api/settings and the resolve_attribution seam."""

import pytest
from fastapi.testclient import TestClient
from kb_core import Attribution

import kb_service.attribution as attribution_module
from kb_service.attribution import is_machine_principal, resolve_attribution
from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import StatefulFakeDbPool, fake_admin_user, fake_user

# ---------------------------------------------------------------------------
# (a) GET default — no row stored
# ---------------------------------------------------------------------------


def test_get_settings_default_null(client: TestClient) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    assert resp.json()["team"] is None


# ---------------------------------------------------------------------------
# (b) PUT round-trip — value stored and reflected on GET
# ---------------------------------------------------------------------------


def test_put_settings_round_trip(client: TestClient) -> None:
    app.dependency_overrides[get_current_user] = fake_admin_user
    resp = client.put("/api/settings", json={"team": "docs-platform"})
    assert resp.status_code == 200
    assert resp.json()["team"] == "docs-platform"

    # Subsequent GET as a regular user reflects the stored value.
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    assert resp.json()["team"] == "docs-platform"


# ---------------------------------------------------------------------------
# (c) PUT clear cases — null, blank string, empty body, absent-row no-op
# ---------------------------------------------------------------------------


def test_put_settings_clear_via_null(client: TestClient) -> None:
    # First store something, then clear with null.
    app.dependency_overrides[get_current_user] = fake_admin_user
    client.put("/api/settings", json={"team": "docs-platform"})

    resp = client.put("/api/settings", json={"team": None})
    assert resp.status_code == 200
    assert resp.json()["team"] is None

    app.dependency_overrides[get_current_user] = fake_user
    assert client.get("/api/settings").json()["team"] is None


def test_put_settings_clear_via_blank_string(client: TestClient) -> None:
    app.dependency_overrides[get_current_user] = fake_admin_user
    client.put("/api/settings", json={"team": "docs-platform"})

    resp = client.put("/api/settings", json={"team": "  "})
    assert resp.status_code == 200
    assert resp.json()["team"] is None

    app.dependency_overrides[get_current_user] = fake_user
    assert client.get("/api/settings").json()["team"] is None


def test_put_settings_clear_via_empty_body(client: TestClient) -> None:
    # Omitted field == null (full-replace semantics).
    app.dependency_overrides[get_current_user] = fake_admin_user
    client.put("/api/settings", json={"team": "docs-platform"})

    resp = client.put("/api/settings", json={})
    assert resp.status_code == 200
    assert resp.json()["team"] is None

    app.dependency_overrides[get_current_user] = fake_user
    assert client.get("/api/settings").json()["team"] is None


def test_put_settings_clear_absent_row_is_noop(client: TestClient) -> None:
    # delete_setting on a row that never existed must not error.
    app.dependency_overrides[get_current_user] = fake_admin_user
    resp = client.put("/api/settings", json={"team": None})
    assert resp.status_code == 200
    assert resp.json()["team"] is None


# ---------------------------------------------------------------------------
# (d) PUT as non-admin → 403
# ---------------------------------------------------------------------------


def test_put_settings_non_admin_forbidden(client: TestClient) -> None:
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.put("/api/settings", json={"team": "x"})
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# (e) GET without Authorization header → 401 (FastAPI 0.136 HTTPBearer)
# ---------------------------------------------------------------------------


def test_get_settings_unauthed_returns_401(client: TestClient) -> None:
    # No dependency override — auth goes through the real HTTPBearer dependency.
    resp = client.get("/api/settings")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# (f) resolve_attribution unit tests (async — asyncio_mode = "auto")
# ---------------------------------------------------------------------------


async def test_resolve_attribution_team_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = StatefulFakeDbPool({"team": "docs-platform"})

    async def _fake_get_db() -> StatefulFakeDbPool:
        return pool

    monkeypatch.setattr(attribution_module, "get_db", _fake_get_db)
    result = await resolve_attribution(fake_user())
    assert isinstance(result, Attribution)
    assert result.contributor == "tester@example.com"
    assert result.team == "docs-platform"


async def test_resolve_attribution_team_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = StatefulFakeDbPool({})

    async def _fake_get_db() -> StatefulFakeDbPool:
        return pool

    monkeypatch.setattr(attribution_module, "get_db", _fake_get_db)
    result = await resolve_attribution(fake_user())
    assert result.contributor == "tester@example.com"
    assert result.team is None


async def test_resolve_attribution_team_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stored blank string ("  ") normalises to None.
    pool = StatefulFakeDbPool({"team": "  "})

    async def _fake_get_db() -> StatefulFakeDbPool:
        return pool

    monkeypatch.setattr(attribution_module, "get_db", _fake_get_db)
    result = await resolve_attribution(fake_user())
    assert result.team is None


async def test_resolve_attribution_team_whitespace_trimmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Out-of-band whitespace in the stored value is trimmed.
    pool = StatefulFakeDbPool({"team": " docs "})

    async def _fake_get_db() -> StatefulFakeDbPool:
        return pool

    monkeypatch.setattr(attribution_module, "get_db", _fake_get_db)
    result = await resolve_attribution(fake_user())
    assert result.team == "docs"


# ---------------------------------------------------------------------------
# (g) PUT trim — leading/trailing whitespace is stripped on write and read
# ---------------------------------------------------------------------------


def test_put_settings_trims_whitespace(client: TestClient) -> None:
    app.dependency_overrides[get_current_user] = fake_admin_user
    resp = client.put("/api/settings", json={"team": "  docs-platform  "})
    assert resp.status_code == 200
    assert resp.json()["team"] == "docs-platform"

    app.dependency_overrides[get_current_user] = fake_user
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    assert resp.json()["team"] == "docs-platform"


# ---------------------------------------------------------------------------
# (h) is_machine_principal — config-absent/-present via the same
#     machine_principal_email app_config key resolve_attribution's sibling
#     accessor reads. Not enforced anywhere yet (see cli.py / this item's
#     acceptance criteria) — these are pure accessor tests.
# ---------------------------------------------------------------------------


async def test_is_machine_principal_config_absent_is_false_for_everyone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No app_config row at all — must be False even for an admin.
    pool = StatefulFakeDbPool({})

    async def _fake_get_db() -> StatefulFakeDbPool:
        return pool

    monkeypatch.setattr(attribution_module, "get_db", _fake_get_db)
    assert await is_machine_principal(fake_user()) is False
    assert await is_machine_principal(fake_admin_user()) is False


async def test_is_machine_principal_config_present_identifies_exactly_that_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = StatefulFakeDbPool({"machine_principal_email": "tester@example.com"})

    async def _fake_get_db() -> StatefulFakeDbPool:
        return pool

    monkeypatch.setattr(attribution_module, "get_db", _fake_get_db)
    assert await is_machine_principal(fake_user()) is True
    # A different (even admin) user is not the machine principal.
    assert await is_machine_principal(fake_admin_user()) is False


async def test_is_machine_principal_config_blank_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stored blank string normalises to None, same as resolve_attribution's team.
    pool = StatefulFakeDbPool({"machine_principal_email": "   "})

    async def _fake_get_db() -> StatefulFakeDbPool:
        return pool

    monkeypatch.setattr(attribution_module, "get_db", _fake_get_db)
    assert await is_machine_principal(fake_user()) is False


# ---------------------------------------------------------------------------
# client_install_spec — MCP thin-client install spec from env
# ---------------------------------------------------------------------------


def test_get_settings_client_install_spec_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KB_SERVICE_CLIENT_INSTALL_SPEC", raising=False)
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.get("/api/settings")
    assert resp.json()["client_install_spec"] == (
        "personal-kb @ git+https://github.com/jason-weddington/personal-kb-mcp"
    )


def test_get_settings_client_install_spec_override(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = "personal-kb @ git+ssh://git@git-host.example.com/repos/personal_kb@main"
    monkeypatch.setenv("KB_SERVICE_CLIENT_INSTALL_SPEC", spec)
    app.dependency_overrides[get_current_user] = fake_user
    resp = client.get("/api/settings")
    assert resp.json()["client_install_spec"] == spec
