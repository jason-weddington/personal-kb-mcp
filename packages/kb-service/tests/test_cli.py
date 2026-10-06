"""Hermetic tests for the create-user / set-machine-principal CLI commands.

``kb_service.cli`` talks DIRECTLY to the service-auth asyncpg pool (imported
lazily inside each command coroutine), so these tests fake the pool at the
``kb_service.database`` module level — the same seam ``kb_service.attribution``
uses — rather than spinning up a real Postgres. ``cli.py`` is excluded from the
coverage metric (see pyproject.toml) as operational/not-hermetically-testable
for the *serve*/*create-admin* wiring, but the pure logic added here (non-admin
insert, duplicate-email error) is straightforward to fake and worth covering
directly, per this item's acceptance criteria.
"""

from typing import Any

import pytest

import kb_service.attribution as attribution_module
import kb_service.database as database
from kb_service.cli import _create_user, _set_machine_principal


class FakeConn:
    """Fake asyncpg connection backed by an in-memory users dict."""

    def __init__(self, users: dict[str, dict[str, Any]]) -> None:
        self.users = users

    async def fetchrow(self, sql: str, *args: Any) -> Any | None:
        if "SELECT id FROM users WHERE email" in sql:
            email = args[0]
            row = self.users.get(email)
            return {"id": row["id"]} if row is not None else None
        return None

    async def execute(self, sql: str, *args: Any) -> str:
        if "INSERT INTO users" in sql:
            user_id, email, hashed_password, is_admin, created_at = args
            self.users[email] = {
                "id": user_id,
                "email": email,
                "hashed_password": hashed_password,
                "is_admin": is_admin,
                "created_at": created_at,
            }
        return "OK"


class FakeAcquire:
    """Fake asyncpg ``pool.acquire()`` async context manager."""

    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> FakeConn:
        return self._conn

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakePool:
    """Fake asyncpg pool exposing only what cli.py's create-user path needs."""

    def __init__(self, users: dict[str, dict[str, Any]]) -> None:
        self._conn = FakeConn(users)

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self._conn)

    async def close(self) -> None:
        return None


@pytest.fixture
def fake_pool(monkeypatch: pytest.MonkeyPatch) -> FakePool:
    """Patch kb_service.database's get_db/init_db/close_db for CLI tests."""
    users: dict[str, dict[str, Any]] = {}
    pool = FakePool(users)

    async def _fake_get_db() -> FakePool:
        return pool

    async def _fake_init_db() -> None:
        return None

    async def _fake_close_db() -> None:
        return None

    monkeypatch.setattr(database, "get_db", _fake_get_db)
    monkeypatch.setattr(database, "init_db", _fake_init_db)
    monkeypatch.setattr(database, "close_db", _fake_close_db)
    return pool


async def test_create_user_is_non_admin(fake_pool: FakePool) -> None:
    msg = await _create_user("worker@example.com", "hunter2")
    assert msg == "created user worker@example.com"
    stored = fake_pool._conn.users["worker@example.com"]
    assert stored["is_admin"] == 0


async def test_create_user_duplicate_email_errors_cleanly(fake_pool: FakePool) -> None:
    await _create_user("worker@example.com", "hunter2")
    with pytest.raises(ValueError, match="already exists"):
        await _create_user("worker@example.com", "different-password")


async def test_set_machine_principal_writes_app_config(
    monkeypatch: pytest.MonkeyPatch, fake_pool: FakePool
) -> None:
    app_config: dict[str, str] = {}

    class FakeConfigPool(FakePool):
        async def fetchrow(self, sql: str, *args: Any) -> Any | None:
            if "app_config" in sql:
                val = app_config.get(args[0])
                return {"value": val} if val is not None else None
            return None

        async def execute(self, sql: str, *args: Any) -> str:  # type: ignore[override]
            if "INSERT INTO app_config" in sql:
                app_config[args[0]] = args[1]
                return "OK"
            return await super().execute(sql, *args)

    config_pool = FakeConfigPool({})

    async def _fake_attribution_get_db() -> FakeConfigPool:
        return config_pool

    monkeypatch.setattr(attribution_module, "get_db", _fake_attribution_get_db)

    msg = await _set_machine_principal("nightly@example.com")
    assert msg == "set machine principal to nightly@example.com"
    assert app_config["machine_principal_email"] == "nightly@example.com"
