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

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import kb_service.attribution as attribution_module
import kb_service.database as database
from kb_service import cli
from kb_service.cli import _create_user, _set_machine_principal
from tests.repeat_rate_fixtures import insert_rows_pool


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


# ─── metrics repeat-rate ─────────────────────────────────────────────────────

_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
)


@pytest.fixture
def local_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Local no-auth mode with the data DB and service DB in *tmp_path*."""
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setattr(database, "_pool", None)
    yield tmp_path / "service.db"


@pytest.mark.parametrize(
    ("argv", "want"),
    [
        (
            [
                "metrics",
                "repeat-rate",
                "--weeks",
                "3",
                "--project",
                "p",
                "--min-gap-hours",
                "12",
                "--json",
            ],
            (3, "p", 12.0, True),
        ),
        (["metrics", "repeat-rate"], (8, None, 24.0, False)),
    ],
)
def test_metrics_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    want: tuple[Any, ...],
) -> None:
    seen: list[tuple[Any, ...]] = []

    async def fake(*args: Any) -> str:
        seen.append(args)
        return "OUT"

    monkeypatch.setattr("sys.argv", ["kb-service", *argv])
    monkeypatch.setattr(cli, "_metrics_repeat_rate", fake)
    cli.main()
    assert capsys.readouterr().out == "OUT\n"
    assert seen == [want]


def test_metrics_without_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["kb-service", "metrics"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


async def test_metrics_repeat_rate_real_dbs(local_env: Path) -> None:
    await database.init_db()
    try:
        await insert_rows_pool(await database.get_db())
        out = await cli._metrics_repeat_rate(
            2, None, 24.0, True, now=datetime(2026, 10, 7, 12, tzinfo=UTC)
        )
        data = json.loads(out)
        assert data["weeks"][1]["sessions"] == 6
        assert data["weeks"][1]["repeat_sessions"] == 2
        assert data["failure_rows"] == 11 and data["resolutions_loaded"] == 0
        assert data["diagnostics"]["resolutions_scanned"] == 0
        md = await cli._metrics_repeat_rate(
            2, None, 24.0, False, now=datetime(2026, 10, 7, 12, tzinfo=UTC)
        )
        assert md.startswith("# Cross-session repeat rate: 2026-W40 to 2026-W41")
    finally:
        if database._pool is not None:
            await database.close_db()


def test_metrics_invalid_weeks(
    local_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv", ["kb-service", "metrics", "repeat-rate", "--weeks", "0"]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert capsys.readouterr().err.startswith(
        "Error: weeks must be between 1 and 104, got 0"
    )
