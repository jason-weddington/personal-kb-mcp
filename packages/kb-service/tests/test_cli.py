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

import kb_core.knowledge_base
import pytest
from kb_core.models.entry import EntryType
from kb_core.search.embeddings import EmbeddingClient

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
    "KB_EMBEDDING_PROVIDER",
    "KB_EMBEDDING_MODEL",
    "KB_EMBEDDING_DIM",
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


# --- write policy: list-keys / set-key-surface --------------------------------


async def _seed_keys(service_db: Path) -> None:
    import sqlite3

    await database.init_db()
    await database.close_db()
    conn = sqlite3.connect(service_db)
    try:
        conn.executemany(
            "INSERT INTO users (id, email, hashed_password, is_admin, created_at)"
            " VALUES (?, ?, 'x', 0, '2020-01-01')",
            [("u1", "a@example.com"), ("u2", "b@example.com")],
        )
        conn.executemany(
            "INSERT INTO api_keys (id, user_id, key_hash, name, created_at, surface)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("k1", "u1", "abcdef0123456789", "laptop", "2020-01-02", None),
                ("k2", "u1", "1234567890abcdef", "talos", "2020-01-03", "headless"),
                ("k3", "u2", "ffffffff00000000", "ci", "2020-01-01", "autonomous"),
            ],
        )
        conn.commit()
    finally:
        conn.close()


async def test_list_keys(local_env: Path) -> None:
    await _seed_keys(local_env)
    out = await cli._list_keys(None)
    assert out.split("\n") == [
        "id\temail\tname\thash_prefix\tsurface\tcreated_at",
        "k1\ta@example.com\tlaptop\tabcdef01\tdefault\t2020-01-02",
        "k2\ta@example.com\ttalos\t12345678\theadless\t2020-01-03",
        "k3\tb@example.com\tci\tffffffff\tautonomous\t2020-01-01",
    ]
    filtered = await cli._list_keys("b@example.com")
    assert filtered.split("\n")[1:] == [
        "k3\tb@example.com\tci\tffffffff\tautonomous\t2020-01-01"
    ]
    assert await cli._list_keys("nobody@example.com") == "no API keys"


async def test_list_keys_empty(local_env: Path) -> None:
    assert await cli._list_keys(None) == "no API keys"


async def test_set_key_surface(local_env: Path) -> None:
    import sqlite3

    await _seed_keys(local_env)
    assert await cli._set_key_surface("k1", "headless") == (
        "set surface of API key k1 (a@example.com, 'laptop') to headless"
    )
    assert await cli._set_key_surface("k2", "default") == (
        "cleared surface of API key k2 (a@example.com, 'talos'); it follows"
        " KB_WRITE_POLICY_DEFAULT_SURFACE"
    )
    conn = sqlite3.connect(local_env)
    try:
        rows = dict(conn.execute("SELECT id, surface FROM api_keys").fetchall())
    finally:
        conn.close()
    assert rows == {"k1": "headless", "k2": None, "k3": "autonomous"}
    with pytest.raises(ValueError, match="no API key with id nope"):
        await cli._set_key_surface("nope", "headless")


def test_key_commands_dispatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[tuple[Any, ...]] = []

    async def fake_list(*args: Any) -> str:
        seen.append(("list", *args))
        return "L"

    async def fake_set(*args: Any) -> str:
        seen.append(("set", *args))
        return "S"

    monkeypatch.setattr(cli, "_list_keys", fake_list)
    monkeypatch.setattr(cli, "_set_key_surface", fake_set)
    monkeypatch.setattr("sys.argv", ["kb-service", "list-keys", "--email", "a@b"])
    cli.main()
    monkeypatch.setattr(
        "sys.argv",
        ["kb-service", "set-key-surface", "--key-id", "k1", "--surface", "default"],
    )
    cli.main()
    assert seen == [("list", "a@b"), ("set", "k1", "default")]
    assert capsys.readouterr().out == "L\nS\n"
    monkeypatch.setattr(
        "sys.argv",
        ["kb-service", "set-key-surface", "--key-id", "k1", "--surface", "bogus"],
    )
    with pytest.raises(SystemExit):
        cli.main()


def test_set_key_surface_unknown_exits_1(
    local_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["kb-service", "set-key-surface", "--key-id", "nope", "--surface", "headless"],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_create_api_key_request_has_only_name() -> None:
    from kb_service.models import CreateApiKeyRequest

    assert set(CreateApiKeyRequest.model_fields) == {"name"}


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """_reembed calls configure_logging(); undo its global handler/levels."""
    import logging

    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    saved = {n: logging.getLogger(n).level for n in ("kb_service", "kb_core")}
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    for n, lv in saved.items():
        logging.getLogger(n).setLevel(lv)


# --- reembed -------------------------------------------------------------------


class FakeEmbedder(EmbeddingClient):
    """Deterministic embedder; ``fail_texts`` get None, ``available`` toggles probe."""

    def __init__(
        self,
        db: Any,
        config: Any,
        *,
        out_dim: int,
        fail_texts: frozenset[str] = frozenset(),
        available: bool = True,
    ) -> None:
        super().__init__(db, config)
        self.out_dim = out_dim
        self.fail_texts = fail_texts
        self.available = available

    async def is_available(self) -> bool:
        return self.available

    async def embed(self, text: str) -> list[float] | None:
        if text in self.fail_texts:
            return None
        return [1.0] + [0.0] * (self.out_dim - 1)


def _install_fake(monkeypatch: pytest.MonkeyPatch, **kw: Any) -> None:
    monkeypatch.setattr(
        kb_core.knowledge_base,
        "build_embedder",
        lambda db, cfg: FakeEmbedder(db, cfg, **kw),
    )


async def _seed(n: int, *, embedded: int = 0) -> list[Any]:
    import kb_service.main

    kb = await kb_service.main._open_kb()
    entries = []
    try:
        for i in range(n):
            e = await kb.knowledge_store.create_entry(
                short_title=f"t{i}",
                long_title=f"t{i}",
                knowledge_details=f"details {i}",
                entry_type=EntryType.FACTUAL_REFERENCE,
            )
            entries.append(e)
        for e in entries[:embedded]:
            assert kb.embedder is not None
            await kb.embedder.store_embedding(e.id, [0.5] + [0.0] * 1023)
            await kb.knowledge_store.mark_embedding(e.id, True)
    finally:
        await kb.close()
    return entries


async def _state() -> tuple[dict[str, int], int]:
    import kb_service.main

    kb = await kb_service.main._open_kb()
    try:
        cur = await kb.db.execute("SELECT id, has_embedding FROM knowledge_entries")
        flags = {r["id"]: r["has_embedding"] for r in await cur.fetchall()}
        cur = await kb.db.execute("SELECT COUNT(*) AS n FROM knowledge_vec")
        row = await cur.fetchone()
        return flags, int(row["n"])
    finally:
        await kb.close()


async def _model_row() -> str | None:
    import kb_service.main

    kb = await kb_service.main._open_kb()
    try:
        cur = await kb.db.execute(
            "SELECT value FROM deployment_config WHERE key = 'embedding_model'"
        )
        row = await cur.fetchone()
        return None if row is None else str(row["value"])
    finally:
        await kb.close()


async def _set_model_row(value: str) -> None:
    import kb_service.main

    kb = await kb_service.main._open_kb()
    try:
        await kb.db.execute(
            "INSERT INTO deployment_config (key, value, set_at)"
            " VALUES ('embedding_model', ?, 'then')",
            (value,),
        )
        await kb.db.commit()
    finally:
        await kb.close()


@pytest.mark.parametrize(("flag", "want"), [([], False), (["--missing"], True)])
def test_reembed_dispatch(
    monkeypatch: pytest.MonkeyPatch, flag: list[str], want: bool
) -> None:
    seen: list[bool] = []

    async def fake(only_missing: bool) -> str:
        seen.append(only_missing)
        return "ok"

    monkeypatch.setattr(cli, "_reembed", fake)
    monkeypatch.setattr("sys.argv", ["kb-service", "reembed", *flag])
    cli.main()
    assert seen == [want]


async def test_reembed_all(local_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, out_dim=1024)
    await _seed(3)
    out = await cli._reembed(False)
    assert out.startswith("Re-embedded all entries: 3 processed, 3 succeeded, 0 failed")
    flags, vecs = await _state()
    assert set(flags.values()) == {1}
    assert vecs == 3


async def test_reembed_missing(
    local_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake(monkeypatch, out_dim=1024)
    await _seed(3, embedded=2)
    out = await cli._reembed(True)
    assert "1 processed" in out
    assert out.startswith("Re-embedded entries without embeddings")


async def test_reembed_table_dim_guard(
    local_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake(monkeypatch, out_dim=1024)
    await _seed(2, embedded=2)
    before = await _state()
    monkeypatch.setenv("KB_EMBEDDING_DIM", "512")
    _install_fake(monkeypatch, out_dim=512)
    with pytest.raises(RuntimeError, match="knowledge_vec stores 1024"):
        await cli._reembed(False)
    monkeypatch.delenv("KB_EMBEDDING_DIM")
    assert await _state() == before


async def test_reembed_probe_length_guard(
    local_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake(monkeypatch, out_dim=1024)
    await _seed(2)
    before = await _state()
    _install_fake(monkeypatch, out_dim=512)
    with pytest.raises(RuntimeError, match="512-dimension"):
        await cli._reembed(False)
    assert await _state() == before


async def test_reembed_unavailable(
    local_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake(monkeypatch, out_dim=1024, available=False)
    with pytest.raises(RuntimeError, match="embedder unavailable"):
        await cli._reembed(False)


async def test_reembed_failure_exit_code(
    local_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake(monkeypatch, out_dim=1024)
    entries = await _seed(2)
    _install_fake(
        monkeypatch, out_dim=1024, fail_texts=frozenset({entries[0].embedding_text})
    )
    with pytest.raises(RuntimeError, match="1 failed") as exc:
        await cli._reembed(False)
    assert entries[0].id in str(exc.value)
    monkeypatch.setattr("sys.argv", ["kb-service", "reembed"])
    with pytest.raises(SystemExit) as se:
        cli.main()
    assert se.value.code == 1


async def test_reembed_deployment_config(
    local_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake(monkeypatch, out_dim=1024)
    monkeypatch.setenv("KB_EMBEDDING_MODEL", "fake-model")
    # No row present: none created, even after a clean all-mode run.
    await _seed(1)
    await cli._reembed(False)
    assert await _model_row() is None
    await _set_model_row("qwen3-embedding:0.6b")
    await cli._reembed(True)
    assert await _model_row() == "qwen3-embedding:0.6b"
    await cli._reembed(False)
    assert await _model_row() == "fake-model"


async def test_reembed_empty_leaves_config(
    local_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake(monkeypatch, out_dim=1024)
    monkeypatch.setenv("KB_EMBEDDING_MODEL", "fake-model")
    await _set_model_row("qwen3-embedding:0.6b")
    assert await cli._reembed(False) == "No entries need embedding."
    assert await _model_row() == "qwen3-embedding:0.6b"
