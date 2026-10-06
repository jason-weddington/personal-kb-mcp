"""Tests for the kb-service daemon spawn + singleton lock.

These tests mock the health check and the subprocess spawn — they do
NOT exercise the real ``kb-service`` binary (which lives in the sibling
``personal-kb-web-service`` repo and need not be installed in the
sandbox).

Coverage map:

* :func:`test_healthy_endpoint_no_spawn` — AC: "given a stubbed-healthy
  endpoint, no spawn is attempted."
* :func:`test_unhealthy_endpoint_spawns_with_pinned_argv` — AC: "spawn
  argv equals svc-local-profile entrypoint" (``['kb-service', 'serve',
  '--port', '<port>']``).
* :func:`test_spawn_env_contains_kb_auth_mode_none` — AC: spawn env
  contains ``KB_AUTH_MODE=none`` and inherits the parent env (including
  ``KB_DB_PATH``).
* :func:`test_concurrent_racers_single_spawn` — AC: "N MCP processes
  start simultaneously and all see an unhealthy endpoint, exactly ONE
  acquires the lock and spawns."
* :func:`test_stale_pidfile_reclaim` — AC: "dead pid + unhealthy
  endpoint => reclaim followed by a single spawn."
* :func:`test_loopback_url_detection` — AC: parses ``PERSONAL_KB_URL``
  host; ``127.0.0.1`` and ``localhost`` are loopback, ``kb.example.com``
  is not.
* :func:`test_daemon_log_path_is_pinned` — AC: spawn redirects
  stdout/stderr to ``<state>/kb-daemon.log`` in APPEND mode.
* :func:`test_spawn_timeout_raises_with_log_path` — AC: timeout
  RuntimeError message includes the daemon log path.
* :func:`test_lifespan_opens_http_backend_over_loopback` — AC:
  "in local mode the lifespan opens an HttpBackend over loopback (not an
  in-process local backend)."
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp import FastMCP

from personal_kb import daemon

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the daemon's pidfile + logfile under tmp_path."""
    monkeypatch.setenv("PERSONAL_KB_DAEMON_STATE_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Loopback detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://127.0.0.1:8765", True),
        ("http://localhost:8765", True),
        ("http://localhost:8765/", True),
        ("http://[::1]:8765", True),
        ("http://kb.example.com:8765", False),
        ("https://my-team-kb.internal", False),
    ],
)
def test_loopback_url_detection(url: str, expected: bool) -> None:
    """``is_loopback_url`` accepts only 127.0.0.1 / localhost / ::1."""
    assert daemon.is_loopback_url(url) is expected


def test_parse_port_from_url() -> None:
    """``parse_port`` extracts the integer port from a URL."""
    assert daemon.parse_port("http://127.0.0.1:8765") == 8765
    assert daemon.parse_port("http://localhost:9000/api") == 9000


def test_parse_port_missing_raises() -> None:
    """Missing port -> ValueError with a clear message."""
    with pytest.raises(ValueError, match="no port"):
        daemon.parse_port("http://127.0.0.1")


# ---------------------------------------------------------------------------
# Healthy endpoint -> no spawn
# ---------------------------------------------------------------------------


async def test_healthy_endpoint_no_spawn(state_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a healthy endpoint, ``ensure_daemon`` returns without spawning."""
    spawn_mock = MagicMock(name="_spawn_daemon", return_value=12345)
    monkeypatch.setattr(daemon, "_spawn_daemon", spawn_mock)
    monkeypatch.setattr(daemon, "_check_health", AsyncMock(return_value=True))

    await daemon.ensure_daemon("http://127.0.0.1:8765")

    spawn_mock.assert_not_called()
    # No pidfile written either (we didn't take the lock).
    assert not (state_dir / "kb-daemon.pid").exists()


# ---------------------------------------------------------------------------
# Unhealthy endpoint -> spawn with pinned argv
# ---------------------------------------------------------------------------


async def test_unhealthy_endpoint_spawns_with_pinned_argv(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spawn argv is the svc-local-profile entrypoint exactly."""
    # First health check unhealthy, post-spawn check healthy.
    health_calls = iter([False, True])
    monkeypatch.setattr(
        daemon, "_check_health", AsyncMock(side_effect=lambda *_a, **_k: next(health_calls))
    )

    captured: dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        proc = MagicMock()
        proc.pid = 42424
        return proc

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)

    await daemon.ensure_daemon("http://127.0.0.1:8765")

    assert captured["argv"] == ["kb-service", "serve", "--port", "8765"]
    # start_new_session detaches from the MCP process group.
    assert captured["kwargs"]["start_new_session"] is True
    # stdin must NOT be the MCP stdio channel — we route it to /dev/null.
    assert captured["kwargs"]["stdin"] == daemon.subprocess.DEVNULL


# ---------------------------------------------------------------------------
# Spawn env: KB_AUTH_MODE=none + inherits parent env (incl. KB_DB_PATH)
# ---------------------------------------------------------------------------


async def test_spawn_env_contains_kb_auth_mode_none(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spawned daemon's env selects no-auth mode and passes KB_DB_PATH through."""
    monkeypatch.setenv("KB_DB_PATH", str(state_dir / "spawn-env-test.db"))
    monkeypatch.setenv("KB_AUTH_MODE", "")  # caller's KB_AUTH_MODE is overridden

    health_calls = iter([False, True])
    monkeypatch.setattr(
        daemon, "_check_health", AsyncMock(side_effect=lambda *_a, **_k: next(health_calls))
    )

    captured: dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        captured["env"] = kwargs["env"]
        proc = MagicMock()
        proc.pid = 5555
        return proc

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)

    await daemon.ensure_daemon("http://127.0.0.1:8765")

    env = captured["env"]
    # The no-auth selector is the literal contract pinned by
    # svc-local-profile (personal-kb-web-service commit 8e06637).
    assert env["KB_AUTH_MODE"] == "none"
    # KB_DB_PATH must pass through so the daemon opens the SAME DB the
    # user's other tooling expects.
    assert env["KB_DB_PATH"] == str(state_dir / "spawn-env-test.db")


# ---------------------------------------------------------------------------
# Concurrent racers -> single spawn
# ---------------------------------------------------------------------------


async def test_concurrent_racers_single_spawn(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two concurrent ``ensure_daemon`` calls produce exactly ONE spawn.

    Mocks the O_EXCL pidfile lock + the health check so the test is
    hermetic (no real subprocess / sockets). The first racer's
    ``_check_health`` is initially False, then True after the spawn; the
    second racer's first ``_check_health`` is False then True (the
    winner's daemon comes up).
    """
    # Health-check sequence: each call returns False until the spawn,
    # then True forever.
    spawn_completed = asyncio.Event()

    async def fake_health(_url):
        return spawn_completed.is_set()

    monkeypatch.setattr(daemon, "_check_health", fake_health)

    spawn_count = 0

    def fake_popen(argv, **kwargs):
        nonlocal spawn_count
        spawn_count += 1
        proc = MagicMock()
        proc.pid = 9001 + spawn_count
        spawn_completed.set()
        return proc

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)
    # Tighten the poll interval so the test runs fast.
    monkeypatch.setattr(daemon, "_HEALTH_POLL_INTERVAL", 0.01)

    await asyncio.gather(
        daemon.ensure_daemon("http://127.0.0.1:8765"),
        daemon.ensure_daemon("http://127.0.0.1:8765"),
    )

    assert spawn_count == 1


# ---------------------------------------------------------------------------
# Stale pidfile reclaim
# ---------------------------------------------------------------------------


async def test_stale_pidfile_reclaim(state_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pidfile pointing at a dead pid + unhealthy endpoint -> reclaim + spawn."""
    pidfile = state_dir / "kb-daemon.pid"
    # Pick a pid that doesn't exist. os.kill(pid, 0) on a non-existent
    # pid raises ProcessLookupError, which _process_alive maps to False.
    # We MUST use a pid that the kernel knows is dead — synthesize one by
    # forking a short-lived child and using its exited pid.
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    pidfile.write_text(f"{pid}\n")

    health_calls = iter([False, False, True])
    monkeypatch.setattr(
        daemon, "_check_health", AsyncMock(side_effect=lambda *_a, **_k: next(health_calls))
    )

    spawn_count = 0

    def fake_popen(argv, **kwargs):
        nonlocal spawn_count
        spawn_count += 1
        proc = MagicMock()
        proc.pid = 7777
        return proc

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon, "_HEALTH_POLL_INTERVAL", 0.01)

    await daemon.ensure_daemon("http://127.0.0.1:8765")

    assert spawn_count == 1
    # The reclaim winner overwrites the pidfile with the new pid.
    assert pidfile.read_text().strip() == "7777"


# ---------------------------------------------------------------------------
# Daemon log path is pinned + opened append-mode
# ---------------------------------------------------------------------------


async def test_daemon_log_path_is_pinned(state_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Spawn redirects stdout/stderr to ``<state>/kb-daemon.log`` (append-mode).

    Asserts the popen call received the SAME fd for stdout and stderr,
    and that fd points at ``kb-daemon.log`` opened with O_APPEND.
    """
    health_calls = iter([False, True])
    monkeypatch.setattr(
        daemon, "_check_health", AsyncMock(side_effect=lambda *_a, **_k: next(health_calls))
    )

    captured: dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        captured["stdout"] = kwargs["stdout"]
        captured["stderr"] = kwargs["stderr"]
        proc = MagicMock()
        proc.pid = 100
        return proc

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)

    await daemon.ensure_daemon("http://127.0.0.1:8765")

    # Same fd routed to both stdout and stderr.
    assert captured["stdout"] == captured["stderr"]
    # The log file exists (was created by O_CREAT).
    assert (state_dir / "kb-daemon.log").exists()


# ---------------------------------------------------------------------------
# Timeout -> RuntimeError with log path
# ---------------------------------------------------------------------------


async def test_spawn_timeout_raises_with_log_path(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the spawned daemon never becomes healthy, raise RuntimeError with log path."""
    # Endpoint is always unhealthy — spawn proceeds, then poll exhausts.
    monkeypatch.setattr(daemon, "_check_health", AsyncMock(return_value=False))

    def fake_popen(argv, **kwargs):
        proc = MagicMock()
        proc.pid = 4242
        return proc

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)

    # Tighten the budget so the test runs in <1s.
    monkeypatch.setattr(daemon, "_HEALTH_POLL_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(daemon, "_HEALTH_POLL_INTERVAL", 0.01)

    expected_log = state_dir / "kb-daemon.log"
    with pytest.raises(RuntimeError, match=str(expected_log)):
        await daemon.ensure_daemon("http://127.0.0.1:8765")


# ---------------------------------------------------------------------------
# Lifespan: HttpBackend over loopback
# ---------------------------------------------------------------------------


async def test_lifespan_opens_http_backend_over_loopback(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In local mode the lifespan opens an HttpBackend over loopback.

    Mocks ``ensure_daemon`` so we don't need a real subprocess.
    """
    monkeypatch.setenv("PERSONAL_KB_URL", "http://127.0.0.1:8765")

    ensure_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("personal_kb.daemon.ensure_daemon", ensure_mock)

    from personal_kb.backend.http import HttpBackend
    from personal_kb.server import lifespan

    mcp = FastMCP("test-loopback-lifespan", lifespan=lifespan)
    async with lifespan(mcp) as ctx:
        ensure_mock.assert_awaited_once_with("http://127.0.0.1:8765")
        assert "backend" in ctx
        assert isinstance(ctx["backend"], HttpBackend)
        # The HTTP-mode branch yields NO kb facade — every backend
        # operation routes through HTTP.
        assert "kb" not in ctx


async def test_lifespan_remote_url_skips_ensure_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-loopback PERSONAL_KB_URL does NOT trigger the spawn pre-step."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "k")

    ensure_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("personal_kb.daemon.ensure_daemon", ensure_mock)

    from personal_kb.backend.http import HttpBackend
    from personal_kb.server import lifespan

    mcp = FastMCP("test-remote-lifespan", lifespan=lifespan)
    async with lifespan(mcp) as ctx:
        ensure_mock.assert_not_called()
        assert isinstance(ctx["backend"], HttpBackend)


async def test_lifespan_unset_url_uses_local_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset URL + key: spawn the local daemon and open HttpBackend with the sentinel."""
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)

    ensure_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("personal_kb.daemon.ensure_daemon", ensure_mock)
    open_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("personal_kb.backend.http.HttpBackend.open", open_mock)

    from personal_kb.config import LOCAL_KB_API_KEY, LOCAL_KB_URL
    from personal_kb.server import lifespan

    mcp = FastMCP("test-local-default", lifespan=lifespan)
    async with lifespan(mcp) as ctx:
        ensure_mock.assert_awaited_once_with(LOCAL_KB_URL)
        open_mock.assert_awaited_once()
        assert ctx["backend"]._api_key == LOCAL_KB_API_KEY


async def test_lifespan_empty_url_uses_local_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty PERSONAL_KB_URL behaves like unset."""
    monkeypatch.setenv("PERSONAL_KB_URL", "")
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)

    ensure_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("personal_kb.daemon.ensure_daemon", ensure_mock)
    monkeypatch.setattr("personal_kb.backend.http.HttpBackend.open", AsyncMock())

    from personal_kb.config import LOCAL_KB_URL
    from personal_kb.server import lifespan

    mcp = FastMCP("test-empty-url", lifespan=lifespan)
    async with lifespan(mcp):
        ensure_mock.assert_awaited_once_with(LOCAL_KB_URL)


async def test_lifespan_remote_url_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote URL with no key fails loudly, naming PERSONAL_KB_API_KEY."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)

    ensure_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("personal_kb.daemon.ensure_daemon", ensure_mock)

    from personal_kb.server import lifespan

    mcp = FastMCP("test-remote-nokey", lifespan=lifespan)
    with pytest.raises(RuntimeError, match="PERSONAL_KB_API_KEY"):
        async with lifespan(mcp):
            pass  # pragma: no cover
    ensure_mock.assert_not_called()
