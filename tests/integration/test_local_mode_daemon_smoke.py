"""End-to-end smoke for the documented local-mode daemon spawn.

The sibling ``tests/test_local_mode_install_contract.py`` covers the
**paper** side of the onboarding contract — every README local-mode
``mcpServers`` block sets the required env vars and the ``[local]``
extra pulls in the package shipping the ``kb-service`` console script.
This module covers the **real** side: when ``kb-service`` IS on PATH
(developer machine, release-time smoke, CI envs that run
``uv sync --extra local``), it actually boots on loopback and answers
``/api/health`` with ``{"status": "ok"}`` — the exact health probe
``personal_kb.daemon._check_health`` performs from the MCP lifespan.

The test is **auto-skipped** when ``kb-service`` is not on PATH so the
default in-workspace test run (which does NOT install the ``[local]``
extra; see the dev group in ``pyproject.toml``) stays green. Install
the extra with::

    uv sync --extra local

and the smoke will run alongside the rest of the suite, catching
regressions that would otherwise only surface when a real user runs the
documented quick-start.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import socket
import subprocess
import time
from typing import TYPE_CHECKING

import httpx
import pytest

from personal_kb import daemon

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    shutil.which("kb-service") is None,
    reason=(
        "kb-service binary not on PATH — install the documented local-mode "
        "bundle with `uv sync --extra local` to run this smoke (it boots the "
        "daemon and hits /api/health, the same probe the MCP lifespan uses)."
    ),
)


# Mirror the lifespan's poll budget. If the documented install path is
# healthy at all, /api/health flips well inside this window — we use the
# same numbers as the MCP server's ``ensure_daemon`` so the smoke fails
# whenever the real lifespan would.
_BOOT_TIMEOUT = daemon._HEALTH_POLL_TIMEOUT
_BOOT_POLL_INTERVAL = daemon._HEALTH_POLL_INTERVAL


def _free_loopback_port() -> int:
    """Bind ``127.0.0.1:0`` and return the kernel-assigned port.

    A bind-and-release race is theoretically possible but the daemon
    rebinds within milliseconds, so this is a non-issue in practice.
    Using a fixed ``8765`` (the documented port) would collide with a
    developer's already-running personal daemon — the smoke MUST stand
    apart from real state.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


async def _wait_until_healthy(base_url: str) -> bool:
    """Poll ``/api/health`` on the same cadence the MCP lifespan does."""
    deadline = time.monotonic() + _BOOT_TIMEOUT
    while time.monotonic() < deadline:
        if await daemon._check_health(base_url):
            return True
        await _async_sleep(_BOOT_POLL_INTERVAL)
    return False


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


@pytest.mark.asyncio
async def test_documented_local_daemon_boots_and_serves_health(
    tmp_path: Path,
) -> None:
    """``kb-service serve --port <p>`` boots and answers /api/health=ok.

    This exercises the **exact** spawn the MCP lifespan performs:
    ``_build_spawn_argv`` argv, ``KB_AUTH_MODE=none`` in the env, stdout
    + stderr redirected to a logfile, loopback host. The daemon is
    killed in the teardown so the smoke cannot pollute the developer's
    machine state.
    """
    port = _free_loopback_port()
    base_url = f"http://127.0.0.1:{port}"

    # Match the spawn env / argv the MCP server uses, but redirect the
    # data DB and daemon-state dir under tmp_path so the smoke is fully
    # hermetic — no clobbering of ``~/.local/share/personal_kb/``.
    env = daemon._build_spawn_env()
    env["KB_DB_PATH"] = str(tmp_path / "smoke.db")
    env["PERSONAL_KB_DAEMON_STATE_DIR"] = str(tmp_path)
    env["KB_AUTO_EXPLORE"] = "FALSE"  # don't drag in the explorer port
    # The daemon honours KB_DATABASE_URL by design; the smoke must not inherit
    # a real one, or the "local" daemon opens that Postgres (it opened the live
    # KB once). Force SQLite explicitly rather than relying on its absence.
    for var in ("KB_DATABASE_URL", "KB_SERVICE_DATABASE_URL"):
        env.pop(var, None)

    logfile = tmp_path / "kb-daemon.log"
    log_fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        proc = subprocess.Popen(  # noqa: S603
            daemon._build_spawn_argv(port),
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=log_fd,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(log_fd)

    try:
        healthy = await _wait_until_healthy(base_url)
        if not healthy:
            log_text = logfile.read_text(errors="replace") if logfile.exists() else "<empty>"
            pytest.fail(
                f"kb-service spawned (pid={proc.pid}) on {base_url} but "
                f"/api/health never returned 200+status=ok within "
                f"{_BOOT_TIMEOUT:.0f}s.\n--- daemon log ---\n{log_text}"
            )

        # Second probe: the response shape ``ensure_daemon`` cares about
        # is locked here too. A daemon that flips ``status`` to something
        # other than ``ok`` would silently break the lifespan even though
        # the HTTP code is 200.
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url}/api/health")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert isinstance(body, dict), body
        assert body.get("status") == "ok", body
    finally:
        # Kill the whole process group — the daemon runs in its own
        # session (start_new_session=True), so a plain proc.terminate()
        # would miss any uvicorn worker children.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=2)
