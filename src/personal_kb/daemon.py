"""Local kb-service daemon spawn + singleton management.

The MCP server's lifespan invokes :func:`ensure_daemon` before opening an
HTTP backend over a loopback URL. The flow:

1. ``GET <base>/api/health`` — if 200 with ``{"status": "ok"}``, return.
2. Otherwise acquire the O_EXCL pidfile lock at
   ``~/.local/share/personal_kb/kb-daemon.pid``. The winner spawns the
   daemon detached (``start_new_session=True``, stdout+stderr redirected to
   ``~/.local/share/personal_kb/kb-daemon.log`` in append mode). The losers
   poll ``/api/health`` until the winner's daemon is up.
3. Both the winner and the losers poll every 0.5s up to a 30s budget.
4. Stale-pidfile reclaim: if the pid in the pidfile is dead
   (``os.kill(pid, 0)`` raises ``ProcessLookupError``) AND the endpoint is
   unhealthy, the pidfile is unlinked and the lock is re-attempted.

The daemon NEVER shuts down on MCP session end. ``ensure_daemon`` writes
no state to the MCP process — its job is only to make a healthy daemon
exist before the lifespan opens the HTTP backend.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


# Pinned literal paths — the daemon outlives sessions, so these MUST be
# stable across MCP processes. Override via env only in tests.
_STATE_DIR = Path(
    os.environ.get(
        "PERSONAL_KB_DAEMON_STATE_DIR",
        os.path.expanduser("~/.local/share/personal_kb"),
    )
)
_PIDFILE = _STATE_DIR / "kb-daemon.pid"
_LOGFILE = _STATE_DIR / "kb-daemon.log"

# Loopback host literals — Reading B treats only these as "spawn me a local
# daemon"; any other host is a remote KB the user owns separately.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Spawn argv (svc-local-profile entrypoint).  The web-service console script
# ``kb-service serve --port <n>`` was added in personal-kb-web-service commit
# 8e06637 alongside KB_AUTH_MODE=none. The MCP server has no opinion on the
# host — the daemon binds 127.0.0.1 by default.
_SPAWN_CMD = "kb-service"
_SPAWN_SUBCOMMAND = "serve"

# Health poll budget.  60 attempts of 0.5s each = 30s hard cap.
_HEALTH_POLL_INTERVAL = 0.5
_HEALTH_POLL_TIMEOUT = 30.0
_HEALTH_POLL_MAX_ATTEMPTS = int(_HEALTH_POLL_TIMEOUT / _HEALTH_POLL_INTERVAL)

# HTTP client timeout for the health check.  Connection-refused must
# return fast so polling stays cheap.
_HEALTH_HTTP_TIMEOUT = 2.0


def _pidfile_path() -> Path:
    """Return the pidfile path (read at call time so tests can override)."""
    state_dir = Path(
        os.environ.get(
            "PERSONAL_KB_DAEMON_STATE_DIR",
            os.path.expanduser("~/.local/share/personal_kb"),
        )
    )
    return state_dir / "kb-daemon.pid"


def _logfile_path() -> Path:
    """Return the daemon log path (read at call time so tests can override)."""
    state_dir = Path(
        os.environ.get(
            "PERSONAL_KB_DAEMON_STATE_DIR",
            os.path.expanduser("~/.local/share/personal_kb"),
        )
    )
    return state_dir / "kb-daemon.log"


def is_loopback_url(base_url: str) -> bool:
    """Return True when *base_url* targets a loopback host.

    Reading B: only loopback hosts trigger the spawn pre-step. A remote
    KB URL connects directly.
    """
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return host in _LOOPBACK_HOSTS


def parse_port(base_url: str) -> int:
    """Return the port from *base_url* (raises if absent).

    Reading B: port is parsed from the URL — there is no separate
    ``PERSONAL_KB_DAEMON_PORT`` env var.
    """
    parsed = urlparse(base_url)
    if parsed.port is None:
        msg = (
            f"PERSONAL_KB_URL {base_url!r} has no port. "
            "Local mode requires an explicit port (e.g. http://127.0.0.1:8765)."
        )
        raise ValueError(msg)
    return parsed.port


async def _check_health(base_url: str) -> bool:
    """Return True iff ``GET <base>/api/health`` returns 200 + ``status=ok``.

    Connection-refused / timeout / non-2xx are all "unhealthy" — the caller
    proceeds to the spawn path.
    """
    url = base_url.rstrip("/") + "/api/health"
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_HTTP_TIMEOUT) as client:
            resp = await client.get(url)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
        return False
    if resp.status_code != 200:
        return False
    try:
        body = resp.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("status") == "ok"


def _read_pidfile(path: Path) -> int | None:
    """Return the pid recorded in *path*, or None if absent/unreadable."""
    try:
        raw = path.read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _process_alive(pid: int) -> bool:
    """Return True iff a process with *pid* is currently running.

    ``os.kill(pid, 0)`` raises :class:`ProcessLookupError` for a dead pid
    and :class:`PermissionError` for one we can't signal (which still
    indicates the process exists).
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _build_spawn_argv(port: int) -> list[str]:
    """Return the argv for the daemon spawn.

    Pinned by svc-local-profile (personal-kb-web-service commit 8e06637):
    ``kb-service serve --port <port>``. The host stays at the
    ``kb-service`` default (``127.0.0.1``) — we want loopback-only.
    """
    return [_SPAWN_CMD, _SPAWN_SUBCOMMAND, "--port", str(port)]


def _build_spawn_env() -> dict[str, str]:
    """Return the env mapping for the daemon spawn.

    Inherits the parent env so the user's ``KB_DATABASE_URL`` /
    ``KB_DB_PATH`` / Ollama config / Anthropic key all flow through to the
    daemon. Layered on top: ``KB_AUTH_MODE=none`` (the no-auth selector
    landed in personal-kb-web-service commit 8e06637) so the daemon
    serves a single synthetic local user without a JWT/API-key wall.
    """
    env = os.environ.copy()
    env["KB_AUTH_MODE"] = "none"
    return env


def _open_logfile(logfile: Path) -> int:
    """Open *logfile* in append mode and return its fd.

    Append mode (``O_APPEND``) is required: the daemon outlives MCP
    sessions, and truncating would lose history across restarts. ``0o644``
    matches the convention of every other file we write under
    ``~/.local/share/personal_kb/``.
    """
    logfile.parent.mkdir(parents=True, exist_ok=True)
    return os.open(
        logfile,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )


def _spawn_daemon(port: int, logfile: Path) -> int:
    """Spawn the daemon detached and return its pid.

    ``start_new_session=True`` puts the daemon in its own process group, so
    a Ctrl-C in the MCP terminal (which signals the foreground group) does
    NOT reach the daemon. ``stdout``/``stderr`` are redirected to *logfile*
    (NOT inherited) so the daemon doesn't pollute the MCP stdio transport.
    ``stdin`` is ``/dev/null`` — the daemon never reads.
    """
    log_fd = _open_logfile(logfile)
    try:
        proc = subprocess.Popen(  # noqa: S603
            _build_spawn_argv(port),
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=log_fd,
            start_new_session=True,
            env=_build_spawn_env(),
            close_fds=True,
        )
    finally:
        os.close(log_fd)
    return proc.pid


def _try_acquire_lock(pidfile: Path, pid_to_write: int) -> bool:
    """O_EXCL-create *pidfile* with *pid_to_write*. Return True iff we won.

    The atomic O_EXCL create is the singleton primitive: exactly one
    racer's create succeeds, so exactly one racer spawns the daemon. A
    losing racer sees ``FileExistsError`` and falls through to the
    health-poll path.
    """
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(pidfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, f"{pid_to_write}\n".encode())
    finally:
        os.close(fd)
    return True


async def _poll_until_healthy(base_url: str) -> bool:
    """Poll ``/api/health`` every 0.5s up to 30s.  Return True iff healthy."""
    for _ in range(_HEALTH_POLL_MAX_ATTEMPTS):
        if await _check_health(base_url):
            return True
        await asyncio.sleep(_HEALTH_POLL_INTERVAL)
    return False


def _reclaim_if_stale(pidfile: Path, base_url_healthy: bool) -> bool:
    """Unlink *pidfile* if it points at a dead pid AND endpoint is unhealthy.

    Return True iff the reclaim happened (the caller should re-attempt the
    O_EXCL create). A live pid OR a healthy endpoint means SOMEONE is
    serving — we MUST NOT delete the pidfile under either condition.
    """
    if base_url_healthy:
        return False
    pid = _read_pidfile(pidfile)
    if pid is None:
        # Empty/missing pidfile but a previous O_EXCL collided — treat as
        # stale so the next attempt can win.
        with contextlib.suppress(FileNotFoundError):
            pidfile.unlink()
        return True
    if _process_alive(pid):
        return False
    with contextlib.suppress(FileNotFoundError):
        pidfile.unlink()
    logger.info("Reclaimed stale kb-daemon pidfile (dead pid %d)", pid)
    return True


async def ensure_daemon(base_url: str) -> None:
    """Make sure a healthy kb-service daemon is reachable at *base_url*.

    Pre-step for the MCP server's lifespan when the resolved
    ``PERSONAL_KB_URL`` is a loopback host. On return, the daemon is up
    and serving ``/api/health``. The daemon outlives the MCP session.

    Raises:
        RuntimeError: When the health-poll budget (30s) is exhausted. The
            message includes the daemon log path so the user can inspect
            the spawn failure.
    """
    if await _check_health(base_url):
        logger.debug("kb-daemon already healthy at %s — no spawn", base_url)
        return

    port = parse_port(base_url)
    pidfile = _pidfile_path()
    logfile = _logfile_path()

    # Up to two attempts: the second runs only if a stale pidfile blocked
    # the first.  Two is sufficient — a third collision implies a healthy
    # daemon (covered by the inner _check_health), not a stale lock.
    for attempt in (1, 2):
        # Sentinel: lock acquisition writes the SPAWNED process's pid.  We
        # don't know it before Popen, so we write a sentinel ('0') and
        # rewrite the real pid on success.  Collisions during the rewrite
        # window are harmless — losers only care that the file EXISTS.
        won = _try_acquire_lock(pidfile, pid_to_write=os.getpid())
        if won:
            try:
                spawned_pid = _spawn_daemon(port, logfile)
            except (FileNotFoundError, PermissionError, OSError) as exc:
                # Spawn failed (e.g. kb-service not on PATH).  Release the
                # lock so the next MCP process can retry — and surface the
                # log path so the user knows where to look.
                pidfile.unlink(missing_ok=True)
                msg = f"Failed to spawn kb-service daemon ({exc}). Daemon log: {logfile}"
                raise RuntimeError(msg) from exc
            # Overwrite the pidfile with the real pid for stale-detection
            # by future MCP processes. A non-atomic overwrite is fine here
            # — the lock is held; nothing else writes to this file.
            try:
                pidfile.write_text(f"{spawned_pid}\n")
            except OSError:
                # Pid mismatch in the pidfile only affects future stale
                # reclaim accuracy; current callers poll /api/health.
                logger.warning("Could not record spawned pid %d in %s", spawned_pid, pidfile)
            logger.info(
                "Spawned kb-daemon pid=%d port=%d log=%s",
                spawned_pid,
                port,
                logfile,
            )
            if await _poll_until_healthy(base_url):
                return
            msg = (
                f"kb-service daemon spawned (pid={spawned_pid}) but "
                f"/api/health never returned 200 within "
                f"{_HEALTH_POLL_TIMEOUT:.0f}s. Daemon log: {logfile}"
            )
            raise RuntimeError(msg)

        # Lost the race (someone else holds the pidfile). First, check for
        # staleness — a dead pid + unhealthy endpoint means the previous
        # winner crashed before bringing the daemon up.
        healthy = await _check_health(base_url)
        if healthy:
            return
        if attempt == 1 and _reclaim_if_stale(pidfile, base_url_healthy=False):
            continue
        # Otherwise the winner is presumably bringing the daemon up —
        # poll until healthy or budget exhausted.
        if await _poll_until_healthy(base_url):
            return
        msg = (
            f"kb-service daemon at {base_url} did not become healthy within "
            f"{_HEALTH_POLL_TIMEOUT:.0f}s (another MCP process holds the "
            f"singleton lock). Daemon log: {logfile}"
        )
        raise RuntimeError(msg)
