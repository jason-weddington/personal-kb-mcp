"""Harness-event forwarding: ``PostToolUseFailure`` → ``POST /api/kb/event``.

Record-only feed for the server-side failure-cue index. :func:`post_failure`
builds one ``post_tool`` event body from the Claude Code hook payload and
sends it once (1.5 s timeout, no retry). It never raises. This module never
writes stdout; the PostToolUseFailure failure context is produced separately
by prevention.failure_context.

Every event that could not be delivered is appended as one jsonl line to the
bounded drop log (:func:`personal_kb_hook.paths.get_event_drop_log_path`), so
a quiet server-side heartbeat can be told apart from a broken pipeline.

Stdlib-only, like the rest of the hook.
"""

from __future__ import annotations

import importlib.metadata
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from personal_kb_hook import telemetry
from personal_kb_hook.defaults import resolve_url_key
from personal_kb_hook.paths import get_event_drop_log_path
from personal_kb_hook.resolver import resolve_project

_EVENT_PATH = "/api/kb/event"
_TIMEOUT: float = 1.5
_MAX_INPUT_STR = 2000
_MAX_ERROR = 4000
_DROP_LOG_MAX_BYTES = 262144


def _hook_version() -> str | None:
    try:
        return importlib.metadata.version("personal-kb-hook")
    except Exception:
        return None


def _scalar_input(tool_input: object) -> dict[str, Any]:
    """Keep only top-level str/int/float/bool values; cut strings to 2000 chars."""
    if not isinstance(tool_input, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in tool_input.items():
        if isinstance(value, str):
            out[key] = value[:_MAX_INPUT_STR]
        elif isinstance(value, (bool, int, float)):
            out[key] = value
    return out


def _record_drop(session_id: object, tool_use_id: object, reason: str, elapsed_ms: int) -> None:
    """Append one drop line; swallow every failure."""
    try:
        path = get_event_drop_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > _DROP_LOG_MAX_BYTES:
            path.unlink()
        line = {
            "ts": telemetry.now_ts(),
            "session_id": session_id if isinstance(session_id, str) else None,
            "tool_use_id": tool_use_id if isinstance(tool_use_id, str) else None,
            "reason": reason,
            "elapsed_ms": elapsed_ms,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
    except Exception:
        return


def _build_body(payload: dict[str, Any]) -> dict[str, Any]:
    session_id: str = payload["session_id"]
    tool_use_id: str = payload["tool_use_id"]
    error: str = payload["error"]
    cwd = payload.get("cwd")
    cwd_str = cwd if isinstance(cwd, str) else None
    engine = telemetry.build_engine()
    is_interrupt = payload.get("is_interrupt")
    duration_ms = payload.get("duration_ms")
    return {
        "type": "post_tool",
        "event_id": f"cc:{session_id}:{tool_use_id}",
        "session_id": session_id,
        "harness": "claude-code",
        "mode": "headless" if engine else "interactive",
        "engine": engine,
        "host": socket.gethostname(),
        "hook_version": _hook_version(),
        "cwd": cwd_str,
        "project": resolve_project(cwd_str),
        "ts": telemetry.now_ts(),
        "tool_name": payload["tool_name"],
        "tool_input": _scalar_input(payload.get("tool_input")),
        "tool_use_id": tool_use_id,
        "error": error[:_MAX_ERROR],
        "is_error": True,
        "is_interrupt": is_interrupt if isinstance(is_interrupt, bool) else False,
        "duration_ms": (
            duration_ms
            if isinstance(duration_ms, int) and not isinstance(duration_ms, bool)
            else None
        ),
    }


def _drop_reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_{exc.code}"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, urllib.error.URLError):
        return "timeout" if isinstance(exc.reason, TimeoutError) else "urlerror"
    return "error"


def post_failure(payload: dict[str, Any]) -> None:
    """Forward one ``PostToolUseFailure`` payload as a ``post_tool`` event.

    Never raises and never writes stdout. Undeliverable events land in the
    drop log with a reason (``missing_fields``, ``no_url_key``, ``timeout``,
    ``urlerror``, ``http_<code>`` or ``error``).
    """
    session_id = payload.get("session_id")
    tool_use_id = payload.get("tool_use_id")
    start = time.monotonic()
    try:
        required = ("session_id", "tool_use_id", "tool_name", "error")
        if not all(isinstance(payload.get(k), str) and payload.get(k) for k in required):
            _record_drop(session_id, tool_use_id, "missing_fields", 0)
            return
        url_key = resolve_url_key()
        if url_key is None:
            _record_drop(session_id, tool_use_id, "no_url_key", 0)
            return
        url, key = url_key
        body = _build_body(payload)
        req = urllib.request.Request(  # noqa: S310
            url.rstrip("/") + _EVENT_PATH,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
            status = int(getattr(resp, "status", None) or resp.getcode())
        if not 200 <= status < 300:
            elapsed = int((time.monotonic() - start) * 1000)
            _record_drop(session_id, tool_use_id, f"http_{status}", elapsed)
    except Exception as exc:
        elapsed = int((time.monotonic() - start) * 1000)
        _record_drop(session_id, tool_use_id, _drop_reason(exc), elapsed)
