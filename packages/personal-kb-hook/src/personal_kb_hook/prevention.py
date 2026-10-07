"""Prevention channels: SessionStart gotcha slice + PreToolUse soft gate.

* **SessionStart** (:func:`session_start`) — one ``GET /api/kb/prevention``
  (1.5 s, no retry). The returned gate settings and Bash cue index are cached
  in ``prevention-<session>.json``; the returned ``slice_text`` (known gotchas
  for the project) is handed back to the CLI to inject as context.
* **PreToolUse** (:func:`pre_tool`) — NO network, no model. Reads the cache
  and, when a Bash call's two-word ``target_class`` exactly matches a cue,
  denies it ONCE with the corrected fact as the reason (at most
  ``max_denies`` per session). In shadow mode it only records ``would_deny``
  — consuming the same deny-once and cap budget, so would-deny counts equal
  what live denies would have been. An identical retry is allowed.
* **Stop** — :func:`stop` records an abandoned pending retry and a summary
  row, :func:`flush_gate_log` POSTs the decision log to
  ``/api/kb/prevention/decisions`` in chunks of 500, and :func:`refresh`
  re-fetches the settings so a server switch flip reaches a live session at
  its next turn end.

Every decision is appended to ``gate-log-<session>.jsonl``; every failed
fetch / flush lands in the shared ``event-drops.jsonl`` drop log. Every public
function here is silent-on-failure and never raises.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any

from personal_kb_hook import cues_lite, telemetry
from personal_kb_hook.defaults import resolve_url_key
from personal_kb_hook.paths import (
    get_event_drop_log_path,
    get_gate_log_path,
    get_prevention_cache_path,
)
from personal_kb_hook.resolver import resolve_project

if TYPE_CHECKING:
    from pathlib import Path

_PREVENTION_PATH = "/api/kb/prevention"
_DECISIONS_PATH = "/api/kb/prevention/decisions"
_FETCH_TIMEOUT: float = 1.5
_FLUSH_TIMEOUT: float = 3.0
_CHUNK = 500
_LOG_MAX_BYTES = 262144
_ORPHAN_MIN_AGE_SECONDS = 3600.0
_CACHE_GC_AGE_SECONDS = 7 * 24 * 3600.0
_REASON_CAP = 1000
_TARGET_CAP = 500

REASON_PREFIX = "KB soft gate (deny once): "
REASON_PREFIX_OBSERVED_ONCE = (
    "KB soft gate (deny once; an earlier session observed this once, unconfirmed): "
)
REASON_SUFFIX = " If you still intend this exact call, retry it unchanged and it will be allowed."

_STATE_DEFAULTS: dict[str, Any] = {
    "denied_resolution_ids": [],
    "deny_count": 0,
    "pending_retry": None,
    "pre_tool_calls": 0,
    "pre_tool_errors": 0,
    "last_error_type": None,
}


# --- small helpers ----------------------------------------------------------


def _hook_version() -> str | None:
    try:
        return importlib.metadata.version("personal-kb-hook")
    except Exception:
        return None


def _session_id(payload: dict[str, Any]) -> str | None:
    sid = payload.get("session_id")
    return sid if isinstance(sid, str) and sid else None


def _record_drop(session_id: str | None, op: str, reason: str, elapsed_ms: int) -> None:
    """Append one line to the shared harness-event drop log; swallow failures."""
    try:
        path = get_event_drop_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > _LOG_MAX_BYTES:
            path.unlink()
        line = {
            "ts": telemetry.now_ts(),
            "session_id": session_id,
            "tool_use_id": None,
            "op": op,
            "reason": reason,
            "elapsed_ms": elapsed_ms,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
    except Exception:
        return


def _drop_reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_{exc.code}"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, urllib.error.URLError):
        return "timeout" if isinstance(exc.reason, TimeoutError) else "urlerror"
    return "error"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_cache(session_id: str) -> dict[str, Any] | None:
    try:
        data = json.loads(get_prevention_cache_path(session_id).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("gate"), dict):
        return None
    return data


def _write_cache(session_id: str, cache: dict[str, Any]) -> None:
    _atomic_write(get_prevention_cache_path(session_id), json.dumps(cache))


# --- fetch (SessionStart + Stop refresh) ------------------------------------


def _fetch(payload: dict[str, Any], session_id: str) -> dict[str, Any] | None:
    """One ``GET /api/kb/prevention``; ``None`` (plus a drop line) on any failure."""
    start = time.monotonic()
    try:
        url_key = resolve_url_key()
        if url_key is None:
            _record_drop(session_id, "prevention_fetch", "no_url_key", 0)
            return None
        url, key = url_key
        cwd = payload.get("cwd")
        cwd_str = cwd if isinstance(cwd, str) else None
        query = urllib.parse.urlencode(
            {
                "project": resolve_project(cwd_str) or "",
                "cwd": cwd_str or "",
                "session_id": session_id,
            }
        )
        req = urllib.request.Request(  # noqa: S310
            url.rstrip("/") + _PREVENTION_PATH + "?" + query,
            headers={"Authorization": f"Bearer {key}"},
            method="GET",
        )
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:  # noqa: S310
            status = int(getattr(resp, "status", None) or resp.getcode())
            raw = resp.read()
        if not 200 <= status < 300:
            elapsed = int((time.monotonic() - start) * 1000)
            _record_drop(session_id, "prevention_fetch", f"http_{status}", elapsed)
            return None
        data = json.loads(raw)
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("gate"), dict)
            or not isinstance(data.get("index"), list)
        ):
            raise ValueError("malformed prevention response")
        return data
    except Exception as exc:
        elapsed = int((time.monotonic() - start) * 1000)
        _record_drop(session_id, "prevention_fetch", _drop_reason(exc), elapsed)
        return None


def _apply_fetch(session_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Rewrite project/gate/index/fetched_ts, preserving the deny-once state."""
    gate = data["gate"]
    cache = _load_cache(session_id) or {}
    for key, default in _STATE_DEFAULTS.items():
        if key not in cache:
            cache[key] = list(default) if isinstance(default, list) else default
    project = data.get("project")
    max_denies = gate.get("max_denies")
    cache["project"] = project if isinstance(project, str) else ""
    cache["gate"] = {
        "enabled": gate.get("enabled") is True,
        "shadow": gate.get("shadow") is not False,
        "max_denies": max_denies if isinstance(max_denies, int) else 0,
    }
    cache["index"] = [e for e in data["index"] if isinstance(e, dict)]
    cache["fetched_ts"] = telemetry.now_ts()
    _write_cache(session_id, cache)
    return cache


def _arm(payload: dict[str, Any], session_id: str, tool: str) -> dict[str, Any] | None:
    """Fetch, cache and record an ``armed`` row; return the response or ``None``."""
    data = _fetch(payload, session_id)
    if data is None:
        return None
    cache = _apply_fetch(session_id, data)
    slice_items = data.get("slice")
    _record(
        session_id,
        cache,
        "armed",
        decision_id=f"cc:{session_id}:armed:{cache['fetched_ts']}",
        tool=tool,
        index_len=len(cache["index"]),
        slice_len=len(slice_items) if isinstance(slice_items, list) else 0,
    )
    return data


def session_start(payload: dict[str, Any]) -> str | None:
    """SessionStart: fetch + cache the gate index; return the slice text or ``None``."""
    try:
        session_id = _session_id(payload)
        if session_id is None:
            return None
        data = _arm(payload, session_id, "SessionStart")
        if data is None:
            return None
        text = data.get("slice_text")
        return text if isinstance(text, str) and text else None
    except Exception:
        return None


def refresh(payload: dict[str, Any]) -> None:
    """Stop: re-fetch settings + index, preserving state (switch flips land here)."""
    try:
        session_id = _session_id(payload)
        if session_id is not None:
            _arm(payload, session_id, "Stop")
    except Exception:
        return


# --- decision rows ----------------------------------------------------------


def _append_row(session_id: str, row: dict[str, Any]) -> None:
    path = get_gate_log_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > _LOG_MAX_BYTES:
        path.unlink()
        _record_drop(session_id, "gate_log_rotated", "size_cap", 0)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _record(session_id: str, cache: dict[str, Any], decision: str, **fields: Any) -> dict[str, Any]:
    """Build a ``GateDecisionRow``-shaped dict and append it to the gate log."""
    engine = telemetry.build_engine()
    gate = cache.get("gate") or {}
    project = cache.get("project")
    row: dict[str, Any] = {
        "session_id": session_id,
        "harness": "claude-code",
        "mode": "headless" if engine else "interactive",
        "engine": engine,
        "host": socket.gethostname(),
        "hook_version": _hook_version(),
        "project": project if isinstance(project, str) else "",
        "shadow": gate.get("shadow") is not False,
        "decision": decision,
        "ts": telemetry.now_ts(),
    }
    row.update(fields)
    _append_row(session_id, row)
    return row


def build_reason(entry: dict[str, Any]) -> str:
    """The deny reason shown to the agent; at most 1000 chars, pinned suffix."""
    prefix = REASON_PREFIX_OBSERVED_ONCE if entry.get("observed_once") else REASON_PREFIX
    wrong = str(entry.get("wrong_belief") or "")
    head = (
        prefix
        + str(entry.get("corrected_fact") or "")
        + (" Earlier wrong belief: " + wrong if wrong else "")
        + f" [{entry.get('provenance_label', '')}; {entry.get('resolution_id', '')}]"
    )
    return head[: _REASON_CAP - len(REASON_SUFFIX)] + REASON_SUFFIX


# --- PreToolUse -------------------------------------------------------------


def _matches(entry: object, tool_name: str, tc: str, target: str) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("tool") != tool_name or entry.get("target_class") != tc:
        return False
    prefix = entry.get("args_prefix")
    if not isinstance(prefix, str) or not prefix.strip():
        return True
    wanted = prefix.split()
    args = cues_lite.bash_args_after_class(target)
    return args[: len(wanted)] == wanted


def _gate(
    session_id: str,
    tool_name: str,
    tool_use_id: str,
    tool_input: dict[str, Any],
    cache: dict[str, Any],
) -> str | None:
    target = cues_lite.extract_target(tool_name, tool_input)
    per_call = {"tool_use_id": tool_use_id, "tool": tool_name, "target": target[:_TARGET_CAP]}

    pending = cache.get("pending_retry")
    if isinstance(pending, dict) and pending.get("tool") == tool_name:
        prior = str(pending.get("target", ""))
        _record(
            session_id,
            cache,
            "retry",
            decision_id=f"cc:{session_id}:{tool_use_id}:retry",
            target_class=cues_lite.target_class(tool_name, target),
            resolution_id=str(pending.get("resolution_id", "")),
            observed_once=False,
            retry_changed_command=target != prior,
            prior_target=prior[:_TARGET_CAP],
            **per_call,
        )
        cache["pending_retry"] = None

    tc = cues_lite.target_class(tool_name, target)
    if not tc:
        return None
    index = cache.get("index")
    entries = index if isinstance(index, list) else []
    match = next((e for e in entries if _matches(e, tool_name, tc, target)), None)
    if match is None:
        return None

    rid = str(match.get("resolution_id", ""))
    matched = {
        **per_call,
        "target_class": tc,
        "resolution_id": rid,
        "resolution_updated_at": match.get("updated_at"),
        "observed_once": match.get("observed_once") is True,
        "retry_changed_command": None,
    }

    def _row(decision: str, **extra: Any) -> None:
        _record(
            session_id,
            cache,
            decision,
            decision_id=f"cc:{session_id}:{tool_use_id}:{decision}",
            **matched,
            **extra,
        )

    denied_ids = cache.get("denied_resolution_ids")
    if not isinstance(denied_ids, list):
        denied_ids = []
    if rid in denied_ids:
        _row("skipped_already_denied")
        return None
    gate = cache["gate"]
    deny_count = int(cache.get("deny_count") or 0)
    if deny_count >= int(gate.get("max_denies") or 0):
        _row("skipped_cap")
        return None

    cache["denied_resolution_ids"] = [*denied_ids, rid]
    cache["deny_count"] = deny_count + 1
    reason = build_reason(match)
    if gate.get("shadow") is not False:
        _row("would_deny", reason_excerpt=reason)
        return None
    cache["pending_retry"] = {
        "tool": tool_name,
        "target": target,
        "resolution_id": rid,
        "tool_use_id": tool_use_id,
    }
    _row("denied", reason_excerpt=reason)
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
    )


def pre_tool(payload: dict[str, Any]) -> str | None:
    """PreToolUse soft gate: a deny envelope (JSON string) or ``None``. No network."""
    try:
        session_id = _session_id(payload)
        tool_name = payload.get("tool_name")
        tool_use_id = payload.get("tool_use_id")
        if session_id is None or not isinstance(tool_name, str) or not tool_name:
            return None
        if not isinstance(tool_use_id, str) or not tool_use_id:
            return None
        raw_input = payload.get("tool_input")
        tool_input = raw_input if isinstance(raw_input, dict) else {}
        cache = _load_cache(session_id)
        if cache is None or cache["gate"].get("enabled") is not True:
            return None
        cache["pre_tool_calls"] = int(cache.get("pre_tool_calls") or 0) + 1
        try:
            result = _gate(session_id, tool_name, tool_use_id, tool_input, cache)
        except Exception as exc:
            cache["pre_tool_errors"] = int(cache.get("pre_tool_errors") or 0) + 1
            cache["last_error_type"] = type(exc).__name__
            _write_cache(session_id, cache)
            return None
        _write_cache(session_id, cache)
        return result
    except Exception:
        return None


# --- Stop: summary, flush ---------------------------------------------------


def stop(payload: dict[str, Any]) -> None:
    """Record an abandoned pending retry and a summary row; reset the counters."""
    try:
        session_id = _session_id(payload)
        if session_id is None:
            return
        cache = _load_cache(session_id)
        if cache is None:
            return
        pending = cache.get("pending_retry")
        if isinstance(pending, dict):
            _record(
                session_id,
                cache,
                "retry",
                decision_id=f"cc:{session_id}:{pending.get('tool_use_id')}:retry",
                tool=str(pending.get("tool") or "Bash"),
                target="",
                prior_target=str(pending.get("target", ""))[:_TARGET_CAP],
                resolution_id=str(pending.get("resolution_id", "")),
                retry_changed_command=None,
                tool_use_id=pending.get("tool_use_id"),
            )
            cache["pending_retry"] = None
        ts = telemetry.now_ts()
        _record(
            session_id,
            cache,
            "summary",
            decision_id=f"cc:{session_id}:summary:{ts}",
            tool="Stop",
            pre_tool_calls=int(cache.get("pre_tool_calls") or 0),
            pre_tool_errors=int(cache.get("pre_tool_errors") or 0),
            last_error_type=cache.get("last_error_type"),
        )
        cache["pre_tool_calls"] = 0
        cache["pre_tool_errors"] = 0
        cache["last_error_type"] = None
        _write_cache(session_id, cache)
    except Exception:
        return


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _post_rows(url: str, key: str, rows: list[dict[str, Any]]) -> str | None:
    """POST one chunk; ``None`` on 2xx, else a drop reason."""
    try:
        req = urllib.request.Request(  # noqa: S310
            url.rstrip("/") + _DECISIONS_PATH,
            data=json.dumps({"rows": rows}).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_FLUSH_TIMEOUT) as resp:  # noqa: S310
            status = int(getattr(resp, "status", None) or resp.getcode())
        return None if 200 <= status < 300 else f"http_{status}"
    except Exception as exc:
        return _drop_reason(exc)


def _flush_file(path: Path, session_id: str | None) -> None:
    """POST a (flushing) gate log; delete it only when every chunk got a 2xx."""
    start = time.monotonic()
    url_key = resolve_url_key()
    if url_key is None:
        _record_drop(session_id, "gate_flush", "no_url_key", 0)
        return
    url, key = url_key
    rows = _read_rows(path)
    failure: str | None = None
    for i in range(0, len(rows), _CHUNK):
        reason = _post_rows(url, key, rows[i : i + _CHUNK])
        if reason is not None and failure is None:
            failure = reason
    if failure is None:
        path.unlink(missing_ok=True)
    else:
        elapsed = int((time.monotonic() - start) * 1000)
        _record_drop(session_id, "gate_flush", failure, elapsed)


def _flushing_path(session_id: str) -> Path:
    return get_gate_log_path(session_id).with_name(f"gate-log-{session_id}.flushing.jsonl")


def flush_gate_log(session_id: str) -> None:
    """Stop: move the gate log to ``.flushing`` and POST it in chunks of 500."""
    try:
        log = get_gate_log_path(session_id)
        flushing = _flushing_path(session_id)
        if log.exists():
            if flushing.exists():
                combined = flushing.read_text(encoding="utf-8")
                if combined and not combined.endswith("\n"):
                    combined += "\n"
                combined += log.read_text(encoding="utf-8")
                _atomic_write(flushing, combined)
                log.unlink(missing_ok=True)
            else:
                os.replace(log, flushing)
        if flushing.exists():
            _flush_file(flushing, session_id)
    except Exception:
        return


def _session_from_log_name(name: str) -> str | None:
    if not name.startswith("gate-log-") or not name.endswith(".jsonl"):
        return None
    inner = name[len("gate-log-") : -len(".jsonl")]
    inner = inner.removesuffix(".flushing")
    return inner or None


def orphan_sweep(current_session_id: str) -> None:
    """SessionStart: flush other sessions' stale gate logs; GC old caches.

    Only files untouched for an hour are flushed (a live parallel session's
    log is left alone), bounded by :data:`telemetry._ORPHAN_SWEEP_CAP` files
    and :data:`telemetry._ORPHAN_SWEEP_BUDGET_SECONDS` of wall time. Prevention
    caches older than seven days are unlinked.
    """
    try:
        cache_dir = get_gate_log_path("placeholder").parent
        if not cache_dir.exists():
            return
        now = time.time()
        for cache_file in cache_dir.glob("prevention-*.json"):
            with contextlib.suppress(OSError):
                if now - cache_file.stat().st_mtime > _CACHE_GC_AGE_SECONDS:
                    cache_file.unlink(missing_ok=True)
        swept = 0
        deadline = time.monotonic() + telemetry._ORPHAN_SWEEP_BUDGET_SECONDS
        for path in sorted(cache_dir.glob("gate-log-*.jsonl")):
            if swept >= telemetry._ORPHAN_SWEEP_CAP:
                return
            sid = _session_from_log_name(path.name)
            if sid is None or sid == current_session_id:
                continue
            try:
                if now - path.stat().st_mtime < _ORPHAN_MIN_AGE_SECONDS:
                    continue
            except OSError:
                continue
            if time.monotonic() >= deadline:
                return
            swept += 1
            _flush_file(path, sid)
    except Exception:
        return
