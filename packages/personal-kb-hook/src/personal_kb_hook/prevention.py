"""Prevention channels: SessionStart gotcha slice + PreToolUse soft gate.

* **SessionStart** (:func:`session_start`) — one ``GET /api/kb/prevention``
  (1.5 s, no retry). The returned gate settings and Bash cue index are cached
  in ``prevention-<session>.json``; the returned ``slice_text`` (known gotchas
  for the project) is handed back to the CLI to inject as context.
* **PreToolUse** (:func:`pre_tool`) — NO network, no model. Reads the cache
  and, when a Bash call's two-word ``target_class`` exactly matches a cue,
  denies it with the corrected fact as the reason. Per-lesson state lives in
  ``deny_state`` (``{resolution_id: {last_deny_ts, overridden}}``): a lesson
  may deny again once ``rearm_hours`` have passed since its last deny, or
  after a SessionStart with source ``compact`` / ``resume`` / ``clear``
  re-arms every lesson (a ``rearmed`` row). Denies are rate limited to
  ``max_denies_per_turn`` per turn (reset by UserPromptSubmit and Stop) and
  ``max_denies_per_hour`` in any rolling 60 minutes; an over-limit match
  records ``skipped_cap`` with ``reason`` ``per_turn`` or ``per_hour``. In
  shadow mode it only records ``would_deny`` — consuming the same per-lesson
  state and limits, so would-deny counts equal what live denies would have
  been. An identical retry is allowed and marks the lesson ``overridden``
  (quiet until its next re-arm).
* **Stop** — :func:`stop` records an abandoned pending retry and a summary
  row, :func:`flush_gate_log` POSTs the decision log to
  ``/api/kb/prevention/decisions`` in chunks of 500, and :func:`refresh`
  re-fetches the settings (including the top-level ``surprise_capture`` mode)
  so a server switch flip reaches a live session at its next turn end.
* **PostToolUseFailure** (:func:`failure_context`) — opt-in via
  ``KB_FAILURE_CONTEXT`` (off when unset), cache-only with NO network. A
  failed Bash call is checked against the cached index with the same matcher
  as the gate (:func:`_find_match`); on a match the corrected fact is returned
  as same-turn ``additionalContext``. A resolution is not repeated within the
  gate's ``rearm_hours`` of its last delivery (tracked as id -> timestamp in
  ``failure-context-<session>.json``); it delivers again once the window has
  passed, and a SessionStart with source compact / resume / clear clears the
  state. It records ``failure_context`` (delivered),
  ``failure_context_repeat`` (delivered within the window, same failure
  again) and ``failure_context_error`` rows, and is independent of shadow mode and the deny limits.

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
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from personal_kb_hook import cues_lite, telemetry
from personal_kb_hook.defaults import resolve_url_key
from personal_kb_hook.paths import (
    get_event_drop_log_path,
    get_failure_context_state_path,
    get_gate_log_path,
    get_prevention_cache_path,
)
from personal_kb_hook.resolver import resolve_project
from personal_kb_hook.tool_inventory import _OFF_VALUES

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
# Longer than the server's 30-day turn_events retention, so a counter removed
# by GC cannot restart and reuse an event_id the server still retains.
_TURN_STATE_GC_AGE_SECONDS = 31 * 24 * 3600.0
# Server surprise-capture modes; lives here (not turn_digest) to avoid a cycle.
SURPRISE_CAPTURE_MODES: tuple[str, ...] = ("off", "shadow", "on")
_REASON_CAP = 1000
_TARGET_CAP = 500

REASON_PREFIX = "KB soft gate (deny once): "
REASON_PREFIX_OBSERVED_ONCE = (
    "KB soft gate (deny once; an earlier session observed this once, unconfirmed): "
)
REASON_SUFFIX = " If you still intend this exact call, retry it unchanged and it will be allowed."

FAILURE_CONTEXT_PREFIX = "KB: this failure matches a known correction: "
FAILURE_CONTEXT_PREFIX_OBSERVED_ONCE = (
    "KB: this failure matches a correction an earlier session observed once (unconfirmed): "
)
_FAILURE_CONTEXT_ON_VALUES = frozenset({"1", "true", "yes", "on"})

# Gate rate-limit defaults, used when the server omits a setting (older server).
DEFAULT_MAX_DENIES_PER_TURN = 1
DEFAULT_MAX_DENIES_PER_HOUR = 6
DEFAULT_REARM_HOURS = 24
_SETTING_MAX = 1000
_DENY_TS_CAP = 1000
# SessionStart sources after which the agent may have lost a lesson's deny
# reason from its context; each one re-arms every lesson.
REARM_SOURCES = frozenset({"compact", "resume", "clear"})

_STATE_DEFAULTS: dict[str, Any] = {
    "deny_state": {},
    "deny_timestamps": [],
    "turn_denies": 0,
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


def _parse_ts(raw: object) -> datetime | None:
    """Parse an ISO timestamp (naive means UTC); ``None`` when unparseable."""
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _migrate_state(data: dict[str, Any]) -> None:
    """Normalise per-lesson state in place; migrate a legacy deny-once cache.

    An old cache's ``denied_resolution_ids`` list becomes ``deny_state``
    entries stamped now (not overridden), so those lessons re-arm after
    ``rearm_hours`` rather than never.
    """
    state = data.get("deny_state")
    if not isinstance(state, dict):
        state = {}
    legacy = data.pop("denied_resolution_ids", None)
    if isinstance(legacy, list):
        now = telemetry.now_ts()
        for rid in legacy:
            if isinstance(rid, str) and rid not in state:
                state[rid] = {"last_deny_ts": now, "overridden": False}
    data.pop("deny_count", None)
    data["deny_state"] = {
        k: v for k, v in state.items() if isinstance(k, str) and isinstance(v, dict)
    }
    stamps = data.get("deny_timestamps")
    data["deny_timestamps"] = (
        [t for t in stamps if isinstance(t, str)][-_DENY_TS_CAP:]
        if isinstance(stamps, list)
        else []
    )
    if not isinstance(data.get("turn_denies"), int):
        data["turn_denies"] = 0


def _load_cache(session_id: str) -> dict[str, Any] | None:
    try:
        data = json.loads(get_prevention_cache_path(session_id).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("gate"), dict):
        return None
    _migrate_state(data)
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


def _setting(gate: dict[str, Any], key: str, default: int) -> int:
    """A gate integer setting in 1..1000, else *default* (an older server omits it)."""
    value = gate.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value if 1 <= value <= _SETTING_MAX else default


def _apply_fetch(session_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Rewrite project/gate/index/fetched_ts, preserving the per-lesson state."""
    gate = data["gate"]
    cache = _load_cache(session_id) or {}
    for key, default in _STATE_DEFAULTS.items():
        if key not in cache:
            cache[key] = type(default)(default) if isinstance(default, list | dict) else default
    project = data.get("project")
    cache["project"] = project if isinstance(project, str) else ""
    cache["gate"] = {
        "enabled": gate.get("enabled") is True,
        "shadow": gate.get("shadow") is not False,
        "max_denies_per_turn": _setting(gate, "max_denies_per_turn", DEFAULT_MAX_DENIES_PER_TURN),
        "max_denies_per_hour": _setting(gate, "max_denies_per_hour", DEFAULT_MAX_DENIES_PER_HOUR),
        "rearm_hours": _setting(gate, "rearm_hours", DEFAULT_REARM_HOURS),
    }
    cache["index"] = [e for e in data["index"] if isinstance(e, dict)]
    mode = data.get("surprise_capture")
    cache["surprise_capture"] = (
        mode if isinstance(mode, str) and mode in SURPRISE_CAPTURE_MODES else "off"
    )
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


def _rearm(session_id: str, source: str) -> None:
    """Clear every lesson's deny state and record one ``rearmed`` row."""
    cache = _load_cache(session_id)
    cleared = 0
    if cache is not None:
        cleared = len(cache.get("deny_state") or {})
        cache["deny_state"] = {}
        cache["turn_denies"] = 0
        _write_cache(session_id, cache)
    fc_cleared = len(_load_failure_state(session_id))
    with contextlib.suppress(OSError):
        get_failure_context_state_path(session_id).unlink()
    ts = telemetry.now_ts()
    _record(
        session_id,
        cache if cache is not None else {},
        "rearmed",
        decision_id=f"cc:{session_id}:rearmed:{ts}",
        tool="SessionStart",
        source=source,
        cleared=cleared,
        failure_context_cleared=fc_cleared,
    )


def session_start(payload: dict[str, Any]) -> str | None:
    """SessionStart: fetch + cache the gate index; return the slice text or ``None``.

    A ``compact`` / ``resume`` / ``clear`` source first re-arms every lesson
    (the agent may no longer remember earlier deny reasons); ``startup``
    does not.
    """
    try:
        session_id = _session_id(payload)
        if session_id is None:
            return None
        source = payload.get("source")
        if isinstance(source, str) and source in REARM_SOURCES:
            with contextlib.suppress(Exception):
                _rearm(session_id, source)
        data = _arm(payload, session_id, "SessionStart")
        if data is None:
            return None
        if os.environ.get("KB_GOTCHA_SLICE", "").strip().lower() in _OFF_VALUES:
            return None  # gate_only arm: index stays armed, slice text is withheld
        text = data.get("slice_text")
        return text if isinstance(text, str) and text else None
    except Exception:
        return None


def new_turn(payload: dict[str, Any]) -> None:
    """UserPromptSubmit: reset the per-turn deny counter. No network."""
    try:
        session_id = _session_id(payload)
        if session_id is None:
            return
        cache = _load_cache(session_id)
        if cache is None or not cache.get("turn_denies"):
            return
        cache["turn_denies"] = 0
        _write_cache(session_id, cache)
    except Exception:
        return


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


def _reason_body(entry: dict[str, Any]) -> str:
    """Corrected fact, earlier wrong belief and provenance — shared by both channels."""
    wrong = str(entry.get("wrong_belief") or "")
    return (
        str(entry.get("corrected_fact") or "")
        + (" Earlier wrong belief: " + wrong if wrong else "")
        + f" [{entry.get('provenance_label', '')}; {entry.get('resolution_id', '')}]"
    )


def build_reason(entry: dict[str, Any]) -> str:
    """The deny reason shown to the agent; at most 1000 chars, pinned suffix."""
    prefix = REASON_PREFIX_OBSERVED_ONCE if entry.get("observed_once") else REASON_PREFIX
    return (prefix + _reason_body(entry))[: _REASON_CAP - len(REASON_SUFFIX)] + REASON_SUFFIX


def build_failure_context(entry: dict[str, Any]) -> str:
    """The PostToolUseFailure context text; at most 1000 chars, no retry suffix."""
    prefix = (
        FAILURE_CONTEXT_PREFIX_OBSERVED_ONCE
        if entry.get("observed_once")
        else FAILURE_CONTEXT_PREFIX
    )
    return (prefix + _reason_body(entry))[:_REASON_CAP]


# --- PreToolUse -------------------------------------------------------------


def _matches(entry: object, tool_name: str, tc: str, args: list[str]) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("tool") != tool_name or entry.get("target_class") != tc:
        return False
    prefix = entry.get("args_prefix")
    if not isinstance(prefix, str) or not prefix.strip():
        return True
    wanted = prefix.split()
    return args[: len(wanted)] == wanted


def _find_match(
    tool_name: str, target: str, entries: list[Any]
) -> tuple[dict[str, Any] | None, str]:
    """First index entry matching a candidate (segment order, then index order)."""
    if tool_name == "Bash":
        candidates = cues_lite.bash_segments(target)
    else:
        tc0 = cues_lite.target_class(tool_name, target)
        candidates = [(tc0, cues_lite.bash_args_after_class(target))] if tc0 else []
    for seg_class, seg_args in candidates:
        found = next((e for e in entries if _matches(e, tool_name, seg_class, seg_args)), None)
        if found is not None:
            return found, seg_class
    return None, ""


def _gate(
    session_id: str,
    tool_name: str,
    tool_use_id: str,
    tool_input: dict[str, Any],
    cache: dict[str, Any],
) -> str | None:
    target = cues_lite.extract_target(tool_name, tool_input)
    per_call = {"tool_use_id": tool_use_id, "tool": tool_name, "target": target[:_TARGET_CAP]}

    deny_state: dict[str, Any] = cache["deny_state"]
    pending = cache.get("pending_retry")
    if isinstance(pending, dict) and pending.get("tool") == tool_name:
        prior = str(pending.get("target", ""))
        pending_rid = str(pending.get("resolution_id", ""))
        retry_tc = cues_lite.target_class(tool_name, target)
        _record(
            session_id,
            cache,
            "retry",
            decision_id=f"cc:{session_id}:{tool_use_id}:retry",
            target_class=retry_tc,
            resolution_id=pending_rid,
            observed_once=False,
            retry_changed_command=target != prior,
            prior_target=prior[:_TARGET_CAP],
            **per_call,
        )
        cache["pending_retry"] = None
        if target == prior and pending_rid:
            lesson = deny_state.get(pending_rid)
            if not isinstance(lesson, dict):
                lesson = {"last_deny_ts": telemetry.now_ts()}
            deny_state[pending_rid] = {**lesson, "overridden": True}
            _record(
                session_id,
                cache,
                "overridden",
                decision_id=f"cc:{session_id}:{tool_use_id}:overridden",
                target_class=retry_tc,
                resolution_id=pending_rid,
                observed_once=False,
                retry_changed_command=False,
                prior_target=prior[:_TARGET_CAP],
                **per_call,
            )

    index = cache.get("index")
    entries = index if isinstance(index, list) else []
    match, tc = _find_match(tool_name, target, entries)
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

    gate = cache["gate"]
    now = datetime.now(UTC)
    rearm = timedelta(hours=_setting(gate, "rearm_hours", DEFAULT_REARM_HOURS))
    lesson = deny_state.get(rid)
    if isinstance(lesson, dict):
        last = _parse_ts(lesson.get("last_deny_ts"))
        if last is not None and now - last < rearm:
            _row(
                "skipped_already_denied",
                reason="overridden" if lesson.get("overridden") is True else "not_rearmed",
            )
            return None
    hour_ago = now - timedelta(hours=1)
    recent = [
        t for t in cache["deny_timestamps"] if (p := _parse_ts(t)) is not None and p > hour_ago
    ]
    cache["deny_timestamps"] = recent
    turn_denies = int(cache.get("turn_denies") or 0)
    if turn_denies >= _setting(gate, "max_denies_per_turn", DEFAULT_MAX_DENIES_PER_TURN):
        _row("skipped_cap", reason="per_turn")
        return None
    if len(recent) >= _setting(gate, "max_denies_per_hour", DEFAULT_MAX_DENIES_PER_HOUR):
        _row("skipped_cap", reason="per_hour")
        return None

    now_iso = now.isoformat()
    deny_state[rid] = {"last_deny_ts": now_iso, "overridden": False}
    cache["deny_timestamps"] = [*recent, now_iso][-_DENY_TS_CAP:]
    cache["turn_denies"] = turn_denies + 1
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


# --- PostToolUseFailure: failure context ------------------------------------


def _load_failure_state(session_id: str) -> dict[str, str]:
    """Map of resolution id -> ISO UTC delivery timestamp for this session.

    A legacy ``delivered_resolution_ids`` file maps each id to the file's
    mtime. Anything unparseable loads as ``{}`` (worst case: one extra delivery).
    """
    path = get_failure_context_state_path(session_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        delivered = data.get("delivered")
        if isinstance(delivered, dict):
            return {k: v for k, v in delivered.items() if isinstance(k, str) and isinstance(v, str)}
        ids = data.get("delivered_resolution_ids")
        if isinstance(ids, list):
            mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
            return {i: mtime for i in ids if isinstance(i, str)}
    except Exception:
        return {}
    return {}


def failure_context(payload: dict[str, Any]) -> str | None:
    """PostToolUseFailure: an ``additionalContext`` envelope or ``None``. No network.

    Opt-in via ``KB_FAILURE_CONTEXT``. A failed Bash call that matches the
    session's cached gate index gets the corrected fact, at most once per
    resolution within the gate's ``rearm_hours`` of its last delivery (state is
    cleared on a compact / resume / clear SessionStart), independent of shadow
    mode and the deny limits.
    Never raises and never writes the prevention cache.
    """
    sid: str | None = None
    tool_use_id: object = None
    cache: dict[str, Any] | None = None
    try:
        if os.environ.get("KB_FAILURE_CONTEXT", "").strip().lower() not in (
            _FAILURE_CONTEXT_ON_VALUES
        ):
            return None
        sid = _session_id(payload)
        if sid is None:
            return None
        if payload.get("tool_name") != "Bash":
            return None
        tool_use_id = payload.get("tool_use_id")
        if not isinstance(tool_use_id, str) or not tool_use_id:
            return None
        if payload.get("is_interrupt") is True:
            return None
        raw_input = payload.get("tool_input")
        tool_input = raw_input if isinstance(raw_input, dict) else {}
        target = cues_lite.extract_target("Bash", tool_input)
        if target == "":
            return None
        cache = _load_cache(sid)
        if cache is None:
            return None
        if cache["gate"].get("enabled") is not True:
            return None
        pending = cache.get("pending_retry")
        if isinstance(pending, dict) and pending.get("tool_use_id") == tool_use_id:
            return None
        index = cache.get("index")
        entries = index if isinstance(index, list) else []
        match, tc = _find_match("Bash", target, entries)
        if match is None:
            return None

        rid = str(match.get("resolution_id", ""))
        fields: dict[str, Any] = {
            "tool": "Bash",
            "tool_use_id": tool_use_id,
            "target": target[:_TARGET_CAP],
            "target_class": tc,
            "resolution_id": rid,
            "resolution_updated_at": match.get("updated_at"),
            "observed_once": match.get("observed_once") is True,
            "retry_changed_command": None,
            "shadow": False,
        }
        state = _load_failure_state(sid)
        rearm = timedelta(hours=_setting(cache["gate"], "rearm_hours", DEFAULT_REARM_HOURS))
        last = _parse_ts(state.get(rid))
        if last is not None and datetime.now(UTC) - last < rearm:
            _record(
                sid,
                cache,
                "failure_context_repeat",
                decision_id=f"cc:{sid}:{tool_use_id}:failure_context_repeat",
                **fields,
            )
            return None

        try:
            _atomic_write(
                get_failure_context_state_path(sid),
                json.dumps({"delivered": {**state, rid: telemetry.now_ts()}}),
            )
        except Exception:
            _record_drop(sid, "failure_context", "state_write", 0)
            return None
        text = build_failure_context(match)
        try:
            _record(
                sid,
                cache,
                "failure_context",
                decision_id=f"cc:{sid}:{tool_use_id}:failure_context",
                reason_excerpt=text,
                **fields,
            )
        except Exception:
            _record_drop(sid, "failure_context", "record_failed", 0)
        return json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUseFailure",
                    "additionalContext": text,
                }
            }
        )
    except Exception as exc:
        _record_drop(sid, "failure_context", type(exc).__name__, 0)
        if isinstance(sid, str) and sid and isinstance(tool_use_id, str) and tool_use_id:
            with contextlib.suppress(Exception):
                _record(
                    sid,
                    cache if cache is not None else {},
                    "failure_context_error",
                    decision_id=f"cc:{sid}:{tool_use_id}:failure_context_error",
                    tool="Bash",
                    tool_use_id=tool_use_id,
                    last_error_type=type(exc).__name__,
                    shadow=False,
                )
        return None


# --- Stop: summary, flush ---------------------------------------------------


def stop(payload: dict[str, Any]) -> None:
    """Record an abandoned pending retry and a summary row; reset the counters.

    Stop also ends the turn, so the per-turn deny counter resets here.
    """
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
        cache["turn_denies"] = 0
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
    caches older than seven days are unlinked, as are ``turn-digest-log-*``
    files older than seven days, other sessions' ``turn-state-*`` files
    older than 31 days, and failure-context-* state files older than seven days.
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
        for fc_file in cache_dir.glob("failure-context-*.json"):
            with contextlib.suppress(OSError):
                if now - fc_file.stat().st_mtime > _CACHE_GC_AGE_SECONDS:
                    fc_file.unlink(missing_ok=True)
        own_state = f"turn-state-{current_session_id}.json"
        for state_file in cache_dir.glob("turn-state-*.json"):
            if state_file.name == own_state:
                continue
            with contextlib.suppress(OSError):
                if now - state_file.stat().st_mtime > _TURN_STATE_GC_AGE_SECONDS:
                    state_file.unlink(missing_ok=True)
        for log_file in cache_dir.glob("turn-digest-log-*.jsonl"):
            with contextlib.suppress(OSError):
                if now - log_file.stat().st_mtime > _CACHE_GC_AGE_SECONDS:
                    log_file.unlink(missing_ok=True)
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
