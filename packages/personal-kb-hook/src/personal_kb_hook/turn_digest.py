"""Stop-time turn digest for surprise capture.

At every ``Stop`` the hook builds a TURN DIGEST from the transcript window and
``last_assistant_message`` and hands it to a detached
``python -m personal_kb_hook.turn_sender``, which POSTs it to ``/api/kb/turn``.

* ``event_id`` is exactly ``<session_id>:<turn_index>`` with NO ``cc:`` prefix,
  unlike the hook's other ids. ``turn_index`` counts every Stop in the session,
  whatever the capture mode.
* Window rule: the records newer than the later of the previous digest's
  ``last_uuid`` and the latest human prompt (see :func:`human_prompt_text`).
* The hook sends raw text; the server redacts secrets before storage.
* The send is gated only on the cached ``surprise_capture`` being ``shadow`` or
  ``on`` (decision R10), never on the listener switches.
* The send is fire-and-forget through a detached sender process.
* Items are truncated oldest-first to fit 65536 bytes.
* The caps are drift-guarded against ``kb_service`` by
  ``tests/test_turn_digest_drift_guard.py``.
"""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from personal_kb_hook import cues_lite, prevention, telemetry
from personal_kb_hook.defaults import resolve_url_key
from personal_kb_hook.paths import get_turn_digest_log_path, get_turn_state_path

TURN_USER_PROMPT_MAX = 4000
TURN_FINAL_MESSAGE_MAX = 4000
TURN_TEXT_MAX = 2000
TURN_TARGET_MAX = 500
TURN_EXCERPT_MAX = 1500
TURN_ITEMS_MAX = 200
TURN_DIGEST_MAX_BYTES = 65536
TURN_TAIL_BYTES = 8388608
EXCERPT_HEAD = 746
EXCERPT_MARKER = "\n[...]\n"
EXCERPT_TAIL = 747
SEND_MODES = frozenset({"shadow", "on"})
KNOWN_BLOCK_TYPES = frozenset(
    {"text", "tool_use", "tool_result", "thinking", "redacted_thinking", "image"}
)
SENDER_MODULE = "personal_kb_hook.turn_sender"
DROP_OP = "turn_digest"
PROMPT_EXCLUDED_PREFIXES: tuple[str, ...] = (
    "<" + "command-name>",
    "<" + "command-message>",
    "<" + "local-command-",
    "[Request interrupted by user",
)


def human_prompt_text(record: object) -> str | None:
    """Return the human prompt text of a transcript record, else ``None``."""
    if not isinstance(record, dict):
        return None
    if record.get("type") != "user":
        return None
    if record.get("isSidechain") is True or record.get("isMeta") is True:
        return None
    if record.get("isCompactSummary") is True:
        return None
    origin = record.get("origin")
    if isinstance(origin, dict) and origin.get("kind") != "human":
        return None
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        text = "\n".join(
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        )
    else:
        return None
    if text.strip() == "" or text.lstrip().startswith(PROMPT_EXCLUDED_PREFIXES):
        return None
    return text


def result_excerpt(content: object) -> str:
    """Head/tail excerpt of a tool_result's text content (<= TURN_EXCERPT_MAX)."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        )
    else:
        return ""
    if len(text) <= TURN_EXCERPT_MAX:
        return text
    return text[:EXCERPT_HEAD] + EXCERPT_MARKER + text[-EXCERPT_TAIL:]


@dataclass(frozen=True)
class TurnWindow:
    """The transcript window of one turn."""

    user_prompt: str | None
    items: list[dict[str, Any]]
    newest_uuid: str | None
    cut: bool
    boundary: Literal["last_uuid", "prompt", "none"]
    last_uuid_missing: bool
    records_parsed: int
    unknown_block_types: dict[str, int]


def _count_unknown(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        if rec.get("isSidechain") is True or rec.get("type") not in ("user", "assistant"):
            continue
        message = rec.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                key = "<non-dict>"
            else:
                btype = block.get("type")
                if not isinstance(btype, str):
                    key = "<no-type>"
                elif btype in KNOWN_BLOCK_TYPES:
                    continue
                else:
                    key = btype
            counts[key] = counts.get(key, 0) + 1
    return counts


def _build_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("isSidechain") is True:
            continue
        message = rec.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        rtype = rec.get("type")
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if rtype == "assistant":
                if btype == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text.strip() != "":
                        items.append({"kind": "assistant_text", "text": text[:TURN_TEXT_MAX]})
                elif btype == "tool_use":
                    tid, name = block.get("id"), block.get("name")
                    if isinstance(tid, str) and tid and isinstance(name, str) and name:
                        target = cues_lite.extract_target(name, block.get("input"))
                        items.append(
                            {
                                "kind": "tool_call",
                                "tool_use_id": tid,
                                "tool": name,
                                "target": target[:TURN_TARGET_MAX],
                                "target_class": cues_lite.target_class(name, target),
                            }
                        )
            elif rtype == "user" and btype == "tool_result":
                tid = block.get("tool_use_id")
                if isinstance(tid, str) and tid:
                    items.append(
                        {
                            "kind": "tool_result",
                            "tool_use_id": tid,
                            "is_error": block.get("is_error") is True,
                            "excerpt": result_excerpt(block.get("content")),
                        }
                    )
    return items


def read_turn(transcript_path: str, last_uuid: str | None) -> TurnWindow | None:
    """Read the newest turn window from the transcript tail; ``None`` on OSError."""
    try:
        with open(transcript_path, "rb") as fh:
            size = fh.seek(0, 2)
            offset = max(0, size - TURN_TAIL_BYTES)
            fh.seek(offset)
            raw = fh.read()
    except OSError:
        return None
    tail_text = raw.decode("utf-8", errors="replace")
    lines = tail_text.split("\n")
    if offset > 0:
        lines = lines[1:]
    want_uuid = last_uuid if isinstance(last_uuid, str) and last_uuid else None
    newest_uuid: str | None = None
    boundary: Literal["last_uuid", "prompt", "none"] = "none"
    user_prompt: str | None = None
    newer: list[dict[str, Any]] = []
    parsed = 0
    for line in reversed(lines):
        if line.strip() == "":
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        parsed += 1
        uid = rec.get("uuid")
        if newest_uuid is None and isinstance(uid, str) and uid:
            newest_uuid = uid
        if want_uuid is not None and uid == want_uuid:
            boundary = "last_uuid"
            break
        prompt = human_prompt_text(rec)
        if prompt is not None:
            boundary = "prompt"
            user_prompt = prompt
            break
        newer.append(rec)
    newer.reverse()
    return TurnWindow(
        user_prompt=user_prompt,
        items=_build_items(newer),
        newest_uuid=newest_uuid,
        cut=(boundary == "none" and offset > 0),
        boundary=boundary,
        last_uuid_missing=(
            want_uuid is not None and boundary != "last_uuid" and want_uuid not in tail_text
        ),
        records_parsed=parsed,
        unknown_block_types=_count_unknown(newer),
    )


def build_digest(
    payload: dict[str, Any],
    *,
    turn_index: int,
    project: str | None,
    window: TurnWindow | None,
) -> bytes | None:
    """Return the UTF-8 digest body (<= TURN_DIGEST_MAX_BYTES) or ``None``."""
    sid = payload["session_id"]
    engine = telemetry.build_engine()
    lam = payload.get("last_assistant_message")
    final = lam[:TURN_FINAL_MESSAGE_MAX] if isinstance(lam, str) and lam.strip() != "" else None
    prompt = window.user_prompt if window is not None else None
    all_items = window.items[-TURN_ITEMS_MAX:] if window is not None else []
    truncated = window is not None and (window.cut or len(window.items) > TURN_ITEMS_MAX)
    body: dict[str, Any] = {
        "event_id": f"{sid}:{turn_index}",
        "session_id": sid,
        "harness": "claude-code",
        "mode": "headless" if engine else "interactive",
        "engine": engine,
        "host": socket.gethostname(),
        "hook_version": prevention._hook_version(),
        "project": project,
        "turn_index": turn_index,
        "ts": telemetry.now_ts(),
        "user_prompt": prompt[:TURN_USER_PROMPT_MAX] if prompt is not None else None,
        "items": all_items,
        "final_message": final,
        "truncated": truncated,
    }

    def encode(items: list[dict[str, Any]], trunc: bool) -> bytes:
        body["items"] = items
        body["truncated"] = trunc
        return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8", "replace"
        )

    raw = encode(all_items, bool(truncated))
    if len(raw) <= TURN_DIGEST_MAX_BYTES:
        return raw
    if len(encode([], True)) > TURN_DIGEST_MAX_BYTES:
        return None
    lo, hi = 0, len(all_items)  # smallest drop count whose suffix fits
    while lo < hi:
        mid = (lo + hi) // 2
        if len(encode(all_items[mid:], True)) <= TURN_DIGEST_MAX_BYTES:
            hi = mid
        else:
            lo = mid + 1
    return encode(all_items[lo:], True)


def log_row(session_id: str, row: dict[str, Any]) -> None:
    """Append one row to the local turn-digest decision log; swallow failures."""
    try:
        path = get_turn_digest_log_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > prevention._LOG_MAX_BYTES:
            path.unlink()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        return


def spawn_sender(body_path: str, session_id: str) -> None:
    """Start the detached sender; never waits on it."""
    subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", SENDER_MODULE, body_path, session_id],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _load_state(session_id: str) -> tuple[int, str | None, str]:
    """Return ``(turn_index, last_uuid, state)``; state is ok/missing/corrupt."""
    try:
        data = json.loads(get_turn_state_path(session_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return 0, None, "missing"
    except (OSError, ValueError):
        return 0, None, "corrupt"
    idx = data.get("next_turn_index") if isinstance(data, dict) else None
    if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
        return 0, None, "corrupt"
    last = data.get("last_uuid")
    return idx, (last if isinstance(last, str) and last else None), "ok"


def stop(payload: dict[str, Any]) -> None:
    """Stop entry: count the turn and, when enabled, spawn the digest sender."""
    start = time.monotonic()
    sid = payload.get("session_id")
    if not isinstance(sid, str) or not sid:
        return
    turn_index, prev_last_uuid, state = _load_state(sid)
    new_last_uuid: str | None = None
    capture: str | None = None
    action: str | None = None
    error: str | None = None
    window: TurnWindow | None = None
    raw: bytes | None = None
    caught = False
    try:
        cache = prevention._load_cache(sid)
        cap = cache.get("surprise_capture") if cache else None
        capture = cap if isinstance(cap, str) else None
        if capture not in SEND_MODES:
            return
        if resolve_url_key() is None:
            prevention._record_drop(sid, DROP_OP, "no_url_key", 0)
            action = "no_url_key"
            return
        tp = payload.get("transcript_path")
        if isinstance(tp, str) and tp:
            window = read_turn(tp, prev_last_uuid)
        if window is None:
            prevention._record_drop(sid, DROP_OP, "transcript_unreadable", 0)
        else:
            new_last_uuid = window.newest_uuid
        proj = cache.get("project") if cache else None
        raw = build_digest(
            payload,
            turn_index=turn_index,
            project=proj if isinstance(proj, str) and proj else None,
            window=window,
        )
        if raw is None:
            prevention._record_drop(sid, DROP_OP, "too_large", 0)
            action = "too_large"
            return
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="kb-turn-", suffix=".json", delete=False
        ) as fh:
            fh.write(raw)
            body_path = fh.name
        try:
            spawn_sender(body_path, sid)
            action = "sent"
        except OSError:
            prevention._record_drop(sid, DROP_OP, "spawn_failed", 0)
            action = "spawn_failed"
            with contextlib.suppress(OSError):
                Path(body_path).unlink(missing_ok=True)
    except Exception as exc:
        caught = True
        action = "error"
        error = type(exc).__name__
        prevention._record_drop(sid, DROP_OP, "error", 0)
    finally:
        try:
            try:
                prevention._atomic_write(
                    get_turn_state_path(sid),
                    json.dumps({"next_turn_index": turn_index + 1, "last_uuid": new_last_uuid}),
                )
            except OSError:
                prevention._record_drop(sid, DROP_OP, "state_write_failed", 0)
            if capture in SEND_MODES or caught:
                body = json.loads(raw) if raw is not None else None
                log_row(
                    sid,
                    {
                        "ts": telemetry.now_ts(),
                        "op": "stop",
                        "session_id": sid,
                        "turn_index": turn_index,
                        "event_id": body["event_id"] if body else None,
                        "capture": capture,
                        "action": action,
                        "error": error,
                        "state": state,
                        "boundary": window.boundary if window else None,
                        "last_uuid_missing": window.last_uuid_missing if window else None,
                        "cut": window.cut if window else None,
                        "records_parsed": window.records_parsed if window else None,
                        "unknown_block_types": window.unknown_block_types if window else None,
                        "items_built": len(window.items) if window else None,
                        "items_sent": len(body["items"]) if body else None,
                        "bytes": len(raw) if raw is not None else None,
                        "truncated": body["truncated"] if body else None,
                        "user_prompt_null": body["user_prompt"] is None if body else None,
                        "final_message_null": body["final_message"] is None if body else None,
                        "elapsed_ms": int((time.monotonic() - start) * 1000),
                    },
                )
        except Exception:
            return
