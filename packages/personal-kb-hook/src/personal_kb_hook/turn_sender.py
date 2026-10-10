"""Detached turn-digest sender: ``python -m personal_kb_hook.turn_sender <body> <sid>``.

POSTs a prebuilt digest body file to ``/api/kb/turn`` (one attempt, no retry),
records failures in the shared drop log (op ``turn_digest``), appends one
``send`` row to the local turn-digest log and always deletes the body file.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
import urllib.request
from pathlib import Path

from personal_kb_hook import prevention, telemetry
from personal_kb_hook.defaults import resolve_url_key
from personal_kb_hook.turn_digest import DROP_OP, log_row

TURN_PATH = "/api/kb/turn"
SEND_TIMEOUT: float = 10.0
REJECT_REASONS = ("redaction-unavailable", "write-failed", "duplicate-mismatch")


def send_file(body_path: str, session_id: str) -> str | None:
    """POST the body file; return the drop reason or ``None``. Never raises."""
    path = Path(body_path)
    reason: str | None = None
    event_id: str | None = None
    http_status: int | None = None
    server_reason: str | None = None
    redactions: int | None = None
    elapsed_ms = 0
    try:
        try:
            raw = path.read_bytes()
        except OSError:
            reason = "body_unreadable"
        else:
            with contextlib.suppress(Exception):
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and isinstance(parsed.get("event_id"), str):
                    event_id = parsed["event_id"]
            uk = resolve_url_key()
            if uk is None:
                reason = "no_url_key"
            else:
                url, key = uk
                start = time.monotonic()
                try:
                    req = urllib.request.Request(  # noqa: S310
                        url.rstrip("/") + TURN_PATH,
                        data=raw,
                        headers={
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=SEND_TIMEOUT) as resp:  # noqa: S310
                        http_status = int(resp.status)
                        resp_body = resp.read()
                    elapsed_ms = int((time.monotonic() - start) * 1000)
                    if not 200 <= http_status < 300:
                        reason = f"http_{http_status}"
                    else:
                        with contextlib.suppress(Exception):
                            data = json.loads(resp_body)
                            if isinstance(data, dict):
                                r = data.get("reason")
                                if isinstance(r, str):
                                    server_reason = r
                                reds = data.get("redactions")
                                if isinstance(reds, list):
                                    redactions = len(reds)
                                if server_reason in REJECT_REASONS:
                                    reason = f"rejected_{server_reason}"
                except Exception as exc:
                    elapsed_ms = int((time.monotonic() - start) * 1000)
                    reason = prevention._drop_reason(exc)
                    code = getattr(exc, "code", None)
                    if isinstance(code, int):
                        http_status = code
    except Exception:
        reason = reason or "error"
    finally:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
    if reason is not None:
        prevention._record_drop(session_id, DROP_OP, reason, elapsed_ms)
    log_row(
        session_id,
        {
            "ts": telemetry.now_ts(),
            "op": "send",
            "session_id": session_id,
            "event_id": event_id,
            "http_status": http_status,
            "server_reason": server_reason,
            "redactions": redactions,
            "drop_reason": reason,
            "elapsed_ms": elapsed_ms,
        },
    )
    return reason


def main() -> None:
    """Entry point: argv[1] body path, argv[2] session id."""
    if len(sys.argv) < 3:
        return
    with contextlib.suppress(Exception):
        send_file(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
