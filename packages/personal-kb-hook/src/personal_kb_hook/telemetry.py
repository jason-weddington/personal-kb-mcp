"""Whisper-efficacy telemetry helpers (stdlib-only, silent-on-failure).

Implements the hook side of GTD ccc05354 — measuring whether whispers
actually help. Four hook touchpoints + a service sink:

1. EMIT — :func:`append_row` writes one jsonl row per roster map shown
   (incl. cross-project) and per listener whisper. Roster rows are appended
   by the SessionStart / UserPromptSubmit pipeline in :mod:`personal_kb_hook.cli`;
   listener rows are appended by the worker in
   :mod:`personal_kb_hook.listener_worker` at pointer-CACHE time (not POST
   time — the POST can return ``null``).
2. CONSUME — :func:`mark_consumed` is called from the PostToolUse branch of
   :mod:`personal_kb_hook.cli` when ``mcp__personal-kb__kb_get`` or
   ``mcp__team-kb__team_kb_get`` fires. Marks the matching scratch row
   ``consumed=true`` (atomic rewrite).
3. FLUSH — :func:`flush_session` is called on Stop, OUTSIDE the
   listener-enabled guard (roster rows accrue regardless of the gate).
   POSTs the whole jsonl as ``{"rows": [...]}`` to
   ``/api/kb/telemetry/whispers``. Leaves the file on disk on a 2xx — the
   composite-key ON CONFLICT path makes re-flush idempotent and keeps the
   SessionStart orphan sweep correct.
4. SINK — the service-side ``whisper_telemetry`` table with composite PK
   ``(session_id, surface, map_id)``.

build_engine is read defensively from ``HEADLESS_BUILD_ENGINE`` — unset =>
``None`` (interactive / control-plane). The hook ships correct today with
``build_engine`` staying null; the agent-gtd task that sets the var is a
separate, non-blocking dependency.

This module is **stdlib-only**. ``dependencies = []`` is preserved.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from personal_kb_hook.paths import get_whisper_log_path

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_TIMEOUT: float = 30.0
_TELEMETRY_PATH = "/api/kb/telemetry/whispers"
_ORPHAN_SWEEP_CAP = 20


def now_ts() -> str:
    """Return an ISO 8601 UTC timestamp string."""
    return datetime.now(UTC).isoformat()


def build_engine() -> str | None:
    """Defensive read of ``HEADLESS_BUILD_ENGINE``.

    Unset / empty => ``None`` (interactive/control-plane). The agent-gtd
    task that sets this on dispatch is a separate, non-blocking dependency.
    """
    raw = os.environ.get("HEADLESS_BUILD_ENGINE", "")
    return raw or None


def append_row(session_id: str, row: dict[str, Any]) -> None:
    """Append one jsonl row to the session's whisper-log. Silent-on-failure.

    The hook must NEVER break a session, so every error path is swallowed
    (logged at DEBUG only). Stdlib ``json.dumps`` only.
    """
    try:
        path = get_whisper_log_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
    except (OSError, TypeError, ValueError) as exc:
        logger.debug("whisper-log append failed: %s", exc)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a jsonl file into a list of dicts. Tolerant — skips bad lines."""
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj: Any = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically rewrite a jsonl file via tempfile + ``os.replace``.

    Mirrors :func:`personal_kb_hook.listener_worker._merge_into_cache`'s
    tempfile + ``os.replace`` in the same parent dir.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as fh:
        tmp_path_str = fh.name
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    os.replace(tmp_path_str, path)


def mark_consumed(session_id: str, map_ids: set[str], consumed_ts: str) -> None:
    """Mark rows whose ``map_id`` or a pointer is in ``map_ids`` as consumed.

    Atomic rewrite of the session's whisper-log: for each row where
    ``consumed`` is falsy AND either

    * the row's ``map_id`` is in ``map_ids`` (direct map fetch), OR
    * any id in the row's ``pointers`` list is in ``map_ids`` (chain
      fetch — the model pulled a detail entry the map points to; GTD
      88441f9c),

    ``consumed`` is set to ``True`` and ``consumed_ts`` to ``consumed_ts``.
    ``trigger_context.consumed_via`` records ``"map"`` for a direct match
    or ``"pointer"`` for a chain-credit — a direct match wins if both hold,
    since the map itself is the more specific signal. ``trigger_context``
    is a plain ``dict[str, Any]`` on the wire (see
    :class:`kb_service.models.WhisperTelemetryRow`), so this piggybacks
    the direct-vs-chain signal to the server with ZERO whisper_telemetry
    schema change. Everything else is written through unchanged.
    Silent-on-failure.

    Ultra-cheap by design: pure in-memory matching over the already-read
    jsonl rows. Does NOT call ``http_index.load_index`` or
    ``resolve_project`` — fires on EVERY kb_get / team_kb_get.
    """
    try:
        path = get_whisper_log_path(session_id)
        if not path.exists():
            return
        rows = _read_jsonl(path)
        if not rows:
            return
        changed = False
        for row in rows:
            if row.get("consumed"):
                continue
            row_map_id = row.get("map_id")
            direct_hit = isinstance(row_map_id, str) and row_map_id in map_ids
            pointer_hit = False
            if not direct_hit:
                raw_pointers = row.get("pointers")
                if isinstance(raw_pointers, list):
                    for ptr in raw_pointers:
                        if isinstance(ptr, str) and ptr in map_ids:
                            pointer_hit = True
                            break
            if not direct_hit and not pointer_hit:
                continue
            row["consumed"] = True
            row["consumed_ts"] = consumed_ts
            ctx = row.get("trigger_context")
            if not isinstance(ctx, dict):
                ctx = {}
                row["trigger_context"] = ctx
            ctx["consumed_via"] = "map" if direct_hit else "pointer"
            changed = True
        if changed:
            _atomic_write_jsonl(path, rows)
    except (OSError, TypeError, ValueError) as exc:
        logger.debug("whisper-log mark_consumed failed: %s", exc)


def _post_telemetry(url: str, key: str, rows: list[dict[str, Any]]) -> bool:
    """POST ``{"rows": rows}`` to ``{url}/api/kb/telemetry/whispers``.

    Returns ``True`` iff the response status is 2xx. Stdlib ``urllib`` only;
    silent-on-failure. 30s timeout, Bearer auth, no retries.
    """
    try:
        endpoint = url.rstrip("/") + _TELEMETRY_PATH
        body_bytes = json.dumps({"rows": rows}).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310
            endpoint,
            data=body_bytes,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
            status = getattr(resp, "status", None) or resp.getcode()
        return 200 <= int(status) < 300
    except Exception as exc:
        logger.debug("whisper telemetry POST failed: %s", exc)
        return False


def _legacy_url_key() -> tuple[str, str] | None:
    """Return the legacy (url, key) pair, or ``None`` if either is unset.

    Mirrors :mod:`personal_kb_hook.roster`'s legacy synthesis path
    (``PERSONAL_KB_URL`` / ``PERSONAL_KB_API_KEY``). v1 routing: ALL rows
    POST to this pair regardless of ``source_kb`` — cross-KB telemetry
    routing waits until team-kb gets its own endpoint.
    """
    url = os.environ.get("PERSONAL_KB_URL", "")
    key = os.environ.get("PERSONAL_KB_API_KEY", "")
    if url and key:
        return url, key
    return None


def flush_session(session_id: str) -> None:
    """Stop-flush the session's whisper-log to the personal endpoint.

    Reads all rows; POSTs them as one batch to PERSONAL_KB_URL +
    ``/api/kb/telemetry/whispers`` with PERSONAL_KB_API_KEY bearer auth. On
    a 2xx, LEAVES the file in place (the table's composite PK makes re-flush
    idempotent; not deleting keeps the SessionStart orphan-sweep correct).
    Skips silently if either env var is unset, the file is missing, or it
    is empty. Runs OUTSIDE the listener-enabled guard — roster rows accrue
    regardless of the gate.
    """
    try:
        creds = _legacy_url_key()
        if creds is None:
            return
        url, key = creds
        path = get_whisper_log_path(session_id)
        if not path.exists():
            return
        rows = _read_jsonl(path)
        if not rows:
            return
        _post_telemetry(url, key, rows)
    except Exception as exc:
        logger.debug("whisper telemetry flush_session failed: %s", exc)


def _parse_session_id_from_filename(name: str) -> str | None:
    """Pull the session_id out of a ``whisper-log-<session_id>.jsonl`` filename."""
    prefix = "whisper-log-"
    suffix = ".jsonl"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    inner = name[len(prefix) : -len(suffix)]
    return inner or None


def orphan_sweep(current_session_id: str) -> None:
    """SessionStart: opportunistically flush + delete prior-session logs.

    Globs ``~/.cache/personal_kb/whisper-log-*.jsonl`` for files whose
    session_id (parsed from the filename) differs from
    ``current_session_id``. For each (capped at :data:`_ORPHAN_SWEEP_CAP`
    files) POSTs its rows to the same telemetry endpoint and, on a 2xx,
    deletes the orphaned file. Safety net for sessions where Stop did not
    fire. Silent-on-failure; stdlib only.
    """
    try:
        creds = _legacy_url_key()
        if creds is None:
            return
        url, key = creds

        cache_dir = get_whisper_log_path("placeholder").parent
        if not cache_dir.exists():
            return

        candidates = sorted(cache_dir.glob("whisper-log-*.jsonl"))
        swept = 0
        for path in candidates:
            if swept >= _ORPHAN_SWEEP_CAP:
                return
            sid = _parse_session_id_from_filename(path.name)
            if sid is None or sid == current_session_id:
                continue
            swept += 1
            try:
                rows = _read_jsonl(path)
                if not rows:
                    # Empty / malformed: just unlink so it doesn't recur.
                    path.unlink(missing_ok=True)
                    continue
                if _post_telemetry(url, key, rows):
                    path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("orphan sweep iteration failed for %s: %s", path, exc)
    except Exception as exc:
        logger.debug("whisper telemetry orphan_sweep failed: %s", exc)
