"""Ephemeral local whisper-decision debug log (stdlib-only, silent-on-failure).

Real-time, human-readable per-session ``tail -f``'able log of the
anticipatory-listener whisper decisions, written at two points in the
hook pipeline:

* The listener RUN — listener_worker.main() writes a multi-line block
  after arbitration: a header line carrying ``cwd_project`` +
  ``operating`` manifest, a ``transcript:`` excerpt, one ``<label> ->
  <winner|none> (<reason>)`` line per KB in roster order, and an outcome
  line (``=> WHISPER <id> next turn`` per winner, or a single ``=> no
  whisper``).
* The PROMPT inject — cli.py's UserPromptSubmit whisper block writes
  ``PROMPT inject <id> "<short_title>"`` for every entry it commits to
  ``ordered``, and ``PROMPT suppress <id> (already whispered this
  session)`` for every entry the already-whispered ``pre_set`` filter
  drops (NOT for cap-dropped entries — a different layer).

This module is DELIBERATELY DISTINCT from
:mod:`personal_kb_hook.telemetry` (the whisper-efficacy analytics sink
that POSTs to ``/api/kb/telemetry/whispers``). No DB, no POST, no
:func:`flush_session`-style batching, no orphan-sweep, no import of
``telemetry``. The file lives under ``~/.cache/personal_kb/`` and is
Jason's debugging tool, not analytics data — he tails + deletes manually.

Stdlib-only. Imports only stdlib + :mod:`personal_kb_hook.paths`. Every
write path is wrapped silent-on-failure (mirrors telemetry.py:83-84) so
a write failure NEVER propagates into the hook / session.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from personal_kb_hook.paths import get_whisper_debug_log_path

logger = logging.getLogger(__name__)

_TRANSCRIPT_EXCERPT_CAP: int = 200


def now_iso() -> str:
    """Return an ISO 8601 UTC timestamp string."""
    return datetime.now(UTC).isoformat()


def _format_transcript_excerpt(text: Any) -> str:
    """Render a request-text value as a transcript-line tail.

    Mirrors the spec: collapse internal whitespace runs to single spaces
    (``" ".join(text.split())``); head-truncate to ``_TRANSCRIPT_EXCERPT_CAP``
    chars; append a trailing ``…`` ONLY when the collapsed string exceeded
    the cap. When ``text`` is absent or not a str, render ``<none>``.
    """
    if not isinstance(text, str):
        return "<none>"
    collapsed = " ".join(text.split())
    if len(collapsed) > _TRANSCRIPT_EXCERPT_CAP:
        return collapsed[:_TRANSCRIPT_EXCERPT_CAP] + "…"
    return collapsed


def _operating_repr(operating: Any) -> str:
    """Render the operating manifest as ``[<comma-joined>]`` for the header.

    Defensive: non-str members are skipped (the hook never trusts blob
    shapes). When ``operating`` is not a list, render an empty list.
    """
    if not isinstance(operating, list):
        return "[]"
    parts = [item for item in operating if isinstance(item, str)]
    return "[" + ",".join(parts) + "]"


def _append_lines(session_id: str, lines: list[str]) -> None:
    """Append ``lines`` to the per-session debug log. Silent-on-failure.

    Mirrors :func:`personal_kb_hook.telemetry.append_row`'s error-swallow
    discipline — any :class:`OSError`, :class:`TypeError`, or
    :class:`ValueError` is swallowed and logged at DEBUG only. The
    debug log must NEVER raise into a session.

    The file is opened in append mode (``"a"``) under a ``with`` block;
    :func:`io.IOBase.flush` is called inside the ``with`` AND the
    ``with``-close flushes/closes the handle, so a concurrent
    ``tail -f`` sees newly written lines without waiting for buffering.
    """
    try:
        path = get_whisper_debug_log_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line)
                if not line.endswith("\n"):
                    fh.write("\n")
            fh.flush()
    except (OSError, TypeError, ValueError) as exc:
        logger.debug("whisper-debug log append failed: %s", exc)


def append_run_block(
    session_id: str,
    *,
    cwd_project: Any,
    operating: Any,
    text: Any,
    per_kb_lines: list[str],
    outcome_lines: list[str],
) -> None:
    """Write the listener-RUN block to the debug log. Silent-on-failure.

    Block shape:

    * header — ``=== <ISO-8601 UTC ts> listener run | cwd_project=<val-or-None> |
      operating=[<comma-joined>] ===``
    * ``transcript: <excerpt-or-none>``
    * one ``  <label> -> <winner-id-or-none> (<reason>)`` per roster entry
      (already formatted by the caller and passed as ``per_kb_lines``).
    * outcome — one ``  => WHISPER <id> next turn`` per surviving winner,
      OR a single ``  => no whisper`` line (caller-built; passed as
      ``outcome_lines``).
    """
    cwd_str = cwd_project if isinstance(cwd_project, str) else None
    header = (
        f"=== {now_iso()} listener run | "
        f"cwd_project={cwd_str} | "
        f"operating={_operating_repr(operating)} ==="
    )
    excerpt = _format_transcript_excerpt(text)
    lines = [header, f"transcript: {excerpt}", *per_kb_lines, *outcome_lines]
    _append_lines(session_id, lines)


def append_prompt_inject(session_id: str, entry_id: str, short_title: str) -> None:
    """Write a ``PROMPT inject <id> "<short_title>"`` line. Silent-on-failure.

    ``short_title`` is rendered verbatim inside double quotes — including
    the empty string, which yields ``PROMPT inject kb-X ""``. Do NOT
    fall back to the id when ``short_title`` is empty: the empty-string
    rendering is itself a useful debugging signal.
    """
    _append_lines(session_id, [f'PROMPT inject {entry_id} "{short_title}"'])


def append_prompt_suppress(session_id: str, entry_id: str) -> None:
    """Write a ``PROMPT suppress <id> (already whispered this session)`` line.

    Silent-on-failure. Used ONLY for entries dropped by the
    already-whispered ``pre_set`` filter (NOT for entries dropped by the
    defensive one-per-KB ``capped`` step — a different suppression
    layer NOT in scope for this debug log).
    """
    _append_lines(session_id, [f"PROMPT suppress {entry_id} (already whispered this session)"])
