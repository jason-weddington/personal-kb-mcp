"""Listener module: env gate, transcript extraction, cache helpers, worker spawn.

Provides the building blocks used by ``cli.py`` to implement the
whisper-next-turn listener gate:

* :func:`is_listener_enabled` — env gate (three-var check).
* :func:`extract_manifest` — bounded tail-scan of a Claude Code JSONL
  transcript, returning ``(text, operated)`` from the last assistant record.
* :func:`read_listener_cache` / :func:`write_listener_cache` — tolerant
  JSON cache helpers.
* :func:`spawn_worker` — fire-and-forget ``Popen`` wrapper for the worker.

This module is **stdlib-only**. The package's ``dependencies = []``
invariant is preserved.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any

from personal_kb_hook.defaults import resolve_url_key

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---- Constants ---------------------------------------------------------------

_MCP_PATTERN: re.Pattern[str] = re.compile(r"^mcp__(.+?)__")
_TAIL_WINDOW: int = 262144  # bytes — read at most this many bytes from EOF
_MIN_TEXT_LEN: int = 200  # skip worker spawn when extracted text is shorter
_TEXT_HEAD_CAP: int = 4000  # keep only the first N chars of the text


# ---- Env gate ----------------------------------------------------------------


def is_listener_enabled() -> bool:
    """Return True iff the listener flag is on and a URL/key resolves.

    Requires ALL of:
    * a resolvable URL/key (unset means the local-mode defaults; a remote
      URL still needs ``PERSONAL_KB_API_KEY``).
    * ``PERSONAL_KB_LISTENER`` — lowercased value in ``{'1', 'true'}``.

    Reads from the environment at call time; no module-level caching.
    """
    flag = os.environ.get("PERSONAL_KB_LISTENER", "").lower()
    return flag in {"1", "true"} and resolve_url_key() is not None


# ---- Transcript extraction ---------------------------------------------------


def extract_manifest(transcript_path: str) -> tuple[str, list[str]] | None:
    """Extract ``(text, operated)`` from the last assistant record in the file.

    Reads at most :data:`_TAIL_WINDOW` bytes from the *end* of the file.
    When the seek offset is > 0 the first (possibly truncated) line is
    discarded so partial JSON lines don't pollute the parse.

    Record schema (Claude Code transcript JSONL):
    * A record qualifies as an assistant record when
      ``record.get('type') == 'assistant'``,
      ``record.get('message')`` is a :class:`dict`, and
      ``record['message'].get('content')`` is a :class:`list`.
    * **text block** — content element where ``block.get('type') == 'text'``
      and ``block.get('text')`` is a non-empty :class:`str`.
    * **tool_use block** — content element where
      ``block.get('type') == 'tool_use'`` and ``block.get('name')`` is a
      :class:`str`.

    Returns:
        ``(text, operated)`` where *text* is the newline-joined text-block
        strings from the **last** assistant record that has at least one
        non-empty text block, head-truncated to
        ``text[:_TEXT_HEAD_CAP]``; *operated* is a sorted,
        de-duplicated list of ``'mcp:<server>'`` strings derived from ALL
        tool_use block names across all assistant records in the tail window
        via the regex ``^mcp__(.+?)__``.

        Returns ``None`` when the transcript is missing / unreadable, contains
        no assistant record with text blocks, or the resulting text is shorter
        than :data:`_MIN_TEXT_LEN` characters.
    """
    try:
        with open(transcript_path, "rb") as fh:
            fh.seek(0, 2)  # seek to EOF
            size = fh.tell()
            offset = max(0, size - _TAIL_WINDOW)
            fh.seek(offset)
            raw_bytes = fh.read()
    except OSError:
        return None

    try:
        text_raw = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        return None

    lines = text_raw.split("\n")

    # Discard the first line when we seeked into the middle of the file
    # (it may be a truncated record that will fail JSON parsing).
    if offset > 0:
        lines = lines[1:]

    last_text: str | None = None
    operated_set: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record: Any = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            continue

        # Qualify as assistant record
        if not isinstance(record, dict):
            continue
        if record.get("type") != "assistant":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        # Process content blocks
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_val = block.get("text")
                if isinstance(text_val, str) and text_val:
                    text_parts.append(text_val)
            elif block_type == "tool_use":
                name = block.get("name")
                if isinstance(name, str):
                    m = _MCP_PATTERN.match(name)
                    if m:
                        operated_set.add(f"mcp:{m.group(1)}")

        if text_parts:
            last_text = "\n".join(text_parts)

    if last_text is None:
        return None

    text = last_text[:_TEXT_HEAD_CAP]

    if len(text) < _MIN_TEXT_LEN:
        return None

    operated = sorted(operated_set)
    return text, operated


# ---- Cache helpers -----------------------------------------------------------


def read_listener_cache(path: Path) -> dict[str, Any]:
    """Read the listener cache file. Returns fresh state on missing/corrupt.

    Fresh state: ``{"pending": [], "whispered_map_ids": []}``.

    The on-disk schema (since P2) is a per-KB-provenance shape:

    * ``pending`` — a JSON list of per-KB pointer objects
      ``{"label": <str>, "id": <str>, "short_title": <str>, "long_title": <str>}``
      with AT MOST one element per ``label`` after worker-side arbitration.
      An empty list ``[]`` means no pending whisper.
    * ``whispered_map_ids`` — a list of two-element ``[label, id]`` lists,
      JSON-native (matches :func:`personal_kb_hook.suppression._write_scratch`).

    Pre-P2 cache files used ``pending`` as a single dict (or ``None``) and
    ``whispered_map_ids`` as a list of bare ``id`` strings. The cli's whisper
    block and the worker's ``_merge_into_cache`` both back-parse those legacy
    shapes tolerantly to label ``'personal'`` (mirroring
    :func:`personal_kb_hook.suppression._read_scratch`'s legacy fallback).

    Any :class:`OSError`, :class:`json.JSONDecodeError`, or non-dict top-level
    value yields the fresh state without raising.
    """
    _fresh: dict[str, Any] = {"pending": [], "whispered_map_ids": []}
    try:
        if not path.exists():
            return _fresh
        raw = path.read_text(encoding="utf-8", errors="replace")
        obj: Any = json.loads(raw)
    except (OSError, json.JSONDecodeError, ValueError):
        return _fresh
    if not isinstance(obj, dict):
        return _fresh
    return obj


def write_listener_cache(path: Path, data: dict[str, Any]) -> None:
    """Atomically write the listener cache file via temp-file + :func:`os.replace`.

    Best-effort; logs at DEBUG on failure, never raises.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = fh.name
            json.dump(data, fh)
        os.replace(tmp_path, path)
    except OSError as exc:
        logger.debug("listener cache write failed at %s: %s", path, exc)


# ---- Worker spawn ------------------------------------------------------------


def spawn_worker(request_tmp_path: str, cache_path: str) -> None:
    """Spawn the listener worker as a fully detached subprocess.

    Invokes ``python -m personal_kb_hook.listener_worker
    <request_tmp_path> <cache_path>`` with ``start_new_session=True`` and
    both stdout / stderr redirected to :data:`subprocess.DEVNULL`. Does NOT
    call :meth:`~subprocess.Popen.wait` or
    :meth:`~subprocess.Popen.communicate`; the parent returns immediately.
    """
    subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "personal_kb_hook.listener_worker",
            request_tmp_path,
            cache_path,
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
