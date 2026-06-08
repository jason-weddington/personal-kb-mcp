"""Per-session suppression scratch file.

Hooks are stateless between turns; without a scratch file "only on change" is
unimplementable. The scratch tracks the last resolved scope and the set of
map ids surfaced so far in this session. We re-emit when the scope drifts or
when new map ids appear. The ``source=compact`` payload bypasses the
subset check so a compaction event re-seeds the context.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

from personal_kb_hook.paths import get_hook_scratch_path

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ScratchState:
    """In-memory mirror of the on-disk scratch file."""

    last_scope: str | None
    surfaced_map_ids: set[str]


def _read_scratch(path: Path) -> ScratchState:
    """Read the scratch file. Missing/corrupt -> fresh state."""
    try:
        if not path.exists():
            return ScratchState(last_scope=None, surfaced_map_ids=set())
        raw = path.read_text(encoding="utf-8", errors="replace")
        obj = json.loads(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.debug("scratch unreadable at %s: %s", path, exc)
        return ScratchState(last_scope=None, surfaced_map_ids=set())
    if not isinstance(obj, dict):
        return ScratchState(last_scope=None, surfaced_map_ids=set())
    last_scope = obj.get("last_scope")
    if last_scope is not None and not isinstance(last_scope, str):
        last_scope = None
    raw_ids = obj.get("surfaced_map_ids") or []
    surfaced: set[str] = set()
    if isinstance(raw_ids, list):
        for item in raw_ids:
            if isinstance(item, str):
                surfaced.add(item)
    return ScratchState(last_scope=last_scope, surfaced_map_ids=surfaced)


def _write_scratch(path: Path, state: ScratchState) -> None:
    """Atomically write the scratch file. Best-effort; logs on failure."""
    payload = {
        "last_scope": state.last_scope,
        "surfaced_map_ids": sorted(state.surfaced_map_ids),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via temp file + os.replace.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = fh.name
            json.dump(payload, fh)
        os.replace(tmp_path, path)
    except OSError as exc:
        logger.debug("scratch write failed at %s: %s", path, exc)


def should_emit(
    *,
    session_id: str,
    scope: str,
    map_ids: list[str],
    source: str | None,
    scratch_path: Path | None = None,
) -> bool:
    """Decide whether to emit the directory string.

    Emit only when ``map_ids`` is non-empty AND any of:
      * ``source == 'compact'`` (compaction event — always re-seed),
      * no scratch file yet,
      * the resolved scope differs from ``last_scope``,
      * the resolved map ids are NOT already a subset of ``surfaced_map_ids``.
    """
    if not map_ids:
        return False
    target = scratch_path or get_hook_scratch_path(session_id)
    state = _read_scratch(target)
    if source == "compact":
        return True
    if state.last_scope is None:
        return True
    if state.last_scope != scope:
        return True
    return not set(map_ids).issubset(state.surfaced_map_ids)


def mark_emitted(
    *,
    session_id: str,
    scope: str,
    map_ids: list[str],
    scratch_path: Path | None = None,
) -> None:
    """Update the scratch file to reflect a successful emission.

    Unions the new ids into ``surfaced_map_ids`` and sets ``last_scope`` to
    the freshly resolved scope. Tolerant of missing/corrupt scratch files
    (treated as fresh).
    """
    target = scratch_path or get_hook_scratch_path(session_id)
    state = _read_scratch(target)
    state.last_scope = scope
    state.surfaced_map_ids |= set(map_ids)
    _write_scratch(target, state)
