"""Per-session suppression scratch file.

Hooks are stateless between turns; without a scratch file "only on change" is
unimplementable. The scratch tracks the last resolved scope and the set of
map keys surfaced so far in this session. We re-emit when the scope drifts or
when new map keys appear. The ``source=compact`` payload bypasses the
subset check so a compaction event re-seeds the context.

A "map key" is a :class:`personal_kb_hook.index_reader.MapKey` — a
``(label, id)`` :class:`typing.NamedTuple` carrying the roster label
alongside the per-KB map id. Pre-P1 scratch files stored bare ``id`` strings;
:func:`_read_scratch` back-parses those tolerantly to
``MapKey(label='personal', id=<s>)`` so an in-flight session that wrote its
scratch under the old format continues to suppress correctly after upgrade.
The on-disk shape since P1 is a sorted JSON list of two-element ``[label, id]``
lists.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from personal_kb_hook.index_reader import MapKey
from personal_kb_hook.paths import get_hook_scratch_path

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class EmitReason(StrEnum):
    """Why :func:`should_emit` decided to emit the roster.

    Members mirror the four emitting branches in :func:`should_emit`, in
    the SAME precedence order the code evaluates them: precedence decides
    which reason a multi-cause emission reports (e.g. a compact event on a
    session whose scope also drifted reports ``COMPACT``, not
    ``SCOPE_CHANGE``, because the compact check runs first).

    When :func:`should_emit` decides NOT to emit, it returns ``None``
    rather than a member of this enum — callers must use an explicit
    ``is not None`` check, not truthiness, since a falsy-but-not-None
    sentinel would be an easy trap here.
    """

    COMPACT = "compact"
    FIRST_EMISSION = "first-emission"
    SCOPE_CHANGE = "scope-change"
    NEW_MAPS = "new-maps"


# Legacy label assigned to bare-string ids found in pre-P1 scratch files.
# Matches the label used by roster.load_roster() when it synthesizes the
# legacy PERSONAL_KB_URL/KEY entry, so an upgraded session migrates its
# suppression set without losing entries.
_LEGACY_LABEL = "personal"


@dataclass
class ScratchState:
    """In-memory mirror of the on-disk scratch file."""

    last_scope: str | None
    surfaced_map_ids: set[MapKey] = field(default_factory=set)


def _read_scratch(path: Path) -> ScratchState:
    """Read the scratch file. Missing/corrupt -> fresh state.

    Back-parses on-disk ``surfaced_map_ids`` elements tolerantly:

    * ``[label, id]`` — both strings, non-empty: kept as ``MapKey(label, id)``.
    * ``"<id>"`` (bare string) — legacy pre-P1 shape: kept as
      ``MapKey(_LEGACY_LABEL, <id>)``.
    * Anything else: silently skipped.
    """
    try:
        if not path.exists():
            return ScratchState(last_scope=None)
        raw = path.read_text(encoding="utf-8", errors="replace")
        obj = json.loads(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.debug("scratch unreadable at %s: %s", path, exc)
        return ScratchState(last_scope=None)
    if not isinstance(obj, dict):
        return ScratchState(last_scope=None)
    last_scope = obj.get("last_scope")
    if last_scope is not None and not isinstance(last_scope, str):
        last_scope = None
    raw_ids = obj.get("surfaced_map_ids") or []
    surfaced: set[MapKey] = set()
    if isinstance(raw_ids, list):
        for item in raw_ids:
            if isinstance(item, str) and item:
                # Legacy pre-P1 bare-id form: assume the legacy 'personal' label.
                surfaced.add(MapKey(label=_LEGACY_LABEL, id=item))
            elif isinstance(item, list) and len(item) == 2:
                label, ident = item[0], item[1]
                if isinstance(label, str) and label and isinstance(ident, str) and ident:
                    surfaced.add(MapKey(label=label, id=ident))
            # Anything else: silently skipped.
    return ScratchState(last_scope=last_scope, surfaced_map_ids=surfaced)


def _write_scratch(path: Path, state: ScratchState) -> None:
    """Atomically write the scratch file. Best-effort; logs on failure.

    The on-disk ``surfaced_map_ids`` shape (since P1) is a deterministic
    sorted list of two-element ``[label, id]`` lists — JSON-native (no
    tuple type) and stable across runs.
    """
    payload = {
        "last_scope": state.last_scope,
        "surfaced_map_ids": sorted([k.label, k.id] for k in state.surfaced_map_ids),
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
    map_ids: list[MapKey],
    source: str | None,
    scratch_path: Path | None = None,
) -> EmitReason | None:
    """Decide whether to emit the directory string, and WHY.

    Returns the :class:`EmitReason` for the first matching branch, in this
    precedence order (do not reorder — precedence decides which reason a
    multi-cause emission reports), or ``None`` when not emitting:
      * ``map_ids`` empty -> ``None`` (nothing to show),
      * ``source == 'compact'`` -> :attr:`EmitReason.COMPACT` (compaction
        event — always re-seed),
      * no scratch file yet -> :attr:`EmitReason.FIRST_EMISSION`,
      * the resolved scope differs from ``last_scope`` ->
        :attr:`EmitReason.SCOPE_CHANGE`,
      * the resolved map keys are NOT already a subset of
        ``surfaced_map_ids`` -> :attr:`EmitReason.NEW_MAPS`,
      * otherwise -> ``None`` (already surfaced; suppressed).
    """
    if not map_ids:
        return None
    target = scratch_path or get_hook_scratch_path(session_id)
    state = _read_scratch(target)
    if source == "compact":
        return EmitReason.COMPACT
    if state.last_scope is None:
        return EmitReason.FIRST_EMISSION
    if state.last_scope != scope:
        return EmitReason.SCOPE_CHANGE
    if not set(map_ids).issubset(state.surfaced_map_ids):
        return EmitReason.NEW_MAPS
    return None


def get_surfaced_map_ids(
    *,
    session_id: str,
    scratch_path: Path | None = None,
) -> frozenset[MapKey]:
    """Return the map keys already surfaced this session (read-only).

    Lets a caller (the ``new-maps`` delta path in
    :mod:`personal_kb_hook.cli`) compute which of a freshly resolved map
    set are genuinely new, without reaching into the private
    :func:`_read_scratch`. Does not mutate the scratch file. Missing/corrupt
    scratch -> empty set, matching :func:`should_emit`'s fresh-state
    treatment.
    """
    target = scratch_path or get_hook_scratch_path(session_id)
    return frozenset(_read_scratch(target).surfaced_map_ids)


def mark_emitted(
    *,
    session_id: str,
    scope: str,
    map_ids: list[MapKey],
    scratch_path: Path | None = None,
) -> None:
    """Update the scratch file to reflect a successful emission.

    Unions the new keys into ``surfaced_map_ids`` and sets ``last_scope``
    to the freshly resolved scope. Tolerant of missing/corrupt scratch
    files (treated as fresh).
    """
    target = scratch_path or get_hook_scratch_path(session_id)
    state = _read_scratch(target)
    state.last_scope = scope
    state.surfaced_map_ids |= set(map_ids)
    _write_scratch(target, state)
