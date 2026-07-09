"""Shared types for the maps index.

The on-disk JSONL reader (``read_index`` / ``_parse_one_file``) has been
removed. Maps are now fetched exclusively from the live KB service via
:mod:`personal_kb_hook.http_index`. This module is kept solely to provide
shared types used across the package:

* :class:`MapEntry` — TypedDict describing one mental_map row.
* :class:`MapKey` — NamedTuple ``(label, id)`` identifying a map within a
  multi-KB roster. Carrying the source KB ``label`` alongside the map ``id``
  prevents two distinct KBs that happen to share the same ``id`` namespace
  from colliding in suppression state, and lets :mod:`personal_kb_hook.render`
  prefix Line-2 entries with their source ``label`` when more than one KB
  is present.

This module is **dependency-free within the package** — it imports nothing
from :mod:`personal_kb_hook`, so :class:`MapKey` and :class:`MapEntry` may
be imported at runtime from any other module here without forming a cycle.
"""

from __future__ import annotations

from typing import NamedTuple, NotRequired, TypedDict


class MapEntry(TypedDict):
    """One mental_map row in the maps index.

    ``pointers`` is the list of ``kb-XXXXX`` ids the map's body mentions —
    the DETAIL entries the map points to. Populated by
    :mod:`personal_kb_hook.http_index`'s ``_map_projects`` from the service
    payload; a missing / malformed ``pointers`` field in the service
    response falls through to an empty list so an OLD server payload
    (pre-pointers) parses cleanly (rollout order: service deploys may lag
    hook upgrades and vice versa).

    Whisper-telemetry consume matches the fetched kb-id against row
    ``map_id`` OR any id in ``pointers`` — a session that pulls the
    detail-chain (map → detail entries) still credits the map row as
    consumed. See GTD 88441f9c.
    """

    id: str
    short_title: str
    long_title: str
    pointers: NotRequired[list[str]]


class MapKey(NamedTuple):
    """A typed key identifying one mental_map within a multi-KB roster.

    The pair ``(label, id)`` is the unit of identity for suppression-state
    bookkeeping (``ScratchState.surfaced_map_ids``) and for Line-2 render
    grouping. ``label`` is the KB roster label (e.g. ``"personal"``,
    ``"team"``); ``id`` is the per-KB ``kb-XXXXX`` map id.

    Deliberately distinct from a bare ``str``: passing a bare id where a
    :class:`MapKey` is expected is a mypy ``error`` (under the package's
    strict mode), preventing the legacy bare-id leak from re-emerging in
    the directory pipeline.
    """

    label: str
    id: str
