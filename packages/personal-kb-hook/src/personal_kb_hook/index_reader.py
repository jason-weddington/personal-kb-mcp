"""TypedDict for a single mental_map entry in the maps index.

The on-disk JSONL reader (``read_index`` / ``_parse_one_file``) has been
removed. Maps are now fetched exclusively from the live KB service via
:mod:`personal_kb_hook.http_index`. This module is kept solely to provide
the :class:`MapEntry` TypedDict, which is imported by
:mod:`personal_kb_hook.http_index` at runtime and by
:mod:`personal_kb_hook.render` under ``TYPE_CHECKING``.
"""

from __future__ import annotations

from typing import TypedDict


class MapEntry(TypedDict):
    """One mental_map row in the maps index."""

    id: str
    short_title: str
    long_title: str
