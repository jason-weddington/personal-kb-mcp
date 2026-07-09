"""Fetch the maps index from each roster KB over HTTP, concurrently.

Provides :func:`load_index`, which takes a roster (a list of
:class:`personal_kb_hook.roster.KbEntry` records — ``(label, url, key)``)
and issues, for each entry CONCURRENTLY::

    GET {url.rstrip('/')}/api/kb/maps-index
    Authorization: Bearer {key}

The per-KB network call uses ``timeout=1.5`` seconds (per-KB network bound).
The overall fan-out is wrapped in a SINGLE
``concurrent.futures.wait(timeout=3.0)`` absolute wall deadline. A KB that
errors, times out, or simply has not completed by the 3.0-second deadline
contributes an EMPTY result for THAT label only — it never raises.

A successful HTTP response is authoritative even when its ``projects`` list
is empty.

Return shape
------------
``dict[str, list[tuple[str, MapEntry]]]`` — mapping ``project_ref`` to a
list of ``(label, MapEntry)`` tuples merged across all roster labels.
A single ``project_ref`` appearing under two labels concatenates both
labels' tuples under that key. Within each label, ``_map_projects``'s
existing per-record tolerance and ``last-wins`` duplicate-project_ref
semantics are preserved unchanged.

When the roster is empty (``[]``) — including the case where P0's loader
returned no entries because legacy env vars were unset — :func:`load_index`
returns ``{}`` without ever calling ``urllib``.

This module is **stdlib-only** (``urllib.request``, ``urllib.parse``,
``json``, ``concurrent.futures``, ``logging``). The package's
``dependencies = []`` invariant is preserved.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

from personal_kb_hook.index_reader import MapEntry

if TYPE_CHECKING:
    from personal_kb_hook.roster import KbEntry

logger = logging.getLogger(__name__)

# Per-KB network read bound (per urllib.request.urlopen call).
_PER_KB_TIMEOUT: float = 1.5

# Single absolute wall-clock deadline for the entire fan-out. A KB whose
# future is NOT in the returned `done` set after this many seconds is
# treated as empty for THAT label only — the fan-out is bounded by this
# value, not by N * _PER_KB_TIMEOUT.
_WALL_DEADLINE: float = 3.0


def _map_projects(projects: list[object]) -> dict[str, list[MapEntry]]:
    """Map a raw project list from the service response to the index shape.

    Applies the same per-record tolerance as
    ``index_reader._parse_one_file`` (lines 61-86): non-dict project
    records, absent or non-str ``project_ref``, non-list ``maps``,
    non-dict map elements, non-non-empty-str ``id``, non-str
    ``short_title``, missing-or-None ``long_title`` (coerced to
    ``""``), and present-but-not-str ``long_title`` (e.g. an ``int``)
    are all silently skipped. Duplicate ``project_ref`` values follow
    last-wins semantics, matching ``read_index``'s documented merge
    policy.

    ``pointers`` is tolerantly parsed: an absent field, a non-list
    value, or non-str / empty-str elements all fold to a clean ``[]`` /
    are dropped — an OLD server payload without ``pointers`` is
    guaranteed to still parse (rollout order: service deploys may lag
    hook upgrades and vice versa).

    Args:
        projects: Raw list from the parsed JSON response body.

    Returns:
        Mapping from project_ref to cleaned list of MapEntry records.
    """
    result: dict[str, list[MapEntry]] = {}
    for record in projects:
        if not isinstance(record, dict):
            continue
        project_ref = record.get("project_ref")
        maps = record.get("maps")
        if not isinstance(project_ref, str) or not project_ref:
            continue
        if not isinstance(maps, list):
            continue
        cleaned: list[MapEntry] = []
        for raw_map in maps:
            if not isinstance(raw_map, dict):
                continue
            entry_id = raw_map.get("id")
            short_title = raw_map.get("short_title")
            long_title = raw_map.get("long_title", "")
            if not isinstance(entry_id, str) or not entry_id:
                continue
            if not isinstance(short_title, str):
                continue
            if long_title is None:
                long_title = ""
            if not isinstance(long_title, str):
                continue
            # `pointers` is tolerantly parsed: an absent field, a non-list,
            # or non-str elements are all coerced to a clean [] / dropped —
            # so a pre-pointers server payload still parses (rollout order:
            # service deploys may lag hook upgrades and vice versa).
            raw_pointers = raw_map.get("pointers")
            pointers: list[str] = []
            if isinstance(raw_pointers, list):
                for item in raw_pointers:
                    if isinstance(item, str) and item:
                        pointers.append(item)
            cleaned.append(
                MapEntry(
                    id=entry_id,
                    short_title=short_title,
                    long_title=long_title,
                    pointers=pointers,
                )
            )
        result[project_ref] = cleaned
    return result


def _fetch_one_kb(entry: KbEntry) -> dict[str, list[MapEntry]]:
    """Fetch and parse the maps index for ONE KB. Never raises.

    On any failure — unsupported scheme, connection error, non-2xx status,
    timeout, undecodable body, JSON decode error, or wrong top-level
    shape — returns ``{}``. Failure is always silent (``DEBUG`` log only,
    no stdout).
    """
    url = entry.url
    key = entry.key
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.debug(
                "roster label %r: URL scheme %r is not http/https; empty for this label",
                entry.label,
                parsed.scheme,
            )
            return {}

        endpoint = url.rstrip("/") + "/api/kb/maps-index"
        req = urllib.request.Request(  # noqa: S310
            endpoint,
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=_PER_KB_TIMEOUT) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")

        data = json.loads(body)
        if not isinstance(data, dict):
            logger.debug(
                "roster label %r: response body is not a JSON object; empty for this label",
                entry.label,
            )
            return {}
        projects = data.get("projects")
        if not isinstance(projects, list):
            logger.debug(
                "roster label %r: response body missing list 'projects' key; empty for this label",
                entry.label,
            )
            return {}

        return _map_projects(projects)

    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        logger.debug(
            "roster label %r: HTTP maps-index fetch failed (%s); empty for this label",
            entry.label,
            exc,
        )
        return {}
    except Exception as exc:
        logger.debug(
            "roster label %r: unexpected error in HTTP maps-index fetch (%s); empty for this label",
            entry.label,
            exc,
        )
        return {}


def load_index(roster: list[KbEntry]) -> dict[str, list[tuple[str, MapEntry]]]:
    """Fan out a maps-index GET to every roster KB CONCURRENTLY.

    For each :class:`KbEntry` in ``roster``, issues an authenticated
    ``GET {url}/api/kb/maps-index`` with ``timeout=1.5`` per call. The
    overall fan-out is bounded by a SINGLE
    ``concurrent.futures.wait(timeout=3.0)`` absolute wall deadline — a
    future not in ``done`` after 3.0 seconds is treated as empty for that
    label and its underlying future is cancelled without blocking. Per-KB
    errors (HTTPError, URLError, TimeoutError, JSONDecodeError, etc.) are
    individually caught and contribute the empty result for THAT label only.

    Args:
        roster: List of :class:`KbEntry` records to query. An empty list
            short-circuits with ``{}`` without ever invoking ``urllib``.

    Returns:
        ``dict[str, list[tuple[str, MapEntry]]]`` — mapping ``project_ref``
        to a list of ``(label, MapEntry)`` tuples merged across all roster
        labels. The same ``project_ref`` appearing under two labels yields
        a single key whose value concatenates both labels' tuples
        (preserving each KB's internal map ordering and the merge order
        induced by roster order).

    Never raises.
    """
    if not roster:
        return {}

    # max_workers must be at least 1; with len(roster) == 0 we already
    # returned above. ThreadPoolExecutor below scales to the roster length.
    max_workers = max(1, len(roster))

    # The single concurrent.futures.wait below enforces a 3.0s absolute
    # wall deadline across the whole fan-out — NOT a sum of per-future
    # 1.5s timeouts (which would be N * 1.5s). 1.5s is the per-KB network
    # bound; 3.0s is the single absolute wall deadline.
    #
    # Critically we do NOT use the executor as a context manager:
    # ``ThreadPoolExecutor.__exit__`` calls ``shutdown(wait=True)``, which
    # blocks until every submitted future has completed — defeating the
    # purpose of the 3.0s wall cap when one KB is hung. We manage the
    # executor lifecycle manually and call ``shutdown(wait=False,
    # cancel_futures=True)`` to return immediately after the deadline.
    per_kb_results: dict[KbEntry, dict[str, list[MapEntry]]] = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_entry: dict[concurrent.futures.Future[dict[str, list[MapEntry]]], KbEntry] = {
            executor.submit(_fetch_one_kb, entry): entry for entry in roster
        }
        done, not_done = concurrent.futures.wait(
            future_to_entry.keys(),
            timeout=_WALL_DEADLINE,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        for future in done:
            entry = future_to_entry[future]
            try:
                per_kb_results[entry] = future.result()
            except Exception as exc:  # defence-in-depth: _fetch_one_kb itself never raises.
                logger.debug(
                    "roster label %r: unexpected future.result() failure (%s); "
                    "empty for this label",
                    entry.label,
                    exc,
                )
                per_kb_results[entry] = {}
        for future in not_done:
            entry = future_to_entry[future]
            logger.debug(
                "roster label %r: wall deadline (%.1fs) exceeded; empty for this label",
                entry.label,
                _WALL_DEADLINE,
            )
            # Best-effort cancel — if the future is already running the
            # cancel is a no-op, but we never .result() it so the thread
            # cannot delay our return.
            future.cancel()
            per_kb_results[entry] = {}
    finally:
        # wait=False so a hung urlopen thread does NOT delay the caller.
        # cancel_futures=True (Python 3.9+) drops any not-yet-started futures.
        executor.shutdown(wait=False, cancel_futures=True)

    # Merge per-KB results into the namespaced shape, preserving roster
    # order so that a project_ref present under multiple labels yields a
    # stable concatenation order.
    merged: dict[str, list[tuple[str, MapEntry]]] = {}
    for entry in roster:
        per_kb = per_kb_results.get(entry, {})
        for project_ref, entries in per_kb.items():
            bucket = merged.setdefault(project_ref, [])
            for map_entry in entries:
                bucket.append((entry.label, map_entry))
    return merged
