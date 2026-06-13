"""Fetch the maps index from the personal-kb web service over HTTP.

Provides :func:`load_index`, which attempts an authenticated HTTP GET
request to ``{PERSONAL_KB_URL}/api/kb/maps-index`` when both
``PERSONAL_KB_URL`` and ``PERSONAL_KB_API_KEY`` are present and
non-empty in the environment at call time.

On any failure — or when either env var is absent or empty, or the URL
scheme is unsupported, or the response shape is wrong — ``load_index``
returns an empty dict ``{}``. A successful HTTP response is authoritative
even when its ``projects`` list is empty.

This module is **stdlib-only** (``urllib.request``, ``urllib.parse``,
``json``, ``os``, ``socket``, ``logging``). The package's
``dependencies = []`` invariant is preserved.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

from personal_kb_hook.index_reader import MapEntry

logger = logging.getLogger(__name__)

_TIMEOUT: float = 3.0


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
            cleaned.append(MapEntry(id=entry_id, short_title=short_title, long_title=long_title))
        result[project_ref] = cleaned
    return result


def load_index() -> dict[str, list[MapEntry]]:
    """Return the maps index, fetching from the KB service when env vars are set.

    Reads ``PERSONAL_KB_URL`` and ``PERSONAL_KB_API_KEY`` from
    :data:`os.environ` at call time (no module-level caching). If both
    are present and non-empty, validates that the URL scheme is
    ``http`` or ``https``, then issues::

        GET {PERSONAL_KB_URL.rstrip('/')}/api/kb/maps-index
        Authorization: Bearer {PERSONAL_KB_API_KEY}

    with a 3.0-second timeout (connect + read via ``urllib``'s single
    ``timeout`` parameter).

    A successful HTTP response is **authoritative even when empty**
    (``{"projects": []}``) — no fallback to local disk in that case.

    On any failure — or when either env var is absent or empty, or the
    URL scheme is unsupported, or the response shape is wrong — returns
    an empty dict ``{}``. Failure is always silent (``DEBUG`` log only,
    no stdout, no exception raised).

    Failure taxonomy (all silent — ``DEBUG`` log only, no stdout):
    connection error / DNS failure (``URLError``), non-2xx status
    (``HTTPError``), socket timeout (``TimeoutError`` /
    ``socket.timeout``), undecodable body (``UnicodeDecodeError``),
    ``json.JSONDecodeError``, wrong top-level shape, unsupported URL
    scheme, or any unexpected ``Exception``.

    Returns:
        Mapping from project_ref to list of MapEntry records, or ``{}``
        on any failure or when env vars are absent/empty.
    """
    url = os.environ.get("PERSONAL_KB_URL", "")
    key = os.environ.get("PERSONAL_KB_API_KEY", "")
    if not url or not key:
        return {}

    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.debug(
                "PERSONAL_KB_URL scheme %r is not http/https; returning empty index",
                parsed.scheme,
            )
            return {}

        endpoint = url.rstrip("/") + "/api/kb/maps-index"
        req = urllib.request.Request(  # noqa: S310
            endpoint,
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")

        data = json.loads(body)
        if not isinstance(data, dict):
            logger.debug("HTTP response body is not a JSON object; returning empty index")
            return {}
        projects = data.get("projects")
        if not isinstance(projects, list):
            logger.debug("HTTP response body missing list 'projects' key; returning empty index")
            return {}

        return _map_projects(projects)

    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        logger.debug("HTTP maps-index fetch failed (%s); returning empty index", exc)
        return {}
    except Exception as exc:
        logger.debug(
            "Unexpected error in HTTP maps-index fetch (%s); returning empty index",
            exc,
        )
        return {}
