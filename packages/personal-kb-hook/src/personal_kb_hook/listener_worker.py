"""Listener worker: POST assistant transcript manifest to the KB service.

Run as a detached subprocess spawned by the hook on a Stop event::

    python -m personal_kb_hook.listener_worker <request_tmp_path> <cache_path>

Behaviour:
* Reads the request JSON from ``argv[1]`` (the tmp file) and deletes it.
* Issues a single POST to ``{PERSONAL_KB_URL}/api/kb/listener`` with a
  ``Bearer`` token and ``Content-Type: application/json`` (30-second timeout,
  stdlib ``urllib`` only, no retries).
* Validates the response per the pinned shape rules (mirroring
  ``http_index._map_projects`` tolerance).
* On a valid non-null ``map``: atomically merges the map into the cache at
  ``argv[2]`` (``pending=<map>``, ``whispered_map_ids`` preserved).
* On null ``map`` or any failure: writes nothing; cache is left unchanged.
* **Always** deletes the request tmp file.
* **Always** exits 0 (top-level broad ``except``).

This module is **stdlib-only**. The package's ``dependencies = []``
invariant is preserved.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TIMEOUT: float = 30.0


def _validate_map(map_obj: object) -> dict[str, str] | None:
    """Validate a ``map`` dict from the service response.

    Applies the same tolerance as :func:`~personal_kb_hook.http_index._map_projects`:
    * ``id`` — non-empty :class:`str` (required).
    * ``short_title`` — :class:`str` (required).
    * ``long_title`` — :class:`str` or ``None``; ``None`` is coerced to
      ``""``; any other non-str value is a failure.

    Returns a normalised ``{"id": ..., "short_title": ..., "long_title": ...}``
    dict on success, or ``None`` on any validation failure.
    """
    if not isinstance(map_obj, dict):
        return None
    entry_id: Any = map_obj.get("id")
    short_title: Any = map_obj.get("short_title")
    long_title: Any = map_obj.get("long_title", "")

    if not isinstance(entry_id, str) or not entry_id:
        return None
    if not isinstance(short_title, str):
        return None
    if long_title is None:
        long_title = ""
    if not isinstance(long_title, str):
        return None  # present-but-non-str long_title is a failure

    return {"id": entry_id, "short_title": short_title, "long_title": long_title}


def _merge_into_cache(cache_path: Path, validated_map: dict[str, str]) -> None:
    """Atomically set ``pending`` in the cache while preserving ``whispered_map_ids``."""
    existing: dict[str, Any] = {}
    try:
        if cache_path.exists():
            raw = cache_path.read_text(encoding="utf-8", errors="replace")
            parsed: Any = json.loads(raw)
            if isinstance(parsed, dict):
                existing = parsed
    except (OSError, json.JSONDecodeError, ValueError):
        pass

    raw_ids: Any = existing.get("whispered_map_ids", [])
    whispered_ids: list[str] = []
    if isinstance(raw_ids, list):
        whispered_ids = [i for i in raw_ids if isinstance(i, str)]

    new_cache: dict[str, Any] = {
        "pending": validated_map,
        "whispered_map_ids": whispered_ids,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=cache_path.parent,
        prefix=cache_path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as fh:
        tmp_path_str = fh.name
        json.dump(new_cache, fh)
    os.replace(tmp_path_str, cache_path)


def main() -> None:
    """Entry point for ``python -m personal_kb_hook.listener_worker``."""
    request_tmp_path: str | None = None
    try:
        if len(sys.argv) < 3:  # 0=prog, 1=req_tmp, 2=cache_path
            return
        request_tmp_path = sys.argv[1]
        cache_path_str = sys.argv[2]

        # Read request data
        request_data: Any = None
        try:
            with open(request_tmp_path, encoding="utf-8") as fh:
                request_data = json.load(fh)
        except (OSError, json.JSONDecodeError, ValueError):
            return

        if not isinstance(request_data, dict):
            return

        url = os.environ.get("PERSONAL_KB_URL", "")
        key = os.environ.get("PERSONAL_KB_API_KEY", "")
        if not url or not key:
            return

        endpoint = url.rstrip("/") + "/api/kb/listener"
        body_bytes = json.dumps(request_data).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310
            endpoint,
            data=body_bytes,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
                body = resp.read().decode("utf-8")
        except Exception:
            return

        # Parse and validate response
        try:
            data: Any = json.loads(body)
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict):
            return
        if "pointer" not in data:
            return

        map_obj: Any = data["pointer"]
        if map_obj is None:
            return  # null map — no-op success

        validated = _validate_map(map_obj)
        if validated is None:
            return

        _merge_into_cache(Path(cache_path_str), validated)

    except Exception:
        logger.debug("listener_worker: unhandled error", exc_info=True)
    finally:
        if request_tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(request_tmp_path)


if __name__ == "__main__":
    main()
