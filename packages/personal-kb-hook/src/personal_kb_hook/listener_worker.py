"""Listener worker: fan out the assistant manifest across the KB roster.

Run as a detached subprocess spawned by the hook on a Stop event::

    python -m personal_kb_hook.listener_worker <request_tmp_path> <cache_path>

Behaviour (P2 multi-KB fan-out, BUILT DARK):

* Reads the request JSON from ``argv[1]`` (the tmp file) and deletes it.
* Loads the multi-KB roster via :func:`personal_kb_hook.roster.load_roster`.
  An empty roster (``[]``) is a complete no-op: ZERO POSTs, the tmp file
  is still deleted in the ``finally`` block, the cache is left untouched,
  exit 0.
* For each :class:`~personal_kb_hook.roster.KbEntry` in the roster, in
  order, issues ONE POST to ``{entry.url.rstrip('/')}/api/kb/listener``
  with a ``Bearer {entry.key}`` token and ``Content-Type: application/json``
  (``_TIMEOUT`` = 30 seconds, stdlib ``urllib`` only, no retries). The
  entire per-KB iteration — request build, urlopen, body read/decode,
  ``json.loads``, dict check, ``'pointer'``-key check, and
  :func:`_validate_map` — is wrapped in ONE broad ``try / except
  Exception ⇒ None`` block so ANY failure for that label yields ``None``
  and the loop continues. (``URLError`` / ``HTTPError`` /
  ``TimeoutError`` / ``JSONDecodeError`` / ``UnicodeDecodeError`` /
  non-dict body / missing ``'pointer'`` / validation failure are
  illustrative members of that catch, not an exhaustive enumeration.)
* Collects ``(label, validated_pointer-or-None)`` pairs in roster order,
  then runs CLIENT-SIDE SUPPRESS-ONLY arbitration: drop nulls, title-dedup
  by ``str.strip().casefold()`` of ``short_title`` (winner per the
  detached-worker tie-break: source_label-in-roster → ``'personal'`` →
  first roster entry), and a defensive one-per-KB cap. Arbitration NEVER
  elevates, re-scores, synthesizes, or reorders by relevance.
* When at least one pointer survives arbitration, atomically merges the
  winners into the cache at ``argv[2]`` as a per-KB-provenance ``pending``
  list ``[{label, id, short_title, long_title}, ...]`` (≤ 1 per label),
  with ``whispered_map_ids`` preserved in the ``[label, id]`` shape
  (tolerantly back-parsing any pre-P2 bare-id strings).
* On empty arbitration result or any failure: cache is left unchanged.
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
import urllib.request
from pathlib import Path
from typing import Any

from personal_kb_hook import roster

logger = logging.getLogger(__name__)

_TIMEOUT: float = 30.0

# Hardcoded legacy label (NOT imported from roster._LEGACY_LABEL, per AC-11)
# — matches the literal used by suppression._LEGACY_LABEL and the roster
# loader's synthesized 'personal' fallback entry.
_LEGACY_LABEL = "personal"


def _validate_map(map_obj: object) -> dict[str, str] | None:
    """Validate a ``pointer`` dict from the service response.

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


def _post_one_kb(
    entry: roster.KbEntry,
    body_bytes: bytes,
) -> dict[str, str] | None:
    """POST the manifest to ONE KB and validate the response.

    Returns the validated pointer dict on success, or ``None`` on:
    null ``pointer``, missing ``pointer`` key, non-dict body, JSON decode
    error, HTTPError / URLError / TimeoutError, validation failure, or
    any unexpected exception. NEVER raises.
    """
    try:
        endpoint = entry.url.rstrip("/") + "/api/kb/listener"
        req = urllib.request.Request(  # noqa: S310
            endpoint,
            data=body_bytes,
            headers={
                "Authorization": f"Bearer {entry.key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
        data: Any = json.loads(body)
        if not isinstance(data, dict):
            return None
        if "pointer" not in data:
            return None
        pointer: Any = data["pointer"]
        if pointer is None:
            return None
        return _validate_map(pointer)
    except Exception:
        return None


def _arbitrate(
    candidates: list[tuple[str, dict[str, str] | None]],
    roster_entries: list[roster.KbEntry],
    source_label: str | None,
) -> list[tuple[str, dict[str, str]]]:
    """Suppress-only client-side arbitration.

    Deterministic steps, in order (NEVER elevates or re-scores):

    (a) Drop every null pointer.
    (b) Title-dedup: group surviving candidates by
        ``str.strip().casefold()`` of ``short_title`` and keep the
        tie-break winner per group.
    (c) One-per-KB cap: at most one pointer survives per ``label`` (a
        defensive no-op today, since each KB returns exactly one pointer
        per request).

    Tie-break order (detached-worker fallback — does NOT call
    ``resolve_project`` / ``http_index.load_index``):

    1. The KB whose ``label`` equals the request's ``source_label`` IF that
       label is present in the roster.
    2. The KB whose ``label`` equals ``'personal'`` (the legacy literal)
       IF present in the roster.
    3. The first KB in roster order.
    """
    # (a) Drop nulls.
    surviving = [(label, p) for label, p in candidates if p is not None]
    if not surviving:
        return []

    roster_labels = [e.label for e in roster_entries]
    if not roster_labels:
        return []

    # Resolve the winner label per AC-6.
    if source_label and source_label in roster_labels:
        winner_label = source_label
    elif _LEGACY_LABEL in roster_labels:
        winner_label = _LEGACY_LABEL
    else:
        winner_label = roster_labels[0]

    # Build a stable label-rank map: winner first, then 'personal' if
    # it's in-roster and not already the winner, then any remaining roster
    # entries in roster order. Labels not appearing in the roster get a
    # sentinel rank that pushes them to the end (defence-in-depth — the
    # only labels in ``surviving`` already came from the roster).
    label_rank: dict[str, int] = {}
    next_rank = 0
    label_rank[winner_label] = next_rank
    next_rank += 1
    if _LEGACY_LABEL in roster_labels and _LEGACY_LABEL not in label_rank:
        label_rank[_LEGACY_LABEL] = next_rank
        next_rank += 1
    for e in roster_entries:
        if e.label not in label_rank:
            label_rank[e.label] = next_rank
            next_rank += 1

    # (b) Title-dedup with tie-break.
    by_title: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for label, pointer in surviving:
        title_key = pointer["short_title"].strip().casefold()
        by_title.setdefault(title_key, []).append((label, pointer))

    deduped: list[tuple[str, dict[str, str]]] = []
    for group in by_title.values():
        group.sort(key=lambda lp: label_rank.get(lp[0], 1_000_000))
        deduped.append(group[0])

    # (c) One-per-KB cap (defensive).
    seen_labels: set[str] = set()
    final: list[tuple[str, dict[str, str]]] = []
    for label, pointer in deduped:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        final.append((label, pointer))

    return final


def _coerce_whispered(raw_ids: Any) -> list[list[str]]:
    """Coerce on-disk ``whispered_map_ids`` to ``[[label, id], ...]``.

    Tolerant back-parse (mirrors
    :func:`personal_kb_hook.suppression._read_scratch`):

    * ``[label, id]`` — both strings, non-empty: kept verbatim.
    * ``"<id>"`` (bare string) — legacy pre-P2 shape: kept as
      ``[_LEGACY_LABEL, "<id>"]``.
    * Anything else: silently dropped.
    """
    out: list[list[str]] = []
    if not isinstance(raw_ids, list):
        return out
    for item in raw_ids:
        if isinstance(item, str) and item:
            out.append([_LEGACY_LABEL, item])
        elif isinstance(item, list) and len(item) == 2:
            label, ident = item[0], item[1]
            if isinstance(label, str) and label and isinstance(ident, str) and ident:
                out.append([label, ident])
    return out


def _merge_into_cache(
    cache_path: Path,
    winners: list[tuple[str, dict[str, str]]],
) -> None:
    """Atomically write the post-arbitration ``pending`` list to the cache.

    Preserves ``whispered_map_ids`` in the ``[label, id]`` shape (with
    tolerant legacy bare-id back-parse). ``pending`` becomes a list of
    per-KB pointer objects ``{label, id, short_title, long_title}`` with
    at most one element per label.
    """
    existing: dict[str, Any] = {}
    try:
        if cache_path.exists():
            raw = cache_path.read_text(encoding="utf-8", errors="replace")
            parsed: Any = json.loads(raw)
            if isinstance(parsed, dict):
                existing = parsed
    except (OSError, json.JSONDecodeError, ValueError):
        pass

    whispered_pairs = _coerce_whispered(existing.get("whispered_map_ids", []))

    pending_list: list[dict[str, str]] = [
        {
            "label": label,
            "id": pointer["id"],
            "short_title": pointer["short_title"],
            "long_title": pointer["long_title"],
        }
        for label, pointer in winners
    ]

    new_cache: dict[str, Any] = {
        "pending": pending_list,
        "whispered_map_ids": whispered_pairs,
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

        # Load roster ourselves — the Stop hook does NOT pass it in.
        # An empty roster is a complete no-op (zero POSTs, cache untouched).
        roster_entries = roster.load_roster()
        if not roster_entries:
            return

        body_bytes = json.dumps(request_data).encode("utf-8")

        raw_source_label = request_data.get("source_label")
        source_label = (
            raw_source_label if isinstance(raw_source_label, str) and raw_source_label else None
        )

        # Sequential fan-out across the roster, in order. Each iteration is
        # individually guarded so a failure for one KB never aborts the loop.
        candidates: list[tuple[str, dict[str, str] | None]] = []
        for entry in roster_entries:
            result = _post_one_kb(entry, body_bytes)
            candidates.append((entry.label, result))

        # Client-side, suppress-only arbitration.
        winners = _arbitrate(candidates, roster_entries, source_label)

        if winners:
            _merge_into_cache(Path(cache_path_str), winners)

    except Exception:
        logger.debug("listener_worker: unhandled error", exc_info=True)
    finally:
        if request_tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(request_tmp_path)


if __name__ == "__main__":
    main()
