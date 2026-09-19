#!/usr/bin/env python3
r"""Turn ``listener_decisions`` telemetry (GTD 268e2af3) into a verdict.

The listener writes a best-effort row on EVERY request (kill-switch,
no-candidates, rule-a, rule-b, no-llm, vote-split, vote-none, whispered) into
the SERVICE/AUTH database's ``listener_decisions`` table (see
``kb_service/database.py`` and ``kb_service/routes/listener_routes.py``).
That table has no production reader — this script IS the reader (scope_out:
no reader API, no SPA surface; a script plus SQL is the deliverable).

Reports, over a time window:
  - decisions by reason
  - whisper rate
  - jury-reaching rate (candidates survived rule A + rule B, an LLM was
    available, and 3 votes were cast — i.e. reason in {whispered,
    vote-split, vote-none})
  - vote-none share of jury-reaching decisions
  - rule-A and rule-B candidate attrition (aggregate counts, from the
    n_retrieved / n_after_a / n_after_b columns)
  - fallback retrieval-path rate
  - THE LOAD-BEARING ONE: how often a given map id appears in
    ``candidate_ids`` (considered) versus ``whispered_ids`` (surfaced) —
    this is what the falsifying experiment (docs/nightly-map-maintenance-design.md)
    reads kill/reshape conditions off of.

Usage:
  KB_SERVICE_DATABASE_URL=postgresql://... \\
    uv run python scripts/listener_decision_report.py [--since 14d]
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from kb_service.database import close_db, get_db

# Reasons reached only once retrieval + rule A + rule B all survived AND an
# LLM was available to cast votes -- i.e. the request actually reached the
# jury, whichever way the jury came down.
_JURY_REACHING_REASONS = {"whispered", "vote-split", "vote-none"}

_UNIT_SECONDS = {"h": 3600, "d": 86400, "w": 604800}


def _parse_since(spec: str) -> datetime:
    """Parse a ``<int><h|d|w>`` window spec (e.g. ``'14d'``) into a UTC cutoff."""
    unit = spec[-1:]
    if unit not in _UNIT_SECONDS:
        raise SystemExit(
            f"--since must end in h/d/w, e.g. '24h', '14d', '2w' (got {spec!r})"
        )
    try:
        amount = int(spec[:-1])
    except ValueError:
        raise SystemExit(
            f"--since must be <int><h|d|w>, e.g. '14d' (got {spec!r})"
        ) from None
    return datetime.now(UTC) - timedelta(seconds=amount * _UNIT_SECONDS[unit])


def _pct(numerator: int, denominator: int) -> str:
    if not denominator:
        return "n/a"
    return f"{(100.0 * numerator / denominator):.1f}%"


def _ids(raw: str | None) -> list[str]:
    """Best-effort JSON-array decode; malformed/empty values read as []."""
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(x) for x in decoded] if isinstance(decoded, list) else []


def render_report(rows: list[dict[str, Any]], since: datetime) -> str:
    """Build the report text from decoded ``listener_decisions`` rows."""
    lines: list[str] = []
    total = len(rows)
    lines.append(f"listener_decisions since {since.isoformat()} — {total} decision(s)")
    if total == 0:
        return "\n".join(lines)
    lines.append("")

    reason_counts = Counter(r["reason"] for r in rows)
    lines.append("Decisions by reason:")
    for reason, count in reason_counts.most_common():
        lines.append(f"  {reason:<14} {count:>5}  ({_pct(count, total)})")

    whisper_count = sum(1 for r in rows if r["decision"] == "whisper")
    lines.append("")
    lines.append(
        f"Whisper rate: {whisper_count}/{total} ({_pct(whisper_count, total)})"
    )

    jury_rows = [r for r in rows if r["reason"] in _JURY_REACHING_REASONS]
    jury_count = len(jury_rows)
    lines.append(
        "Jury-reaching rate (whispered/vote-split/vote-none): "
        f"{jury_count}/{total} ({_pct(jury_count, total)})"
    )
    vote_none_count = sum(1 for r in jury_rows if r["reason"] == "vote-none")
    lines.append(
        f"  vote-none share of jury-reaching: {vote_none_count}/{jury_count} "
        f"({_pct(vote_none_count, jury_count)})"
    )

    n_retrieved_total = sum(r["n_retrieved"] for r in rows)
    n_after_a_total = sum(r["n_after_a"] for r in rows)
    n_after_b_total = sum(r["n_after_b"] for r in rows)
    dropped_a = n_retrieved_total - n_after_a_total
    dropped_b = n_after_a_total - n_after_b_total
    lines.append("")
    lines.append("Per-stage candidate attrition:")
    lines.append(f"  retrieved:          {n_retrieved_total}")
    lines.append(
        f"  dropped by rule-A:  {dropped_a} ({_pct(dropped_a, n_retrieved_total)}) "
        f"-> {n_after_a_total} remain"
    )
    lines.append(
        f"  dropped by rule-B:  {dropped_b} ({_pct(dropped_b, n_after_a_total)}) "
        f"-> {n_after_b_total} remain"
    )

    fallback_count = sum(1 for r in rows if r["retrieval_path"] == "fallback-direct")
    lines.append("")
    lines.append(
        f"Fallback retrieval path used: {fallback_count}/{total} "
        f"({_pct(fallback_count, total)})"
    )

    # Which SIGNAL produced a whisper (GTD be964e94's AC4 question). Only
    # whisper rows carry a meaningful signal — a decline has no surfaced
    # pointer to attribute — so this counts over whispers, not all rows.
    # .get, not [], deliberately: the three instances deploy one at a time, so
    # this reader can legitimately run against a host whose schema predates
    # candidate_signal. A missing column should degrade to "(unset)", not
    # crash the whole report on the one column that is nice-to-have.
    signal_counts: Counter[str] = Counter()
    whisper_rows = [r for r in rows if r["decision"] == "whisper"]
    for r in whisper_rows:
        signal_counts[r.get("candidate_signal") or "(unset)"] += 1
    lines.append("")
    lines.append(f"Whispers by candidate signal ({len(whisper_rows)} whispers):")
    if whisper_rows:
        for signal, count in signal_counts.most_common():
            lines.append(f"  {signal:<12} {count} ({_pct(count, len(whisper_rows))})")
    else:
        lines.append("  (no whispers in window)")

    considered: Counter[str] = Counter()
    whispered: Counter[str] = Counter()
    for r in rows:
        for map_id in _ids(r["candidate_ids"]):
            considered[map_id] += 1
        for map_id in _ids(r["whispered_ids"]):
            whispered[map_id] += 1

    lines.append("")
    lines.append("Map id: considered vs whispered (candidate_ids vs whispered_ids):")
    if not considered:
        lines.append("  (no candidate_ids recorded in this window)")
    for map_id, considered_count in considered.most_common():
        whispered_count = whispered.get(map_id, 0)
        lines.append(
            f"  {map_id:<14} considered={considered_count:>4}  "
            f"whispered={whispered_count:>4}  "
            f"({_pct(whispered_count, considered_count)})"
        )

    return "\n".join(lines)


async def _fetch_rows(since: datetime) -> list[dict[str, Any]]:
    pool = await get_db()
    records = await pool.fetch(
        # candidate_signal (GTD be964e94) is selected here because this report
        # is the ONLY consumer of listener_decisions in the codebase: that item
        # added the column and populates it, this item shipped the reader, and
        # neither spec mentioned the other — so without this line the question
        # "how many whispers came from lexical vs detail vs fallback" is only
        # answerable by hand-typed SQL. Same seam, opposite direction, as the
        # retrieval_path filter below.
        "SELECT reason, decision, candidates_considered, candidate_ids,"
        " whispered_ids, n_retrieved, n_after_a, n_after_b, retrieval_path,"
        " candidate_signal"
        " FROM listener_decisions WHERE decided_ts >= $1"
        # Exclude pre-telemetry rows. ~151 legacy rows predate these columns
        # and carry defaults, so including them silently dilutes exactly the
        # fallback and jury-reaching rates this instrument exists to expose —
        # for the whole first two weeks of readings, which is the measurement
        # window that matters.
        " AND retrieval_path <> ''"
        " ORDER BY decided_ts",
        since.isoformat(),
    )
    return [dict(r) for r in records]


async def _run(since: datetime) -> None:
    try:
        rows = await _fetch_rows(since)
    finally:
        await close_db()
    print(render_report(rows, since))


def main() -> None:
    """Parse CLI args, connect via ``KB_SERVICE_DATABASE_URL``, and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        default="14d",
        help="Time window to report over, e.g. '24h', '14d', '2w' (default: 14d)",
    )
    args = parser.parse_args()

    if not os.environ.get("KB_SERVICE_DATABASE_URL"):
        sys.exit("KB_SERVICE_DATABASE_URL must be set (service-auth DB DSN).")

    since = _parse_since(args.since)
    asyncio.run(_run(since))


if __name__ == "__main__":
    main()
