#!/usr/bin/env python3
r"""Offline failure-cue baseline: cross-session repeat rate from transcripts.

Read-only: no DB, no network. Walks Claude Code transcript ``.jsonl`` files,
pulls every failed tool call (``tool_result`` blocks with ``is_error: true``),
normalizes each with :func:`kb_core.cues.build_cue` (the SAME normalizer the
live ``POST /api/kb/event`` route uses — this script carries no rules of its
own) and reports how often the same failure cue recurs in a later session.

Population
----------
A failure is a ``tool_result`` block with ``is_error`` true on a ``user``
line. Its text is the block's ``content`` (a str, or the ``text`` items of a
list joined with ``\n``); the line's ``toolUseResult`` is NOT used. Results
whose text starts with ``<tool_use_error>``, ``Permission to use ``,
``The user doesn't want to proceed`` or ``[Request interrupted by user`` are
EXCLUDED (counted under ``excluded``): Claude Code never fires
``PostToolUseFailure`` for validation rejections, permission denials,
cancellations or interrupts, so excluding them keeps this offline population
the same as the live one. Interrupted results are therefore excluded from the
rate; any future live-side query over ``failure_events`` must likewise use
``WHERE is_interrupt = 0``. Failures are deduplicated globally by
``tool_use_id`` (first seen wins).

Repeat-rate definition
----------------------
For a cue ``c``:

* ``s0`` is the session with the minimum ``(earliest ts on c, session_id)``
  — a tie in ts goes to the lexicographically smaller session_id;
* ``t0`` is ``s0``'s earliest ts on ``c``;
* a session ``s != s0`` is a REPEAT session iff its earliest ts on ``c`` is
  ``>= t0 + min_gap_hours``;
* ``sessions(c)`` is the number of distinct sessions that hit ``c``;
* ``repeat_rate(c) = repeat_sessions / sessions``.

Aggregates: ``aggregate_rate = sum(repeat_sessions) / sum(sessions)`` over all
cues (single-session cues stay in the denominator), and ``aggregate_rate_ge2``
is the same over cues with ``sessions >= 2``. Both are ``round(x, 4)``.

Usage:
  uv run python packages/kb-service/scripts/failure_cue_baseline.py PATH [PATH ...] \\
    [--min-gap-hours 24] [--since YYYY-MM-DD] [--json OUT] [--top 30] \\
    [--project-map FILE] [--headless-cwd-prefix /home/dispatch/]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from kb_core.cues import CUE_NORMALIZER_VERSION, FailureCue, build_cue

EXCLUDED_PREFIXES = (
    "<tool_use_error>",
    "Permission to use ",
    "The user doesn't want to proceed",
    "[Request interrupted by user",
)
_RAW_CAP = 300
_SAMPLES = 3


@dataclass(frozen=True)
class Failure:
    """One failed tool call, normalized."""

    cue: FailureCue
    session_id: str
    ts: datetime
    mode: str
    raw: str
    tool_use_id: str


@dataclass
class ParseResult:
    """Failures plus the parse counters reported in ``totals``."""

    failures: list[Failure] = field(default_factory=list)
    transcripts: int = 0
    excluded: int = 0
    malformed: int = 0
    unknown_tool_use: int = 0
    duplicates_skipped: int = 0


def _iter_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(x for x in p.rglob("*.jsonl") if x.is_file()))
        elif p.is_file():
            files.append(p)
    return files


def _read_kb_project(cwd: str) -> str | None:
    """Walk up from *cwd* to a ``.kb_project`` (same rule as the hook resolver)."""
    start = Path(cwd)
    if not start.exists():
        return None
    for directory in [start, *start.parents]:
        marker = directory / ".kb_project"
        try:
            if not marker.is_file():
                continue
            text = marker.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                return line
        return None
    return None


def _result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        )
    return ""


def _parse_ts(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _content_blocks(line: dict[str, Any]) -> list[dict[str, Any]]:
    message = line.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict)]


def parse_paths(
    paths: list[str],
    project_map: dict[str, str] | None,
    headless_prefix: str,
) -> ParseResult:
    """Parse transcripts under *paths* into normalized failures. Never raises."""
    result = ParseResult()
    seen: set[str] = set()
    project_cache: dict[str, str | None] = {}
    project_map = project_map or {}

    def kb_project_for(cwd: str | None) -> str | None:
        if not cwd:
            return None
        if cwd not in project_cache:
            mapped = project_map.get(PurePosixPath(cwd).name)
            project_cache[cwd] = mapped if mapped else _read_kb_project(cwd)
        return project_cache[cwd]

    for path in _iter_files(paths):
        result.transcripts += 1
        try:
            raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        lines: list[dict[str, Any]] = []
        for raw_line in raw_lines:
            if not raw_line.strip():
                continue
            try:
                obj = json.loads(raw_line)
            except (json.JSONDecodeError, ValueError):
                result.malformed += 1
                continue
            if isinstance(obj, dict):
                lines.append(obj)
            else:
                result.malformed += 1

        tool_uses: dict[str, tuple[str, object]] = {}
        for line in lines:
            if line.get("type") != "assistant":
                continue
            for block in _content_blocks(line):
                if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                    tool_uses[block["id"]] = (
                        str(block.get("name", "")),
                        block.get("input"),
                    )

        for line in lines:
            if line.get("type") != "user":
                continue
            for block in _content_blocks(line):
                if (
                    block.get("type") != "tool_result"
                    or block.get("is_error") is not True
                ):
                    continue
                text = _result_text(block.get("content"))
                if text.lstrip().startswith(EXCLUDED_PREFIXES):
                    result.excluded += 1
                    continue
                tool_use_id = block.get("tool_use_id")
                if not isinstance(tool_use_id, str) or tool_use_id not in tool_uses:
                    result.unknown_tool_use += 1
                    continue
                if tool_use_id in seen:
                    result.duplicates_skipped += 1
                    continue
                ts = _parse_ts(line.get("timestamp"))
                session_id = line.get("sessionId")
                if ts is None or not isinstance(session_id, str):
                    result.malformed += 1
                    continue
                seen.add(tool_use_id)
                cwd_raw = line.get("cwd")
                cwd = cwd_raw if isinstance(cwd_raw, str) else None
                name, tool_input = tool_uses[tool_use_id]
                cue = build_cue(name, tool_input, text, kb_project_for(cwd), cwd)
                mode = (
                    "headless"
                    if cwd and cwd.startswith(headless_prefix)
                    else "interactive"
                )
                result.failures.append(
                    Failure(
                        cue=cue,
                        session_id=session_id,
                        ts=ts,
                        mode=mode,
                        raw=text,
                        tool_use_id=tool_use_id,
                    )
                )
    return result


def _iso(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")


def _week(ts: datetime) -> str:
    year, week, _ = ts.date().isocalendar()
    return f"{year}-W{week:02d}"


def compute_report(
    failures: list[Failure],
    min_gap_hours: float,
    parse_result: ParseResult | None = None,
) -> dict[str, Any]:
    """Compute the repeat-rate report (see the module docstring for the definition)."""
    ordered = sorted(failures, key=lambda f: (f.ts, f.session_id, f.tool_use_id))
    by_cue: dict[str, list[Failure]] = defaultdict(list)
    for f in ordered:
        by_cue[f.cue.cue_key].append(f)

    gap = timedelta(hours=min_gap_hours)
    cues: list[dict[str, Any]] = []
    weeks: Counter[str] = Counter()
    # (cue_key, session_id) pairs that are repeats, keyed to their first failure.
    repeat_firsts: list[Failure] = []
    total_sessions = total_repeats = ge2_sessions = ge2_repeats = 0

    for key, fs in by_cue.items():
        first_by_session: dict[str, Failure] = {}
        for f in fs:
            first_by_session.setdefault(f.session_id, f)
        s0 = min(first_by_session, key=lambda s: (first_by_session[s].ts, s))
        t0 = first_by_session[s0].ts
        repeats = [
            first
            for s, first in first_by_session.items()
            if s != s0 and first.ts >= t0 + gap
        ]
        for first in repeats:
            weeks[_week(first.ts)] += 1
        repeat_firsts.extend(repeats)
        n_sessions = len(first_by_session)
        total_sessions += n_sessions
        total_repeats += len(repeats)
        if n_sessions >= 2:
            ge2_sessions += n_sessions
            ge2_repeats += len(repeats)
        distinct_raw: list[str] = []
        for f in fs:
            sample = f.raw[:_RAW_CAP]
            if sample not in distinct_raw:
                distinct_raw.append(sample)
        head = first_by_session[s0].cue
        cues.append(
            {
                "cue_key": key,
                "tool": head.tool,
                "target_class": head.target_class,
                "project": head.project,
                "host_class": head.host_class,
                "normalized_error": head.normalized_error,
                "failures": len(fs),
                "sessions": n_sessions,
                "repeat_sessions": len(repeats),
                "repeat_rate": round(len(repeats) / n_sessions, 4),
                "first_ts": _iso(fs[0].ts),
                "last_ts": _iso(fs[-1].ts),
                "distinct_raw": len(distinct_raw),
                "samples": distinct_raw[:_SAMPLES],
            }
        )
    cues.sort(key=lambda c: (-c["repeat_sessions"], -c["sessions"], c["cue_key"]))

    def cut(attr: str) -> list[dict[str, Any]]:
        fails: Counter[str] = Counter()
        sessions: dict[str, set[str]] = defaultdict(set)
        reps: Counter[str] = Counter()
        for f in ordered:
            k = f.mode if attr == "mode" else f.cue.host_class
            fails[k] += 1
            sessions[k].add(f.session_id)
        for f in repeat_firsts:
            reps[f.mode if attr == "mode" else f.cue.host_class] += 1
        return [
            {
                "key": k,
                "failures": fails[k],
                "sessions": len(sessions[k]),
                "repeat_sessions": reps[k],
            }
            for k in sorted(fails)
        ]

    n_failures = len(ordered)
    distinct = len(by_cue)
    singletons = sum(1 for fs in by_cue.values() if len(fs) == 1)
    largest = max((len(fs) for fs in by_cue.values()), default=0)
    pr = parse_result or ParseResult()
    totals: dict[str, Any] = {
        "normalizer_version": CUE_NORMALIZER_VERSION,
        "harness": "claude-code",
        "transcripts": pr.transcripts,
        "failures": n_failures,
        "excluded": pr.excluded,
        "malformed": pr.malformed,
        "unknown_tool_use": pr.unknown_tool_use,
        "duplicates_skipped": pr.duplicates_skipped,
        "distinct_cues": distinct,
        "cues_ge2_sessions": sum(1 for c in cues if c["sessions"] >= 2),
        "aggregate_rate": round(total_repeats / total_sessions, 4)
        if total_sessions
        else 0.0,
        "aggregate_rate_ge2": round(ge2_repeats / ge2_sessions, 4)
        if ge2_sessions
        else 0.0,
        "singleton_cue_fraction": round(singletons / distinct, 4) if distinct else 0.0,
        "top_cue_share": round(largest / n_failures, 4) if n_failures else 0.0,
        "by_project_source": dict(Counter(f.cue.project_source for f in ordered)),
        "by_error_rule": dict(Counter(f.cue.error_rule for f in ordered)),
        "anomalies": {
            "empty_error": sum(1 for f in ordered if f.cue.normalized_error == ""),
            "empty_bash_target": sum(
                1
                for f in ordered
                if f.cue.normalized_error != ""
                and f.cue.tool == "Bash"
                and f.cue.target_class == ""
            ),
        },
    }
    return {
        "totals": totals,
        "cues": cues,
        "weeks": [{"week": w, "repeat_sessions": weeks[w]} for w in sorted(weeks)],
        "by_host_class": cut("host_class"),
        "by_mode": cut("mode"),
    }


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(out)


def render_markdown(report: dict[str, Any], top: int) -> str:
    """Render the report as Markdown tables (totals, top cues, weeks, cuts)."""
    totals = report["totals"]
    parts = [
        "## Totals",
        _md_table(["metric", "value"], [[k, v] for k, v in totals.items()]),
    ]
    top_cues = [c for c in report["cues"] if c["sessions"] >= 2][:top]
    parts += [
        "## Top cues",
        _md_table(
            [
                "cue_key",
                "tool",
                "target_class",
                "project",
                "sessions",
                "repeats",
                "rate",
                "normalized_error",
            ],
            [
                [
                    c["cue_key"],
                    c["tool"],
                    c["target_class"],
                    c["project"],
                    c["sessions"],
                    c["repeat_sessions"],
                    c["repeat_rate"],
                    c["normalized_error"],
                ]
                for c in top_cues
            ],
        ),
        "## Weeks",
        _md_table(
            ["week", "repeat_sessions"],
            [[w["week"], w["repeat_sessions"]] for w in report["weeks"]],
        ),
    ]
    for name in ("by_host_class", "by_mode"):
        parts += [
            f"## {name}",
            _md_table(
                ["key", "failures", "sessions", "repeat_sessions"],
                [
                    [r["key"], r["failures"], r["sessions"], r["repeat_sessions"]]
                    for r in report[name]
                ],
            ),
        ]
    if totals["singleton_cue_fraction"] > 0.9:
        parts.append("WARN under-collapse: singleton_cue_fraction > 0.9")
    if totals["top_cue_share"] > 0.2:
        parts.append("WARN over-collapse: top_cue_share > 0.2")
    return "\n\n".join(parts) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0] if __doc__ else ""
    )
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--min-gap-hours", type=float, default=24.0)
    parser.add_argument("--since", default=None, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--project-map", default=None)
    parser.add_argument("--headless-cwd-prefix", default="/home/dispatch/")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; prints Markdown, optionally writes ``--json``."""
    args = _parse_args(argv)
    project_map: dict[str, str] = {}
    if args.project_map:
        project_map = json.loads(Path(args.project_map).read_text(encoding="utf-8"))
    parsed = parse_paths(args.paths, project_map, args.headless_cwd_prefix)
    failures = parsed.failures
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=UTC)
        failures = [f for f in failures if f.ts >= since]
    report = compute_report(failures, args.min_gap_hours, parsed)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    sys.stdout.write(render_markdown(report, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
