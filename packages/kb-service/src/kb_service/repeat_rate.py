"""Weekly cross-session repeat-mistake rate from the failure-cue index.

Reads ``failure_events`` (service DB) and the resolution entries in the data DB,
and computes, per ISO week, how often a session hit a failure cue that an
earlier session had already hit. The repeat rule and the per-pair booking match
``scripts/failure_cue_baseline.py`` so the numbers are comparable. The module
makes no LLM call and writes nothing to either database.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from kb_core.cues import bash_segments

from kb_service import prevention
from kb_service.models import (
    RepeatRateCounts,
    RepeatRateCue,
    RepeatRateCutRow,
    RepeatRateDiagnostics,
    RepeatRateResponse,
    RepeatRateWeek,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from kb_service.db_types import DbPool

logger = logging.getLogger(__name__)

DEFAULT_WEEKS = 8
MAX_WEEKS = 104
DEFAULT_MIN_GAP_HOURS = 24.0
MAX_MIN_GAP_HOURS = 720.0
TOP_CUES_CAP = 20

_FAILURE_ROWS_SQL = (
    "SELECT id, session_id, cue_key, normalizer_version, harness, mode, engine, "
    "host, host_class, project, tool, target, target_class, normalized_error, ts "
    "FROM failure_events WHERE is_interrupt = 0 AND ts < $1 ORDER BY ts, id"
)
_FAILURE_ROWS_PROJECT_SQL = (
    "SELECT id, session_id, cue_key, normalizer_version, harness, mode, engine, "
    "host, host_class, project, tool, target, target_class, normalized_error, ts "
    "FROM failure_events WHERE is_interrupt = 0 AND ts < $1 AND project = $2 "
    "ORDER BY ts, id"
)
_RESOLUTION_CUES_SQL = (
    "SELECT id, created_at, hints, project_ref FROM knowledge_entries"
    " WHERE entry_type != 'mental_map'"
    " AND hints LIKE '%\"resolution\"%'"
    " ORDER BY id"
)


@dataclass(frozen=True)
class FailureRow:
    """One non-interrupt failure event (``ts`` is always UTC-aware)."""

    id: int
    session_id: str
    cue_key: str
    normalizer_version: int
    harness: str
    mode: str
    engine: str | None
    host: str | None
    host_class: str
    project: str
    tool: str
    target: str
    target_class: str
    normalized_error: str
    ts: datetime


@dataclass(frozen=True)
class ResolutionCue:
    """A resolution entry reduced to the cue fields used for coverage."""

    entry_id: str
    project: str
    scope: str
    tool: str
    target_class: str
    args_prefix: str
    created_at: datetime


@dataclass
class FailureFetchStats:
    """Counters from :func:`fetch_failure_rows`."""

    scanned: int = 0
    skipped_bad_ts: int = 0
    noncanonical_ts: int = 0
    first_noncanonical_id: int | None = None


@dataclass
class ResolutionCueLoadStats:
    """Counters from :func:`load_resolution_cues`."""

    scanned: int = 0
    skipped_malformed: int = 0
    skipped_no_cue: int = 0
    skipped_bad_created_at: int = 0


def window_bounds(now: datetime, weeks: int) -> tuple[datetime, datetime]:
    """Return ``(start, end)`` covering *weeks* whole ISO weeks incl. now's week.

    A naive *now* is treated as UTC. ``end`` is 00:00 UTC on the Monday after
    the week containing *now*.

    Raises:
        ValueError: If *weeks* is outside ``1..MAX_WEEKS``.
    """
    if weeks < 1 or weeks > MAX_WEEKS:
        raise ValueError(f"weeks must be between 1 and {MAX_WEEKS}, got {weeks}")
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    day = now.astimezone(UTC).date()
    monday = day - timedelta(days=day.weekday())
    end = datetime(monday.year, monday.month, monday.day, tzinfo=UTC) + timedelta(
        days=7
    )
    return end - timedelta(days=7 * weeks), end


async def fetch_failure_rows(
    pool: DbPool, window_end: datetime, project: str | None
) -> tuple[list[FailureRow], FailureFetchStats]:
    """Read non-interrupt failure rows with ``ts < window_end`` in (ts, id) order.

    Raises:
        ValueError: If *window_end* is naive.
    """
    if window_end.tzinfo is None:
        raise ValueError("window_end must be timezone-aware")
    end = window_end.astimezone(UTC).isoformat(timespec="seconds")
    if project:
        raw = await pool.fetch(_FAILURE_ROWS_PROJECT_SQL, end, project)
    else:
        raw = await pool.fetch(_FAILURE_ROWS_SQL, end)
    stats = FailureFetchStats()
    out: list[FailureRow] = []
    for row in raw:
        stats.scanned += 1
        try:
            ts = datetime.fromisoformat(row["ts"])
        except ValueError:
            stats.skipped_bad_ts += 1
            logger.warning(
                "repeat_rate skipped_row id=%d reason=bad_ts", int(row["id"])
            )
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        ts = ts.astimezone(UTC)
        if row["ts"] != ts.isoformat(timespec="seconds"):
            stats.noncanonical_ts += 1
            if stats.first_noncanonical_id is None:
                stats.first_noncanonical_id = int(row["id"])
        out.append(
            FailureRow(
                id=int(row["id"]),
                session_id=row["session_id"],
                cue_key=row["cue_key"],
                normalizer_version=int(row["normalizer_version"]),
                harness=row["harness"],
                mode=row["mode"],
                engine=row["engine"],
                host=row["host"],
                host_class=row["host_class"],
                project=row["project"],
                tool=row["tool"],
                target=row["target"],
                target_class=row["target_class"],
                normalized_error=row["normalized_error"],
                ts=ts,
            )
        )
    if stats.noncanonical_ts > 0:
        logger.warning(
            "repeat_rate tripwire=noncanonical_ts count=%d first_id=%d",
            stats.noncanonical_ts,
            stats.first_noncanonical_id,
        )
    return out, stats


async def load_resolution_cues(
    db: Any,
) -> tuple[list[ResolutionCue], ResolutionCueLoadStats]:
    """Load every resolution entry that carries a cue, active or not.

    A resolution counts from its own ``created_at`` even if it was later
    superseded or deactivated, so there is deliberately no active filter.
    """
    stats = ResolutionCueLoadStats()
    cursor = await db.execute(_RESOLUTION_CUES_SQL)
    rows = await cursor.fetchall()
    out: list[ResolutionCue] = []
    first_skipped: str | None = None
    for row in rows:
        stats.scanned += 1
        entry_id = str(row[0])
        try:
            parsed = prevention.parse_resolution(entry_id, str(row[1]), row[2])
        except prevention._SkipError:
            stats.skipped_malformed += 1
            if first_skipped is None:
                first_skipped = entry_id
            continue
        if parsed.cue_tool == "" or parsed.cue_target_class == "":
            stats.skipped_no_cue += 1
            continue
        try:
            created = datetime.fromisoformat(str(row[1]))
        except ValueError:
            stats.skipped_bad_created_at += 1
            if first_skipped is None:
                first_skipped = entry_id
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        out.append(
            ResolutionCue(
                entry_id=entry_id,
                project=row[3] or "",
                scope=parsed.scope,
                tool=parsed.cue_tool,
                target_class=parsed.cue_target_class,
                args_prefix=parsed.cue_args_prefix,
                created_at=created.astimezone(UTC),
            )
        )
    if stats.skipped_malformed + stats.skipped_bad_created_at > 0:
        logger.warning(
            "repeat_rate skipped_resolutions first_entry_id=%s malformed=%d"
            " bad_created_at=%d",
            first_skipped,
            stats.skipped_malformed,
            stats.skipped_bad_created_at,
        )
    return out, stats


def cue_matches(res: ResolutionCue, row: FailureRow) -> bool:
    """True when *res* describes the failure *row* (ignoring creation time)."""
    if res.scope != "global" and res.project != row.project:
        return False
    if res.tool != row.tool:
        return False
    if row.tool == "Bash":
        words = res.args_prefix.split()
        for cls, args in bash_segments(row.target):
            if cls == res.target_class and (
                not res.args_prefix or args[: len(words)] == words
            ):
                return True
        return False
    return row.target_class == res.target_class


def resolution_matches(res: ResolutionCue, row: FailureRow) -> bool:
    """True when *res* already existed at the failure and describes it."""
    return res.created_at <= row.ts and cue_matches(res, row)


@dataclass
class _Pair:
    session_id: str
    cue_key: str
    first_ts: datetime
    earliest: FailureRow
    rows: int = 0
    repeat: bool = False
    covered: bool = False
    covering: tuple[str, ...] = ()
    near_miss: bool = False
    week: str = ""


def _iso_week(ts: datetime) -> str:
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _counts(pairs: Sequence[_Pair]) -> RepeatRateCounts:
    sessions = {p.session_id for p in pairs}
    repeat_s = {p.session_id for p in pairs if p.repeat}
    covered_s = {p.session_id for p in pairs if p.covered}
    covered_repeat_s = {p.session_id for p in pairs if p.covered and p.repeat}
    repeat_pairs = sum(1 for p in pairs if p.repeat)
    return RepeatRateCounts(
        sessions=len(sessions),
        repeat_sessions=len(repeat_s),
        repeat_rate=_rate(len(repeat_s), len(sessions)),
        distinct_cue_keys=len({p.cue_key for p in pairs}),
        pairs=len(pairs),
        repeat_pairs=repeat_pairs,
        aggregate_rate=_rate(repeat_pairs, len(pairs)),
        covered_sessions=len(covered_s),
        covered_repeat_sessions=len(covered_repeat_s),
        covered_repeat_rate=_rate(len(covered_repeat_s), len(covered_s)),
    )


_COUNT_FIELDS = (
    "sessions",
    "repeat_sessions",
    "pairs",
    "repeat_pairs",
    "covered_sessions",
    "covered_repeat_sessions",
)


def _cuts(resp: RepeatRateResponse) -> list[list[RepeatRateCutRow]]:
    return [
        resp.by_harness,
        resp.by_mode,
        resp.by_engine,
        resp.by_host_class,
        resp.by_host,
    ]


def audit_repeat_rate(resp: RepeatRateResponse) -> list[str]:
    """Return the sorted names of failed invariant checks on *resp*."""
    failed: set[str] = set()
    for cut in _cuts(resp):
        for week in resp.weeks:
            cells = [r for r in cut if r.week == week.week]
            for name in _COUNT_FIELDS:
                if sum(getattr(r, name) for r in cells) != getattr(week, name):
                    failed.add("cut_sum_mismatch")
    if (
        sum(w.pairs for w in resp.weeks) != resp.total.pairs
        or sum(w.repeat_pairs for w in resp.weeks) != resp.total.repeat_pairs
    ):
        failed.add("pairs_total_mismatch")
    every: list[RepeatRateCounts] = [*resp.weeks, resp.total]
    for cut in _cuts(resp):
        every.extend(cut)
    for c in every:
        if (
            c.repeat_sessions > c.sessions
            or c.repeat_pairs > c.pairs
            or c.sessions > c.pairs
            or c.distinct_cue_keys > c.pairs
            or c.covered_sessions > c.sessions
            or c.covered_repeat_sessions > c.covered_sessions
            or c.covered_repeat_sessions > c.repeat_sessions
        ):
            failed.add("count_bounds")
    return sorted(failed)


def compute_repeat_rate(
    rows: Sequence[FailureRow],
    resolutions: Sequence[ResolutionCue],
    *,
    weeks: int,
    now: datetime,
    min_gap_hours: float = DEFAULT_MIN_GAP_HOURS,
    project: str | None = None,
    fetch_stats: FailureFetchStats | None = None,
    load_stats: ResolutionCueLoadStats | None = None,
) -> RepeatRateResponse:
    """Aggregate the weekly repeat rate; pure (no I/O). See how_it_works.md.

    Raises:
        ValueError: If *weeks* or *min_gap_hours* is out of range.
    """
    if min_gap_hours < 0 or min_gap_hours > MAX_MIN_GAP_HOURS:
        raise ValueError(
            f"min_gap_hours must be between 0 and {MAX_MIN_GAP_HOURS}, "
            f"got {min_gap_hours}"
        )
    window_start, window_end = window_bounds(now, weeks)
    kept = sorted((r for r in rows if r.ts < window_end), key=lambda r: (r.ts, r.id))

    pairs: dict[tuple[str, str], _Pair] = {}
    session_first: dict[str, FailureRow] = {}
    cue_first: dict[str, FailureRow] = {}
    for r in kept:
        key = (r.session_id, r.cue_key)
        pair = pairs.get(key)
        if pair is None:
            pair = pairs[key] = _Pair(r.session_id, r.cue_key, r.ts, r)
        pair.rows += 1
        session_first.setdefault(r.session_id, r)
        cue_first.setdefault(r.cue_key, r)

    first_by_cue: dict[str, tuple[datetime, str]] = {}
    for p in pairs.values():
        cand = (p.first_ts, p.session_id)
        cur = first_by_cue.get(p.cue_key)
        if cur is None or cand < cur:
            first_by_cue[p.cue_key] = cand
    gap = timedelta(hours=min_gap_hours)
    for p in pairs.values():
        t0, s0 = first_by_cue[p.cue_key]
        p.repeat = p.session_id != s0 and p.first_ts >= t0 + gap
        matching = sorted(
            {r.entry_id for r in resolutions if resolution_matches(r, p.earliest)}
        )
        p.covering = tuple(matching)
        p.covered = bool(matching)
        p.near_miss = (not p.covered) and any(
            cue_matches(r, p.earliest) for r in resolutions
        )
        p.week = _iso_week(p.first_ts)

    booked = [p for p in pairs.values() if window_start <= p.first_ts < window_end]

    week_rows: list[RepeatRateWeek] = []
    week_labels: list[str] = []
    for i in range(weeks):
        start = window_start + timedelta(days=7 * i)
        label = _iso_week(start)
        week_labels.append(label)
        c = _counts([p for p in booked if p.week == label])
        week_rows.append(
            RepeatRateWeek(
                week=label, week_start=start.date().isoformat(), **c.model_dump()
            )
        )

    def cut(keyfn: Callable[[FailureRow], str]) -> list[RepeatRateCutRow]:
        groups: dict[tuple[str, str], list[_Pair]] = defaultdict(list)
        for p in booked:
            groups[(p.week, keyfn(session_first[p.session_id]))].append(p)
        return [
            RepeatRateCutRow(week=w, key=k, **_counts(groups[(w, k)]).model_dump())
            for (w, k) in sorted(groups)
        ]

    by_cue: dict[str, list[_Pair]] = defaultdict(list)
    for p in booked:
        by_cue[p.cue_key].append(p)
    cue_entries = []
    for cue_key, ps in by_cue.items():
        repeat_n = sum(1 for p in ps if p.repeat)
        if repeat_n < 1:
            continue
        first = cue_first[cue_key]
        cue_entries.append(
            RepeatRateCue(
                cue_key=cue_key,
                tool=first.tool,
                target_class=first.target_class,
                project=first.project,
                normalized_error=first.normalized_error,
                sessions=len(ps),
                repeat_sessions=repeat_n,
                covered_sessions=sum(1 for p in ps if p.covered),
                covering_resolution_ids=sorted({i for p in ps for i in p.covering}),
            )
        )
    cue_entries.sort(key=lambda c: (-c.repeat_sessions, -c.sessions, c.cue_key))

    booked_keys = {(p.session_id, p.cue_key) for p in booked}
    booked_rows = [r for r in kept if (r.session_id, r.cue_key) in booked_keys]
    versions = sorted({r.normalizer_version for r in booked_rows})
    if len(versions) > 1:
        logger.warning(
            "repeat_rate warning=mixed_normalizer_versions versions=%s", versions
        )
    cue_row_counts: dict[str, int] = defaultdict(int)
    for r in booked_rows:
        cue_row_counts[r.cue_key] += 1
    singleton = top_share = None
    if booked_rows:
        singleton = round(
            sum(1 for n in cue_row_counts.values() if n == 1) / len(cue_row_counts), 4
        )
        top_share = round(max(cue_row_counts.values()) / len(booked_rows), 4)

    noncanonical = fetch_stats.noncanonical_ts if fetch_stats else 0
    diagnostics = RepeatRateDiagnostics(
        failure_rows_scanned=fetch_stats.scanned if fetch_stats else len(rows),
        failure_rows_skipped_bad_ts=fetch_stats.skipped_bad_ts if fetch_stats else 0,
        failure_rows_noncanonical_ts=noncanonical,
        failure_rows_before_window=sum(1 for r in kept if r.ts < window_start),
        resolutions_scanned=load_stats.scanned if load_stats else len(resolutions),
        resolutions_skipped_malformed=(
            load_stats.skipped_malformed if load_stats else 0
        ),
        resolutions_skipped_no_cue=load_stats.skipped_no_cue if load_stats else 0,
        resolutions_skipped_bad_created_at=(
            load_stats.skipped_bad_created_at if load_stats else 0
        ),
        coverage_near_miss_created_after=sum(1 for p in booked if p.near_miss),
        normalizer_versions=versions,
        singleton_cue_fraction=singleton,
        top_cue_share=top_share,
        tripwires=[],
    )
    resp = RepeatRateResponse(
        weeks=week_rows,
        total=_counts(booked),
        by_harness=cut(lambda r: r.harness),
        by_mode=cut(lambda r: r.mode),
        by_engine=cut(lambda r: r.engine or "none"),
        by_host_class=cut(lambda r: r.host_class),
        by_host=cut(lambda r: r.host or "unknown"),
        top_cues=cue_entries[:TOP_CUES_CAP],
        project=project or None,
        min_gap_hours=min_gap_hours,
        window_start=window_start.isoformat(timespec="seconds"),
        window_end=window_end.isoformat(timespec="seconds"),
        failure_rows=len(kept),
        resolutions_loaded=len(resolutions),
        diagnostics=diagnostics,
    )
    names = set(audit_repeat_rate(resp))
    for name in sorted(names):
        logger.warning("repeat_rate tripwire=%s", name)
    if noncanonical > 0:
        names.add("noncanonical_ts")
    resp.diagnostics.tripwires = sorted(names)
    return resp


async def build_repeat_rate(
    pool: DbPool,
    data_db: Any,
    *,
    weeks: int,
    project: str | None,
    min_gap_hours: float,
    now: datetime,
) -> RepeatRateResponse:
    """Fetch inputs from both databases and compute the repeat rate."""
    proj = project or None
    _, window_end = window_bounds(now, weeks)
    if min_gap_hours < 0 or min_gap_hours > MAX_MIN_GAP_HOURS:
        raise ValueError(
            f"min_gap_hours must be between 0 and {MAX_MIN_GAP_HOURS}, "
            f"got {min_gap_hours}"
        )
    t0 = time.perf_counter()
    rows, fstats = await fetch_failure_rows(pool, window_end, proj)
    t1 = time.perf_counter()
    resolutions, lstats = await load_resolution_cues(data_db)
    t2 = time.perf_counter()
    resp = compute_repeat_rate(
        rows,
        resolutions,
        weeks=weeks,
        now=now,
        min_gap_hours=min_gap_hours,
        project=proj,
        fetch_stats=fstats,
        load_stats=lstats,
    )
    t3 = time.perf_counter()
    logger.info(
        "repeat_rate weeks=%d project=%s min_gap_hours=%s window_end=%s"
        " failure_rows_scanned=%d resolutions_scanned=%d resolutions_loaded=%d"
        " sessions=%d repeat_sessions=%d covered_sessions=%d"
        " covered_repeat_sessions=%d tripwires=%s fetch_ms=%d load_ms=%d"
        " compute_ms=%d",
        weeks,
        proj or "",
        min_gap_hours,
        resp.window_end,
        resp.diagnostics.failure_rows_scanned,
        resp.diagnostics.resolutions_scanned,
        resp.resolutions_loaded,
        resp.total.sessions,
        resp.total.repeat_sessions,
        resp.total.covered_sessions,
        resp.total.covered_repeat_sessions,
        ",".join(resp.diagnostics.tripwires) or "none",
        int((t1 - t0) * 1000),
        int((t2 - t1) * 1000),
        int((t3 - t2) * 1000),
    )
    return resp


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.4f}"


def _table(heading: str, header: list[str], body: list[list[object]]) -> list[str]:
    lines = ["", f"## {heading}", "| " + " | ".join(header) + " |"]
    lines.append("|" + "---|" * len(header))
    for row in body:
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return lines


def render_repeat_rate_markdown(resp: RepeatRateResponse) -> str:
    """Render *resp* as markdown (no trailing newline)."""
    d = resp.diagnostics
    lines = [
        f"# Cross-session repeat rate: {resp.weeks[0].week} to {resp.weeks[-1].week}"
        f" (project={resp.project or 'all'}, min_gap_hours={resp.min_gap_hours})"
    ]
    if d.tripwires:
        lines += ["", "**TRIPWIRE: " + ", ".join(d.tripwires) + "**"]

    def week_row(label: str, c: RepeatRateCounts) -> list[object]:
        return [
            label,
            c.sessions,
            c.repeat_sessions,
            _fmt(c.repeat_rate),
            c.pairs,
            c.repeat_pairs,
            _fmt(c.aggregate_rate),
            c.distinct_cue_keys,
            c.covered_sessions,
            c.covered_repeat_sessions,
            _fmt(c.covered_repeat_rate),
        ]

    lines += _table(
        "weeks",
        [
            "week",
            "sessions",
            "repeat_sessions",
            "repeat_rate",
            "pairs",
            "repeat_pairs",
            "aggregate_rate",
            "distinct_cue_keys",
            "covered_sessions",
            "covered_repeat_sessions",
            "covered_repeat_rate",
        ],
        [week_row(w.week, w) for w in resp.weeks] + [week_row("total", resp.total)],
    )
    cut_header = [
        "week",
        "key",
        "sessions",
        "repeat_sessions",
        "repeat_rate",
        "aggregate_rate",
        "covered_sessions",
        "covered_repeat_sessions",
        "covered_repeat_rate",
    ]
    for name, cut_rows in (
        ("by_harness", resp.by_harness),
        ("by_mode", resp.by_mode),
        ("by_engine", resp.by_engine),
        ("by_host_class", resp.by_host_class),
        ("by_host", resp.by_host),
    ):
        lines += _table(
            name,
            cut_header,
            [
                [
                    r.week,
                    r.key,
                    r.sessions,
                    r.repeat_sessions,
                    _fmt(r.repeat_rate),
                    _fmt(r.aggregate_rate),
                    r.covered_sessions,
                    r.covered_repeat_sessions,
                    _fmt(r.covered_repeat_rate),
                ]
                for r in cut_rows
            ],
        )
    lines += _table(
        "top_cues",
        [
            "cue_key",
            "tool",
            "target_class",
            "project",
            "sessions",
            "repeat_sessions",
            "covered_sessions",
            "covering_resolution_ids",
            "normalized_error",
        ],
        [
            [
                c.cue_key,
                c.tool,
                c.target_class,
                c.project,
                c.sessions,
                c.repeat_sessions,
                c.covered_sessions,
                ", ".join(c.covering_resolution_ids) or "-",
                c.normalized_error,
            ]
            for c in resp.top_cues
        ],
    )
    skipped = (
        d.resolutions_skipped_malformed
        + d.resolutions_skipped_no_cue
        + d.resolutions_skipped_bad_created_at
    )
    versions = ",".join(str(v) for v in d.normalizer_versions) or "-"
    lines += [
        "",
        f"failure_rows={resp.failure_rows}"
        f" failure_rows_scanned={d.failure_rows_scanned}"
        f" skipped_bad_ts={d.failure_rows_skipped_bad_ts}"
        f" noncanonical_ts={d.failure_rows_noncanonical_ts}"
        f" before_window={d.failure_rows_before_window}"
        f" resolutions_loaded={resp.resolutions_loaded}"
        f" resolutions_scanned={d.resolutions_scanned}"
        f" resolutions_skipped={skipped}"
        f" near_miss_created_after={d.coverage_near_miss_created_after}"
        f" normalizer_versions={versions}"
        f" tripwires={','.join(d.tripwires) or 'none'}",
    ]
    return "\n".join(lines)
