"""Prevention channels: gate index + gotcha slice, gate decisions, gate stats.

* ``GET /api/kb/prevention`` — the per-session payload the hook fetches at
  SessionStart (and refreshes at Stop): soft-gate settings, the Bash cue
  index the PreToolUse gate matches against (empty unless the gate is
  enabled), and the SessionStart gotcha slice (always returned).
* ``POST /api/kb/prevention/decisions`` — batch ingest of hook-recorded gate
  decisions into the SERVICE DB ``gate_decisions`` table. A batch endpoint,
  because PreToolUse must never touch the network: the hook logs locally and
  flushes at Stop.
* ``GET /api/kb/prevention/stats`` — efficacy and health of the gate.

Server switches are read per request from ``os.environ`` and fail closed:

* ``KB_SOFT_GATE_ENABLED`` — literally ``TRUE`` (any case) turns the index on.
* ``KB_SOFT_GATE_SHADOW`` — default shadow; only ``FALSE`` enables real denies.
* ``KB_SOFT_GATE_DISABLED_PROJECTS`` — comma-separated per-project kill switch.
* ``KB_DELIVER_OBSERVED_ONCE`` — default on; ``FALSE`` withholds autonomous
  resolutions observed in a single session.
* ``KB_SURPRISE_CAPTURE`` — ``off|shadow|on``, default off; any other value
  means off. Reported as the top-level ``surprise_capture`` field.

No route answers with a 5xx: a failure is logged at WARNING and answered with
an inert, empty response. No model or LLM call is made anywhere here.
"""

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from kb_core.cues import resolve_cue_project

from kb_service.auth import get_current_user
from kb_service.database import get_db
from kb_service.models import (
    GateDecisionBatch,
    GateDecisionResponse,
    GateDecisionRow,
    GateInvariantViolations,
    GateSettings,
    GateStatsHostRow,
    GateStatsResolutionRow,
    GateStatsResponse,
    PreventionDiagnostics,
    PreventionResponse,
    SurpriseCaptureMode,
    User,
)
from kb_service.prevention import (
    FIRST_SIGHTING_SHAPES,
    build_gate_index,
    build_slice,
    count_index_excluded_observed_once,
    load_corrections,
    load_resolutions,
    render_slice,
)
from kb_service.turn_digest import surprise_capture_mode

router = APIRouter(prefix="/api/kb", tags=["kb"])

logger = logging.getLogger(__name__)

MAX_DENIES_PER_SESSION = 2
_CORRECTIONS_LIMIT = 20
_DECISIONS = (
    "denied",
    "would_deny",
    "skipped_already_denied",
    "skipped_cap",
    "retry",
    "armed",
    "summary",
)

_INSERT_SQL = (
    "INSERT INTO gate_decisions (decision_id, session_id, harness, mode, engine,"
    " host, hook_version, project, resolution_id, resolution_updated_at, tool,"
    " target, target_class, decision, shadow, reason_excerpt,"
    " retry_changed_command, prior_target, observed_once, index_len, slice_len,"
    " pre_tool_calls, pre_tool_errors, last_error_type, tool_use_id, ts,"
    " received_ts)"
    " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,"
    " $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27)"
    " ON CONFLICT (decision_id) DO NOTHING"
)

_SESSION_DENIES_SQL = (
    "SELECT COUNT(*) AS n FROM gate_decisions"
    " WHERE session_id = $1 AND decision IN ('denied', 'would_deny')"
)

_COUNTS_SQL = (
    "SELECT decision, COUNT(*) AS n FROM gate_decisions"
    " WHERE received_ts >= $1 GROUP BY decision"
)

_SUMMARY_SQL = (
    "SELECT"
    " COUNT(DISTINCT CASE WHEN decision = 'armed' THEN session_id END)"
    " AS armed_sessions,"
    " SUM(CASE WHEN decision = 'retry' AND retry_changed_command IS NOT NULL"
    " THEN 1 ELSE 0 END) AS retries,"
    " SUM(CASE WHEN decision = 'retry' AND retry_changed_command = 1"
    " THEN 1 ELSE 0 END) AS retries_changed,"
    " SUM(CASE WHEN decision = 'retry' AND retry_changed_command IS NULL"
    " THEN 1 ELSE 0 END) AS retries_abandoned,"
    " COALESCE(SUM(CASE WHEN decision = 'summary' THEN pre_tool_errors END), 0)"
    " AS pre_tool_errors_total"
    " FROM gate_decisions WHERE received_ts >= $1"
)

# One row per denied / would_deny / retry decision with a resolution, plus
# whether a matching failure followed it (meaningful for denied/would_deny).
_RESOLUTION_ROWS_SQL = (
    "SELECT g.resolution_id, g.decision, g.retry_changed_command,"
    " CASE WHEN EXISTS (SELECT 1 FROM failure_events f"
    " WHERE f.session_id = g.session_id AND f.tool = g.tool"
    " AND f.target_class = g.target_class AND f.is_interrupt = 0"
    " AND f.ts >= g.ts) THEN 1 ELSE 0 END AS followed_by_failure"
    " FROM gate_decisions g"
    " WHERE g.received_ts >= $1 AND g.resolution_id != ''"
    " AND g.decision IN ('denied', 'would_deny', 'retry')"
)

_HOSTS_SQL = (
    "SELECT harness, mode, host, COUNT(*) AS armed, MAX(received_ts) AS last_ts,"
    " MAX(hook_version) AS hook_version FROM gate_decisions"
    " WHERE received_ts >= $1 AND decision = 'armed'"
    " GROUP BY harness, mode, host ORDER BY harness, mode, host"
)

_OVER_CAP_SQL = (
    "SELECT COUNT(*) AS n FROM (SELECT session_id FROM gate_decisions"
    " WHERE received_ts >= $1 AND decision IN ('denied', 'would_deny')"
    " GROUP BY session_id HAVING COUNT(*) > $2) AS over_cap"
)

_REPEAT_PAIRS_SQL = (
    "SELECT COUNT(*) AS n FROM (SELECT session_id, resolution_id FROM gate_decisions"
    " WHERE received_ts >= $1 AND decision IN ('denied', 'would_deny')"
    " GROUP BY session_id, resolution_id HAVING COUNT(*) > 1) AS repeats"
)


async def _count(pool: Any, sql: str, *args: Any) -> int:
    row = await pool.fetchrow(sql, *args)
    return int(row["n"] or 0) if row is not None else 0


# --- switches ---------------------------------------------------------------


def _project_disabled(project: str) -> bool:
    raw = os.environ.get("KB_SOFT_GATE_DISABLED_PROJECTS", "")
    disabled = {p.strip().lower() for p in raw.split(",") if p.strip()}
    return project.strip().lower() in disabled


def gate_settings(project: str) -> GateSettings:
    """Read the soft-gate switches for *project* (fail closed)."""
    enabled = os.environ.get("KB_SOFT_GATE_ENABLED", "").strip().upper() == "TRUE"
    shadow = os.environ.get("KB_SOFT_GATE_SHADOW", "").strip().upper() != "FALSE"
    if _project_disabled(project):
        enabled = False
    return GateSettings(
        enabled=enabled, shadow=shadow, max_denies=MAX_DENIES_PER_SESSION
    )


def _deliver_observed_once() -> bool:
    return os.environ.get("KB_DELIVER_OBSERVED_ONCE", "").strip().upper() != "FALSE"


def _inert(
    project: str, surprise_capture: SurpriseCaptureMode = "off"
) -> PreventionResponse:
    return PreventionResponse(
        project=project,
        gate=GateSettings(
            enabled=False, shadow=True, max_denies=MAX_DENIES_PER_SESSION
        ),
        index=[],
        slice=[],
        slice_text="",
        diagnostics=PreventionDiagnostics(),
        surprise_capture=surprise_capture,
    )


# --- GET /api/kb/prevention -------------------------------------------------


@router.get("/prevention", response_model=PreventionResponse)
async def get_prevention(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    project: str | None = None,
    cwd: str | None = None,
    session_id: str | None = None,
) -> PreventionResponse:
    """Return the gate settings, Bash cue index and gotcha slice for a session."""
    del user  # auth gate only
    mode = surprise_capture_mode()
    effective = ""
    try:
        effective = resolve_cue_project(project, cwd)[0]
        gate = gate_settings(effective)
        if not effective:
            logger.info(
                "prevention_fetch project= session_id=%s no_project=true"
                " surprise_capture=%s",
                session_id,
                mode,
            )
            return PreventionResponse(
                project="",
                gate=gate,
                index=[],
                slice=[],
                slice_text="",
                diagnostics=PreventionDiagnostics(),
                surprise_capture=mode,
            )
        db = request.app.state.kb.db
        resolutions, stats = await load_resolutions(
            db, effective, _deliver_observed_once()
        )
        corrections = await load_corrections(db, effective, _CORRECTIONS_LIMIT)
        if gate.enabled:
            index, index_truncated = build_gate_index(resolutions)
            excluded = count_index_excluded_observed_once(resolutions)
        else:
            index, index_truncated = [], 0
            excluded = 0
        untrusted_once = {
            r.entry_id
            for r in resolutions
            if r.observed_once and r.shape not in FIRST_SIGHTING_SHAPES
        }
        leaked = [c.resolution_id for c in index if c.resolution_id in untrusted_once]
        if leaked:
            logger.warning(
                "prevention tripwire=observed_once_in_index project=%s"
                " resolution_ids=%s",
                effective,
                leaked,
            )
        items, slice_truncated = build_slice(resolutions, corrections)
        slice_text = render_slice(effective, items)
        diagnostics = PreventionDiagnostics(
            resolutions_total=stats.resolutions_total,
            skipped_malformed=stats.skipped_malformed,
            skipped_observed_once=stats.skipped_observed_once,
            index_truncated=index_truncated,
            slice_truncated=slice_truncated,
            index_excluded_observed_once=excluded,
        )
    except Exception as exc:
        logger.warning(
            "prevention_fetch failed project=%s session_id=%s exc=%s",
            effective,
            session_id,
            type(exc).__name__,
        )
        return _inert(effective, mode)
    logger.info(
        "prevention_fetch project=%s session_id=%s enabled=%s shadow=%s"
        " index_len=%d slice_len=%d resolutions_total=%d skipped_malformed=%d"
        " skipped_observed_once=%d index_truncated=%d slice_truncated=%d"
        " slice_ids=%s surprise_capture=%s index_excluded_observed_once=%d"
        " index_ids=%s",
        effective,
        session_id,
        gate.enabled,
        gate.shadow,
        len(index),
        len(items),
        diagnostics.resolutions_total,
        diagnostics.skipped_malformed,
        diagnostics.skipped_observed_once,
        diagnostics.index_truncated,
        diagnostics.slice_truncated,
        [i.entry_id for i in items],
        mode,
        diagnostics.index_excluded_observed_once,
        [c.resolution_id for c in index],
    )
    return PreventionResponse(
        project=effective,
        gate=gate,
        index=index,
        slice=items,
        slice_text=slice_text,
        diagnostics=diagnostics,
        surprise_capture=mode,
    )


# --- POST /api/kb/prevention/decisions --------------------------------------


def _normalize_ts(raw: str | None, fallback: str) -> str:
    """Parse an ISO timestamp to UTC, second precision; *fallback* on failure."""
    if not raw:
        return fallback
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat(timespec="seconds")


def _insert_args(row: GateDecisionRow, received_ts: str) -> tuple[Any, ...]:
    changed = row.retry_changed_command
    return (
        row.decision_id,
        row.session_id,
        row.harness,
        row.mode,
        row.engine,
        row.host,
        row.hook_version,
        row.project,
        row.resolution_id,
        row.resolution_updated_at,
        row.tool,
        row.target[:500],
        row.target_class,
        row.decision,
        1 if row.shadow else 0,
        row.reason_excerpt[:1000] if row.reason_excerpt is not None else None,
        None if changed is None else (1 if changed else 0),
        row.prior_target[:500] if row.prior_target is not None else None,
        1 if row.observed_once else 0,
        row.index_len,
        row.slice_len,
        row.pre_tool_calls,
        row.pre_tool_errors,
        row.last_error_type,
        row.tool_use_id,
        _normalize_ts(row.ts, received_ts),
        received_ts,
    )


@router.post("/prevention/decisions", response_model=GateDecisionResponse)
async def post_decisions(
    body: GateDecisionBatch,
    user: Annotated[User, Depends(get_current_user)],
) -> GateDecisionResponse:
    """Record a batch of hook gate decisions (idempotent on ``decision_id``)."""
    del user  # auth gate only
    inserted = 0
    try:
        pool = await get_db()
        received_ts = datetime.now(UTC).isoformat(timespec="seconds")
        deny_sessions: set[str] = set()
        for row in body.rows:
            status = await pool.execute(_INSERT_SQL, *_insert_args(row, received_ts))
            if status == "INSERT 0 1":
                inserted += 1
                if row.decision in ("denied", "would_deny"):
                    deny_sessions.add(row.session_id)
        for sid in sorted(deny_sessions):
            count = await _count(pool, _SESSION_DENIES_SQL, sid)
            if count > MAX_DENIES_PER_SESSION:
                logger.warning(
                    "gate_invariant_violation session_id=%s denies=%d max=%d",
                    sid,
                    count,
                    MAX_DENIES_PER_SESSION,
                )
    except Exception as exc:
        logger.warning(
            "gate_decisions write_failed rows=%d exc=%s",
            len(body.rows),
            type(exc).__name__,
        )
        return GateDecisionResponse(inserted=0, duplicates=0)
    return GateDecisionResponse(inserted=inserted, duplicates=len(body.rows) - inserted)


# --- GET /api/kb/prevention/stats -------------------------------------------


def _empty_stats(since: str) -> GateStatsResponse:
    return GateStatsResponse(
        since=since,
        counts=dict.fromkeys(_DECISIONS, 0),
        armed_sessions=0,
        retries=0,
        retries_changed=0,
        retries_abandoned=0,
        retry_changed_share=None,
        would_deny_precision=None,
        pre_tool_errors_total=0,
        by_resolution=[],
        by_host=[],
        invariant_violations=GateInvariantViolations(
            over_cap_sessions=0, repeat_deny_pairs=0
        ),
    )


def _by_resolution(records: list[Any]) -> tuple[list[GateStatsResolutionRow], int, int]:
    """Aggregate per-resolution rows; also return (would_deny, would_deny_followed)."""
    acc: dict[str, dict[str, int]] = {}
    would_deny = 0
    would_deny_followed = 0
    for r in records:
        rid = str(r["resolution_id"])
        bucket = acc.setdefault(
            rid,
            {
                "denied": 0,
                "would_deny": 0,
                "retries": 0,
                "retries_changed": 0,
                "followed_by_failure": 0,
            },
        )
        decision = r["decision"]
        followed = int(r["followed_by_failure"] or 0)
        if decision in ("denied", "would_deny"):
            bucket[decision] += 1
            bucket["followed_by_failure"] += followed
            if decision == "would_deny":
                would_deny += 1
                would_deny_followed += followed
        elif r["retry_changed_command"] is not None:
            bucket["retries"] += 1
            if int(r["retry_changed_command"]) == 1:
                bucket["retries_changed"] += 1
    rows = [
        GateStatsResolutionRow(resolution_id=rid, **acc[rid]) for rid in sorted(acc)
    ]
    return rows, would_deny, would_deny_followed


@router.get("/prevention/stats", response_model=GateStatsResponse)
async def prevention_stats(
    user: Annotated[User, Depends(get_current_user)],
    hours: Annotated[int, Query(ge=1, le=2160)] = 168,
) -> GateStatsResponse:
    """Soft-gate efficacy (retry share, would-deny precision) and health."""
    del user  # auth gate only
    since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat(timespec="seconds")
    try:
        pool = await get_db()
        counts = dict.fromkeys(_DECISIONS, 0)
        for r in await pool.fetch(_COUNTS_SQL, since):
            counts[str(r["decision"])] = int(r["n"])
        summary = await pool.fetchrow(_SUMMARY_SQL, since) or {}
        retries = int(summary.get("retries") or 0)
        retries_changed = int(summary.get("retries_changed") or 0)
        by_resolution, would_deny, would_deny_followed = _by_resolution(
            list(await pool.fetch(_RESOLUTION_ROWS_SQL, since))
        )
        by_host = [
            GateStatsHostRow(
                harness=r["harness"],
                mode=r["mode"],
                host=r["host"],
                armed=int(r["armed"]),
                last_ts=r["last_ts"],
                hook_version=r["hook_version"],
            )
            for r in await pool.fetch(_HOSTS_SQL, since)
        ]
        over_cap = await _count(pool, _OVER_CAP_SQL, since, MAX_DENIES_PER_SESSION)
        repeats = await _count(pool, _REPEAT_PAIRS_SQL, since)
    except Exception as exc:
        logger.warning("prevention_stats failed exc=%s", type(exc).__name__)
        return _empty_stats(since)
    if over_cap or repeats:
        logger.warning(
            "gate_invariant_violation over_cap_sessions=%d repeat_deny_pairs=%d",
            over_cap,
            repeats,
        )
    return GateStatsResponse(
        since=since,
        counts=counts,
        armed_sessions=int(summary.get("armed_sessions") or 0),
        retries=retries,
        retries_changed=retries_changed,
        retries_abandoned=int(summary.get("retries_abandoned") or 0),
        retry_changed_share=round(retries_changed / retries, 4) if retries else None,
        would_deny_precision=(
            round(would_deny_followed / would_deny, 4) if would_deny else None
        ),
        pre_tool_errors_total=int(summary.get("pre_tool_errors_total") or 0),
        by_resolution=by_resolution,
        by_host=by_host,
        invariant_violations=GateInvariantViolations(
            over_cap_sessions=over_cap, repeat_deny_pairs=repeats
        ),
    )
