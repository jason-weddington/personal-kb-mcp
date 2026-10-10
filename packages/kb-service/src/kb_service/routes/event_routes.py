"""Harness event ingest: POST /api/kb/event and GET /api/kb/event/heartbeat.

The failure-cue index. The personal-kb-hook posts one ``post_tool`` event per
tool failure (Claude Code's ``PostToolUseFailure`` hook); this route
normalizes it with :func:`kb_core.cues.build_cue` and records one
``failure_events`` row (SERVICE DB), keyed for idempotency by ``event_id``.

Record-only: the response carries no delivery content (no additionalContext,
no decision) — delivery belongs to later items. No model/LLM call is made.
The full event envelope (``session_start``, ``pre_tool``, ``post_tool``,
``turn_end``) is accepted, but only ``post_tool`` is acted on; the rest are
answered with ``reason='unsupported-type'``.

The route always answers HTTP 200 for a schema-valid body: a DB failure is
logged at WARNING and reported as ``reason='write-failed'``, never a 5xx, so
the fire-and-forget hook never has to interpret server errors.

``GET /api/kb/event/heartbeat`` groups recent rows by (harness, mode, host)
plus the per-process ``_OUTCOMES`` counter. It counts failures only, so a
zero row-count cannot be told apart from a dead pipeline on its own — check
the hook's ``event-drops.jsonl`` drop log as the cross-check.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from kb_core.cues import FailureCue, build_cue

from kb_service import harness_tools
from kb_service.auth import get_current_user
from kb_service.database import get_db
from kb_service.models import (
    EventHeartbeatResponse,
    EventHeartbeatRow,
    EventRequest,
    EventResponse,
    User,
)

router = APIRouter(prefix="/api/kb", tags=["kb"])

logger = logging.getLogger(__name__)

# Count of each returned ``reason`` since process start (heartbeat surface).
_OUTCOMES: dict[str, int] = {}

_INSERT_SQL = (
    "INSERT INTO failure_events (event_id, cue_key, normalizer_version,"
    " session_id, harness, mode, engine, host, hook_version, host_class,"
    " project, project_source, tool, target, target_class, normalized_error,"
    " error_rule, anomaly, raw_error_excerpt, is_interrupt, ts, received_ts)"
    " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,"
    " $15, $16, $17, $18, $19, $20, $21, $22)"
    " ON CONFLICT (event_id) DO NOTHING"
)

_HEARTBEAT_SQL = (
    "SELECT harness, mode, host, COUNT(*) AS count,"
    " SUM(CASE WHEN anomaly IS NOT NULL THEN 1 ELSE 0 END) AS anomalies,"
    " MAX(received_ts) AS last_ts, MAX(hook_version) AS hook_version"
    " FROM failure_events WHERE received_ts >= $1"
    " GROUP BY harness, mode, host ORDER BY harness, mode, host"
)


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


def _anomaly(tool_name: str, cue: FailureCue) -> str | None:
    if cue.normalized_error == "":
        return "empty_error"
    if tool_name == "Bash" and cue.target_class == "":
        return "empty_bash_target"
    return None


async def _record(
    body: EventRequest, tool_name: str, error: str
) -> tuple[EventResponse, FailureCue | None]:
    """Normalize and insert one failure; never raises."""
    try:
        cue = build_cue(tool_name, body.tool_input, error, body.project, body.cwd)
        anomaly = _anomaly(tool_name, cue)
        if anomaly is not None:
            logger.warning(
                "failure_event anomaly=%s event_id=%s error=%r",
                anomaly,
                body.event_id,
                error[:200],
            )
        received_ts = datetime.now(UTC).isoformat(timespec="seconds")
        ts = _normalize_ts(body.ts, received_ts)
        pool = await get_db()
        status = await pool.execute(
            _INSERT_SQL,
            body.event_id,
            cue.cue_key,
            cue.normalizer_version,
            body.session_id,
            body.harness,
            body.mode,
            body.engine,
            body.host,
            body.hook_version,
            cue.host_class,
            cue.project,
            cue.project_source,
            cue.tool,
            cue.target[:500],
            cue.target_class,
            cue.normalized_error,
            cue.error_rule,
            anomaly,
            error[:2000],
            1 if body.is_interrupt else 0,
            ts,
            received_ts,
        )
    except Exception as exc:
        logger.warning(
            "failure_event write_failed event_id=%s session_id=%s tool_name=%s exc=%s",
            body.event_id,
            body.session_id,
            tool_name,
            type(exc).__name__,
        )
        return EventResponse(recorded=False, cue_key=None, reason="write-failed"), None
    recorded = status == "INSERT 0 1"
    response = EventResponse(
        recorded=recorded,
        cue_key=cue.cue_key,
        normalizer_version=cue.normalizer_version,
        reason="recorded" if recorded else "duplicate",
    )
    return response, cue


async def _decide(body: EventRequest) -> tuple[EventResponse, FailureCue | None]:
    if body.type != "post_tool":
        return EventResponse(recorded=False, reason="unsupported-type"), None
    if not body.is_error:
        return EventResponse(recorded=False, reason="not-failure"), None
    tool_name, error = body.tool_name, body.error
    if tool_name is None or error is None or not tool_name.strip() or not error.strip():
        return EventResponse(recorded=False, reason="missing-fields"), None
    return await _record(
        body, harness_tools.canonical_tool(body.harness, tool_name), error
    )


@router.post("/event", response_model=EventResponse)
async def post_event(
    body: EventRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> EventResponse:
    """Record a harness event (only ``post_tool`` failures today)."""
    del user  # auth gate only
    response, cue = await _decide(body)
    _OUTCOMES[response.reason] = _OUTCOMES.get(response.reason, 0) + 1
    logger.info(
        "failure_event reason=%s type=%s tool_name=%s cue_key=%s"
        " project_source=%s error_rule=%s",
        response.reason,
        body.type,
        body.tool_name,
        response.cue_key,
        cue.project_source if cue else None,
        cue.error_rule if cue else None,
    )
    return response


@router.get("/event/heartbeat", response_model=EventHeartbeatResponse)
async def event_heartbeat(
    user: Annotated[User, Depends(get_current_user)],
    hours: Annotated[int, Query(ge=1, le=720)] = 24,
) -> EventHeartbeatResponse:
    """Recent failure-event counts by (harness, mode, host) plus route outcomes."""
    del user  # auth gate only
    since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat(timespec="seconds")
    pool = await get_db()
    records = await pool.fetch(_HEARTBEAT_SQL, since)
    rows = [
        EventHeartbeatRow(
            harness=r["harness"],
            mode=r["mode"],
            host=r["host"],
            count=int(r["count"]),
            anomalies=int(r["anomalies"] or 0),
            last_ts=r["last_ts"],
            hook_version=r["hook_version"],
        )
        for r in records
    ]
    return EventHeartbeatResponse(
        since=since, rows=rows, route_outcomes=dict(_OUTCOMES)
    )
