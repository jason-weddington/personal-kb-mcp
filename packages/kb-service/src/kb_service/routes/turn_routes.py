"""Surprise-capture turn digest ingest: POST /api/kb/turn and its heartbeat.

Contract: ``event_id`` is exactly ``'<session_id>:<turn_index>'`` (no harness
prefix); ingest is idempotent on ``event_id`` (first write wins). The
``KB_SURPRISE_CAPTURE`` switch (off|shadow|on, default off) is the only switch
read on this path. Free-text fields are secret-redacted before storage. A body
over 64 KiB is rejected with 413 regardless of the switch. The route answers
HTTP 200 for any schema-valid body (DB failure -> ``write-failed``). No model
call is made. The hook's drop log is the cross-check for a zero heartbeat.
"""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from kb_core.cues import resolve_cue_project

from kb_service.auth import get_current_user
from kb_service.database import get_db
from kb_service.models import (
    TurnDigestRequest,
    TurnDigestResponse,
    TurnHeartbeatResponse,
    TurnHeartbeatRow,
    TurnReasoningItem,
    TurnToolCallItem,
    TurnToolResultItem,
    User,
)
from kb_service.turn_digest import (
    TURN_DIGEST_MAX_BYTES,
    insert_turn_digest,
    redact_turn_digest,
    surprise_capture_mode,
    turn_digest_anomalies,
)

router = APIRouter(prefix="/api/kb", tags=["kb"])

logger = logging.getLogger(__name__)

_Reason = Literal[
    "recorded",
    "duplicate",
    "duplicate-mismatch",
    "capture-off",
    "redaction-unavailable",
    "write-failed",
]

# Count of each outcome since process start (heartbeat surface).
_OUTCOMES: dict[str, int] = {}

_HEARTBEAT_SQL = (
    "SELECT harness, mode, host, COUNT(*) AS count,"
    " SUM(truncated) AS truncated,"
    " SUM(CASE WHEN redactions <> '[]' THEN 1 ELSE 0 END) AS redacted,"
    " SUM(CASE WHEN anomaly IS NOT NULL THEN 1 ELSE 0 END) AS anomalies,"
    " SUM(CASE WHEN project = '' THEN 1 ELSE 0 END) AS empty_project,"
    " MAX(received_ts) AS last_ts, MAX(hook_version) AS hook_version"
    " FROM turn_events WHERE received_ts >= $1"
    " GROUP BY harness, mode, host ORDER BY harness, mode, host"
)

_PENDING_SQL = (
    "SELECT COUNT(*) AS n, MIN(received_ts) AS oldest FROM turn_events"
    " WHERE processed_at IS NULL"
)

_GAPS_SQL = (
    "SELECT COUNT(*) AS n FROM (SELECT session_id FROM turn_events"
    " WHERE received_ts >= $1 GROUP BY session_id"
    " HAVING MAX(turn_index) - MIN(turn_index) + 1"
    " > COUNT(DISTINCT turn_index)) AS gaps"
)


def record_validation_failure(errors: Sequence[Any]) -> None:
    """Log a /api/kb/turn 422 (loc and type only) and count it."""
    logger.warning(
        "turn_event reason=invalid errors=%s",
        [(".".join(map(str, e["loc"])), e["type"]) for e in errors][:10],
    )
    _OUTCOMES["invalid"] = _OUTCOMES.get("invalid", 0) + 1


@router.post("/turn", response_model=TurnDigestResponse)
async def post_turn(
    body: TurnDigestRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> TurnDigestResponse:
    """Record one per-turn digest (idempotent on ``event_id``)."""
    del user  # auth gate only
    nbytes = len(await request.body())
    n_calls = sum(1 for i in body.items if isinstance(i, TurnToolCallItem))
    n_reasoning = sum(1 for i in body.items if isinstance(i, TurnReasoningItem))
    n_results = sum(1 for i in body.items if isinstance(i, TurnToolResultItem))
    n_errors = sum(
        1 for i in body.items if isinstance(i, TurnToolResultItem) and i.is_error
    )
    if nbytes > TURN_DIGEST_MAX_BYTES:
        logger.warning(
            "turn_event reason=too-large session_id=%s turn_index=%d bytes=%d"
            " items=%d truncated=%s hook_version=%s harness=%s reasoning=%d",
            body.session_id,
            body.turn_index,
            nbytes,
            len(body.items),
            body.truncated,
            body.hook_version,
            body.harness,
            n_reasoning,
        )
        _OUTCOMES["too-large"] = _OUTCOMES.get("too-large", 0) + 1
        raise HTTPException(413, detail="turn digest exceeds 65536 bytes")

    mode = surprise_capture_mode()
    reason: _Reason
    types: list[str] = []
    anomaly = ""
    if mode == "off":
        reason = "capture-off"
    else:
        try:
            redacted = redact_turn_digest(body)
            if redacted is None:
                logger.warning(
                    "turn_event redaction unavailable event_id=%s", body.event_id
                )
                reason = "redaction-unavailable"
            else:
                digest, found = redacted
                anomalies = turn_digest_anomalies(digest)
                anomaly = ",".join(anomalies)
                if anomalies:
                    logger.warning(
                        "turn_event anomaly=%s event_id=%s", anomaly, body.event_id
                    )
                reason = await insert_turn_digest(
                    await get_db(),
                    digest,
                    capture_mode=mode,
                    redactions=found,
                    received_ts=datetime.now(UTC).isoformat(timespec="seconds"),
                )
                types = found
        except Exception as exc:
            logger.warning(
                "turn_event write_failed event_id=%s session_id=%s exc=%s harness=%s",
                body.event_id,
                body.session_id,
                type(exc).__name__,
                body.harness,
            )
            reason = "write-failed"
            types = []
    response = TurnDigestResponse(
        recorded=reason == "recorded",
        reason=reason,
        redactions=types,
    )
    _OUTCOMES[reason] = _OUTCOMES.get(reason, 0) + 1
    logger.info(
        "turn_event reason=%s capture_mode=%s session_id=%s turn_index=%d"
        " project=%s host=%s hook_version=%s bytes=%d items=%d tool_calls=%d"
        " tool_results=%d result_errors=%d truncated=%s anomaly=%s redactions=%s"
        " harness=%s reasoning=%d",
        reason,
        mode,
        body.session_id,
        body.turn_index,
        resolve_cue_project(body.project, None)[0],
        body.host,
        body.hook_version,
        nbytes,
        len(body.items),
        n_calls,
        n_results,
        n_errors,
        body.truncated,
        anomaly or None,
        types,
        body.harness,
        n_reasoning,
    )
    return response


@router.get("/turn/heartbeat", response_model=TurnHeartbeatResponse)
async def turn_heartbeat(
    user: Annotated[User, Depends(get_current_user)],
    hours: Annotated[int, Query(ge=1, le=720)] = 24,
) -> TurnHeartbeatResponse:
    """Recent turn-digest counts by (harness, mode, host) plus pending/gaps."""
    del user  # auth gate only
    since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat(timespec="seconds")
    pool = await get_db()
    records = await pool.fetch(_HEARTBEAT_SQL, since)
    rows = [
        TurnHeartbeatRow(
            harness=r["harness"],
            mode=r["mode"],
            host=r["host"],
            count=int(r["count"]),
            truncated=int(r["truncated"] or 0),
            redacted=int(r["redacted"] or 0),
            anomalies=int(r["anomalies"] or 0),
            empty_project=int(r["empty_project"] or 0),
            last_ts=r["last_ts"],
            hook_version=r["hook_version"],
        )
        for r in records
    ]
    pending = await pool.fetchrow(_PENDING_SQL)
    gaps = await pool.fetchrow(_GAPS_SQL, since)
    return TurnHeartbeatResponse(
        since=since,
        rows=rows,
        pending=int(pending["n"] or 0) if pending is not None else 0,
        oldest_pending_received_ts=pending["oldest"] if pending is not None else None,
        sessions_with_gaps=int(gaps["n"] or 0) if gaps is not None else 0,
        route_outcomes=dict(_OUTCOMES),
    )
