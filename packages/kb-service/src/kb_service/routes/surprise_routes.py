"""Surprise capture routes: run a detection pass, audit the candidates.

POST /api/kb/surprise/drain runs one detection pass. It holds the in-process
drain lock that the background worker also holds, so the two never run
concurrently; the per-digest claim covers separate processes. In mode off it
answers 200 with zeros and touches nothing.

GET /api/kb/surprise/candidates reads ``surprise_candidates`` with each
candidate's shadow dry run (``surprise_dry_runs``) and the mode/host of its
last turn event, so an agent can audit what capture WOULD write before it is
switched on. It never returns raw ``turn_events`` items.

Same auth as the other /api/kb routes (any authenticated user).
"""

import asyncio
import dataclasses
import json
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request

from kb_service import surprise_worker
from kb_service.auth import get_current_user
from kb_service.database import get_db
from kb_service.models import (
    SurpriseCandidateAudit,
    SurpriseCandidateOut,
    SurpriseCandidatesResponse,
    SurpriseCandidateStatus,
    SurpriseDrainResponse,
    SurpriseDryRunOut,
    User,
)

CANDIDATES_DEFAULT_LIMIT = 200
CANDIDATES_MAX_LIMIT = 1000
CANDIDATES_DEFAULT_SINCE = timedelta(hours=24)

_CANDIDATES_SQL = (
    "SELECT c.id, c.shape, c.status, c.session_id, c.project, c.created_at,"
    " c.detector_model, c.detector_output, c.turn_event_ids, c.entry_id,"
    " r.id AS dry_run_id, r.would_outcome, r.reason, r.payload,"
    " r.distiller_model, r.distiller_version"
    " FROM surprise_candidates c"
    " LEFT JOIN surprise_dry_runs r ON r.candidate_id = c.id"
    " WHERE c.created_at >= $1"
)

router = APIRouter(prefix="/api/kb", tags=["kb"])


@router.post("/surprise/drain", response_model=SurpriseDrainResponse)
async def drain_surprise(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> SurpriseDrainResponse:
    """Process every pending turn digest now and return what was found."""
    del user  # auth gate only
    lock = getattr(request.app.state, "surprise_drain_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.surprise_drain_lock = lock
    async with lock:
        result = await surprise_worker.drain_once(await get_db(), request.app.state.kb)
    return SurpriseDrainResponse(
        digests_processed=result.digests_processed,
        candidates=[
            SurpriseCandidateOut(**dataclasses.asdict(c)) for c in result.candidates
        ],
        entries_written=result.entries_written,
        entries_merged=result.entries_merged,
    )


def _as_utc_iso(value: datetime) -> str:
    """*value* as the ``created_at`` text form (UTC, seconds, ``+00:00``)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds")


def _json_value(raw: Any, default: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default
    return default if raw is None else raw


@router.get("/surprise/candidates", response_model=SurpriseCandidatesResponse)
async def list_surprise_candidates(
    user: Annotated[User, Depends(get_current_user)],
    since: Annotated[
        datetime | None, Query(description="ISO timestamp; default 24h ago")
    ] = None,
    status: SurpriseCandidateStatus | None = None,
    shape: Annotated[int | None, Query(ge=1, le=3)] = None,
    project: str | None = None,
    limit: Annotated[
        int, Query(ge=1, le=CANDIDATES_MAX_LIMIT)
    ] = CANDIDATES_DEFAULT_LIMIT,
) -> SurpriseCandidatesResponse:
    """List surprise candidates newest first, each with its dry run (or null)."""
    del user  # auth gate only
    floor = since if since is not None else datetime.now(UTC) - CANDIDATES_DEFAULT_SINCE
    sql = _CANDIDATES_SQL
    args: list[Any] = [_as_utc_iso(floor)]
    for column, value in (
        ("c.status", status),
        ("c.shape", shape),
        ("c.project", project),
    ):
        if value is not None:
            args.append(value)
            sql += f" AND {column} = ${len(args)}"
    args.append(limit)
    sql += f" ORDER BY c.created_at DESC, c.id DESC LIMIT ${len(args)}"
    pool = await get_db()
    rows = await pool.fetch(sql, *args)

    parsed: list[tuple[dict[str, Any], list[str]]] = []
    last_events: set[str] = set()
    for row in rows:
        ids = _json_value(row["turn_event_ids"], [])
        event_ids = [str(e) for e in ids] if isinstance(ids, list) else []
        if event_ids:
            last_events.add(event_ids[-1])
        parsed.append((row, event_ids))

    turns: dict[str, dict[str, Any]] = {}
    if last_events:
        wanted = sorted(last_events)
        placeholders = ", ".join(f"${i}" for i in range(1, len(wanted) + 1))
        for t in await pool.fetch(
            "SELECT event_id, mode, host FROM turn_events"  # noqa: S608
            f" WHERE event_id IN ({placeholders})",
            *wanted,
        ):
            turns[str(t["event_id"])] = t

    out: list[SurpriseCandidateAudit] = []
    for row, event_ids in parsed:
        turn = turns.get(event_ids[-1]) if event_ids else None
        dry_run = None
        if row["dry_run_id"] is not None:
            payload = _json_value(row["payload"], {})
            dry_run = SurpriseDryRunOut(
                would_outcome=row["would_outcome"],
                reason=row["reason"],
                payload=payload if isinstance(payload, dict) else {},
                distiller_model=row["distiller_model"],
                distiller_version=int(row["distiller_version"]),
            )
        output = _json_value(row["detector_output"], {})
        out.append(
            SurpriseCandidateAudit(
                id=int(row["id"]),
                shape=row["shape"],
                status=row["status"],
                session_id=row["session_id"],
                project=row["project"],
                created_at=row["created_at"],
                detector_model=row["detector_model"],
                detector_output=output if isinstance(output, dict) else {},
                turn_event_ids=event_ids,
                mode=turn["mode"] if turn is not None else None,
                host=turn["host"] if turn is not None else None,
                dry_run=dry_run,
                entry_id=row["entry_id"],
            )
        )
    return SurpriseCandidatesResponse(candidates=out)
