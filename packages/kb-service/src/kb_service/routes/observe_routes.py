"""Admin-only, read-only, keyset-paginated raw-signal feeds for the observatory.

``GET /api/kb/observe/{turn-events,failure-events,surprise-detections}``.
Nothing here writes or marks anything.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from kb_service.auth import require_admin
from kb_service.database import get_db
from kb_service.models import (
    ObserveFailureEventRow,
    ObserveFailureEventsResponse,
    ObserveSurpriseDetectionRow,
    ObserveSurpriseDetectionsResponse,
    ObserveTurnEventRow,
    ObserveTurnEventsResponse,
    User,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kb/observe", tags=["observe"])

MAX_LIMIT = 500
DEFAULT_LIMIT = 200


def _as_utc_iso(value: datetime | None) -> str | None:
    """*value* in the stored text form (UTC, seconds, ``+00:00``)."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds")


def _parse_json(raw: Any, default: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default
    return default if raw is None else raw


async def _page(
    *,
    table: str,
    ts_column: str,
    json_columns: dict[str, Any],
    admin: User,
    since: datetime | None,
    until: datetime | None,
    after_id: int,
    limit: int,
) -> tuple[list[dict[str, Any]], int | None]:
    """Fetch one page; *table* and *ts_column* are module constants, never input."""
    since_s = _as_utc_iso(since)
    until_s = _as_utc_iso(until)
    args: list[Any] = [after_id]
    sql = f"SELECT * FROM {table} WHERE id > $1"  # noqa: S608
    if since_s is not None:
        args.append(since_s)
        sql += f" AND {ts_column} >= ${len(args)}"
    if until_s is not None:
        args.append(until_s)
        sql += f" AND {ts_column} < ${len(args)}"
    args.append(limit)
    sql += f" ORDER BY id ASC LIMIT ${len(args)}"
    pool = await get_db()
    rows: list[dict[str, Any]] = []
    for record in await pool.fetch(sql, *args):
        row = dict(record)
        for column, default in json_columns.items():
            row[column] = _parse_json(row.get(column), default)
        rows.append(row)
    logger.info(
        "observe table=%s admin=%s since=%s until=%s after_id=%d limit=%d rows=%d",
        table,
        admin.id,
        since_s,
        until_s,
        after_id,
        limit,
        len(rows),
    )
    next_id = rows[-1]["id"] if len(rows) == limit else None
    return rows, next_id


_Since = Annotated[datetime | None, Query(description="ISO-8601, inclusive")]
_Until = Annotated[datetime | None, Query(description="ISO-8601, exclusive")]
_AfterId = Annotated[int, Query(ge=0)]
_Limit = Annotated[int, Query(ge=1, le=MAX_LIMIT)]


@router.get("/turn-events", response_model=ObserveTurnEventsResponse)
async def observe_turn_events(
    admin: Annotated[User, Depends(require_admin)],
    since: _Since = None,
    until: _Until = None,
    after_id: _AfterId = 0,
    limit: _Limit = DEFAULT_LIMIT,
) -> ObserveTurnEventsResponse:
    """Raw turn digests by ``received_ts``, ``id`` ascending."""
    rows, next_id = await _page(
        table="turn_events",
        ts_column="received_ts",
        json_columns={"items": [], "redactions": []},
        admin=admin,
        since=since,
        until=until,
        after_id=after_id,
        limit=limit,
    )
    return ObserveTurnEventsResponse(
        rows=[ObserveTurnEventRow(**r) for r in rows], next_after_id=next_id
    )


@router.get("/failure-events", response_model=ObserveFailureEventsResponse)
async def observe_failure_events(
    admin: Annotated[User, Depends(require_admin)],
    since: _Since = None,
    until: _Until = None,
    after_id: _AfterId = 0,
    limit: _Limit = DEFAULT_LIMIT,
) -> ObserveFailureEventsResponse:
    """Raw failure events by ``received_ts``, ``id`` ascending."""
    rows, next_id = await _page(
        table="failure_events",
        ts_column="received_ts",
        json_columns={},
        admin=admin,
        since=since,
        until=until,
        after_id=after_id,
        limit=limit,
    )
    return ObserveFailureEventsResponse(
        rows=[ObserveFailureEventRow(**r) for r in rows], next_after_id=next_id
    )


@router.get("/surprise-detections", response_model=ObserveSurpriseDetectionsResponse)
async def observe_surprise_detections(
    admin: Annotated[User, Depends(require_admin)],
    since: _Since = None,
    until: _Until = None,
    after_id: _AfterId = 0,
    limit: _Limit = DEFAULT_LIMIT,
) -> ObserveSurpriseDetectionsResponse:
    """Detector decisions (negatives too) by ``ts``, ``id`` ascending."""
    rows, next_id = await _page(
        table="surprise_detections",
        ts_column="ts",
        json_columns={"details": {}},
        admin=admin,
        since=since,
        until=until,
        after_id=after_id,
        limit=limit,
    )
    return ObserveSurpriseDetectionsResponse(
        rows=[ObserveSurpriseDetectionRow(**r) for r in rows], next_after_id=next_id
    )
