"""Experience-loop metrics: ``GET /api/kb/metrics/repeat-rate``.

Catches no exception on purpose: a DB failure is an HTTP 500, never an
all-zero 200 that reads like a quiet week.
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from kb_service.auth import get_current_user
from kb_service.database import get_db
from kb_service.models import RepeatRateResponse, User
from kb_service.repeat_rate import (
    DEFAULT_MIN_GAP_HOURS,
    DEFAULT_WEEKS,
    MAX_MIN_GAP_HOURS,
    MAX_WEEKS,
    build_repeat_rate,
)

router = APIRouter(prefix="/api/kb", tags=["kb"])


def _utcnow() -> datetime:
    """Current UTC time (test seam)."""
    return datetime.now(UTC)


@router.get("/metrics/repeat-rate", response_model=RepeatRateResponse)
async def get_repeat_rate(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    weeks: Annotated[int, Query(ge=1, le=MAX_WEEKS)] = DEFAULT_WEEKS,
    project: str | None = None,
    min_gap_hours: Annotated[
        float, Query(ge=0, le=MAX_MIN_GAP_HOURS)
    ] = DEFAULT_MIN_GAP_HOURS,
) -> RepeatRateResponse:
    """Weekly cross-session repeat-mistake rate from the failure-cue index."""
    del user
    return await build_repeat_rate(
        await get_db(),
        request.app.state.kb.db,
        weeks=weeks,
        project=project,
        min_gap_hours=min_gap_hours,
        now=_utcnow(),
    )
