"""Surprise capture: POST /api/kb/surprise/drain runs one detection pass.

Same auth as the other /api/kb routes (any authenticated user). The pass
holds the in-process drain lock that the background worker also holds, so the
two never run concurrently; the per-digest claim covers separate processes.
In mode off it answers 200 with zeros and touches nothing.
"""

import asyncio
import dataclasses
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from kb_service import surprise_worker
from kb_service.auth import get_current_user
from kb_service.database import get_db
from kb_service.models import SurpriseCandidateOut, SurpriseDrainResponse, User

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
