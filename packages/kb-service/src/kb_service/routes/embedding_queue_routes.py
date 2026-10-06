"""KB embedding retry queue route: GET /api/kb/embedding-queue.

Operator-facing status endpoint for the self-healing embedding retry queue +
background worker (GTD 735a7e1d). Admin-only — the queue depth, oldest
pending age, and worker running state are operational signals, not something
every authenticated user needs to see.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from kb_service.auth import require_admin
from kb_service.config import is_embed_worker_enabled
from kb_service.models import EmbeddingQueueStatusResponse, User

router = APIRouter(prefix="/api/kb", tags=["kb"])


@router.get("/embedding-queue", response_model=EmbeddingQueueStatusResponse)
async def embedding_queue(
    request: Request,
    _admin: Annotated[User, Depends(require_admin)],
) -> EmbeddingQueueStatusResponse:
    """Return the embedding retry queue's operator-facing snapshot.

    Composes ``KnowledgeBase.embedding_queue_stats()`` (pending/exhausted
    counts, oldest pending age, next due time, and the vectorless_unqueued
    invariant tripwire) with ``worker_enabled`` (the ``KB_EMBED_WORKER_ENABLED``
    env adapter) and ``worker_running`` (the live worker task state).
    """
    stats = await request.app.state.kb.embedding_queue_stats()
    return EmbeddingQueueStatusResponse(
        **stats,
        worker_enabled=is_embed_worker_enabled(),
        worker_running=request.app.state.kb.embedding_worker_running,
    )
