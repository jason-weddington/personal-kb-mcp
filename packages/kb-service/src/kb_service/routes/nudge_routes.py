"""Pointer-candidates route: POST /api/kb/pointer-candidates.

Server half of the capture-time map nudge (Phase 1 of
docs/nightly-map-maintenance-design.md). Unpointed entries are both a FLOW
and a STOCK: a nightly sweep handles the stock, but the flow is best
intercepted at capture time, when the agent that just wrote an entry still
holds the context needed to write a good pointer gloss. This endpoint is the
read-only "is this entry pointed at, and if not, what maps might want to
point at it" check the MCP client calls right after a ``kb_store`` — a
*separate* endpoint, not something wedged into the store request path (see
the item description for why: no ANN index exists, so a kNN belongs outside
the store request's event-loop turn).

This route is deliberately read-only: no telemetry rows, no graph mutations,
no entry updates. It must be safe to call on every store.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from kb_service.auth import get_current_user
from kb_service.models import (
    PointerCandidate,
    PointerCandidatesRequest,
    PointerCandidatesResponse,
    User,
)

router = APIRouter(prefix="/api/kb", tags=["kb"])

# How many candidate maps to surface. Pinned rather than client-configurable —
# this is a capture-time nudge, not a search result page.
MAX_POINTER_CANDIDATES = 5


async def _has_owning_map(db: Any, entry_id: str) -> bool:
    """True iff an ACTIVE mental_map entry has a ``references`` edge -> entry_id.

    One graph_edges anti-join, per the acceptance criteria: ``source`` is the
    map, ``target`` is the entry being pointed at.
    """
    cursor = await db.execute(
        "SELECT 1 FROM graph_edges ge"
        " JOIN knowledge_entries e ON e.id = ge.source"
        " WHERE ge.target = ? AND ge.edge_type = 'references'"
        " AND e.entry_type = 'mental_map' AND e.is_active = 1"
        " LIMIT 1",
        (entry_id,),
    )
    rows = await cursor.fetchall()
    return len(rows) > 0


async def _fetch_own_embedding(db: Any, entry_id: str) -> list[float] | None:
    """Return the entry's own stored embedding vector, or None if absent.

    ``kb.db.vector_search`` takes an embedding to search *from* — it has no
    "search from this entry_id" mode — so the entry's own vector has to be
    read out of ``knowledge_vec`` first. Cast to text so this works
    regardless of whether the pgvector client-side codec is registered; the
    text form is the same bracketed-list format ``vector_search`` itself
    builds when it sends a query vector.
    """
    cursor = await db.execute(
        "SELECT embedding::text AS embedding FROM knowledge_vec WHERE entry_id = ?",
        (entry_id,),
    )
    rows = await cursor.fetchall()
    if not rows:
        return None
    raw = rows[0][0]
    if not raw:
        return None
    return [float(v) for v in raw.strip("[]").split(",") if v]


@router.post("/pointer-candidates", response_model=PointerCandidatesResponse)
async def pointer_candidates(
    body: PointerCandidatesRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> PointerCandidatesResponse:
    """Report whether an entry already has an owning map, and if not, candidates.

    Args:
        body: Request body carrying the freshly-stored ``entry_id``.
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user (JWT or API key via Bearer header) — same
            dependency as the other ``/api/kb/*`` routes; this is an ordinary
            agent call, not an admin operation.

    Returns:
        ``PointerCandidatesResponse``. Every degenerate case (no embedding
        yet — the normal state right after a store, before the retry queue
        or worker has embedded it; no ``project_ref``; the project has no
        other maps) returns ``has_owning_map=False, candidates=[]`` rather
        than raising. A genuinely-missing ``entry_id`` is the one case that
        404s.
    """
    kb = request.app.state.kb
    entry = await kb.get(body.entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="entry not found")

    if await _has_owning_map(kb.db, body.entry_id):
        # Owning map already exists — the client stays silent either way, so
        # there's no reason to pay for a kNN here.
        return PointerCandidatesResponse(has_owning_map=True, candidates=[])

    if not entry.has_embedding or not entry.project_ref:
        return PointerCandidatesResponse(has_owning_map=False, candidates=[])

    embedding = await _fetch_own_embedding(kb.db, body.entry_id)
    if embedding is None:
        # has_embedding said yes but the vector row is gone/unreadable —
        # graceful empty, not a 500.
        return PointerCandidatesResponse(has_owning_map=False, candidates=[])

    hits = await kb.db.vector_search(
        embedding,
        limit=MAX_POINTER_CANDIDATES,
        project_ref=entry.project_ref,
        entry_type="mental_map",
    )

    candidates: list[PointerCandidate] = []
    for map_id, distance in hits:
        if map_id == body.entry_id:
            continue
        map_entry = await kb.get(map_id)
        if map_entry is None:
            continue
        candidates.append(
            PointerCandidate(
                map_id=map_id,
                short_title=map_entry.short_title,
                project_ref=map_entry.project_ref,
                distance=distance,
            )
        )

    return PointerCandidatesResponse(has_owning_map=False, candidates=candidates)
