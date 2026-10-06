"""KB API routes: the authed read surface over the kb-core engine.

P1 ships ``POST /api/kb/search``.  P2 adds sibling per-feature routers under
``routes/`` (e.g. ``query_routes``) rather than extending this module.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from kb_core.models.search import SearchQuery

from kb_service.auth import _auth_mode, get_current_user
from kb_service.models import (
    RuntimeResponse,
    SearchRequest,
    SearchResponse,
    User,
)

router = APIRouter(prefix="/api/kb", tags=["kb"])


@router.get("/runtime", response_model=RuntimeResponse)
async def runtime() -> RuntimeResponse:
    """Report the active runtime auth mode (unauthenticated).

    The SPA calls this before any user is known, to decide whether to render
    the login flow. The response shape is the frozen cross-item contract
    ``{"auth": "none"|"jwt"}``.
    """
    return RuntimeResponse(auth=_auth_mode())


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> SearchResponse:
    """Hybrid search over the knowledge base (authenticated).

    ``contributor`` is passed only as a telemetry kwarg — distinct from the
    ``SearchQuery.contributor`` FILTER field, which is left unset because
    physical DB isolation already scopes the data.
    """
    search_query = SearchQuery(
        query=body.query,
        project_ref=body.project_ref,
        entry_type=body.entry_type,  # type: ignore[arg-type]
        tags=body.tags,
        limit=body.limit,
        include_stale=body.include_stale,
        include_expired=body.include_expired,
    )
    results, filtered_count = await request.app.state.kb.search(
        search_query, contributor=user.email
    )
    return SearchResponse(results=results, filtered_count=filtered_count)
