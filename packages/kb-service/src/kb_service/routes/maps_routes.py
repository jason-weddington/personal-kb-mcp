"""KB maps-index route: GET /api/kb/maps-index.

Computes the maps index on request from the live data DB.  No sidecar file,
no in-memory cache, and no LISTEN connection are required — the service reads
the DB directly so the index is always fresh by construction (kb-01744).
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from kb_service.auth import get_current_user
from kb_service.models import MapRef, MapsIndexResponse, ProjectMaps, User

router = APIRouter(prefix="/api/kb", tags=["kb"])


@router.get("/maps-index", response_model=MapsIndexResponse)
async def maps_index(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> MapsIndexResponse:
    """Return the full maps index, computed on request from the live DB.

    Queries ``knowledge_entries`` for all distinct project refs that have at
    least one active mental_map entry, then fetches the map list for each ref
    via ``KnowledgeBase.maps_for_project``.  Projects with zero maps are
    silently omitted (mirrors the legacy ``maps_index_writer`` 'drop the line'
    semantics).  Results are sorted ascending by ``project_ref`` for
    deterministic output.

    Exceptions propagate — FastAPI returns 500.  No per-project error handling
    is ported; the legacy best-effort log-and-skip existed only to protect a
    background file writer.
    """
    cursor = await request.app.state.kb.db.execute(
        "SELECT DISTINCT project_ref FROM knowledge_entries"
        " WHERE is_active = 1 AND entry_type = 'mental_map'"
        " AND project_ref IS NOT NULL"
    )
    rows = await cursor.fetchall()

    refs: list[str] = []
    for row in rows:
        ref = row[0]
        if isinstance(ref, str) and ref:
            refs.append(ref)
    project_refs = sorted(refs)

    projects: list[ProjectMaps] = []
    for ref in project_refs:
        maps = await request.app.state.kb.maps_for_project(ref)
        if not maps:
            continue
        projects.append(
            ProjectMaps(
                project_ref=ref,
                maps=[
                    MapRef(
                        id=m["id"],
                        short_title=m["short_title"],
                        long_title=m["long_title"],
                    )
                    for m in maps
                ],
            )
        )

    return MapsIndexResponse(projects=projects)
