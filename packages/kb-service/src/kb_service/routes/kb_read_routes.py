"""KB API read/meta routes: get, graph exploration, preflight, and list endpoints.

Endpoints:
- POST /api/kb/get      — fetch entries by ID with pointer-rot analysis
- GET  /api/kb/graph/*  — structured-JSON graph surface (neighbors, bfs, path, etc.)
- GET  /api/kb/preflight — project context primer
- GET  /api/kb/projects|/contributors|/teams — aggregate list endpoints

NOTE: The MCP ``kb_explore`` tool launches a browser explorer UI — that behavior is
P3 of this service, NOT this item; these endpoints are the structured-JSON graph
surface over the ``kb.graph`` accessor.
"""

import re
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from kb_core.db.queries import touch_accessed
from kb_core.models.entry import EntryType
from kb_core.ttl import parse_ttl

if TYPE_CHECKING:
    from datetime import timedelta

from kb_service.auth import get_current_user
from kb_service.models import (
    GetEntryResult,
    GetRequest,
    GetResponse,
    GraphBfsEntry,
    GraphBfsResponse,
    GraphNeighbor,
    GraphNeighborsResponse,
    GraphPathHop,
    GraphPathResponse,
    GraphVocabularyResponse,
    KbListItem,
    KbListResponse,
    PointerRotTarget,
    PreflightResponse,
    ScopeEntriesResponse,
    SupersedesChainResponse,
    User,
)

router = APIRouter(prefix="/api/kb", tags=["kb"])

# Mirrors kb_core.graph.queries._KB_ID_RE (private — do not import the underscore name).
_KB_ID_RE = re.compile(r"^kb-\d{5}$")


@router.post("/get", response_model=GetResponse)
async def get_entries(
    body: GetRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> GetResponse:
    """Fetch knowledge entries by ID with optional pointer-rot analysis.

    Args:
        body: Request body containing the list of entry IDs to fetch.
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user (injected via dependency).

    Returns:
        GetResponse with one result slot per requested ID, in request order.
        Inactive entries are reported as not-found.  Pointer-rot is computed
        only for MENTAL_MAP entries.
    """
    kb = request.app.state.kb
    results: list[GetEntryResult] = []
    found_ids: list[str] = []

    for entry_id in body.ids:
        entry = await kb.get(entry_id)

        # None OR inactive → report as not found (kb.get does NOT filter is_active)
        if entry is None or not entry.is_active:
            results.append(
                GetEntryResult(id=entry_id, found=False, entry=None, pointer_rot=[])
            )
            continue

        found_ids.append(entry_id)

        # Pointer-rot analysis — MENTAL_MAP entries only
        pointer_rot: list[PointerRotTarget] = []
        if entry.entry_type == EntryType.MENTAL_MAP:
            neighbors = await kb.graph.neighbors(entry.id, direction="outgoing")
            seen: set[str] = set()
            kb_targets: list[str] = []
            for neighbor_id, _edge_type, _direction in neighbors:
                if neighbor_id not in seen and _KB_ID_RE.match(neighbor_id):
                    seen.add(neighbor_id)
                    kb_targets.append(neighbor_id)

            for target_id in kb_targets:
                t = await kb.get(target_id)
                if t is None:
                    continue  # skip missing targets; a deactivated one IS a rot signal
                if t.superseded_by is not None:
                    # superseded wins even when also inactive
                    pointer_rot.append(
                        PointerRotTarget(
                            target_id=target_id, superseded_by=t.superseded_by
                        )
                    )
                elif not t.is_active:
                    pointer_rot.append(
                        PointerRotTarget(target_id=target_id, superseded_by=None)
                    )

            pointer_rot.sort(key=lambda x: x.target_id)

        results.append(
            GetEntryResult(
                id=entry_id, found=True, entry=entry, pointer_rot=pointer_rot
            )
        )

    if found_ids:
        await touch_accessed(kb.db, found_ids)

    return GetResponse(results=results)


@router.get("/graph/neighbors", response_model=GraphNeighborsResponse)
async def graph_neighbors(
    node_id: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    edge_types: Annotated[list[str] | None, Query()] = None,
    direction: Literal["outgoing", "incoming", "both"] = "both",
    limit: int = Query(50, ge=1, le=200),
) -> GraphNeighborsResponse:
    """List neighbors of a graph node.

    Args:
        node_id: The node whose neighbors to list.
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user.
        edge_types: Optional filter on edge types (multi-value query param).
        direction: One of ``outgoing``, ``incoming``, or ``both`` (default).
            FastAPI validates against the Literal and returns 422 for any
            other value — the engine silently returns ``[]`` for unknown
            directions, which would be indistinguishable from an empty result.
        limit: Maximum neighbors to return (1-200, default 50).

    Returns:
        GraphNeighborsResponse with ``(neighbor_id, edge_type, direction)`` tuples.
    """
    kb = request.app.state.kb
    raw: list[tuple[str, str, str]] = await kb.graph.neighbors(
        node_id, edge_types=edge_types, direction=direction, limit=limit
    )
    neighbors = [
        GraphNeighbor(neighbor_id=nid, edge_type=et, direction=d) for nid, et, d in raw
    ]
    return GraphNeighborsResponse(neighbors=neighbors)


@router.get("/graph/bfs", response_model=GraphBfsResponse)
async def graph_bfs(
    start_node: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    max_depth: int = Query(2, ge=1, le=5),
    edge_types: Annotated[list[str] | None, Query()] = None,
    limit: int = Query(20, ge=1, le=100),
) -> GraphBfsResponse:
    """BFS traversal from a start node, collecting reachable entry nodes.

    Args:
        start_node: Node ID to start BFS from.
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user.
        max_depth: Maximum BFS depth (1-5, default 2).
        edge_types: Optional filter on edge types (multi-value query param).
        limit: Maximum entries to return (1-100, default 20).

    Returns:
        GraphBfsResponse with ``(entry_id, depth, path)`` tuples.
    """
    kb = request.app.state.kb
    raw: list[tuple[str, int, list[str]]] = await kb.graph.bfs_entries(
        start_node, max_depth=max_depth, edge_types=edge_types, limit=limit
    )
    entries = [GraphBfsEntry(entry_id=eid, depth=d, path=p) for eid, d, p in raw]
    return GraphBfsResponse(entries=entries)


@router.get("/graph/path", response_model=GraphPathResponse)
async def graph_path(
    source: str,
    target: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    max_depth: int = Query(4, ge=1, le=10),
) -> GraphPathResponse:
    """Find the shortest path between two graph nodes.

    Args:
        source: Source node ID.
        target: Target node ID.
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user.
        max_depth: Maximum BFS depth for path search (1-10, default 4).

    Returns:
        GraphPathResponse.  ``found=False, hops=[]`` when no path exists;
        ``found=True, hops=[]`` when source == target (zero-hop path).
    """
    kb = request.app.state.kb
    raw: list[tuple[str, str, str]] | None = await kb.graph.find_path(
        source, target, max_depth=max_depth
    )
    if raw is None:
        return GraphPathResponse(found=False, hops=[])
    # raw == [] means source == target (found, but no hops needed)
    hops = [GraphPathHop(source=s, edge_type=e, target=t) for s, e, t in raw]
    return GraphPathResponse(found=True, hops=hops)


@router.get("/graph/supersedes-chain", response_model=SupersedesChainResponse)
async def graph_supersedes_chain(
    entry_id: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> SupersedesChainResponse:
    """Return the full supersedes chain for an entry, oldest first.

    Args:
        entry_id: The entry ID whose supersedes chain to retrieve.
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user.

    Returns:
        SupersedesChainResponse with the chain list (oldest-first).
    """
    kb = request.app.state.kb
    chain: list[str] = await kb.graph.supersedes_chain(entry_id)
    return SupersedesChainResponse(chain=chain)


@router.get("/graph/scope-entries", response_model=ScopeEntriesResponse)
async def graph_scope_entries(
    scope: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    entry_type: str | None = None,
    order_by: str = "created_at",
) -> ScopeEntriesResponse:
    """Return entry IDs for a scope string.

    Args:
        scope: Scope string — ``project:X``, ``tag:Y``, ``person:X``,
            ``tool:X``, ``kb-NNNNN``, a bare entry_type, or a generic node ID.
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user.
        entry_type: Optional entry-type filter (passed through to the engine).
        order_by: Sort column.  The engine whitelist is ``created_at``,
            ``updated_at``, ``confidence_level``, ``short_title``; anything else
            silently falls back to ``created_at`` — the route does NOT validate
            this field, it passes through verbatim.

    Returns:
        ScopeEntriesResponse with matching entry IDs.
    """
    kb = request.app.state.kb
    entry_ids: list[str] = await kb.graph.entries_for_scope(
        scope, entry_type=entry_type, order_by=order_by
    )
    return ScopeEntriesResponse(entry_ids=entry_ids)


@router.get("/graph/vocabulary", response_model=GraphVocabularyResponse)
async def graph_vocabulary(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    max_nodes: int = Query(200, ge=1, le=1000),
) -> GraphVocabularyResponse:
    """Return non-entry graph vocabulary nodes grouped by type.

    Args:
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user.
        max_nodes: Maximum nodes to return (1-1000, default 200).

    Returns:
        GraphVocabularyResponse with a ``{type: [node_id, ...]}`` mapping.
    """
    kb = request.app.state.kb
    nodes: dict[str, list[str]] = await kb.graph.vocabulary(max_nodes=max_nodes)
    return GraphVocabularyResponse(nodes=nodes)


@router.get("/preflight", response_model=PreflightResponse)
async def preflight(
    project_ref: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    since: str | None = None,
) -> PreflightResponse:
    """Return a project context primer (preflight string).

    Args:
        project_ref: Project reference to query (e.g. ``personal-kb``).
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user.
        since: Optional time window in TTL format (``Nh``/``Nd``/``Nw``,
            e.g. ``7d``, ``24h``).  Raises 422 on bad format or zero duration.

    Returns:
        PreflightResponse with the project_ref echoed and the context string.
    """
    kb = request.app.state.kb
    since_td: timedelta | None = None
    if since is not None:
        try:
            since_td = parse_ttl(since)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    context: str = await kb.preflight(project_ref, since=since_td)
    return PreflightResponse(project_ref=project_ref, context=context)


@router.get("/projects", response_model=KbListResponse)
async def list_projects(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> KbListResponse:
    """List active projects with their entry counts.

    Args:
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user.

    Returns:
        KbListResponse ordered by entry count descending.  Empty table returns
        ``{items: []}`` with HTTP 200.
    """
    kb = request.app.state.kb
    cursor = await kb.db.execute(
        "SELECT project_ref, COUNT(*) as cnt FROM knowledge_entries"
        " WHERE is_active = 1 AND project_ref IS NOT NULL"
        " GROUP BY project_ref ORDER BY cnt DESC"
    )
    rows = await cursor.fetchall()
    return KbListResponse(
        items=[KbListItem(name=row[0], entry_count=row[1]) for row in rows]
    )


@router.get("/contributors", response_model=KbListResponse)
async def list_contributors(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> KbListResponse:
    """List active contributors with their entry counts.

    Args:
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user.

    Returns:
        KbListResponse ordered by entry count descending.  Empty table returns
        ``{items: []}`` with HTTP 200.
    """
    kb = request.app.state.kb
    cursor = await kb.db.execute(
        "SELECT contributor, COUNT(*) as cnt FROM knowledge_entries"
        " WHERE is_active = 1 AND contributor IS NOT NULL"
        " GROUP BY contributor ORDER BY cnt DESC"
    )
    rows = await cursor.fetchall()
    return KbListResponse(
        items=[KbListItem(name=row[0], entry_count=row[1]) for row in rows]
    )


@router.get("/teams", response_model=KbListResponse)
async def list_teams(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> KbListResponse:
    """List active teams with their entry counts.

    Args:
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user.

    Returns:
        KbListResponse ordered by entry count descending.  Empty table returns
        ``{items: []}`` with HTTP 200.
    """
    kb = request.app.state.kb
    cursor = await kb.db.execute(
        "SELECT team, COUNT(*) as cnt FROM knowledge_entries"
        " WHERE is_active = 1 AND team IS NOT NULL"
        " GROUP BY team ORDER BY cnt DESC"
    )
    rows = await cursor.fetchall()
    return KbListResponse(
        items=[KbListItem(name=row[0], entry_count=row[1]) for row in rows]
    )
