"""Home-page browse routes: read-only lists for inspecting the KB.

Endpoints (all GET, all user-authenticated):
- /api/kb/browse/recent   — newest or most recently updated entries
- /api/kb/browse/top      — most connected (graph degree) or recently read entries
- /api/kb/browse/stale    — decayed or expired entries
- /api/kb/browse/projects — project summaries
- /api/kb/browse/project/{project_ref:path} — one project's entries

The queries live in ``kb_core.browse``; every route goes through ``_browse``,
which maps ``BrowseParamError`` to 422 and ``ProjectNotFound`` to 404,
re-raises anything else, and logs exactly one ``browse-route`` line.
"""

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from kb_core.browse import (
    BrowsePage,
    BrowseParamError,
    ProjectNotFound,
    ProjectPage,
    browse_project,
    browse_projects,
    browse_recent,
    browse_stale,
    browse_top,
)
from kb_core.models.entry import EntryType

from kb_service.auth import get_current_user
from kb_service.models import (
    BrowseEntriesResponse,
    BrowseEntryItem,
    BrowseProjectItem,
    BrowseProjectsResponse,
    User,
)

router = APIRouter(prefix="/api/kb/browse", tags=["kb"])
logger = logging.getLogger(__name__)

BROWSE_ROUTE_MARKER = "browse-route"


async def _browse(
    view: str,
    mode: str,
    user: User,
    call: Callable[[], Awaitable[BrowsePage | ProjectPage]],
    *,
    entry_type: str | None,
    project_ref: str | None,
    q: str | None,
    limit: int,
    cursor: str | None,
) -> BrowsePage | ProjectPage:
    """Run one browse call, map its errors and log one browse-route line.

    Args:
        view: recent, top, stale, projects or project.
        mode: The sort, signal or kind value, or '-' for projects.
        user: The authenticated user.
        call: Zero-argument closure performing the kb-core call.
        entry_type: The entry-type filter, if any.
        project_ref: The project, for the project view.
        q: The text filter, if any.
        limit: Page size.
        cursor: Offset cursor.

    Returns:
        The kb-core page.

    Raises:
        HTTPException: 422 for a bad cursor or limit, 404 for an unknown project.
    """
    start = time.perf_counter()
    try:
        page = await call()
    except BrowseParamError as exc:
        logger.info(
            "%s view=%s mode=%s outcome=bad_param detail=%r duration_ms=%d user_id=%s",
            BROWSE_ROUTE_MARKER,
            view,
            mode,
            str(exc),
            int((time.perf_counter() - start) * 1000),
            user.id,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ProjectNotFound as exc:
        logger.info(
            "%s view=%s mode=%s outcome=not_found detail=%r duration_ms=%d user_id=%s",
            BROWSE_ROUTE_MARKER,
            view,
            mode,
            str(exc.args[0]) if exc.args else "",
            int((time.perf_counter() - start) * 1000),
            user.id,
        )
        missing = exc.args[0] if exc.args else ""
        raise HTTPException(
            status_code=404, detail=f"Unknown project: {missing}"
        ) from exc
    except Exception as exc:
        logger.warning(
            "%s view=%s mode=%s outcome=error detail=%r duration_ms=%d user_id=%s",
            BROWSE_ROUTE_MARKER,
            view,
            mode,
            type(exc).__name__,
            int((time.perf_counter() - start) * 1000),
            user.id,
        )
        raise
    logger.info(
        "%s view=%s mode=%s entry_type=%s project_ref=%r q_len=%d limit=%d"
        " cursor=%r outcome=ok scanned=%d matched=%d returned=%d skipped=%d"
        " duration_ms=%d user_id=%s",
        BROWSE_ROUTE_MARKER,
        view,
        mode,
        entry_type or "-",
        project_ref,
        len(q.strip()) if q else 0,
        limit,
        cursor,
        page.scanned,
        page.total,
        len(page.items),
        page.skipped,
        int((time.perf_counter() - start) * 1000),
        user.id,
    )
    return page


def _entries_response(
    result: BrowsePage | ProjectPage, signal: Literal["degree", "read"] | None = None
) -> BrowseEntriesResponse:
    page = cast("BrowsePage", result)
    return BrowseEntriesResponse(
        items=[
            BrowseEntryItem(
                id=r.id,
                short_title=r.short_title,
                long_title=r.long_title,
                entry_type=r.entry_type,
                project_ref=r.project_ref,
                created_at=r.created_at.isoformat(),
                updated_at=r.updated_at.isoformat(),
                last_accessed=r.last_accessed.isoformat() if r.last_accessed else None,
                expires_at=r.expires_at.isoformat() if r.expires_at else None,
                confidence_level=r.confidence_level,
                effective_confidence=r.effective_confidence,
                half_life_days=r.half_life_days,
                anchor_age_days=r.anchor_age_days,
                expired=r.expired,
                signal_value=r.signal_value,
            )
            for r in page.items
        ],
        next_cursor=page.next_cursor,
        total=page.total,
        skipped=page.skipped,
        signal=signal,
    )


def _type_value(entry_type: EntryType | None) -> str | None:
    return entry_type.value if entry_type is not None else None


Limit = Annotated[int, Query(ge=1, le=100)]


@router.get("/recent", response_model=BrowseEntriesResponse)
async def get_recent(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    sort: Literal["created", "updated"] = "created",
    entry_type: EntryType | None = None,
    limit: Limit = 25,
    cursor: str | None = None,
) -> BrowseEntriesResponse:
    """List the newest entries, or entries updated after creation."""
    kb = request.app.state.kb
    etype = _type_value(entry_type)
    page = await _browse(
        "recent",
        sort,
        user,
        lambda: browse_recent(
            kb.db, sort=sort, entry_type=etype, limit=limit, cursor=cursor
        ),
        entry_type=etype,
        project_ref=None,
        q=None,
        limit=limit,
        cursor=cursor,
    )
    return _entries_response(page)


@router.get("/top", response_model=BrowseEntriesResponse)
async def get_top(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    signal: Literal["degree", "read"] = "degree",
    entry_type: EntryType | None = None,
    limit: Limit = 25,
    cursor: str | None = None,
) -> BrowseEntriesResponse:
    """List the most connected or most recently read entries."""
    kb = request.app.state.kb
    etype = _type_value(entry_type)
    page = await _browse(
        "top",
        signal,
        user,
        lambda: browse_top(
            kb.db, signal=signal, entry_type=etype, limit=limit, cursor=cursor
        ),
        entry_type=etype,
        project_ref=None,
        q=None,
        limit=limit,
        cursor=cursor,
    )
    return _entries_response(page, signal)


@router.get("/stale", response_model=BrowseEntriesResponse)
async def get_stale(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    kind: Literal["decayed", "expired"] = "decayed",
    entry_type: EntryType | None = None,
    limit: Limit = 25,
    cursor: str | None = None,
) -> BrowseEntriesResponse:
    """List decayed or expired entries."""
    kb = request.app.state.kb
    etype = _type_value(entry_type)
    page = await _browse(
        "stale",
        kind,
        user,
        lambda: browse_stale(
            kb.db, kind=kind, entry_type=etype, limit=limit, cursor=cursor
        ),
        entry_type=etype,
        project_ref=None,
        q=None,
        limit=limit,
        cursor=cursor,
    )
    return _entries_response(page)


@router.get("/projects", response_model=BrowseProjectsResponse)
async def get_projects(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    q: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 24,
    cursor: str | None = None,
) -> BrowseProjectsResponse:
    """List projects with entry counts, most recently active first."""
    kb = request.app.state.kb
    result = await _browse(
        "projects",
        "-",
        user,
        lambda: browse_projects(kb.db, q=q, limit=limit, cursor=cursor),
        entry_type=None,
        project_ref=None,
        q=q,
        limit=limit,
        cursor=cursor,
    )
    page = cast("ProjectPage", result)
    return BrowseProjectsResponse(
        items=[
            BrowseProjectItem(
                project_ref=p.project_ref,
                entry_count=p.entry_count,
                counts_by_type=p.counts_by_type,
                last_activity=p.last_activity.isoformat(),
            )
            for p in page.items
        ],
        next_cursor=page.next_cursor,
        total=page.total,
        skipped=page.skipped,
    )


@router.get("/project/{project_ref:path}", response_model=BrowseEntriesResponse)
async def get_project(
    project_ref: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    sort: Literal["newest", "updated", "confidence"] = "newest",
    entry_type: EntryType | None = None,
    q: str | None = None,
    limit: Limit = 25,
    cursor: str | None = None,
) -> BrowseEntriesResponse:
    """List one project's entries, mental maps first."""
    kb = request.app.state.kb
    etype = _type_value(entry_type)
    page = await _browse(
        "project",
        sort,
        user,
        lambda: browse_project(
            kb.db,
            project_ref,
            sort=sort,
            entry_type=etype,
            q=q,
            limit=limit,
            cursor=cursor,
        ),
        entry_type=etype,
        project_ref=project_ref,
        q=q,
        limit=limit,
        cursor=cursor,
    )
    return _entries_response(page)
