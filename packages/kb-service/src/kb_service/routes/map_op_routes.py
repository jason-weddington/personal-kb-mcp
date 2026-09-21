"""POST /api/kb/map-op — the machine-principal map write path for somnus.

The nightly map-maintenance loop reaches the KB only over HTTP.

Until this endpoint existed it had no way to persist a map op at all.

Its run body shipped a loud ``UnwiredOpSink`` that exits 1, never a silent no-op.

The contract of record is docs/somnus-functional-spec.md, "The write path".

THE SERVER NEVER PARSES THE MAP GRAMMAR.

The request carries the whole composed body, because somnus's Rung-3 code renders it.

A server-side composer would have to parse an existing map body to re-render it.

That would contradict the lint's deliberate grammar-agnosticism.

It would also mangle the 27 hand-written maps that predate any grammar.

So every invariant here is a set comparison over the body's ``kb-`` refs.

The refs come from kb-core's own pointer regex (``kb_core.map_lint.map_pointer_ids``).

The gap ops use a plain substring test — grammar-agnostic in exactly the same way.

Why not reuse ``POST /api/kb/store``: nothing on that path enforces additive-only.

That invariant is non-negotiable precisely because a live Postgres offers no revert.

A body that silently drops half a map's pointers is a valid store there.

Additive-only was a prompt instruction until this endpoint made it machine-checked.

Machine principal ONLY: 403 for everyone else.

The invariants are correct for an unattended loop and hostile to a human editing a map.

Humans keep ``/store``, where the map lint stays advisory.

Status semantics, flat and terminal, matching the ledger and loop-input endpoints.

403 not the machine principal · 404 unknown ``map_id``.

409 invariant violation or stale ``base_version``.

422 lint findings, or an op outside the closed vocabulary.

200/201 success. Nothing here is retryable.

Writes go through the same kb-core seam as ``/store``.

``kb.get`` reads the stored map; ``kb.store`` persists ``create_map``.

``kb.update`` persists the other three, with ``change_reason`` naming the op.

``updated_by`` is set from the caller, exactly as ``/store``'s update path does.
"""

import logging
from typing import Annotated, Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from kb_core.map_lint import count_map_pointers, map_body_budget, map_pointer_ids
from kb_core.models.entry import EntryType, KnowledgeEntry

from kb_service.attribution import is_machine_principal, resolve_attribution
from kb_service.auth import get_current_user
from kb_service.models import (
    MapOpAddPointerRequest,
    MapOpCreateMapRequest,
    MapOpProposeGapRequest,
    MapOpRequest,
    MapOpResponse,
    MapOpStrikeGapRequest,
    User,
)
from kb_service.routes.map_write_guards import (
    _check_machine_principal_map_lint,
    _check_orphan_mental_map,
)

logger = logging.getLogger(__name__)

# Greppable log marker — mirrors MAP_LOOP_INPUT_MARKER / CLUSTER_LEDGER_ROUTE_MARKER.
MAP_OP_ROUTE_MARKER = "map-op-route"

router = APIRouter(prefix="/api/kb", tags=["kb"])


def _log(
    op: str,
    outcome: str,
    status_code: int,
    *,
    project_ref: str | None = None,
    map_id: str | None = None,
    detail: str | None = None,
) -> None:
    """Emit the one operator-readable trail line for a handled op."""
    logger.info(
        "%s op=%s project_ref=%r map_id=%r outcome=%s status=%d%s",
        MAP_OP_ROUTE_MARKER,
        op,
        project_ref,
        map_id,
        outcome,
        status_code,
        f" detail={detail!r}" if detail is not None else "",
    )


def _reject(
    op: str,
    outcome: str,
    status_code: int,
    detail: str,
    *,
    project_ref: str | None = None,
    map_id: str | None = None,
) -> NoReturn:
    """Log the rejection and raise it — one log line, one raise, per branch.

    Args:
        op: The op literal, for the trail line.
        outcome: A stable machine-readable token for the night's log.
        status_code: The HTTP status (403/404/409/422).
        detail: The response ``detail`` string.
        project_ref: The request's project_ref, when it has one.
        map_id: The request's map_id, when it has one.

    Raises:
        HTTPException: Always, with *status_code* and *detail*.
    """
    _log(
        op,
        outcome,
        status_code,
        project_ref=project_ref,
        map_id=map_id,
        detail=detail,
    )
    raise HTTPException(status_code=status_code, detail=detail)


def _envelope(body: str, entry: KnowledgeEntry) -> MapOpResponse:
    """Build the one uniform success envelope from the submitted body."""
    pointer_count = count_map_pointers(body)
    return MapOpResponse(
        map_id=entry.id,
        version=entry.version,
        pointer_count=pointer_count,
        budget=map_body_budget(pointer_count),
    )


def _sorted_ids(ids: set[str]) -> list[str]:
    """Render a ref set deterministically for a rejection detail."""
    return sorted(ids)


async def _load_stored_map(
    op: str,
    kb: Any,
    map_id: str,
    base_version: int | None,
) -> KnowledgeEntry:
    """Fetch the stored map, 404 on unknown ids and non-map entry types.

    A ``map_id`` resolving to a different ``entry_type`` is never silently updated.

    The entry_type check is part of the 404, not a separate error channel.

    The optional ``base_version`` guard is checked here too.

    Supplied and stale, it is a 409; omitted, no version check runs at all.

    That is the cheap guard for the one race the design deferred a lease over.

    Args:
        op: The op literal, for the trail line.
        kb: The kb-core facade from ``app.state.kb``.
        map_id: The request's ``map_id``.
        base_version: The request's optional ``base_version``.

    Returns:
        The stored ``KnowledgeEntry``, guaranteed to be a ``mental_map``.

    Raises:
        HTTPException: 404 unknown or non-``mental_map`` id.
        HTTPException: 409 stale ``base_version``.
    """
    stored: KnowledgeEntry | None = await kb.get(map_id)
    if stored is None or stored.entry_type is not EntryType.MENTAL_MAP:
        _reject(
            op,
            "not_found",
            404,
            f"map {map_id} not found or not a mental_map",
            project_ref=getattr(stored, "project_ref", None),
            map_id=map_id,
        )
    if base_version is not None and base_version != stored.version:
        _reject(
            op,
            "stale_base_version",
            409,
            f"base_version {base_version} != stored version {stored.version}"
            f" for map {map_id}",
            project_ref=stored.project_ref,
            map_id=map_id,
        )
    return stored


def _assert_pointer_sets_unchanged(
    op: str,
    stored_body: str,
    submitted_body: str,
    project_ref: str | None,
    map_id: str,
) -> None:
    """Assert a gap op moved no pointers: ``new == old``, else 409."""
    old = map_pointer_ids(stored_body)
    new = map_pointer_ids(submitted_body)
    if new != old:
        _reject(
            op,
            "pointer_set_changed",
            409,
            "gap op must not move pointers: added"
            f" {_sorted_ids(new - old)}, removed {_sorted_ids(old - new)}",
            project_ref=project_ref,
            map_id=map_id,
        )


async def _create_map(
    kb: Any,
    op_req: MapOpCreateMapRequest,
    user: User,
    contributor: str,
    team: str | None,
    response: Response,
) -> MapOpResponse:
    """Store a brand-new map: cardinal rule, lint, per-night caps, store.

    Args:
        kb: The kb-core facade from ``app.state.kb``.
        op_req: The validated ``create_map`` request.
        user: The authenticated machine principal.
        contributor: ``resolve_attribution(user).contributor`` — what the caps count.
        team: ``resolve_attribution(user).team``.
        response: The FastAPI response, whose status code becomes 201.

    Returns:
        ``MapOpResponse`` for the newly stored map.

    Raises:
        HTTPException: 422 cardinal rule or lint findings, 409 either cap.
    """
    op = "create_map"
    _check_orphan_mental_map(EntryType.MENTAL_MAP, op_req.body, None)
    await _check_machine_principal_map_lint(EntryType.MENTAL_MAP, op_req.body, user)
    # NO PER-NIGHT CREATION CAP, and its removal is a decision rather than an
    # omission (Jason, 2026-09-21). The cap existed solely to bound IRREVERSIBLE
    # damage — the design's words were that it "converts a clustering mistake
    # into a one-map event rather than a corpus-wide one" — and that reasoning
    # died when maps became deletable by the loop that wrote them. What remains
    # of the bound is the admission rules (minimum cluster size, native
    # majority, label shape), which reject bad clusters outright, and the
    # per-run token budget, which bounds spend.
    #
    # It was also actively counterproductive: rung 2 runs one inference per
    # cluster, so a capped run PAID to decide ops for every cluster and then
    # discarded all but one. A project with fourteen valid subject areas needs
    # fourteen maps, and taking fourteen nights to write them is fourteen nights
    # of agents orienting badly in it.
    #
    # `count_maps_created_since` stays in kb-core: nothing calls it for
    # enforcement now, but it is the natural primitive for the telemetry that
    # replaces the cap as the thing watching this loop.
    #
    # The kwargs mirror kb_write_routes.store's create path for a mental_map
    # carrying only the map-op fields.
    # Same entry_type, same confidence level, same attribution, same enrich default.
    # A map written here behaves identically to one written through /store.
    entry = await kb.store(
        short_title=op_req.short_title,
        long_title=op_req.long_title,
        knowledge_details=op_req.body,
        entry_type=EntryType.MENTAL_MAP,
        project_ref=op_req.project_ref,
        source_context=None,
        confidence_level=0.9,
        tags=None,
        hints=None,
        contributor=contributor,
        team=team,
        sensitivity=None,
        expires_at=None,
    )
    entry = await kb.get(entry.id) or entry
    _log(
        op,
        "created",
        201,
        project_ref=op_req.project_ref,
        map_id=entry.id,
    )
    response.status_code = 201
    return _envelope(op_req.body, entry)


async def _add_pointer(
    kb: Any,
    op_req: MapOpAddPointerRequest,
    user: User,
) -> MapOpResponse:
    """Append exactly one pointer, verified over the two ref sets.

    ``old`` and ``new`` are the ref sets of the stored and submitted bodies.

    ``new`` must be a superset of ``old``: additive-only, because there is no revert.

    ``new - old`` must be exactly ``{added_entry_id}``.

    The added entry must be an active entry in the map's own project.

    Raises:
        HTTPException: 404 unknown map or unknown added entry.
        HTTPException: 409 any invariant violation.
        HTTPException: 422 lint findings.
    """
    op = "add_pointer"
    stored = await _load_stored_map(
        kb=kb, op=op, map_id=op_req.map_id, base_version=op_req.base_version
    )
    await _check_machine_principal_map_lint(EntryType.MENTAL_MAP, op_req.body, user)
    old = map_pointer_ids(stored.knowledge_details)
    new = map_pointer_ids(op_req.body)
    removed = old - new
    if removed:
        _reject(
            op,
            "additive_only_violation",
            409,
            "additive-only invariant violated: pointers removed from the map"
            f" body: {', '.join(_sorted_ids(removed))}",
            project_ref=stored.project_ref,
            map_id=op_req.map_id,
        )
    added = new - old
    if not added:
        _reject(
            op,
            "no_pointer_added",
            409,
            "exactly-one-pointer invariant violated: no pointer added"
            " (the submitted body's kb- ref set equals the stored one)",
            project_ref=stored.project_ref,
            map_id=op_req.map_id,
        )
    if len(added) > 1:
        _reject(
            op,
            "more_than_one_pointer_added",
            409,
            "exactly-one-pointer invariant violated: more than one pointer added"
            f" ({len(added)}: {', '.join(_sorted_ids(added))}), expected exactly one",
            project_ref=stored.project_ref,
            map_id=op_req.map_id,
        )
    only_added = next(iter(added))
    if only_added != op_req.added_entry_id:
        _reject(
            op,
            "wrong_pointer_added",
            409,
            "exactly-one-pointer invariant violated: the submitted body adds"
            f" {only_added}, not the claimed added_entry_id"
            f" {op_req.added_entry_id}",
            project_ref=stored.project_ref,
            map_id=op_req.map_id,
        )
    target = await kb.get(op_req.added_entry_id)
    if target is None or not target.is_active:
        _reject(
            op,
            "added_entry_not_found",
            404,
            f"added_entry_id {op_req.added_entry_id} does not resolve to an"
            " active entry",
            project_ref=stored.project_ref,
            map_id=op_req.map_id,
        )
    if target.project_ref != stored.project_ref:
        _reject(
            op,
            "added_entry_in_different_project",
            409,
            f"added_entry_id {op_req.added_entry_id} belongs to project"
            f" {target.project_ref!r}, not the map's different project"
            f" {stored.project_ref!r}",
            project_ref=stored.project_ref,
            map_id=op_req.map_id,
        )
    entry = await kb.update(
        op_req.map_id,
        knowledge_details=op_req.body,
        change_reason=f"map-op add_pointer: added {op_req.added_entry_id}",
        updated_by=user.email,
    )
    entry = await kb.get(entry.id) or entry
    _log(
        op,
        "pointer_added",
        200,
        project_ref=stored.project_ref,
        map_id=op_req.map_id,
    )
    return _envelope(op_req.body, entry)


async def _strike_gap(
    kb: Any,
    op_req: MapOpStrikeGapRequest,
    user: User,
) -> MapOpResponse:
    """Strike one recorded gap, verified by plain substring and never by parse.

    Raises:
        HTTPException: 404 unknown map, 409 any invariant, 422 lint findings.
    """
    op = "strike_gap"
    stored = await _load_stored_map(
        kb=kb, op=op, map_id=op_req.map_id, base_version=op_req.base_version
    )
    await _check_machine_principal_map_lint(EntryType.MENTAL_MAP, op_req.body, user)
    _assert_pointer_sets_unchanged(
        op, stored.knowledge_details, op_req.body, stored.project_ref, op_req.map_id
    )
    old = map_pointer_ids(stored.knowledge_details)
    if op_req.closing_entry_id not in old:
        _reject(
            op,
            "closing_entry_not_pointed_at",
            409,
            f"closing_entry_id {op_req.closing_entry_id} is not pointed at by"
            f" the stored map {op_req.map_id}",
            project_ref=stored.project_ref,
            map_id=op_req.map_id,
        )
    gap = op_req.gap_text.strip()
    if gap not in stored.knowledge_details:
        _reject(
            op,
            "gap_absent_from_stored_body",
            409,
            "strike_gap gap_text is not present in the stored map body, so"
            " there is no gap to strike",
            project_ref=stored.project_ref,
            map_id=op_req.map_id,
        )
    if gap in op_req.body:
        _reject(
            op,
            "gap_still_present",
            409,
            "strike_gap gap_text is still present in the submitted map body",
            project_ref=stored.project_ref,
            map_id=op_req.map_id,
        )
    entry = await kb.update(
        op_req.map_id,
        knowledge_details=op_req.body,
        change_reason=(f"map-op strike_gap: gap closed by {op_req.closing_entry_id}"),
        updated_by=user.email,
    )
    entry = await kb.get(entry.id) or entry
    _log(
        op,
        "gap_struck",
        200,
        project_ref=stored.project_ref,
        map_id=op_req.map_id,
    )
    return _envelope(op_req.body, entry)


async def _propose_gap(
    kb: Any,
    op_req: MapOpProposeGapRequest,
    user: User,
) -> MapOpResponse:
    """Record one new gap, verified by plain substring and never by parse.

    Raises:
        HTTPException: 404 unknown map, 409 any invariant, 422 lint findings.
    """
    op = "propose_gap"
    stored = await _load_stored_map(
        kb=kb, op=op, map_id=op_req.map_id, base_version=op_req.base_version
    )
    await _check_machine_principal_map_lint(EntryType.MENTAL_MAP, op_req.body, user)
    _assert_pointer_sets_unchanged(
        op, stored.knowledge_details, op_req.body, stored.project_ref, op_req.map_id
    )
    gap = op_req.gap_text.strip()
    if gap in stored.knowledge_details:
        _reject(
            op,
            "gap_already_present",
            409,
            "propose_gap gap_text is already present in the stored map body",
            project_ref=stored.project_ref,
            map_id=op_req.map_id,
        )
    if gap not in op_req.body:
        _reject(
            op,
            "gap_absent_from_submitted_body",
            409,
            "propose_gap gap_text is absent from the submitted map body",
            project_ref=stored.project_ref,
            map_id=op_req.map_id,
        )
    entry = await kb.update(
        op_req.map_id,
        knowledge_details=op_req.body,
        change_reason="map-op propose_gap: gap recorded",
        updated_by=user.email,
    )
    entry = await kb.get(entry.id) or entry
    _log(
        op,
        "gap_proposed",
        200,
        project_ref=stored.project_ref,
        map_id=op_req.map_id,
    )
    return _envelope(op_req.body, entry)


@router.post("/map-op", response_model=MapOpResponse)
async def map_op(
    body: MapOpRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    response: Response,
) -> MapOpResponse:
    """Apply one map op from the closed vocabulary, machine principal only.

    Args:
        body: One of the four validated op requests, discriminated on ``op``.
        request: FastAPI request (provides ``app.state.kb``).
        user: The authenticated caller; must be the configured machine principal.
        response: The FastAPI response; 201 on ``create_map``, else 200.

    Returns:
        ``MapOpResponse`` — ``{map_id, version, pointer_count, budget}`` for every op.

    Raises:
        HTTPException: 403, 404, 409 or 422 per the module docstring.
    """
    kb = request.app.state.kb
    op = body.op
    if not await is_machine_principal(user):
        _log(op, "forbidden_not_machine_principal", 403, project_ref=None, map_id=None)
        raise HTTPException(
            status_code=403,
            detail=(
                "POST /api/kb/map-op is restricted to the configured machine"
                " principal; humans keep POST /api/kb/store"
            ),
        )
    attr = await resolve_attribution(user)
    # resolve_attribution always sets contributor to the caller's email; the
    # fallback below is a type guard only, never a value change.
    contributor = attr.contributor or user.email
    if isinstance(body, MapOpCreateMapRequest):
        return await _create_map(
            kb,
            body,
            user,
            contributor=contributor,
            team=attr.team,
            response=response,
        )
    if isinstance(body, MapOpAddPointerRequest):
        return await _add_pointer(kb, body, user)
    if isinstance(body, MapOpStrikeGapRequest):
        return await _strike_gap(kb, body, user)
    return await _propose_gap(kb, body, user)
