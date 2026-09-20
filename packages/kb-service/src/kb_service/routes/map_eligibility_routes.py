"""KB map-eligibility review + override routes (nightly map maintenance).

Operator-facing surface over ``KnowledgeBase.map_eligibility()`` and the
human override table: ``GET /api/kb/map-eligibility`` returns EVERY verdict
with its full evidence (a reviewing agent that cannot explain why a project
was excluded cannot decide whether the exclusion was right), and the two
POST endpoints set / clear a human override.  Admin-only — see
docs/nightly-map-maintenance-design.md.
"""

import logging
from dataclasses import asdict
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request

from kb_service.attribution import is_machine_principal, resolve_attribution
from kb_service.auth import require_admin
from kb_service.models import (
    MapEligibilityOverrideClearRequest,
    MapEligibilityOverrideResponse,
    MapEligibilityOverrideSetRequest,
    MapEligibilityResponse,
    MapEligibilityVerdictModel,
    User,
)

logger = logging.getLogger(__name__)

# Greppable marker — mirrors kb_core.map_eligibility.MAP_ELIGIBILITY_MARKER.
MAP_ELIGIBILITY_ROUTE_MARKER = "map-eligibility-override"

router = APIRouter(prefix="/api/kb", tags=["kb"])


@router.get("/map-eligibility", response_model=MapEligibilityResponse)
async def map_eligibility(
    request: Request,
    _admin: Annotated[User, Depends(require_admin)],
) -> MapEligibilityResponse:
    """Return EVERY map-eligibility verdict with its full evidence.

    No filtering, no rounding, no re-sorting: the endpoint returns eligible
    and ineligible projects alike, in the order kb-core returned
    (``resolve_eligibility`` sorts by ``project_ref`` ascending), with
    ``top_prefix_share`` as the engine's unrounded float.
    """
    verdicts = await request.app.state.kb.map_eligibility()
    return MapEligibilityResponse(
        projects=[MapEligibilityVerdictModel(**asdict(v)) for v in verdicts]
    )


@router.post("/map-eligibility/override", response_model=MapEligibilityOverrideResponse)
async def set_override(
    body: MapEligibilityOverrideSetRequest,
    request: Request,
    user: Annotated[User, Depends(require_admin)],
) -> MapEligibilityOverrideResponse:
    """Upsert the human override for ``body.project_ref``.

    Returns the post-write resolved verdict.  ``set_by`` comes from
    ``resolve_attribution(user).contributor`` — this route is the only
    place the human identity can enter, since the hosted
    ``app.state.kb`` is constructed with a bare ``Attribution()``.  An
    unrecognised ``project_ref`` is accepted (no 404): an override on a ref
    with zero mappable entries is a supported, deliberately-surfaced
    ``orphaned`` state.
    """
    kb = request.app.state.kb
    attr = await resolve_attribution(user)
    machine = await is_machine_principal(user)
    await kb.set_map_eligibility_override(
        body.project_ref,
        eligible=body.eligible,
        reason=body.reason,
        set_by=attr.contributor,
    )
    return await _resolved(
        kb,
        body.project_ref,
        op="set",
        changed=True,
        set_by=attr.contributor,
        machine_principal=machine,
        reason=body.reason,
    )


@router.post(
    "/map-eligibility/override/clear", response_model=MapEligibilityOverrideResponse
)
async def clear_override(
    body: MapEligibilityOverrideClearRequest,
    request: Request,
    user: Annotated[User, Depends(require_admin)],
) -> MapEligibilityOverrideResponse:
    """Delete the human override for ``body.project_ref``.

    ``changed`` is the engine's own return — ``False`` when no override row
    existed — so a no-op clear is a 200 with ``changed: false``, never a
    404.  ``set_by`` is ``None`` because a clear removes attribution rather
    than setting it.
    """
    kb = request.app.state.kb
    machine = await is_machine_principal(user)
    changed = await kb.clear_map_eligibility_override(body.project_ref)
    return await _resolved(
        kb,
        body.project_ref,
        op="clear",
        changed=bool(changed),
        set_by=None,
        machine_principal=machine,
        reason=None,
    )


async def _resolved(
    kb: Any,
    project_ref: str,
    *,
    op: Literal["set", "clear"],
    changed: bool,
    set_by: str | None,
    machine_principal: bool,
    reason: str | None,
) -> MapEligibilityOverrideResponse:
    """Re-resolve the post-write verdict for *project_ref* and log the write."""
    verdicts = await kb.map_eligibility()
    verdict = next((v for v in verdicts if v.evidence.project_ref == project_ref), None)
    if op == "set" and verdict is None:
        logger.warning(
            "%s: set for project_ref %r returned no verdict after write"
            " (invariant breach - write did not land or resolver union is broken)",
            MAP_ELIGIBILITY_ROUTE_MARKER,
            project_ref,
        )
    logger.info(
        "%s: op=%s project_ref=%r changed=%s set_by=%r machine_principal=%s"
        " effective_eligible=%s decided_by=%s orphaned=%s computed_eligible=%s"
        " mappable=%s hand_authored=%s top_prefix=%r top_prefix_share=%s reason=%r",
        MAP_ELIGIBILITY_ROUTE_MARKER,
        op,
        project_ref,
        changed,
        set_by,
        machine_principal,
        verdict.effective_eligible if verdict is not None else "absent",
        verdict.decided_by if verdict is not None else "absent",
        verdict.orphaned if verdict is not None else "absent",
        verdict.evidence.computed_eligible if verdict is not None else "absent",
        verdict.evidence.mappable if verdict is not None else "absent",
        verdict.evidence.hand_authored if verdict is not None else "absent",
        verdict.evidence.top_prefix if verdict is not None else "absent",
        verdict.evidence.top_prefix_share if verdict is not None else "absent",
        (reason or "")[:200],
    )
    return MapEligibilityOverrideResponse(
        changed=changed,
        verdict=MapEligibilityVerdictModel(**asdict(verdict))
        if verdict is not None
        else None,
    )
