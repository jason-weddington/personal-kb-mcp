"""Cluster/decline ledger HTTP routes — the loop's read AND the human's write.

Two endpoints over the ``KnowledgeBase`` facade's cluster-ledger surface
(kb-core 246891c), which is the ONLY matching engine:

* ``POST /api/kb/cluster-ledger/match`` — a BATCH of candidate member-id
  sets matched against one project's ``map_cluster_ledger`` rows. All
  Jaccard arithmetic, the ``CLUSTER_MATCH_JACCARD`` threshold and the
  member-set-doubling reopen rule live in ``kb_core.cluster_ledger`` and
  are deliberately NOT reimplemented here: 0.5 is documented in kb-core as
  INVENTED rather than measured and will be retuned from near-miss data,
  and a second copy of the constant in this repo would drift silently —
  the exact stale-paraphrase-standing-in-for-a-primitive failure this
  pair has hit twice. somnus receives the VERDICT (matched, ledger_id,
  status, jaccard, reopen_eligible), never the inputs to recompute it.
* ``POST /api/kb/cluster-ledger/decline`` — the human's verdict, keyed on
  the member-id set (never the ``cluster_key``: identity is member-set
  overlap and the key is a server-side birth hash somnus cannot and
  should not mint). The route resolves the member set to a row via
  ``kb.record_cluster`` — kb-core's own member-set→row resolver, with its
  declined-preferred ``_best_match`` — and then declines that row by key.
  This is what makes the design's re-arm work: a decline submitted at the
  DOUBLED member size first updates the matched row's stored member set,
  so ``decline_cluster`` re-arms ``declined_member_count`` at the new
  size, which is precisely the one escape from a permanent decline. The
  cost is a ``sightings`` bump and a ``map_cluster_ledger_recorded`` audit
  row on every decline — accepted, because without a record call a
  first-time decline (no prior sighting: this repo deliberately ships NO
  create_map/record endpoint; materialised clusters are visible through
  ``maps[].pointers``) could not create the row the decline needs. The
  decline reason travels as the ``label`` so a declined row's
  ``last_label`` is never empty. ``declined_by`` is the caller's
  contributor identity via ``resolve_attribution`` — the facade passes it
  through verbatim.

There is deliberately NO third endpoint recording a successful
``create_map``: a materialised cluster is already visible to somnus
through ``maps[].pointers`` on the loop-input response, and this table
holds ONLY what grows monotonically and cannot be recomputed.

Auth is ``get_current_user``, NOT ``require_admin`` — for the reason
recorded verbatim in ``map_lint_routes.py``: somnus is a plain non-admin
principal by design and an admin-only gate would force the loop to hold
admin credentials it should not have.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from kb_service.attribution import resolve_attribution
from kb_service.auth import get_current_user
from kb_service.models import (
    ClusterLedgerDeclineRequest,
    ClusterLedgerDeclineResponse,
    ClusterLedgerMatchItem,
    ClusterLedgerMatchRequest,
    ClusterLedgerMatchResponse,
    User,
)

logger = logging.getLogger(__name__)

# Greppable log marker — mirrors MAP_LOOP_INPUT_MARKER.
CLUSTER_LEDGER_ROUTE_MARKER = "cluster-ledger-route"

router = APIRouter(prefix="/api/kb", tags=["kb"])

# The status string carried by an unmatched candidate (see the model docstring).
STATUS_NONE = "none"


async def _admission_gate(kb: Any, project_ref: str) -> None:
    """404/409 admission gate, IDENTICAL to map_loop_input's — one rule, not two.

    404 for a ref with no eligibility verdict, 409 for an ineligible one,
    human overrides honoured in both directions (the verdict's
    ``effective_eligible`` already folds the override in). Both endpoints
    enforce it before touching the ledger, exactly as map-loop-input does
    before issuing any of its own reads.
    """
    verdicts = await kb.map_eligibility()
    verdict = next((v for v in verdicts if v.evidence.project_ref == project_ref), None)
    if verdict is None:
        logger.info(
            "%s outcome=not_found project_ref=%s",
            CLUSTER_LEDGER_ROUTE_MARKER,
            project_ref,
        )
        raise HTTPException(status_code=404, detail="project_ref not found")
    if verdict.effective_eligible is False:
        logger.info(
            "%s outcome=ineligible project_ref=%s decided_by=%s",
            CLUSTER_LEDGER_ROUTE_MARKER,
            project_ref,
            verdict.decided_by,
        )
        raise HTTPException(status_code=409, detail="project_ref not map-eligible")


@router.post("/cluster-ledger/match", response_model=ClusterLedgerMatchResponse)
async def cluster_ledger_match(
    body: ClusterLedgerMatchRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> ClusterLedgerMatchResponse:
    """Match a batch of candidate member-id sets against one project's ledger.

    Args:
        body: ``{project_ref, candidates: [{member_entry_ids}, ...]}``.
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user — plain principal, NOT admin (see module
            docstring).

    Returns:
        200 with ``{results: [{index, matched, ledger_id, status, jaccard,
        reopen_eligible}, ...]}`` — one item per candidate, in request
        order, ``index`` the zero-based request position. An empty batch
        short-circuits to ``{"results": []}`` WITHOUT reaching the facade
        (kb-core raises ``ValueError`` on an empty batch, and 200/empty is
        the honest answer to a project with no pockets tonight). ``jaccard``
        crosses the wire exactly as kb-core computed it — never rounded.

    Raises:
        HTTPException: 404 unknown ref, 409 not map-eligible (before any
            ledger access).
    """
    kb = request.app.state.kb
    await _admission_gate(kb, body.project_ref)
    if not body.candidates:
        return ClusterLedgerMatchResponse(results=[])

    result = await kb.match_clusters(
        body.project_ref, [candidate.member_entry_ids for candidate in body.candidates]
    )
    items = [
        ClusterLedgerMatchItem(
            index=index,
            matched=verdict.matched is not None,
            ledger_id=(
                verdict.matched.cluster_key if verdict.matched is not None else None
            ),
            status=(
                verdict.matched.status if verdict.matched is not None else STATUS_NONE
            ),
            jaccard=verdict.jaccard,
            reopen_eligible=verdict.reopened,
        )
        for index, verdict in enumerate(result.verdicts)
    ]
    logger.info(
        "%s op=match project_ref=%s candidates=%d matched=%d",
        CLUSTER_LEDGER_ROUTE_MARKER,
        body.project_ref,
        len(items),
        sum(1 for item in items if item.matched),
    )
    return ClusterLedgerMatchResponse(results=items)


@router.post(
    "/cluster-ledger/decline",
    status_code=201,
    response_model=ClusterLedgerDeclineResponse,
)
async def cluster_ledger_decline(
    body: ClusterLedgerDeclineRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> ClusterLedgerDeclineResponse:
    """Decline one cluster by member-id set — the human's verdict, 201.

    Resolves the member set to its ledger row via ``kb.record_cluster``
    (see the module docstring for why the decline records first), then
    declines that row by key with ``declined_by`` set to the caller's
    contributor identity.

    Raises:
        HTTPException: 404 unknown ref, 409 not map-eligible (before any
            ledger access); 404 "cluster not found" if the facade reports
            the resolved row vanished mid-request (a kb-core invariant
            breach, not a caller shape).
    """
    kb = request.app.state.kb
    await _admission_gate(kb, body.project_ref)
    attr = await resolve_attribution(user)
    recorded = await kb.record_cluster(
        body.project_ref, body.member_entry_ids, body.reason
    )
    declined = await kb.decline_cluster(
        recorded.cluster_key, reason=body.reason, declined_by=attr.contributor
    )
    if declined is None:
        logger.error(
            "%s invariant=row_vanished project_ref=%s cluster_key=%s",
            CLUSTER_LEDGER_ROUTE_MARKER,
            body.project_ref,
            recorded.cluster_key,
        )
        raise HTTPException(status_code=404, detail="cluster not found")
    logger.info(
        "%s op=decline project_ref=%s ledger_id=%s declined_by=%s members=%d",
        CLUSTER_LEDGER_ROUTE_MARKER,
        body.project_ref,
        declined.cluster_key,
        attr.contributor or "none",
        len(declined.member_entry_ids),
    )
    return ClusterLedgerDeclineResponse(ledger_id=declined.cluster_key)
