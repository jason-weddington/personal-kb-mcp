"""DELETE /api/kb/maps/{map_id} — the loop's one removal path, machine principal only.

The nightly map loop could create maps and add pointers, but never remove a bad one.
So the map graph only accumulated, and a graph nobody can hand-repair rots.

The ruling (Jason, 2026-09-21): the loop that finds a bad map deletes it.
Gone for real, with its outbound edges — a 100% self-healing graph, maps included.
A human cannot maintain a 3000-entry KB; instrumentation replaces the human in the loop.

The distinction the ruling preserves: ``deactivate`` is for real knowledge entries.
Their CONTENT has value and must be recoverable.
A ``mental_map`` holds no facts — a directory card pointing at the entries that do.
So a bad map is DELETED outright, and its edges go with it rather than dangling.

Machine principal ONLY: 403 for everyone else, the same gate as ``POST /api/kb/map-op``.
A human deleting a map is a different workflow, explicitly out of scope here.

All deletions go through kb-core's ``delete_map`` primitive, pinned at kb-core de66b2f.
It is reached via the ``KnowledgeBase`` facade method of the same name.
It refuses any entry_type that is not ``mental_map``.
It deletes the row plus BOTH edge directions plus the version rows in one transaction.
It keeps the ``audit_events`` rows and returns the full record.
That record — body, pointers, edge counts, referrers — is the whole response.

``inbound_referrer_ids`` is load-bearing, and its being non-empty is NOT an error.
``inbound_referrer_ids`` is load-bearing, and its being non-empty is not an error.

Status semantics, flat and terminal, matching ``map_op_routes``.

403 not the machine principal · 404 unknown ``map_id`` · 422 a missing or blank reason.
409 an id that exists but is not a ``mental_map`` · 200 success.
A 404 and a 409 must not be conflated.
One means the loop holds a stale id; the other means it is pointed at real knowledge.
The second is a much more serious bug than the first.

kb-core raises ``ValueError`` for both, so they are mapped apart by message.
That is the same trick ``_map_value_error`` in ``kb_write_routes`` already plays.
Here it is keyed on this primitive's own not-found wording.

THE AUDIT EVENT IS THE ACCEPTANCE CRITERION THAT CARRIES THE RULING.

kb-core's primitive deliberately writes no audit event: the caller names who and why.
So this endpoint writes one ``audit_events`` row per deletion.
It records the map id, its project_ref, its short_title, its FULL knowledge_details.
It records the pointer_ids and the inbound_referrer_ids.
It records the acting contributor and the caller-supplied reason too.
A deletion that cannot be reconstructed from the audit trail must never be produced.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Annotated, Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Request
from kb_core.map_delete import MAP_DELETE_MARKER, DeletedMapRecord

from kb_service.attribution import is_machine_principal, resolve_attribution
from kb_service.auth import get_current_user
from kb_service.models import MapDeleteRequest, MapDeleteResponse, User

logger = logging.getLogger(__name__)

# Greppable log marker — mirrors MAP_OP_ROUTE_MARKER.
MAP_DELETE_ROUTE_MARKER = "map-delete-route"

# The one audit event_type this endpoint writes. kb-core's primitive keeps the
# audit_events rows referring to the deleted map; this row is the durable
# record of the deletion itself.
AUDIT_EVENT_TYPE = "map_deleted"

router = APIRouter(prefix="/api/kb", tags=["kb"])


def _log(
    op: str,
    outcome: str,
    status_code: int,
    *,
    project_ref: str | None = None,
    map_id: str | None = None,
    outbound_edges_deleted: int = 0,
    inbound_edges_deleted: int = 0,
    inbound_referrer_count: int = 0,
    detail: str | None = None,
) -> None:
    """Emit the one operator-readable trail line for a handled deletion.

    The counts are on the line so an operator reading the journal sees
    deletions — including how many referrer bodies need repair — without
    querying the audit table.
    """
    logger.info(
        "%s op=%s project_ref=%r map_id=%r outcome=%s status=%d"
        " outbound_edges_deleted=%d inbound_edges_deleted=%d"
        " inbound_referrer_count=%d%s",
        MAP_DELETE_ROUTE_MARKER,
        op,
        project_ref,
        map_id,
        outcome,
        status_code,
        outbound_edges_deleted,
        inbound_edges_deleted,
        inbound_referrer_count,
        f" detail={detail!r}" if detail is not None else "",
    )


def _reject(
    op: str,
    outcome: str,
    status_code: int,
    detail: str,
    *,
    map_id: str,
) -> NoReturn:
    """Log the rejection and raise it — one log line, one raise, per branch.

    Args:
        op: The op literal, for the trail line.
        outcome: A stable machine-readable token for the night's log.
        status_code: The HTTP status (403/404/409/422).
        detail: The response ``detail`` string.
        map_id: The request's map_id, when it has one.

    Raises:
        HTTPException: Always, with *status_code* and *detail*.
    """
    _log(op, outcome, status_code, map_id=map_id, detail=detail)
    raise HTTPException(status_code=status_code, detail=detail)


def _delete_value_error(exc: ValueError) -> HTTPException:
    """Map a kb-core ``ValueError`` to HTTP 404 (no such row) or 409 (not a map).

    ``_map_value_error`` in ``kb_write_routes`` keys its 404 on ``"not found"``.
    This primitive words its not-found error ``"no entry with id"``, so the same
    trick is keyed on that message instead.
    """
    msg = str(exc)
    if f"{MAP_DELETE_MARKER}: no entry with id" in msg:
        return HTTPException(status_code=404, detail=msg)
    return HTTPException(status_code=409, detail=msg)


def _resolve_reason(
    body: MapDeleteRequest | None, query_reason: str | None
) -> str | None:
    """Pick the body reason, falling back to the query parameter, then strip.

    Returns ``None`` when neither carries a non-blank reason, which the caller
    turns into a 422 — a deletion without a recorded reason is untraceable
    rot-removal.
    """
    candidates = [body.reason if body is not None else None, query_reason]
    for candidate in candidates:
        if candidate is not None and candidate.strip():
            return candidate.strip()
    return None


def _audit_detail(record: DeletedMapRecord, reason: str, contributor: str) -> str:
    """Render the one detail payload the audit row must carry, as JSON.

    ``knowledge_details`` is the FULL body — never a summary — because this row
    is all that survives the map and it has to be sufficient to reconstruct it.

    JSON rather than a delimiter-joined string, and that is the whole point of
    the format choice: a map body contains newlines, semicolons, commas and
    equals signs, and ``reason`` is caller-supplied text that could contain
    ``;project_ref=`` and silently overwrite an earlier field. A record whose
    stated purpose is reconstruction has to be unambiguously parseable, or it
    is a record of something nobody can read back.
    """
    return json.dumps(
        {
            "deleted_by": contributor,
            "reason": reason,
            "map_id": record.entry_id,
            "project_ref": record.project_ref,
            "short_title": record.short_title,
            "long_title": record.long_title,
            "pointer_ids": list(record.pointer_ids),
            "inbound_referrer_ids": list(record.inbound_referrer_ids),
            "outbound_edges_deleted": record.outbound_edges_deleted,
            "inbound_edges_deleted": record.inbound_edges_deleted,
            "knowledge_details": record.knowledge_details,
        },
        sort_keys=True,
    )


async def _write_audit_event(
    kb: Any, record: DeletedMapRecord, reason: str, contributor: str
) -> None:
    """Record the deletion in ``audit_events``, fire-and-forget with a warning.

    The deletion is already committed when this runs, so a failure here cannot
    be rolled back — the map is gone either way — and the response body hands
    the caller the very same record.
    The warning names the map so the night's journal still points a human at
    the gap in the trail, which is the most this endpoint can do.
    """
    try:
        created_at = datetime.now(UTC).isoformat()
        await kb.db.execute(
            "INSERT INTO audit_events"
            " (event_type, entry_id, contributor, detail, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                AUDIT_EVENT_TYPE,
                record.entry_id,
                contributor,
                _audit_detail(record, reason, contributor),
                created_at,
            ),
        )
        await kb.db.commit()
    except Exception:
        logger.warning(
            "%s: failed to record %s audit event for map %s — the deletion is"
            " committed but its trail row is missing",
            MAP_DELETE_ROUTE_MARKER,
            AUDIT_EVENT_TYPE,
            record.entry_id,
            exc_info=True,
        )


@router.delete("/maps/{map_id}", response_model=MapDeleteResponse)
async def delete_map(
    map_id: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    body: MapDeleteRequest | None = None,
    reason: str | None = None,
) -> MapDeleteResponse:
    """Hard-delete one mental_map with its edges, machine principal only.

    Args:
        map_id: The id of the map to delete.
        request: FastAPI request (provides ``app.state.kb``).
        user: The authenticated caller; must be the configured machine principal.
        body: The optional request body, carrying the required ``reason``.
        reason: The same reason as a query parameter; either carrier is fine,
            but one of them must be non-blank.

    Returns:
        ``MapDeleteResponse`` — the full ``DeletedMapRecord`` the caller needs
        to log the map back into existence.

    Raises:
        HTTPException: 403 not the machine principal, 404 unknown ``map_id``,
            409 an id that is not a ``mental_map``, 422 a missing or blank
            reason.
    """
    kb = request.app.state.kb
    op = "delete_map"
    if not await is_machine_principal(user):
        _log(op, "forbidden_not_machine_principal", 403, map_id=map_id)
        raise HTTPException(
            status_code=403,
            detail=(
                "DELETE /api/kb/maps/{map_id} is restricted to the configured"
                " machine principal; a human deleting a map is a different"
                " workflow and is out of scope here"
            ),
        )
    recorded_reason = _resolve_reason(body, reason)
    if recorded_reason is None:
        _reject(
            op,
            "reason_required",
            422,
            "a deletion reason is required — in the body or as the `reason`"
            " query parameter — and it must not be blank: a deletion with no"
            " recorded reason is untraceable rot-removal",
            map_id=map_id,
        )
    try:
        record = await kb.delete_map(map_id)
    except ValueError as exc:
        http_exc = _delete_value_error(exc)
        _reject(
            op,
            "not_found" if http_exc.status_code == 404 else "not_a_mental_map",
            http_exc.status_code,
            http_exc.detail,
            map_id=map_id,
        )
    attr = await resolve_attribution(user)
    # resolve_attribution always sets contributor to the caller's email; the
    # fallback below is a type guard only, never a value change.
    contributor = attr.contributor or user.email
    await _write_audit_event(kb, record, recorded_reason, contributor)
    _log(
        op,
        "deleted",
        200,
        project_ref=record.project_ref,
        map_id=record.entry_id,
        outbound_edges_deleted=record.outbound_edges_deleted,
        inbound_edges_deleted=record.inbound_edges_deleted,
        inbound_referrer_count=len(record.inbound_referrer_ids),
    )
    return MapDeleteResponse(
        map_id=record.entry_id,
        project_ref=record.project_ref,
        short_title=record.short_title,
        long_title=record.long_title,
        knowledge_details=record.knowledge_details,
        pointer_ids=record.pointer_ids,
        outbound_edges_deleted=record.outbound_edges_deleted,
        inbound_edges_deleted=record.inbound_edges_deleted,
        inbound_referrer_ids=record.inbound_referrer_ids,
    )
