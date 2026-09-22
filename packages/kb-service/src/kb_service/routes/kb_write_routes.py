"""KB write endpoints: store, store_batch, deactivate, reactivate, and more.

Endpoints: POST /api/kb/store, /store_batch, /entries/{id}/deactivate,
/entries/{id}/reactivate, /bulk_update, /feedback.

All six endpoints live under ``/api/kb`` and require authentication.
``reactivate`` and ``bulk_update`` additionally require admin privileges
(matching the ``kb_maintain`` KB_MANAGER gate described in kb-01742).

LLM graph enrichment (store, store_batch) runs synchronously inside the
request; a batch may take tens of seconds.  No streaming or timeout machinery
is added in this item.

mental_map writes carry the machine-principal lint gate
(``_check_machine_principal_map_lint``): when the writer is the configured
machine principal, a kb-core map-lint finding rejects the write with 422
(the nightly loop's only mechanical purity check — see
docs/nightly-map-maintenance-design.md).  For every other user the lint is
advisory (rendered by the MCP channel) and never blocks a write here.
That guard and the orphan check beside it live in ``map_write_guards.py``,
shared verbatim with the machine-principal map-op write path.
"""

import logging
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from kb_core.ingest.safety import detect_secrets_in_content
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.ttl import compute_expires_at

from kb_service.attribution import resolve_attribution
from kb_service.auth import get_current_user, require_admin
from kb_service.models import User
from kb_service.models_kb import (
    BulkUpdatePair,
    BulkUpdateRequest,
    BulkUpdateResponse,
    EntryActionResponse,
    FeedbackRequest,
    FeedbackResponse,
    StoreBatchRequest,
    StoreBatchResponse,
    StoreRequest,
    StoreResponse,
)
from kb_service.routes.map_write_guards import (
    _check_machine_principal_map_lint,
    _check_orphan_mental_map,
    _mental_map_has_pointer,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kb", tags=["kb-write"])

# Allowed filter / update keys for the bulk_update endpoint.
# The engine silently ignores unknown keys; this endpoint rejects them (422).
_BULK_FILTER_KEYS: frozenset[str] = frozenset(
    {"contributor", "team", "project_ref", "entry_type", "tags", "entry_ids"}
)
_BULK_UPDATE_KEYS: frozenset[str] = frozenset(
    {
        "project_ref",
        "entry_type",
        "confidence_level",
        "tags_add",
        "tags_remove",
        "team",
    }
)

# ─── internal validation helpers ─────────────────────────────────────────────


def _parse_ttl(ttl: str | None) -> datetime | None:
    """Convert a TTL string to an expiry datetime, raising 422 on parse errors."""
    if ttl is None:
        return None
    try:
        return compute_expires_at(ttl)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _check_secrets(knowledge_details: str, kb: Any) -> None:
    """Scan *knowledge_details* for secrets when safety is enabled.

    A non-empty findings list raises 422 listing the detected types.
    A ``None`` return (library missing) or an empty list means no rejection.
    Gated on ``kb.config.ingest.skip_safety``.
    """
    if kb.config.ingest.skip_safety:
        return
    findings = detect_secrets_in_content(knowledge_details)
    if findings:
        types_str = ", ".join(findings)
        raise HTTPException(
            status_code=422,
            detail=f"Secret scan detected sensitive content: {types_str}",
        )


def _map_value_error(exc: ValueError) -> HTTPException:
    """Map a kb-core ``ValueError`` to HTTP 404 (not found) or 409 (conflict)."""
    msg = str(exc)
    if "not found" in msg:
        return HTTPException(status_code=404, detail=msg)
    return HTTPException(status_code=409, detail=msg)


_MAP_DEACTIVATE_BLOCKED = (
    "mental_map entries cannot be deactivated. A mental_map holds no facts to "
    "recover, so a bad one is deleted outright — edges and all — by the "
    "machine principal via DELETE /api/kb/maps/{map_id}, which removes the "
    "row, both edge directions and the version rows in one transaction. "
    "Deactivating a map is refused because a soft delete would strip its "
    "outbound graph edges and orphan every detail entry the map pointed to "
    "from the listener's detail -> owning-map reverse lookup; the edges go "
    "with a proper delete, never with a deactivation."
)


# ─── endpoints ───────────────────────────────────────────────────────────────


@router.post("/store", response_model=StoreResponse)
async def store(
    body: StoreRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> StoreResponse:
    """Create or update a single knowledge base entry.

    LLM graph enrichment runs synchronously inside the request and may take
    several seconds.  Set ``update_entry_id`` to update an existing entry.
    """
    kb = request.app.state.kb
    attr = await resolve_attribution(user)

    if body.update_entry_id is None:
        # ── CREATE path ──────────────────────────────────────────────────────
        if not body.short_title or not body.long_title or not body.knowledge_details:
            raise HTTPException(
                status_code=422,
                detail=(
                    "short_title, long_title, and knowledge_details are required "
                    "when creating a new entry."
                ),
            )
        expires_at = _parse_ttl(body.ttl)
        _check_secrets(body.knowledge_details, kb)
        entry_type = body.entry_type or EntryType.FACTUAL_REFERENCE
        _check_orphan_mental_map(entry_type, body.knowledge_details, body.hints)
        await _check_machine_principal_map_lint(
            entry_type, body.knowledge_details, user
        )

        entry: KnowledgeEntry = await kb.store(
            short_title=body.short_title,
            long_title=body.long_title,
            knowledge_details=body.knowledge_details,
            entry_type=entry_type,
            project_ref=body.project_ref,
            source_context=body.source_context,
            confidence_level=(
                body.confidence_level if body.confidence_level is not None else 0.9
            ),
            tags=body.tags,
            hints=body.hints,
            contributor=attr.contributor,
            team=attr.team,
            sensitivity=body.sensitivity,
            expires_at=expires_at,
        )
        entry = await kb.get(entry.id) or entry
        return StoreResponse(action="created", entry=entry)

    # ── UPDATE path ──────────────────────────────────────────────────────────
    entry_id = body.update_entry_id
    expires_at = _parse_ttl(body.ttl)
    if body.knowledge_details:
        _check_secrets(body.knowledge_details, kb)

    # The orphan check must run against the EFFECTIVE post-update body, not the
    # raw request body: knowledge_details is legitimately None on a
    # metadata-only update (a tags-only or hints-only change), and entry_type
    # is legitimately None when the update doesn't touch it. Fetch the current
    # row and merge the same way kb-core's own update_entry does (mirrors
    # knowledge_store.py's effective_details / merged_hints computation), so a
    # tags-only update to a map whose STORED body already has a pointer is not
    # rejected as an orphan.
    existing = await kb.get(entry_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")
    effective_entry_type = (
        body.entry_type if body.entry_type is not None else existing.entry_type
    )
    effective_details = (
        body.knowledge_details if body.knowledge_details else existing.knowledge_details
    )
    effective_hints: dict[str, Any] = dict(existing.hints)
    if body.hints:
        effective_hints.update(body.hints)
    _check_orphan_mental_map(effective_entry_type, effective_details, effective_hints)
    # Machine-principal map-lint gate — same EFFECTIVE-body view as the orphan
    # check above, for the same reason: a mental_map is the entry after the
    # update, so any write by the machine principal that would leave a map
    # failing the lint is rejected, including a metadata-only update whose
    # stored body fails. The somnus loop always sends full bodies on automated
    # updates (design: "full-body writes on every automated update"), so this
    # never rejects a legitimate automated write.
    await _check_machine_principal_map_lint(
        effective_entry_type, effective_details, user
    )

    try:
        entry = await kb.update(
            entry_id,
            knowledge_details=body.knowledge_details or None,
            change_reason=body.change_reason,
            confidence_level=body.confidence_level,
            tags=body.tags,
            hints=body.hints,
            updated_by=user.email,
            sensitivity=body.sensitivity,
            expires_at=expires_at,
            short_title=body.short_title or None,
            long_title=body.long_title or None,
            entry_type=body.entry_type,
            project_ref=body.project_ref,
            source_context=body.source_context,
        )
    except ValueError as exc:
        raise _map_value_error(exc) from exc
    entry = await kb.get(entry.id) or entry
    return StoreResponse(action="updated", entry=entry)


@router.post("/store_batch", response_model=StoreBatchResponse)
async def store_batch(
    body: StoreBatchRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> StoreBatchResponse:
    """Create multiple knowledge base entries in a single request.

    LLM graph enrichment runs synchronously inside the request; a batch may
    take tens of seconds.  All entries are validated up-front — any failure
    rejects the entire batch (422 with the failing entry index in the detail).
    """
    kb = request.app.state.kb
    attr = await resolve_attribution(user)

    # ── up-front batch validation ─────────────────────────────────────────
    for i, raw in enumerate(body.entries):
        prefix = f"entry {i}: "
        if raw.ttl is not None:
            try:
                compute_expires_at(raw.ttl)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=f"{prefix}{exc}") from exc
        if not kb.config.ingest.skip_safety:
            findings = detect_secrets_in_content(raw.knowledge_details)
            if findings:
                types_str = ", ".join(findings)
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"{prefix}Secret scan detected sensitive content: {types_str}"
                    ),
                )
        if raw.entry_type is EntryType.MENTAL_MAP and not _mental_map_has_pointer(
            raw.knowledge_details, raw.hints
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{prefix}A mental_map entry requires at least one outbound "
                    "pointer (a kb-XXXXX reference in knowledge_details, or a "
                    "supersedes/related_entities hint)."
                ),
            )
        await _check_machine_principal_map_lint(
            raw.entry_type, raw.knowledge_details, user, prefix=prefix
        )

    # ── build facade dicts ────────────────────────────────────────────────
    entry_dicts: list[dict[str, Any]] = []
    for raw in body.entries:
        expires_at = compute_expires_at(raw.ttl) if raw.ttl is not None else None
        entry_dicts.append(
            {
                "short_title": raw.short_title,
                "long_title": raw.long_title,
                "knowledge_details": raw.knowledge_details,
                "entry_type": raw.entry_type,
                "project_ref": raw.project_ref,
                "source_context": raw.source_context,
                "confidence_level": raw.confidence_level,
                "tags": raw.tags,
                "hints": raw.hints,
                "sensitivity": raw.sensitivity,
                "expires_at": expires_at,
                "contributor": attr.contributor,
                "team": attr.team,
            }
        )

    created: list[KnowledgeEntry] = await kb.store_batch(entry_dicts, enrich=True)

    # Re-fetch each created entry to pick up has_embedding
    refreshed: list[KnowledgeEntry] = []
    for e in created:
        fetched = await kb.get(e.id)
        refreshed.append(fetched if fetched is not None else e)

    return StoreBatchResponse(requested=len(body.entries), created=refreshed)


@router.post("/entries/{entry_id}/deactivate", response_model=EntryActionResponse)
async def deactivate(
    entry_id: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> EntryActionResponse:
    """Deactivate a knowledge base entry and clean up its outbound graph edges.

    mental_map entries are rejected with 422 — see ``_MAP_DEACTIVATE_BLOCKED``.
    A bad map is deleted by the machine principal via
    ``DELETE /api/kb/maps/{map_id}`` instead, never deactivated.
    """
    kb = request.app.state.kb
    existing = await kb.get(entry_id)
    if existing is not None and existing.entry_type is EntryType.MENTAL_MAP:
        raise HTTPException(status_code=422, detail=_MAP_DEACTIVATE_BLOCKED)
    try:
        entry: KnowledgeEntry = await kb.deactivate(entry_id, contributor=user.email)
    except ValueError as exc:
        raise _map_value_error(exc) from exc
    # Replicate the MCP channel's graph cleanup (kb_maintain reactivate path).
    # Scoped (not blanket) to exclude mental_map sources: the guard above
    # already keeps a map from reaching this line via this endpoint, but the
    # delete is scoped too as defense in depth against the same hazard
    # kb-core's own equivalent delete was narrowed for (personal_kb commit
    # 8917cb0, graph/builder.py::_clear_edges_for_source ->
    # Database.delete_deterministic_edges) — a map's outbound "references"
    # edges are exactly what the listener's detail -> owning-map reverse
    # lookup depends on. Non-map entries are unaffected.
    await kb.db.execute(
        "DELETE FROM graph_edges WHERE source = ? AND source NOT IN "
        "(SELECT id FROM knowledge_entries WHERE entry_type = 'mental_map')",
        (entry_id,),
    )
    await kb.db.commit()
    return EntryActionResponse(entry=entry)


@router.post("/entries/{entry_id}/reactivate", response_model=EntryActionResponse)
async def reactivate(
    entry_id: str,
    request: Request,
    user: Annotated[User, Depends(require_admin)],
) -> EntryActionResponse:
    """Reactivate a previously deactivated entry and rebuild its graph (admin only).

    Graph rebuild is best-effort: failures are logged as warnings and never
    propagate to the caller.
    """
    kb = request.app.state.kb
    try:
        entry: KnowledgeEntry = await kb.reactivate(entry_id, contributor=user.email)
    except ValueError as exc:
        raise _map_value_error(exc) from exc

    # Best-effort graph rebuild — mirrors kb_maintain's reactivate action.
    try:
        await kb.graph_builder.build_for_entry(entry)
    except Exception as exc:
        logger.warning("graph_builder.build_for_entry failed for %s: %s", entry_id, exc)
    if kb.graph_enricher is not None:
        try:
            await kb.graph_enricher.enrich_entry(entry)
        except Exception as exc:
            logger.warning(
                "graph_enricher.enrich_entry failed for %s: %s", entry_id, exc
            )
    return EntryActionResponse(entry=entry)


@router.post("/bulk_update", response_model=BulkUpdateResponse)
async def bulk_update(
    body: BulkUpdateRequest,
    request: Request,
    user: Annotated[User, Depends(require_admin)],
) -> BulkUpdateResponse:
    """Bulk-update entries matching *filters* by applying *updates* (admin only).

    ``dry_run`` defaults to ``True``; callers must explicitly set ``False`` to
    persist changes.  Unknown filter or update keys cause a 422 — the engine
    silently ignores them but this endpoint enforces the whitelist.
    """
    kb = request.app.state.kb

    if not body.filters:
        raise HTTPException(
            status_code=422,
            detail="filters must not be empty (mass-update guard).",
        )
    if not body.updates:
        raise HTTPException(status_code=422, detail="updates must not be empty.")

    unknown_filters = set(body.filters.keys()) - _BULK_FILTER_KEYS
    if unknown_filters:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown filter key(s): {', '.join(sorted(unknown_filters))}. "
                f"Allowed: {', '.join(sorted(_BULK_FILTER_KEYS))}."
            ),
        )
    unknown_updates = set(body.updates.keys()) - _BULK_UPDATE_KEYS
    if unknown_updates:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown update key(s): {', '.join(sorted(unknown_updates))}. "
                f"Allowed: {', '.join(sorted(_BULK_UPDATE_KEYS))}."
            ),
        )

    pairs: list[tuple[KnowledgeEntry, KnowledgeEntry]] = await kb.bulk_update(
        body.filters,
        body.updates,
        contributor=user.email,
        dry_run=body.dry_run,
    )
    results = [BulkUpdatePair(before=b, after=a) for b, a in pairs]
    return BulkUpdateResponse(dry_run=body.dry_run, count=len(results), results=results)


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    body: FeedbackRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> FeedbackResponse:
    """Record agent friction/quality feedback to the KB data DB.

    Writes to the ``agent_feedback`` table created by kb-core's own schema
    (postgres_backend.py:769).  The service adds no schema of its own.
    ``feedback_type`` must be one of: ``missing``, ``unhelpful``, ``friction``.
    """
    kb = request.app.state.kb
    attr = await resolve_attribution(user)
    now = datetime.now(UTC).isoformat()
    await kb.db.execute(
        "INSERT INTO agent_feedback (feedback_type, tool_name,"
        " query_or_params, detail, contributor, team, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            body.feedback_type,
            body.tool_name,
            body.query_or_params,
            body.detail,
            attr.contributor,
            attr.team,
            now,
        ),
    )
    await kb.db.commit()
    return FeedbackResponse(status="recorded", feedback_type=body.feedback_type)
