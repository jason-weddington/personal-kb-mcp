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

from fastapi import APIRouter, Body, Depends, HTTPException, Request
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
    DeactivateRequest,
    EntryActionResponse,
    FeedbackRequest,
    FeedbackResponse,
    StoreBatchEntry,
    StoreBatchRequest,
    StoreBatchResponse,
    StoreRequest,
    StoreResponse,
)
from kb_service.routes.map_write_guards import (
    _check_machine_principal_map_lint,
    _check_orphan_mental_map,
    _mental_map_has_pointer,
    check_superseded_map_pointers,
    map_write_pointer_ids,
)
from kb_service.routes.near_duplicate_guard import (
    check_distinct_from,
    collect_distinct_from,
    enforce_near_duplicate_guard,
    record_stored,
)
from kb_service.supersession_log import log_reconcile_report

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


# ─── supersession helpers ────────────────────────────────────────────────────

# Greppable log marker for every supersedes decision this module makes.
SUPERSESSION_ROUTE_MARKER = "supersession-route"

_CHANGE_REASON_UPDATE = (
    "change_reason is required when updating an entry: say what changed and why."
)
_CHANGE_REASON_DEACTIVATE = (
    "change_reason is required when deactivating an entry; if a newer entry "
    "replaces this one, also pass superseded_by."
)
_DISTINCT_FROM_UPDATE = (
    "distinct_from applies only when creating an entry; an update has no "
    "near-duplicate check."
)
_HINTS_SUPERSEDES_SHAPE = (
    "supersedes rejected: hints.supersedes must be a kb-id string or a list of "
    "kb-id strings"
)


def _log_supersedes_decision(
    op: str,
    outcome: str,
    *,
    mode: str,
    writer: str | None,
    targets: list[str],
    problems: list[str],
) -> None:
    """Emit the one INFO trail line for a supersedes decision.

    ``op`` is store|store_batch|deactivate|update; ``outcome`` is
    accepted|rejected|none|change_reason_missing; ``mode`` describes the raw
    request ``supersedes`` field before normalization
    (absent|none_literal|empty|list).
    """
    logger.info(
        "%s op=%s outcome=%s mode=%s writer=%r targets=%r problems=%r",
        SUPERSESSION_ROUTE_MARKER,
        op,
        outcome,
        mode,
        writer,
        targets,
        problems,
    )


def _supersedes_mode(raw: list[str] | str | None) -> str:
    """Describe the raw request ``supersedes`` value for the trail line."""
    if raw is None:
        return "absent"
    if raw == "none":
        return "none_literal"
    if not raw:
        return "empty"
    return "list"


def _norm(value: object) -> list[object]:
    """None -> [], scalar -> [value], list as-is (mirrors kb-core's norm)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _hint_supersedes(hints: dict[str, Any] | None) -> list[str] | None:
    """Return the str items of ``hints['supersedes']``, or None when malformed.

    Nothing is silently filtered: any non-str item makes the whole value
    malformed, which the caller rejects with 422.
    """
    items = _norm((hints or {}).get("supersedes"))
    if not all(isinstance(i, str) for i in items):
        return None
    return [str(i) for i in items]


def _request_supersedes(raw: list[str] | str | None) -> list[str]:
    """The request ``supersedes`` field as a list (None/'none'/[] -> [])."""
    return list(raw) if isinstance(raw, list) else []


async def _warn_if_build_failed(kb: Any, writer: str, targets: list[str]) -> None:
    """WARN when a validated supersedes write left a target unpointed.

    A just-written active non-map superseder always qualifies, so a target
    whose ``superseded_by`` is still NULL means ``_build_graph`` failed
    (best-effort) and the next startup reconcile has to heal it.
    """
    pending: list[str] = []
    for target_id in targets:
        target = await kb.get(target_id)
        if target is not None and target.superseded_by is None:
            pending.append(target_id)
    if pending:
        logger.warning(
            "%s build_failed writer=%s pending_targets=%r"
            " (healed by next startup reconcile)",
            SUPERSESSION_ROUTE_MARKER,
            writer,
            pending,
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
        await check_superseded_map_pointers(
            kb, entry_type, body.knowledge_details, body.hints
        )

        mode = _supersedes_mode(body.supersedes)
        hint_targets = _hint_supersedes(body.hints)
        if hint_targets is None:
            _log_supersedes_decision(
                "store",
                "rejected",
                mode=mode,
                writer=None,
                targets=[],
                problems=[_HINTS_SUPERSEDES_SHAPE],
            )
            raise HTTPException(status_code=422, detail=_HINTS_SUPERSEDES_SHAPE)
        effective = sorted(
            set(hint_targets) | set(_request_supersedes(body.supersedes))
        )
        store_hints = body.hints
        if effective:
            problems = await kb.check_supersedes(
                effective, writer_id=None, writer_entry_type=entry_type
            )
            if problems:
                _log_supersedes_decision(
                    "store",
                    "rejected",
                    mode=mode,
                    writer=None,
                    targets=effective,
                    problems=problems,
                )
                raise HTTPException(
                    status_code=422,
                    detail="supersedes rejected: " + "; ".join(problems),
                )
            store_hints = {**(body.hints or {}), "supersedes": effective}
        _log_supersedes_decision(
            "store",
            "accepted" if effective else "none",
            mode=mode,
            writer=None,
            targets=effective,
            problems=[],
        )

        distinct_from = collect_distinct_from(body.distinct_from, body.hints)
        distinct_problems = await check_distinct_from(
            kb, distinct_from, supersedes=effective
        )
        if distinct_problems:
            raise HTTPException(
                status_code=422,
                detail="distinct_from rejected: " + "; ".join(distinct_problems),
            )
        decision = await enforce_near_duplicate_guard(
            kb,
            op="store",
            entry_index=None,
            contributor=attr.contributor,
            short_title=body.short_title,
            long_title=body.long_title,
            knowledge_details=body.knowledge_details,
            entry_type=entry_type,
            project_ref=body.project_ref,
            supersedes=effective,
            distinct_from=distinct_from,
        )
        if distinct_from:
            store_hints = {**(store_hints or {}), "distinct_from": distinct_from}

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
            hints=store_hints,
            contributor=attr.contributor,
            team=attr.team,
            sensitivity=body.sensitivity,
            expires_at=expires_at,
        )
        if effective:
            await _warn_if_build_failed(kb, entry.id, effective)
        await record_stored(
            kb, decision, entry_id=entry.id, contributor=attr.contributor
        )
        entry = await kb.get(entry.id) or entry
        return StoreResponse(action="created", entry=entry, superseded_ids=effective)

    # ── UPDATE path ──────────────────────────────────────────────────────────
    entry_id = body.update_entry_id
    mode = _supersedes_mode(body.supersedes)
    if body.change_reason is None or not body.change_reason.strip():
        _log_supersedes_decision(
            "update",
            "change_reason_missing",
            mode=mode,
            writer=entry_id,
            targets=[],
            problems=[],
        )
        raise HTTPException(status_code=422, detail=_CHANGE_REASON_UPDATE)
    if body.distinct_from or "distinct_from" in (body.hints or {}):
        raise HTTPException(status_code=422, detail=_DISTINCT_FROM_UPDATE)
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
    # D5: only NEWLY added superseded pointers are rejected (grandfathered).
    await check_superseded_map_pointers(
        kb,
        effective_entry_type,
        effective_details,
        effective_hints,
        existing_ids=map_write_pointer_ids(existing.knowledge_details, existing.hints),
    )

    # Supersedes on update: a MONOTONIC union with what the entry already
    # records. Only targets NEW to this entry are validated, so recorded ones
    # (possibly inactive by now) are grandfathered; nothing is retracted here.
    update_hints = body.hints
    new_targets: list[str] = []
    request_targets = _request_supersedes(body.supersedes)
    if request_targets or (body.hints is not None and "supersedes" in body.hints):
        hint_targets = _hint_supersedes(body.hints)
        if hint_targets is None:
            _log_supersedes_decision(
                "update",
                "rejected",
                mode=mode,
                writer=entry_id,
                targets=[],
                problems=[_HINTS_SUPERSEDES_SHAPE],
            )
            raise HTTPException(status_code=422, detail=_HINTS_SUPERSEDES_SHAPE)
        recorded = {
            str(i)
            for i in _norm(existing.hints.get("supersedes"))
            if isinstance(i, str)
        }
        merged = sorted(recorded | set(hint_targets) | set(request_targets))
        new_targets = sorted(set(merged) - recorded)
        if new_targets:
            problems = await kb.check_supersedes(
                new_targets,
                writer_id=entry_id,
                writer_entry_type=effective_entry_type,
            )
            if problems:
                _log_supersedes_decision(
                    "update",
                    "rejected",
                    mode=mode,
                    writer=entry_id,
                    targets=new_targets,
                    problems=problems,
                )
                raise HTTPException(
                    status_code=422,
                    detail="supersedes rejected: " + "; ".join(problems),
                )
        update_hints = {**(body.hints or {}), "supersedes": merged}
    _log_supersedes_decision(
        "update",
        "accepted" if new_targets else "none",
        mode=mode,
        writer=entry_id,
        targets=new_targets,
        problems=[],
    )

    try:
        entry = await kb.update(
            entry_id,
            knowledge_details=body.knowledge_details or None,
            change_reason=body.change_reason,
            confidence_level=body.confidence_level,
            tags=body.tags,
            hints=update_hints,
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
    if new_targets:
        await _warn_if_build_failed(kb, entry.id, new_targets)
    entry = await kb.get(entry.id) or entry
    return StoreResponse(action="updated", entry=entry, superseded_ids=new_targets)


async def _validate_batch_supersedes(
    kb: Any, i: int, raw: StoreBatchEntry
) -> list[str]:
    """Apply the CREATE supersedes rule to batch entry *i*; return its target set.

    Raises:
        HTTPException: 422 ``entry {i}: supersedes rejected: ...``.
    """
    mode = _supersedes_mode(raw.supersedes)
    hint_targets = _hint_supersedes(raw.hints)
    if hint_targets is None:
        _log_supersedes_decision(
            "store_batch",
            "rejected",
            mode=mode,
            writer=None,
            targets=[],
            problems=[_HINTS_SUPERSEDES_SHAPE],
        )
        raise HTTPException(
            status_code=422, detail=f"entry {i}: {_HINTS_SUPERSEDES_SHAPE}"
        )
    effective = sorted(set(hint_targets) | set(_request_supersedes(raw.supersedes)))
    if effective:
        problems = await kb.check_supersedes(
            effective, writer_id=None, writer_entry_type=raw.entry_type
        )
        if problems:
            _log_supersedes_decision(
                "store_batch",
                "rejected",
                mode=mode,
                writer=None,
                targets=effective,
                problems=problems,
            )
            raise HTTPException(
                status_code=422,
                detail=f"entry {i}: supersedes rejected: " + "; ".join(problems),
            )
    _log_supersedes_decision(
        "store_batch",
        "accepted" if effective else "none",
        mode=mode,
        writer=None,
        targets=effective,
        problems=[],
    )
    return effective


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
    batch_supersedes: list[list[str]] = []
    batch_distinct_from: list[list[str]] = []
    batch_decisions: list[Any] = []
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
                    "related_entities hint)."
                ),
            )
        await _check_machine_principal_map_lint(
            raw.entry_type, raw.knowledge_details, user, prefix=prefix
        )
        await check_superseded_map_pointers(
            kb,
            raw.entry_type,
            raw.knowledge_details,
            raw.hints,
            status_code=422,
            prefix=prefix,
        )
        effective = await _validate_batch_supersedes(kb, i, raw)
        batch_supersedes.append(effective)
        distinct_from = collect_distinct_from(raw.distinct_from, raw.hints)
        distinct_problems = await check_distinct_from(
            kb, distinct_from, supersedes=effective
        )
        if distinct_problems:
            raise HTTPException(
                status_code=422,
                detail=f"{prefix}distinct_from rejected: "
                + "; ".join(distinct_problems),
            )
        batch_distinct_from.append(distinct_from)
        batch_decisions.append(
            await enforce_near_duplicate_guard(
                kb,
                op="store_batch",
                entry_index=i,
                contributor=attr.contributor,
                short_title=raw.short_title,
                long_title=raw.long_title,
                knowledge_details=raw.knowledge_details,
                entry_type=raw.entry_type,
                project_ref=raw.project_ref,
                supersedes=effective,
                distinct_from=distinct_from,
            )
        )

    # ── build facade dicts ────────────────────────────────────────────────
    entry_dicts: list[dict[str, Any]] = []
    for raw, effective, distinct in zip(
        body.entries, batch_supersedes, batch_distinct_from, strict=True
    ):
        expires_at = compute_expires_at(raw.ttl) if raw.ttl is not None else None
        hints = (
            {**(raw.hints or {}), "supersedes": effective} if effective else raw.hints
        )
        if distinct:
            hints = {**(hints or {}), "distinct_from": distinct}
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
                "hints": hints,
                "sensitivity": raw.sensitivity,
                "expires_at": expires_at,
                "contributor": attr.contributor,
                "team": attr.team,
            }
        )

    created: list[KnowledgeEntry] = await kb.store_batch(entry_dicts, enrich=True)
    # The facade may skip failed entries, so only pair decisions with created
    # entries when the counts line up (the normal case).
    if len(created) == len(batch_decisions):
        for decision, e in zip(batch_decisions, created, strict=True):
            await record_stored(
                kb, decision, entry_id=e.id, contributor=attr.contributor
            )

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
    body: Annotated[DeactivateRequest | None, Body()] = None,
) -> EntryActionResponse:
    """Deactivate a knowledge base entry; kb-core cleans up its outbound edges.

    Checked in this order: auth (401); mental_map (422, see
    ``_MAP_DEACTIVATE_BLOCKED`` — a bad map is deleted via
    ``DELETE /api/kb/maps/{map_id}`` instead); a missing or blank
    ``change_reason`` (422 — so an unknown id with no reason is 422, not
    404); an optional ``superseded_by`` that must exist and pass
    ``check_supersedes`` (422); then ``kb.deactivate``, whose ``ValueError``
    maps to 404/409. The edge cleanup and the supersession recompute run
    inside ``kb.deactivate``'s one transaction.
    """
    kb = request.app.state.kb
    existing = await kb.get(entry_id)
    if existing is not None and existing.entry_type is EntryType.MENTAL_MAP:
        raise HTTPException(status_code=422, detail=_MAP_DEACTIVATE_BLOCKED)
    change_reason = body.change_reason if body is not None else None
    superseded_by = body.superseded_by if body is not None else None
    mode = "absent" if superseded_by is None else "list"
    if change_reason is None or not change_reason.strip():
        _log_supersedes_decision(
            "deactivate",
            "change_reason_missing",
            mode=mode,
            writer=superseded_by,
            targets=[],
            problems=[],
        )
        raise HTTPException(status_code=422, detail=_CHANGE_REASON_DEACTIVATE)
    if superseded_by is not None:
        superseder = await kb.get(superseded_by)
        if superseder is None:
            problems = [f"superseded_by {superseded_by} not found"]
        else:
            problems = await kb.check_supersedes(
                [entry_id],
                writer_id=superseded_by,
                writer_entry_type=superseder.entry_type,
            )
            if problems:
                problems = ["superseded_by rejected: " + "; ".join(problems)]
        if problems:
            _log_supersedes_decision(
                "deactivate",
                "rejected",
                mode=mode,
                writer=superseded_by,
                targets=[entry_id],
                problems=problems,
            )
            raise HTTPException(status_code=422, detail=problems[0])
    _log_supersedes_decision(
        "deactivate",
        "accepted" if superseded_by is not None else "none",
        mode=mode,
        writer=superseded_by,
        targets=[entry_id] if superseded_by is not None else [],
        problems=[],
    )
    try:
        entry: KnowledgeEntry = await kb.deactivate(
            entry_id,
            contributor=user.email,
            change_reason=change_reason,
            superseded_by=superseded_by,
        )
    except ValueError as exc:
        raise _map_value_error(exc) from exc
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
    if not body.dry_run:
        # An entry_type change can flip which entries qualify as superseders
        # (e.g. to/from mental_map), so heal superseded_by right away rather
        # than waiting for the next startup reconcile.
        changed_ids = [b.id for b, a in pairs if b.entry_type != a.entry_type]
        if changed_ids:
            await kb.recompute_supersession(changed_ids)
    results = [BulkUpdatePair(before=b, after=a) for b, a in pairs]
    return BulkUpdateResponse(dry_run=body.dry_run, count=len(results), results=results)


@router.post("/admin/reconcile-supersession")
async def reconcile_supersession(
    request: Request,
    user: Annotated[User, Depends(require_admin)],
) -> dict[str, Any]:
    """Run the idempotent supersession reconcile now (admin only).

    Same work as the startup reconcile; every drifted row is logged at WARNING.
    """
    kb = request.app.state.kb
    report = await kb.reconcile_supersession()
    log_reconcile_report(report, logger)
    return {
        "edges_added": report.edges_added,
        "set_count": report.set_count,
        "cleared_count": report.cleared_count,
        "changed": [list(t) for t in report.changed],
    }


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
