"""Server-side write policy: which surface a write comes from, and what it may do.

Every API key carries an optional ``surface`` (interactive | headless |
autonomous). A request's surface is the key's surface, else ``interactive``
for a password (JWT) login, else ``KB_WRITE_POLICY_DEFAULT_SURFACE``. The
``X-KB-Mode`` header can only DOWNGRADE that trust, never raise it.

An interactive surface writes exactly as before. A headless or autonomous
create (``kb_store`` / ``kb_store_batch``) is not written: it is queued as a
shape-5 ``surprise_candidates`` row, and the candidate pipeline's distiller,
critic and lesson TTL decide whether it reaches the KB. Updates,
deactivations, reactivations, non-dry-run bulk updates and non-dry-run
ingests are refused (403) from a non-interactive surface.

Every policy decision emits exactly one ``write-policy`` INFO line.
"""

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import HTTPException, Request
from kb_core import Attribution
from kb_core.ingest.safety import detect_secrets_in_content
from kb_core.models.entry import EntryType
from kb_core.ttl import compute_expires_at

from kb_service import database
from kb_service.models import SurpriseCaptureMode, User
from kb_service.models_kb import (
    StoreBatchQueuedResponse,
    StoreBatchRequest,
    StoreQueuedResponse,
    StoreRequest,
)
from kb_service.resolution_hint import (
    ResolutionHintError,
    validate_and_stamp_resolution,
)
from kb_service.store_distill import (
    STORE_CANDIDATE_DETECTOR_MODEL,
    STORE_CANDIDATE_SHAPE,
)
from kb_service.turn_digest import surprise_capture_mode

logger = logging.getLogger(__name__)

Surface = Literal["interactive", "headless", "autonomous"]
SURFACES: tuple[Surface, ...] = ("interactive", "headless", "autonomous")
SURFACE_TRUST: dict[str, int] = {"interactive": 2, "headless": 1, "autonomous": 0}

WriteOp = Literal[
    "store",
    "store_batch",
    "update",
    "deactivate",
    "reactivate",
    "bulk_update",
    "ingest_text",
    "ingest_url",
    "ingest_file",
    "mint_key",
]

WRITE_POLICY_DEFAULT_SURFACE_ENV = "KB_WRITE_POLICY_DEFAULT_SURFACE"
MODE_HEADER = "X-KB-Mode"
HARNESS_HEADER = "X-KB-Harness"
ENGINE_HEADER = "X-KB-Engine"
HARNESS_MAX = 64
USER_AGENT_MAX = 80
WRITE_POLICY_MARKER = "write-policy"
WRITE_POLICY_DETAIL_PREFIX = "write policy: "


# --- pure surface helpers ----------------------------------------------------


def parse_surface(raw: str | None) -> Surface | None:
    """A surface from *raw*: None when blank, ``headless`` when unknown."""
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    for s in SURFACES:
        if s == value:
            return s
    return "headless"


def is_coerced(raw: str | None) -> bool:
    """True iff *raw* is non-blank and not a known surface (fail closed)."""
    if raw is None or not raw.strip():
        return False
    return raw.strip().lower() not in SURFACES


def downgrade(base: Surface, header: Surface | None) -> Surface:
    """The lower-trust of *base* and *header*; *base* on a tie or no header."""
    if header is None:
        return base
    if SURFACE_TRUST[header] < SURFACE_TRUST[base]:
        return header
    return base


def sanitize_harness(raw: str | None) -> str:
    """An ``X-KB-Harness`` or ``X-KB-Engine`` value as a safe, bounded token."""
    return re.sub(r"[^A-Za-z0-9._:/-]", "", (raw or "").strip())[:HARNESS_MAX]


_WARNED_DEFAULT_VALUES: set[str] = set()


def default_surface() -> Surface:
    """``KB_WRITE_POLICY_DEFAULT_SURFACE``, read on every call.

    Unset or blank is ``interactive`` (back-compat); an unknown value is
    ``headless`` (fail closed) and is warned about once per distinct value.
    """
    raw = os.environ.get(WRITE_POLICY_DEFAULT_SURFACE_ENV)
    if raw is None or not raw.strip():
        return "interactive"
    parsed = parse_surface(raw)
    if is_coerced(raw) and raw not in _WARNED_DEFAULT_VALUES:
        logger.warning(
            "write-policy bad_default_surface value=%r fallback=headless", raw
        )
        _WARNED_DEFAULT_VALUES.add(raw)
    return parsed or "headless"


# --- request context ---------------------------------------------------------


@dataclass(frozen=True)
class WriteContext:
    """The resolved surface of one write request and what it was based on."""

    surface: Surface
    source: Literal["key", "jwt", "default", "header"]
    key_surface: Surface | None
    header_mode: Surface | None
    header_mode_coerced: bool
    harness: str
    api_key_id: str | None
    auth_method: str
    user_id: str
    session_key: str
    user_agent: str
    engine: str = ""


async def resolve_write_context(request: Request, user: User) -> WriteContext:
    """Resolve the write surface of *request* (key, JWT or default, then header)."""
    principal = getattr(request.state, "kb_principal", None)
    api_key_id: str | None = principal.api_key_id if principal is not None else None
    auth_method: str = principal.auth_method if principal is not None else "unknown"
    key_surface: Surface | None = None
    if api_key_id is not None:
        pool = await database.get_db()
        row = await pool.fetchrow(
            "SELECT surface FROM api_keys WHERE id = $1", api_key_id
        )
        key_surface = parse_surface(row["surface"]) if row is not None else None
    base: Surface
    if key_surface is not None:
        base = key_surface
    elif auth_method == "jwt":
        base = "interactive"
    else:
        base = default_surface()
    raw_mode = request.headers.get(MODE_HEADER)
    header_mode = parse_surface(raw_mode)
    surface = downgrade(base, header_mode)
    source: Literal["key", "jwt", "default", "header"]
    if header_mode is not None and SURFACE_TRUST[header_mode] < SURFACE_TRUST[base]:
        source = "header"
    elif key_surface is not None:
        source = "key"
    elif auth_method == "jwt":
        source = "jwt"
    else:
        source = "default"
    return WriteContext(
        surface=surface,
        source=source,
        key_surface=key_surface,
        header_mode=header_mode,
        header_mode_coerced=is_coerced(raw_mode),
        harness=sanitize_harness(request.headers.get(HARNESS_HEADER)),
        api_key_id=api_key_id,
        auth_method=auth_method,
        user_id=user.id,
        session_key=f"key:{api_key_id}" if api_key_id else f"user:{user.id}",
        user_agent=(request.headers.get("user-agent") or "")[:USER_AGENT_MAX],
        engine=sanitize_harness(request.headers.get(ENGINE_HEADER)),
    )


# --- decisions and rejections ------------------------------------------------


def _dash(value: object) -> str:
    return "-" if value is None else str(value)


def log_decision(
    op: WriteOp,
    outcome: Literal["allowed", "queued", "rejected"],
    wctx: WriteContext,
    *,
    reason: str = "",
    candidate_ids: Sequence[int] = (),
    capture_mode: str | None = None,
) -> None:
    """Emit the one INFO line for a write-policy decision (no entry text)."""
    logger.info(
        "%s op=%s outcome=%s surface=%s source=%s key_surface=%s header_mode=%s"
        " header_mode_coerced=%d harness=%r engine=%r key_id=%s user_id=%s auth=%s"
        " capture_mode=%s candidate_ids=%s reason=%s ua=%r",
        WRITE_POLICY_MARKER,
        op,
        outcome,
        wctx.surface,
        wctx.source,
        _dash(wctx.key_surface),
        _dash(wctx.header_mode),
        int(wctx.header_mode_coerced),
        wctx.harness,
        wctx.engine,
        _dash(wctx.api_key_id),
        wctx.user_id,
        wctx.auth_method,
        _dash(capture_mode),
        ",".join(str(i) for i in candidate_ids) or "-",
        reason,
        wctx.user_agent,
    )


REJECTION_TEMPLATES: dict[str, str] = {
    "update": (
        "updating an entry requires an interactive surface; this request is"
        " {surface}. Store a new entry instead and it is queued for review."
    ),
    "deactivate": (
        "deactivating an entry requires an interactive surface; this request is"
        " {surface}."
    ),
    "reactivate": (
        "reactivating an entry requires an interactive surface; this request is"
        " {surface}."
    ),
    "bulk_update": (
        "bulk_update with dry_run=false requires an interactive surface; this"
        " request is {surface}."
    ),
    "ingest": (
        "ingesting content requires an interactive surface; this request is"
        " {surface}. Store the entry with kb_store and it is queued for review."
    ),
    "supersedes": (
        "superseding entries requires an interactive surface; this request is"
        ' {surface}. Store the entry with supersedes "none" and it is queued for'
        " review."
    ),
    "distinct_from": (
        "distinct_from is not accepted from a {surface} surface; the candidate"
        " pipeline resolves near-duplicates."
    ),
    "mental_map": "mental_map entries cannot be stored from a {surface} surface.",
    "no_project": "a store from a {surface} surface needs a project_ref.",
}


def policy_rejection(
    op: WriteOp, reason: str, wctx: WriteContext, *, index: int | None = None
) -> HTTPException:
    """Build the 403 for *reason* and log the rejected decision."""
    detail = (
        WRITE_POLICY_DETAIL_PREFIX
        + (f"entry {index}: " if index is not None else "")
        + REJECTION_TEMPLATES[reason].format(surface=wctx.surface)
    )
    log_decision(op, "rejected", wctx, reason=reason)
    return HTTPException(status_code=403, detail=detail)


def require_interactive(
    op: Literal[
        "update",
        "deactivate",
        "reactivate",
        "bulk_update",
        "ingest_text",
        "ingest_url",
        "ingest_file",
    ],
    wctx: WriteContext,
) -> None:
    """Raise the policy 403 unless *wctx* is interactive; log the decision."""
    if wctx.surface != "interactive":
        raise policy_rejection(op, "ingest" if op.startswith("ingest_") else op, wctx)
    log_decision(op, "allowed", wctx)


# --- queueing ----------------------------------------------------------------

_DROPPED_HINT_KEYS = ("surprise_capture", "write_policy", "supersedes", "distinct_from")


def sanitize_store_hints(
    hints: dict[str, Any] | None, entry_type: EntryType
) -> dict[str, Any] | None:
    """Strip trust-bearing hints from a routed store; a resolution stays untrusted.

    Raises:
        ResolutionHintError: the resolution is malformed.
        RuntimeError: the post-condition tripwire (no row is inserted).
    """
    if hints is None:
        return None
    h = {k: v for k, v in hints.items() if k not in _DROPPED_HINT_KEYS}
    res = h.get("resolution")
    if isinstance(res, dict):
        h["resolution"] = {
            **res,
            "provenance": {"capture": "autonomous", "grounding": "asserted"},
            "observed_sessions": 1,
            "scope": "project",
        }
    result = validate_and_stamp_resolution(
        h, is_machine=True, entry_type=entry_type.value
    )
    if not result:
        return None
    out = result.get("resolution")
    bad = "surprise_capture" in result
    if out is not None:
        prov = out.get("provenance") if isinstance(out, dict) else None
        bad = (
            bad
            or prov != {"capture": "autonomous", "grounding": "asserted"}
            or out.get("observed_sessions") != 1
        )
    if bad:
        logger.error("write-policy tripwire=resolution_not_sanitized")
        raise RuntimeError("write policy: routed resolution was not sanitized")
    return result


_INSERT_STORE_CANDIDATE_SQL = (
    "INSERT INTO surprise_candidates (shape, session_id, project, turn_event_ids,"
    " detector_model, detector_output, status, entry_id, created_at)"
    " VALUES ($1, $2, $3, '[]', $4, $5, $6, NULL, $7) RETURNING id"
)


def _queued_surface(s: Surface) -> Literal["headless", "autonomous"]:
    """Narrow a non-interactive surface for the queued response."""
    if s == "headless" or s == "autonomous":
        return s
    raise RuntimeError("queue_store called for an interactive surface")


def _has_supersedes(supersedes: object, hints: dict[str, Any] | None) -> bool:
    if isinstance(supersedes, list) and supersedes:
        return True
    return (hints or {}).get("supersedes") not in (None, "none", [])


def _has_distinct_from(
    distinct_from: list[str] | None, hints: dict[str, Any] | None
) -> bool:
    return bool(distinct_from) or "distinct_from" in (hints or {})


def _detector_output(
    wctx: WriteContext,
    attr: Attribution,
    mode: SurpriseCaptureMode,
    *,
    op: Literal["store", "store_batch"],
    batch_index: int | None,
    request: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "store",
        "op": op,
        "batch_index": batch_index,
        "surface": wctx.surface,
        "source": wctx.source,
        "harness": wctx.harness,
        "engine": wctx.engine,
        "api_key_id": wctx.api_key_id,
        "user_id": wctx.user_id,
        "auth_method": wctx.auth_method,
        "contributor": attr.contributor,
        "team": attr.team,
        "capture_mode": mode,
        "request": request,
    }


def _request_obj(
    *,
    short_title: str,
    long_title: str,
    knowledge_details: str,
    entry_type: EntryType,
    project_ref: str,
    source_context: str | None,
    confidence_level: float | None,
    tags: list[str] | None,
    hints: dict[str, Any] | None,
    sensitivity: str | None,
    ttl: str | None,
) -> dict[str, Any]:
    return {
        "short_title": short_title,
        "long_title": long_title,
        "knowledge_details": knowledge_details,
        "entry_type": entry_type.value,
        "project_ref": project_ref,
        "source_context": source_context,
        "confidence_level": confidence_level,
        "tags": tags,
        "hints": hints,
        "sensitivity": sensitivity,
        "ttl": ttl,
    }


def _candidate_status(mode: SurpriseCaptureMode) -> str:
    return "pending" if mode == "on" else "shadow"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


async def queue_store(
    wctx: WriteContext,
    body: StoreRequest,
    *,
    entry_type: EntryType,
    attr: Attribution,
) -> StoreQueuedResponse:
    """Queue a non-interactive create as a shape-5 candidate (no KB write)."""
    mode = surprise_capture_mode()
    if entry_type is EntryType.MENTAL_MAP:
        raise policy_rejection("store", "mental_map", wctx)
    if body.project_ref is None or not body.project_ref.strip():
        raise policy_rejection("store", "no_project", wctx)
    if _has_supersedes(body.supersedes, body.hints):
        raise policy_rejection("store", "supersedes", wctx)
    if _has_distinct_from(body.distinct_from, body.hints):
        raise policy_rejection("store", "distinct_from", wctx)
    try:
        hints = sanitize_store_hints(body.hints, entry_type)
    except ResolutionHintError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    project = body.project_ref.strip()
    output = _detector_output(
        wctx,
        attr,
        mode,
        op="store",
        batch_index=None,
        request=_request_obj(
            short_title=body.short_title,
            long_title=body.long_title,
            knowledge_details=body.knowledge_details,
            entry_type=entry_type,
            project_ref=project,
            source_context=body.source_context,
            confidence_level=body.confidence_level,
            tags=body.tags,
            hints=hints,
            sensitivity=body.sensitivity,
            ttl=body.ttl,
        ),
    )
    pool = await database.get_db()
    row = await pool.fetchrow(
        _INSERT_STORE_CANDIDATE_SQL,
        STORE_CANDIDATE_SHAPE,
        wctx.session_key,
        project,
        STORE_CANDIDATE_DETECTOR_MODEL,
        json.dumps(output),
        _candidate_status(mode),
        _now(),
    )
    if row is None:
        raise RuntimeError("write policy: candidate insert returned no id")
    cand_id = int(row["id"])
    log_decision("store", "queued", wctx, candidate_ids=[cand_id], capture_mode=mode)
    return StoreQueuedResponse(
        status="queued",
        candidate_id=cand_id,
        surface=_queued_surface(wctx.surface),
        capture_mode=mode,
    )


async def queue_store_batch(
    wctx: WriteContext,
    body: StoreBatchRequest,
    *,
    attr: Attribution,
    skip_safety: bool,
) -> StoreBatchQueuedResponse:
    """Queue a non-interactive batch: validate every entry, then insert all."""
    mode = surprise_capture_mode()
    prepared: list[tuple[str, dict[str, Any]]] = []
    for i, raw in enumerate(body.entries):
        if raw.ttl is not None:
            try:
                compute_expires_at(raw.ttl)
            except ValueError as exc:
                detail = f"entry {i}: {exc}"
                raise HTTPException(status_code=422, detail=detail) from exc
        if not skip_safety:
            findings = detect_secrets_in_content(raw.knowledge_details)
            if findings:
                types = ", ".join(findings)
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"entry {i}: Secret scan detected sensitive content: {types}"
                    ),
                )
        if raw.entry_type is EntryType.MENTAL_MAP:
            raise policy_rejection("store_batch", "mental_map", wctx, index=i)
        if raw.project_ref is None or not raw.project_ref.strip():
            raise policy_rejection("store_batch", "no_project", wctx, index=i)
        if _has_supersedes(raw.supersedes, raw.hints):
            raise policy_rejection("store_batch", "supersedes", wctx, index=i)
        if _has_distinct_from(raw.distinct_from, raw.hints):
            raise policy_rejection("store_batch", "distinct_from", wctx, index=i)
        try:
            hints = sanitize_store_hints(raw.hints, raw.entry_type)
        except ResolutionHintError as exc:
            raise HTTPException(status_code=422, detail=f"entry {i}: {exc}") from exc
        project = raw.project_ref.strip()
        output = _detector_output(
            wctx,
            attr,
            mode,
            op="store_batch",
            batch_index=i,
            request=_request_obj(
                short_title=raw.short_title,
                long_title=raw.long_title,
                knowledge_details=raw.knowledge_details,
                entry_type=raw.entry_type,
                project_ref=project,
                source_context=raw.source_context,
                confidence_level=raw.confidence_level,
                tags=raw.tags,
                hints=hints,
                sensitivity=raw.sensitivity,
                ttl=raw.ttl,
            ),
        )
        prepared.append((project, output))
    pool = await database.get_db()
    ids: list[int] = []
    status = _candidate_status(mode)
    now = _now()
    async with pool.acquire() as conn, conn.transaction():
        for project, output in prepared:
            row = await conn.fetchrow(
                _INSERT_STORE_CANDIDATE_SQL,
                STORE_CANDIDATE_SHAPE,
                wctx.session_key,
                project,
                STORE_CANDIDATE_DETECTOR_MODEL,
                json.dumps(output),
                status,
                now,
            )
            ids.append(int(row["id"]))
    log_decision("store_batch", "queued", wctx, candidate_ids=ids, capture_mode=mode)
    return StoreBatchQueuedResponse(
        status="queued",
        requested=len(body.entries),
        candidate_ids=ids,
        surface=_queued_surface(wctx.surface),
        capture_mode=mode,
    )


# --- startup -----------------------------------------------------------------


async def log_startup(mode: SurpriseCaptureMode, *, worker_started: bool) -> None:
    """Log the policy's startup line, and warn when queued stores go unattended."""
    counts = {"interactive": 0, "headless": 0, "autonomous": 0, "unset": 0}
    keys: str
    try:
        pool = await database.get_db()
        rows = await pool.fetch(
            "SELECT surface, COUNT(*) AS n FROM api_keys GROUP BY surface"
        )
        for r in rows:
            name = r["surface"] if r["surface"] in SURFACES else "unset"
            counts[name] += int(r["n"])
        keys = (
            f"interactive:{counts['interactive']},headless:{counts['headless']},"
            f"autonomous:{counts['autonomous']},unset:{counts['unset']}"
        )
    except Exception:
        counts = dict.fromkeys(counts, 0)
        keys = "unavailable"
    default = default_surface()
    worker = "started" if worker_started else "not_started"
    logger.info(
        "write-policy started default_surface=%s capture_mode=%s worker=%s keys=%s",
        default,
        mode,
        worker,
        keys,
    )
    queueing = default != "interactive" or counts["headless"] + counts["autonomous"] > 0
    if not queueing:
        return
    reason: str | None = None
    if mode == "off":
        reason = "capture_off"
    elif not worker_started:
        reason = "worker_not_started"
    if reason is not None:
        logger.warning(
            "write-policy queue_unattended reason=%s default_surface=%s"
            " capture_mode=%s worker=%s",
            reason,
            default,
            mode,
            worker,
        )


# --- route classification ----------------------------------------------------

WRITE_ROUTE_CLASSES: dict[str, str] = {
    # policed: resolve_write_context runs in the handler
    "POST /api/kb/store": "policed",
    "POST /api/kb/store_batch": "policed",
    "POST /api/kb/entries/{entry_id}/deactivate": "policed",
    "POST /api/kb/entries/{entry_id}/reactivate": "policed",
    "POST /api/kb/bulk_update": "policed",
    "POST /api/kb/ingest/text": "policed",
    "POST /api/kb/ingest/url": "policed",
    "POST /api/kb/ingest/file": "policed",
    # admin_only
    "POST /api/kb/admin/reconcile-supersession": "admin_only",
    "POST /api/kb/map-eligibility/override": "admin_only",
    "POST /api/kb/map-eligibility/override/clear": "admin_only",
    # machine_principal_only
    "POST /api/kb/map-op": "machine_principal_only",
    "DELETE /api/kb/maps/{map_id}": "machine_principal_only",
    # pipeline
    "POST /api/kb/surprise/drain": "pipeline",
    # no_entry_write
    "POST /api/kb/search": "no_entry_write",
    "POST /api/kb/get": "no_entry_write",
    "POST /api/kb/ask": "no_entry_write",
    "POST /api/kb/summarize": "no_entry_write",
    "POST /api/kb/query/stream": "no_entry_write",
    "POST /api/kb/feedback": "no_entry_write",
    "POST /api/kb/cluster-ledger/match": "no_entry_write",
    "POST /api/kb/cluster-ledger/decline": "no_entry_write",
    "POST /api/kb/listener": "no_entry_write",
    "POST /api/kb/event": "no_entry_write",
    "POST /api/kb/turn": "no_entry_write",
    "POST /api/kb/prevention/decisions": "no_entry_write",
    "POST /api/kb/telemetry/whispers": "no_entry_write",
    "POST /api/kb/map-lint": "no_entry_write",
    "POST /api/kb/pointer-candidates": "no_entry_write",
}
