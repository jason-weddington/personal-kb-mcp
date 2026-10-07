"""Near-duplicate guard for kb_store create (``/store`` and ``/store_batch``).

A create whose text is at or above a cosine floor to an existing same-project
entry is rejected with 409 unless EVERY such candidate is covered by this
request's ``supersedes`` or ``distinct_from``. The guard fails open when no
embedding is available or the search fails, and is skipped for mental_map
entries and for entries without a project.

Every guarded create emits one greppable log line and one ``audit_events`` row
(``near_duplicate_checked``) so the floor can be retuned from real traffic.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import HTTPException
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.near_duplicates import NearDuplicateCheck

from kb_service.config import get_near_duplicate_floor

logger = logging.getLogger(__name__)

NEAR_DUPLICATE_GUARD_MARKER = "near-duplicate-guard"

_KB_ID_RE = re.compile(r"kb-\d{5}")
_DISTINCT_SHAPE = (
    "distinct_from rejected: hints.distinct_from must be a kb-id string or a list "
    "of kb-id strings"
)


def _norm_ids(value: object) -> list[object]:
    """None -> [], list -> list, anything else -> [value]."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def collect_distinct_from(
    field_value: list[str] | None, hints: dict[str, Any] | None
) -> list[str]:
    """Union of the ``distinct_from`` field and ``hints.distinct_from``, sorted.

    Raises:
        HTTPException: 422 when ``hints.distinct_from`` holds a non-str item.
    """
    items = _norm_ids((hints or {}).get("distinct_from"))
    if not all(isinstance(i, str) for i in items):
        raise HTTPException(status_code=422, detail=_DISTINCT_SHAPE)
    return sorted(set(field_value or []) | {str(i) for i in items})


async def check_distinct_from(
    kb: Any, ids: list[str], *, supersedes: list[str]
) -> list[str]:
    """Return one problem string per invalid ``distinct_from`` id (``[]`` if valid).

    Cross-project ids are allowed.
    """
    problems: list[str] = []
    for target in ids:
        if not _KB_ID_RE.fullmatch(target):
            problems.append(f"{target} is not a valid entry id")
            continue
        entry = await kb.get(target)
        if entry is None:
            problems.append(f"{target} not found")
        elif not entry.is_active:
            problems.append(f"{target} is inactive")
        elif entry.entry_type is EntryType.MENTAL_MAP:
            problems.append(f"{target} is a mental_map")
        elif target in supersedes:
            problems.append(f"{target} cannot be both superseded and distinct")
    return problems


@dataclass
class GuardDecision:
    """Outcome of one guard run; ``detail`` is the audit_events payload."""

    outcome: str
    detail: dict[str, Any]


def _text_sha(short_title: str, long_title: str, knowledge_details: str) -> str:
    text = KnowledgeEntry.compose_embedding_text(
        short_title, long_title, knowledge_details
    )
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _build_detail(
    *,
    op: str,
    entry_index: int | None,
    outcome: str,
    project_ref: str | None,
    floor: float,
    check: NearDuplicateCheck | None,
    resolved_by: dict[str, str],
    distinct_from: list[str],
    text_sha: str,
    short_title: str,
) -> dict[str, Any]:
    candidates = [[c.id, c.similarity] for c in check.candidates] if check else []
    candidate_ids = {str(c[0]) for c in candidates}
    return {
        "op": op,
        "entry_index": entry_index,
        "outcome": outcome,
        "project_ref": project_ref,
        "floor": floor,
        "top_similarity": check.top_similarity if check else None,
        "raw_hits": check.raw_hits if check else 0,
        "eligible": check.eligible_count if check else 0,
        "embed_ms": check.embed_ms if check else 0,
        "search_ms": check.search_ms if check else 0,
        "candidates": candidates,
        "resolved_by": resolved_by,
        "distinct_from_unused": sorted(set(distinct_from) - candidate_ids),
        "text_sha": text_sha,
        "short_title": short_title,
    }


def _log_decision(detail: dict[str, Any], contributor: str | None) -> None:
    outcome = detail["outcome"]
    level = (
        logging.WARNING
        if outcome in ("embedder_unavailable", "search_failed")
        else logging.INFO
    )
    top = detail["top_similarity"]
    logger.log(
        level,
        f"{NEAR_DUPLICATE_GUARD_MARKER} op=%s entry_index=%s outcome=%s project_ref=%s "
        "floor=%.4f top_similarity=%s raw_hits=%d eligible=%d embed_ms=%d search_ms=%d "
        "candidates=%r resolved_by=%r distinct_from_unused=%r contributor=%r "
        "short_title=%r text_sha=%s",
        detail["op"],
        detail["entry_index"],
        outcome,
        detail["project_ref"],
        detail["floor"],
        "none" if top is None else f"{top:.4f}",
        detail["raw_hits"],
        detail["eligible"],
        detail["embed_ms"],
        detail["search_ms"],
        [tuple(c) for c in detail["candidates"]],
        detail["resolved_by"],
        detail["distinct_from_unused"],
        contributor,
        detail["short_title"],
        detail["text_sha"],
    )


async def _write_audit(
    kb: Any, detail: dict[str, Any], *, entry_id: str | None, contributor: str | None
) -> None:
    await kb.record_audit_event(
        "near_duplicate_checked",
        entry_id=entry_id,
        contributor=contributor,
        detail=json.dumps(detail),
    )


async def record_stored(
    kb: Any, decision: GuardDecision, *, entry_id: str, contributor: str | None
) -> None:
    """After a successful store: write the audit row and the stored log line."""
    await _write_audit(kb, decision.detail, entry_id=entry_id, contributor=contributor)
    logger.info(
        f"{NEAR_DUPLICATE_GUARD_MARKER}-stored op=%s entry_index=%s entry_id=%s "
        "text_sha=%s outcome=%s",
        decision.detail["op"],
        decision.detail["entry_index"],
        entry_id,
        decision.detail["text_sha"],
        decision.outcome,
    )


async def enforce_near_duplicate_guard(
    kb: Any,
    *,
    op: Literal["store", "store_batch"],
    entry_index: int | None,
    contributor: str | None,
    short_title: str,
    long_title: str,
    knowledge_details: str,
    entry_type: EntryType,
    project_ref: str | None,
    supersedes: list[str],
    distinct_from: list[str],
) -> GuardDecision:
    """Run the guard for one create; raise 409 on an uncovered near-duplicate."""
    floor = get_near_duplicate_floor()
    text_sha = _text_sha(short_title, long_title, knowledge_details)

    def finish(
        outcome: str,
        check: NearDuplicateCheck | None = None,
        resolved_by: dict[str, str] | None = None,
    ) -> GuardDecision:
        detail = _build_detail(
            op=op,
            entry_index=entry_index,
            outcome=outcome,
            project_ref=project_ref,
            floor=floor,
            check=check,
            resolved_by=resolved_by or {},
            distinct_from=distinct_from,
            text_sha=text_sha,
            short_title=short_title,
        )
        _log_decision(detail, contributor)
        return GuardDecision(outcome=outcome, detail=detail)

    if entry_type is EntryType.MENTAL_MAP:
        return finish("exempt_mental_map")
    if project_ref is None or not project_ref.strip():
        return finish("skipped_no_project")

    check = await kb.find_near_duplicates(
        short_title=short_title,
        long_title=long_title,
        knowledge_details=knowledge_details,
        project_ref=project_ref,
        floor=floor,
        limit=5,
    )
    if check.status != "checked":
        return finish(check.status, check)

    covered = set(supersedes) | set(distinct_from)
    unresolved = [c for c in check.candidates if c.id not in covered]
    if not check.candidates:
        return finish("clear", check)
    if not unresolved:
        resolved_by = {
            c.id: "supersedes" if c.id in supersedes else "distinct_from"
            for c in check.candidates
        }
        return finish("resolved", check, resolved_by)

    decision = finish("conflict", check)
    await _write_audit(kb, decision.detail, entry_id=None, contributor=contributor)
    prefix = f"entry {entry_index}: " if op == "store_batch" else ""
    msg = (
        prefix
        + f"near-duplicate: this new entry is at or above cosine {floor:.2f} "
        + f"to existing entries in project {project_ref}: "
        + "; ".join(
            f'{c.id} "{c.short_title}" ({c.similarity:.3f})' for c in unresolved
        )
        + ". Cover EVERY listed id with one of: update_entry_id=<id> (same fact: "
        "update that entry, with change_reason); supersedes=[<id>] (the new entry "
        "replaces it; older clients: hints={'supersedes': [<id>]}); "
        "distinct_from=[<id>] (genuinely different facts; older clients: "
        "hints={'distinct_from': [<id>]})."
    )
    detail: dict[str, Any] = {
        "error": "near_duplicate",
        "message": msg,
        "project_ref": project_ref,
        "floor": floor,
        "candidates": [
            {
                "id": c.id,
                "short_title": c.short_title,
                "entry_type": c.entry_type,
                "similarity": c.similarity,
                "updated_at": c.updated_at,
            }
            for c in unresolved
        ],
    }
    if op == "store_batch":
        detail["entry_index"] = entry_index
    raise HTTPException(status_code=409, detail=detail)
