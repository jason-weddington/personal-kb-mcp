"""Surprise capture: the drain over pending turn digests and its worker.

``drain_once`` reads pending ``turn_events`` digests through the
``kb_service.turn_digest`` helpers, runs the three detectors in
``kb_service.surprise`` on each, claims each digest (``processed_at``) and
records every decision in ``surprise_detections`` and every hit in
``surprise_candidates``. In mode ``on`` it then hands pending candidates to
``distill_candidates`` (the distillation hook). ``KB_SURPRISE_CAPTURE``, read
from the current environment at processing time, is the only switch on this
path.

``SurpriseCaptureWorker`` runs ``drain_once`` on a poll interval, but only
against a Postgres kb-core backend; local SQLite installs process digests
solely through ``POST /api/kb/surprise/drain``.
"""

import asyncio
import contextlib
import dataclasses
import json
import logging
import math
import os
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from kb_core.ingest.safety import detect_secrets_in_content, redact_secrets
from kb_core.llm.provider import LLMProvider
from kb_core.models.entry import EntryType

from kb_service.config import (
    NEAR_DUPLICATE_FLOOR_DEFAULT,
    build_anthropic_config,
    get_near_duplicate_floor,
)
from kb_service.database import get_db
from kb_service.db_types import DbPool
from kb_service.models import SurpriseCaptureMode
from kb_service.prevention import load_resolutions
from kb_service.resolution_hint import (
    ResolutionHintError,
    validate_and_stamp_resolution,
)
from kb_service.surprise import (
    DETECTOR_MIN_CONFIDENCE,
    DETECTOR_MIN_CONFIDENCE_BY_SHAPE,
    RAW_RESPONSE_EXCERPT_MAX,
    SHAPE1_DETECTOR_MODEL,
    SURPRISE_DETECTOR_SYSTEM,
    SURPRISE_DETECTOR_VERSION,
    DetectionRecord,
    DistillResult,
    NewCandidate,
    SurpriseCandidate,
    TurnDigest,
    build_shape2_prompt,
    build_shape3_prompt,
    detect_shape1,
    evidence_grounded,
    parse_detector_response,
    shape2_skip_reason,
    shape3_prompt_details,
    shape3_skip_reason,
)
from kb_service.surprise_distill import (
    DISTILL_CONFIDENCE_LEVEL,
    GATE_DENY_MARKER,
    REDACTION_MARKER,
    RESOLUTION_WRONG_BELIEF_MAX,
    SURPRISE_CONTRIBUTOR,
    SURPRISE_DISTILLER_SYSTEM,
    SURPRISE_DISTILLER_VERSION,
    SURPRISE_HINT_KEY,
    SURPRISE_TAG,
    DistillVerdict,
    build_distill_prompt,
    build_knowledge_details,
    build_resolution,
    find_exact_match,
    known_sessions,
    merge_block_reason,
    merged_surprise_hint,
    not_durable_reason,
    parse_distill_response,
    shape1_cue,
    stored_observed_sessions,
)
from kb_service.turn_digest import (
    get_session_turn_digests,
    list_pending_turn_digests,
    mark_turn_digests_processed,
    prune_turn_events,
    surprise_capture_mode,
)

logger = logging.getLogger(__name__)

SURPRISE_DETECTOR_DEFAULT_MODEL = "claude-sonnet-5-5"
SURPRISE_WORKER_POLL_SECONDS = 60.0

_DETECTOR_LLM_CACHE: dict[str, LLMProvider] = {}

SURPRISE_DISTILLER_DEFAULT_MODEL = "claude-sonnet-5-5"
_DISTILLER_LLM_CACHE: dict[str, LLMProvider] = {}

_CANDIDATE_COLUMNS = (
    "id, shape, session_id, project, turn_event_ids, detector_model,"
    " detector_output, status, entry_id, created_at"
)

_INSERT_CANDIDATE_SQL = (
    "INSERT INTO surprise_candidates (shape, session_id, project, turn_event_ids,"
    " detector_model, detector_output, status, entry_id, created_at)"
    " VALUES ($1, $2, $3, $4, $5, $6, $7, NULL, $8) RETURNING id"
)

_INSERT_DETECTION_SQL = (
    "INSERT INTO surprise_detections (event_id, session_id, project, shape, mode,"
    " outcome, reason, detector_model, detector_version, confidence,"
    " candidate_id, details, raw_response_excerpt, prompt_chars,"
    " response_chars, latency_ms, ts)"
    " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,"
    " $15, $16, $17)"
)

_FAILED_OUTCOMES = frozenset({"llm_error", "unparseable", "invalid_fields"})


# --- helpers -----------------------------------------------------------------


def get_detector_llm() -> LLMProvider | None:
    """Return the detector's own Anthropic client, cached per model.

    The model is ``KB_SURPRISE_DETECTOR_MODEL`` or
    ``SURPRISE_DETECTOR_DEFAULT_MODEL``. Never reads the KB's synthesis or
    query providers and ignores ``KB_QUERY_PROVIDER``. None (nothing cached)
    when the Anthropic client cannot be imported.
    """
    model = (
        os.environ.get("KB_SURPRISE_DETECTOR_MODEL", "").strip()
        or SURPRISE_DETECTOR_DEFAULT_MODEL
    )
    cached = _DETECTOR_LLM_CACHE.get(model)
    if cached is not None:
        return cached
    try:
        from kb_core.llm.anthropic import AnthropicLLMClient
    except ImportError:
        return None
    client = AnthropicLLMClient(build_anthropic_config(model=model))
    _DETECTOR_LLM_CACHE[model] = client
    return client


def get_distiller_llm() -> LLMProvider | None:
    """Return the distiller's own Anthropic client, cached per model.

    The model is ``KB_SURPRISE_DISTILL_MODEL`` or
    ``SURPRISE_DISTILLER_DEFAULT_MODEL``, read on every call. Independent of
    the detector: never calls ``get_detector_llm``, never reads
    ``KB_SURPRISE_DETECTOR_MODEL`` or the KB's synthesis/query providers, and
    ignores ``KB_QUERY_PROVIDER``. None (nothing cached) when the Anthropic
    client cannot be imported.
    """
    model = (
        os.environ.get("KB_SURPRISE_DISTILL_MODEL", "").strip()
        or SURPRISE_DISTILLER_DEFAULT_MODEL
    )
    cached = _DISTILLER_LLM_CACHE.get(model)
    if cached is not None:
        return cached
    try:
        from kb_core.llm.anthropic import AnthropicLLMClient
    except ImportError:
        return None
    client = AnthropicLLMClient(build_anthropic_config(model=model))
    _DISTILLER_LLM_CACHE[model] = client
    return client


def _read_floor(name: str) -> float | None:
    """Parse env var *name* as a 0..1 float; None when unset or invalid (warns)."""
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if math.isfinite(value) and 0.0 <= value <= 1.0:
        return value
    logger.warning("surprise_drain bad_min_confidence value=%r var=%s", raw, name)
    return None


def detector_min_confidence() -> float:
    """Global floor: ``KB_SURPRISE_MIN_CONFIDENCE`` (0..1), else 0.7."""
    value = _read_floor("KB_SURPRISE_MIN_CONFIDENCE")
    return DETECTOR_MIN_CONFIDENCE if value is None else value


def detector_min_confidence_for(shape: int) -> float:
    """Floor for *shape*: per-shape env, then global env, then shape default."""
    value = _read_floor(f"KB_SURPRISE_MIN_CONFIDENCE_SHAPE{shape}")
    if value is not None:
        return value
    value = _read_floor("KB_SURPRISE_MIN_CONFIDENCE")
    if value is not None:
        return value
    return DETECTOR_MIN_CONFIDENCE_BY_SHAPE.get(shape, DETECTOR_MIN_CONFIDENCE)


def detector_min_confidence_by_shape() -> dict[int, float]:
    """Floors for shapes 2 and 3, each env var read (and warned on) once."""
    glob = _read_floor("KB_SURPRISE_MIN_CONFIDENCE")
    out: dict[int, float] = {}
    for shape in (2, 3):
        value = _read_floor(f"KB_SURPRISE_MIN_CONFIDENCE_SHAPE{shape}")
        if value is None:
            value = glob
        if value is None:
            value = DETECTOR_MIN_CONFIDENCE_BY_SHAPE[shape]
        out[shape] = value
    return out


def detector_model_name(llm: Any) -> str:
    """The configured model id of *llm*, else its class name."""
    model = getattr(getattr(llm, "_config", None), "model", None)
    if isinstance(model, str) and model:
        return model
    return type(llm).__name__


def digest_from_row(row: Mapping[str, Any]) -> TurnDigest:
    """Build a ``TurnDigest`` from a ``turn_events``-shaped mapping."""
    raw_items = row.get("items")
    items: list[Any]
    if isinstance(raw_items, list):
        items = raw_items
    else:
        try:
            parsed = json.loads(raw_items) if isinstance(raw_items, str) else None
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            items = parsed
        else:
            logger.warning("surprise_drain bad_items event_id=%s", row.get("event_id"))
            items = []
    ts = row["ts"]
    return TurnDigest(
        event_id=row["event_id"],
        session_id=row["session_id"],
        project=row.get("project") or "",
        turn_index=int(row["turn_index"]),
        user_prompt=row.get("user_prompt"),
        items=[i for i in items if isinstance(i, dict)],
        final_message=row.get("final_message"),
        truncated=bool(row.get("truncated")),
        ts=ts if isinstance(ts, str) else ts.isoformat(),
    )


def candidate_from_row(row: Mapping[str, Any]) -> SurpriseCandidate:
    """Build a ``SurpriseCandidate`` from a ``surprise_candidates`` row."""
    event_ids = row["turn_event_ids"]
    output = row["detector_output"]
    return SurpriseCandidate(
        id=int(row["id"]),
        shape=int(row["shape"]),
        session_id=row["session_id"],
        project=row["project"],
        turn_event_ids=(
            json.loads(event_ids) if isinstance(event_ids, str) else event_ids
        ),
        detector_model=row["detector_model"],
        detector_output=json.loads(output) if isinstance(output, str) else output,
        status=row["status"],
        entry_id=row["entry_id"],
        created_at=row["created_at"],
    )


# --- per-digest detection ----------------------------------------------------


async def _model_record(
    llm: LLMProvider,
    shape: int,
    prompt: str,
    sources: list[str],
    turn_event_ids: list[str],
    threshold: float,
    *,
    extra_details: dict[str, Any] | None = None,
) -> DetectionRecord:
    model = detector_model_name(llm)
    details: dict[str, Any] = {"min_confidence": threshold, **(extra_details or {})}
    prompt_chars = len(SURPRISE_DETECTOR_SYSTEM) + len(prompt)
    start = time.monotonic()
    try:
        raw = await llm.generate(prompt, system=SURPRISE_DETECTOR_SYSTEM)
    except Exception:
        return DetectionRecord(
            shape=shape,
            outcome="llm_error",
            reason="exception",
            detector_model=model,
            details=details,
            prompt_chars=prompt_chars,
            response_chars=0,
            latency_ms=int((time.monotonic() - start) * 1000),
        )
    latency_ms = int((time.monotonic() - start) * 1000)
    common: dict[str, Any] = {
        "shape": shape,
        "detector_model": model,
        "details": details,
        "raw_response_excerpt": (raw or "")[:RAW_RESPONSE_EXCERPT_MAX] or None,
        "prompt_chars": prompt_chars,
        "response_chars": len(raw or ""),
        "latency_ms": latency_ms,
    }
    if raw is None:
        return DetectionRecord(outcome="llm_error", reason="none", **common)
    verdict, reject = parse_detector_response(raw)
    if verdict is None:
        return DetectionRecord(outcome=reject or "unparseable", reason="", **common)
    if verdict.confidence < threshold:
        return DetectionRecord(
            outcome="low_confidence",
            reason="",
            confidence=verdict.confidence,
            **common,
        )
    if not evidence_grounded(verdict.evidence_excerpt, sources):
        return DetectionRecord(
            outcome="ungrounded", reason="", confidence=verdict.confidence, **common
        )
    return DetectionRecord(
        outcome="candidate",
        reason="",
        confidence=verdict.confidence,
        candidate=NewCandidate(shape, turn_event_ids, model, verdict.to_output()),
        **common,
    )


async def detect_digest(
    llm: LLMProvider | None,
    cur: TurnDigest,
    session_digests: list[TurnDigest],
    *,
    min_confidence: float | None = None,
    min_confidence_by_shape: Mapping[int, float] | None = None,
) -> list[DetectionRecord]:
    """Run shapes 1, 2 and 3 on *cur*; return its detection records in order.

    *session_digests* holds the session's digests with ``turn_index`` up to
    the current one. Model calls run sequentially (shape 2, then shape 3).
    """

    def floor(shape: int) -> float:
        if min_confidence is not None:
            return min_confidence
        if min_confidence_by_shape is not None:
            return min_confidence_by_shape[shape]
        return detector_min_confidence_for(shape)

    records: list[DetectionRecord] = []

    shape1 = detect_shape1(session_digests, cur.event_id)
    if shape1.candidates:
        for cand in shape1.candidates:
            records.append(
                DetectionRecord(
                    shape=1,
                    outcome="candidate",
                    reason="",
                    detector_model=SHAPE1_DETECTOR_MODEL,
                    confidence=1.0,
                    candidate=cand,
                    details=dict(shape1.stats),
                )
            )
    else:
        records.append(
            DetectionRecord(
                shape=1,
                outcome="no_surprise",
                reason="",
                detector_model=SHAPE1_DETECTOR_MODEL,
                details=dict(shape1.stats),
            )
        )

    model = detector_model_name(llm) if llm is not None else ""

    prev = next(
        (d for d in session_digests if d.turn_index == cur.turn_index - 1), None
    )
    reason2 = shape2_skip_reason(prev, cur)
    if reason2 is not None or prev is None:
        reason2 = reason2 or "no_prev"
        records.append(DetectionRecord(2, "not_applicable", reason2, model))
    elif llm is None:
        records.append(DetectionRecord(2, "no_llm", "", model))
    else:
        records.append(
            await _model_record(
                llm,
                2,
                build_shape2_prompt(prev, cur),
                [cur.user_prompt or ""],
                [prev.event_id, cur.event_id],
                floor(2),
            )
        )

    reason3 = shape3_skip_reason(cur)
    if reason3 is not None:
        records.append(DetectionRecord(3, "not_applicable", reason3, model))
    elif llm is None:
        records.append(DetectionRecord(3, "no_llm", "", model))
    else:
        sources = [
            str(i.get("excerpt") or "")
            for i in cur.items
            if i.get("kind") == "tool_result"
        ]
        records.append(
            await _model_record(
                llm,
                3,
                build_shape3_prompt(cur),
                sources,
                [cur.event_id],
                floor(3),
                extra_details=shape3_prompt_details(cur),
            )
        )
    return records


# --- drain -------------------------------------------------------------------


@dataclass
class DrainResult:
    """What one ``drain_once`` call did."""

    digests_processed: int
    candidates: list[SurpriseCandidate]
    entries_written: list[str]
    entries_merged: list[str]


async def distill_candidates(
    pool: DbPool,
    kb: Any,
    candidates: list[SurpriseCandidate],
    mode: SurpriseCaptureMode,
) -> DistillResult:
    """Distill pending candidates into autonomous ``lesson_learned`` entries.

    Contract: ``drain_once`` calls this ONLY when mode is 'on', with every
    status='pending' candidate in id order. For any other mode (or no
    candidates) it returns ``DistillResult()`` with no DB write, no KB write
    and no LLM call. Candidates detected in shadow have status 'shadow', are
    never passed and are never distilled, so flipping shadow to on does not
    replay them; S0 also skips any non-pending candidate that is passed.

    Each candidate runs S0-S13 in order: the S0 claim pre-check; no_project;
    the redaction-marker and gate-deny-marker rejects; the failure-context
    lineage reject (a shape-1 wrong belief equal to the target of a
    ``failure_context`` / ``failure_context_repeat`` gate decision in the same
    session is ``gate_induced`` with reason ``failure_context``: a KB-assisted
    recovery must not corroborate the resolution it was handed); an exact match on
    (wrong belief, cue) against the project's resolutions, which merges
    without any LLM call; one distiller call (``get_distiller_llm``,
    ``KB_SURPRISE_DISTILL_MODEL``, default claude-sonnet-5-5, never
    ``get_detector_llm``); parse, secret scan and resolution validation; a
    cosine match at ``KB_NEAR_DUPLICATE_FLOOR``; then a merge (an
    ``observed_sessions`` bump, with the triggering event id appended to
    ``hints.surprise_capture.event_ids``, newest 20 kept) or a new entry.
    With no distiller LLM a candidate without an exact match stays pending.

    Status semantics: written = a new entry (entry_id = its id); merged =
    folded into an existing entry, same_session included (entry_id = that
    entry); rejected = no KB write for any terminal reason (see
    ``surprise_distillations.outcome``), not only a not-durable judgement.
    Every decision is recorded in ``surprise_distillations``. Persistence
    assumes a single writer per service DB (``surprise_drain_lock``), with
    S0 plus the conditional status UPDATE as tripwires. DB exceptions
    propagate.
    """
    if mode != "on" or not candidates:
        return DistillResult()
    result = DistillResult()
    llm = get_distiller_llm()
    counts: Counter[str] = Counter()
    aborted = 0
    try:
        try:
            floor = get_near_duplicate_floor()
        except ValueError:
            floor = NEAR_DUPLICATE_FLOOR_DEFAULT
            logger.warning(
                "surprise_distill bad_near_duplicate_floor fallback=%s", floor
            )
        for cand in candidates:
            row = await pool.fetchrow(_CANDIDATE_STATUS_SQL, cand.id)
            if row is None or row["status"] != "pending":
                logger.warning(
                    "surprise_distill tripwire=double_distill candidate_id=%d",
                    cand.id,
                )
                counts["double_distill"] += 1
                continue
            kb_answered = (
                frozenset(
                    str(r["target"]).strip()
                    for r in await pool.fetch(
                        _FAILURE_CONTEXT_TARGETS_SQL, cand.session_id
                    )
                )
                if cand.shape == 1
                else frozenset()
            )
            decision = _Decision()
            try:
                distilled = await _distill_one(
                    cand,
                    kb,
                    llm,
                    floor,
                    decision,
                    result,
                    counts,
                    kb_answered_targets=kb_answered,
                )
            except Exception as exc:
                decision.outcome = "kb_error"
                decision.reason = type(exc).__name__
                distilled = True
            if not distilled:
                counts["no_llm"] += 1
                continue
            await _persist_decision(pool, cand, decision, counts)
    except BaseException:
        aborted = 1
        raise
    finally:
        logger.info(
            "surprise_distill summary distiller_version=%d distiller_model=%s"
            " input=%d llm_calls=%d written=%d merged=%d same_session=%d"
            " covered=%d not_durable=%d redacted=%d gate_induced=%d"
            " llm_errors=%d unparseable=%d invalid=%d invalid_resolution=%d"
            " secret_detected=%d no_project=%d kb_error=%d no_llm=%d"
            " double_distill=%d exact_matches=%d cosine_matches=%d"
            " near_dup_unavailable=%d promoted=%d llm_ms=%d prompt_chars=%d"
            " response_chars=%d oldest_input_created_at=%s aborted=%d",
            SURPRISE_DISTILLER_VERSION,
            detector_model_name(llm) if llm is not None else "",
            len(candidates),
            counts["llm_calls"],
            counts["written"],
            counts["merged"],
            counts["same_session"],
            counts["covered"],
            counts["not_durable"],
            counts["redacted"],
            counts["gate_induced"],
            counts["llm_error"],
            counts["unparseable"],
            counts["invalid_fields"],
            counts["invalid_resolution"],
            counts["secret_detected"],
            counts["no_project"],
            counts["kb_error"],
            counts["no_llm"],
            counts["double_distill"],
            counts["exact_matches"],
            counts["cosine_matches"],
            counts["near_dup_unavailable"],
            counts["promoted"],
            counts["llm_ms"],
            counts["prompt_chars"],
            counts["response_chars"],
            min(c.created_at for c in candidates),
            aborted,
        )
    return result


# --- distillation internals --------------------------------------------------

_CANDIDATE_STATUS_SQL = "SELECT status FROM surprise_candidates WHERE id = $1"
_FAILURE_CONTEXT_TARGETS_SQL = (
    "SELECT target FROM gate_decisions WHERE session_id = $1"
    " AND decision IN ('failure_context', 'failure_context_repeat')"
)

_CLAIM_CANDIDATE_SQL = (
    "UPDATE surprise_candidates SET status = $1, entry_id = $2"
    " WHERE id = $3 AND status = 'pending'"
)

_INSERT_DISTILLATION_SQL = (
    "INSERT INTO surprise_distillations (candidate_id, session_id, project,"
    " shape, outcome, reason, entry_id, matched_entry_id, match_kind,"
    " similarity, near_duplicate_status, near_duplicate_floor,"
    " cue_target_class, observed_sessions_before, observed_sessions_after,"
    " verdict, distiller_model, distiller_version, raw_response_excerpt,"
    " prompt_chars, response_chars, latency_ms, ts)"
    " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,"
    " $15, $16, $17, $18, $19, $20, $21, $22, $23)"
)

_DISTILL_FAILED_OUTCOMES = frozenset(
    {
        "llm_error",
        "unparseable",
        "invalid_fields",
        "invalid_resolution",
        "secret_detected",
        "kb_error",
    }
)

_CANDIDATE_STATUS = {"written": "written", "merged": "merged", "same_session": "merged"}


@dataclass
class _Decision:
    """One candidate's distill decision: the ``surprise_distillations`` row."""

    outcome: str = ""
    reason: str = ""
    entry_id: str | None = None
    matched_id: str | None = None
    match_kind: str = ""
    similarity: float | None = None
    near_duplicate_status: str = ""
    near_duplicate_floor: float | None = None
    cue: dict[str, str] | None = None
    cue_target_class: str = ""
    before: int | None = None
    after: int | None = None
    verdict: DistillVerdict | None = None
    distiller_model: str = ""
    raw_response_excerpt: str | None = None
    prompt_chars: int | None = None
    response_chars: int | None = None
    latency_ms: int | None = None


async def _distill_one(
    c: SurpriseCandidate,
    kb: Any,
    llm: LLMProvider | None,
    floor: float,
    d: _Decision,
    result: DistillResult,
    counts: Counter[str],
    *,
    kb_answered_targets: frozenset[str] = frozenset(),
) -> bool:
    """Run S1-S13 for one candidate, filling *d*; False means a no_llm skip."""
    if c.project.strip() == "":
        d.outcome = "no_project"
        return True
    wrong_belief = str(c.detector_output.get("wrong_belief") or "").strip()[
        :RESOLUTION_WRONG_BELIEF_MAX
    ]
    if REDACTION_MARKER in wrong_belief:
        d.outcome, d.reason = "redacted", "wrong_belief"
        return True
    if GATE_DENY_MARKER in str(c.detector_output.get("evidence_excerpt") or ""):
        d.outcome = "gate_induced"
        return True
    if c.shape == 1 and wrong_belief != "" and wrong_belief in kb_answered_targets:
        d.outcome, d.reason = "gate_induced", "failure_context"
        return True
    cue = (
        shape1_cue(wrong_belief, str(c.detector_output.get("cue_target_class") or ""))
        if c.shape == 1
        else None
    )
    d.cue = cue
    d.cue_target_class = (cue or {}).get("target_class", "")

    resolutions, _ = await load_resolutions(kb.db, c.project, True)
    match = find_exact_match(resolutions, wrong_belief, cue)
    if match is not None:
        d.matched_id, d.match_kind = match.entry_id, "exact"
        counts["exact_matches"] += 1
        await _apply_match(c, kb, d, result)
        return True

    if llm is None:
        return False
    prompt = build_distill_prompt(c)
    counts["llm_calls"] += 1
    start = time.monotonic()
    raw: str | None
    try:
        raw = await llm.generate(prompt, system=SURPRISE_DISTILLER_SYSTEM)
    except Exception:
        raw, llm_reason = None, "exception"
    else:
        llm_reason = "none"
    d.latency_ms = int((time.monotonic() - start) * 1000)
    d.prompt_chars = len(SURPRISE_DISTILLER_SYSTEM) + len(prompt)
    d.response_chars = len(raw or "")
    d.distiller_model = detector_model_name(llm)
    redacted = redact_secrets(raw) if raw else None
    if redacted is not None:
        d.raw_response_excerpt = redacted[0][:RAW_RESPONSE_EXCERPT_MAX] or None
    counts["llm_ms"] += d.latency_ms
    counts["prompt_chars"] += d.prompt_chars
    counts["response_chars"] += d.response_chars

    verdict, reject = parse_distill_response(raw)
    if verdict is None:
        d.outcome = reject or "unparseable"
        if reject == "llm_error":
            d.reason = llm_reason
        elif reject == "not_durable":
            d.reason = not_durable_reason(raw)
        return True
    d.verdict = verdict
    if any(
        REDACTION_MARKER in text
        for text in (
            verdict.short_title,
            verdict.long_title,
            verdict.corrected_fact,
            verdict.lesson,
        )
    ):
        d.outcome, d.reason = "redacted", "verdict"
        return True

    resolution = build_resolution(c, verdict)
    details = build_knowledge_details(c, verdict, resolution)
    if not kb.config.ingest.skip_safety:
        findings = detect_secrets_in_content(
            "\n".join([verdict.short_title, verdict.long_title, details])
        )
        if findings:
            d.outcome, d.reason = "secret_detected", ",".join(findings)
            return True

    try:
        stamped = validate_and_stamp_resolution(
            {"resolution": resolution}, is_machine=True, entry_type="lesson_learned"
        )
    except ResolutionHintError as exc:
        d.outcome, d.reason = "invalid_resolution", exc.reason
        return True
    if stamped is None:
        d.outcome, d.reason = "invalid_resolution", "no_resolution"
        return True

    check = await kb.find_near_duplicates(
        short_title=verdict.short_title,
        long_title=verdict.long_title,
        knowledge_details=details,
        project_ref=c.project,
        floor=floor,
        limit=5,
    )
    d.near_duplicate_status = check.status
    d.near_duplicate_floor = floor
    if check.status != "checked":
        counts["near_dup_unavailable"] += 1
    elif check.candidates:
        top = check.candidates[0]
        d.matched_id, d.match_kind, d.similarity = top.id, "cosine", top.similarity
        counts["cosine_matches"] += 1
        await _apply_match(c, kb, d, result)
        return True

    entry = await kb.store(
        short_title=verdict.short_title,
        long_title=verdict.long_title,
        knowledge_details=details,
        entry_type=EntryType.LESSON_LEARNED,
        project_ref=c.project,
        source_context=(
            f"surprise_capture candidate {c.id} shape {c.shape} session {c.session_id}"
        ),
        confidence_level=DISTILL_CONFIDENCE_LEVEL,
        tags=[SURPRISE_TAG, f"shape-{c.shape}"],
        hints={**stamped, SURPRISE_HINT_KEY: merged_surprise_hint({}, c, new=True)},
        contributor=SURPRISE_CONTRIBUTOR,
        enrich=False,
    )
    result.entries_written.append(entry.id)
    d.outcome, d.entry_id, d.after = "written", entry.id, 1
    return True


async def _apply_match(
    c: SurpriseCandidate, kb: Any, d: _Decision, result: DistillResult
) -> None:
    """S12: merge into the matched entry, or record why it is not written."""
    entry = await kb.get(d.matched_id)
    if entry is None:
        d.outcome, d.reason = "covered", "missing"
        return
    res_obj = entry.hints.get("resolution")
    if not isinstance(res_obj, dict):
        d.outcome, d.reason = "covered", "no_resolution"
        return
    block = merge_block_reason(entry.hints, d.cue)
    if block is not None:
        d.outcome, d.reason = "covered", block
        return
    before = stored_observed_sessions(entry.hints)
    if c.session_id in known_sessions(entry.hints):
        d.outcome, d.entry_id = "same_session", entry.id
        d.before = d.after = before
        return
    new_res = dict(res_obj)
    new_res["observed_sessions"] = before + 1
    try:
        stamped = validate_and_stamp_resolution(
            {"resolution": new_res},
            is_machine=True,
            entry_type=entry.entry_type.value,
            existing_hints=entry.hints,
        )
    except ResolutionHintError as exc:
        d.outcome, d.reason = "invalid_resolution", exc.reason
        return
    if stamped is None:
        d.outcome, d.reason = "invalid_resolution", "no_resolution"
        return
    await kb.update(
        entry.id,
        hints={**stamped, SURPRISE_HINT_KEY: merged_surprise_hint(entry.hints, c)},
        change_reason=(
            f"surprise_capture: merged candidate {c.id} from session"
            f" {c.session_id}; observed_sessions {before}->{before + 1}"
        ),
        updated_by=SURPRISE_CONTRIBUTOR,
        enrich=False,
    )
    if entry.id not in result.entries_merged:
        result.entries_merged.append(entry.id)
    d.outcome, d.entry_id = "merged", entry.id
    d.before, d.after = before, before + 1


async def _persist_decision(
    pool: DbPool, c: SurpriseCandidate, d: _Decision, counts: Counter[str]
) -> None:
    """Move the candidate to its terminal status and insert its decision row."""
    status = _CANDIDATE_STATUS.get(d.outcome, "rejected")
    verdict = (
        json.dumps(dataclasses.asdict(d.verdict))
        if d.verdict is not None and d.outcome != "secret_detected"
        else None
    )
    inserted = False
    try:
        async with pool.acquire() as conn, conn.transaction():
            status_str = await conn.execute(
                _CLAIM_CANDIDATE_SQL, status, d.entry_id, c.id
            )
            if int(status_str.split()[-1]) != 0:
                await conn.execute(
                    _INSERT_DISTILLATION_SQL,
                    c.id,
                    c.session_id,
                    c.project,
                    c.shape,
                    d.outcome,
                    d.reason,
                    d.entry_id,
                    d.matched_id,
                    d.match_kind,
                    d.similarity,
                    d.near_duplicate_status,
                    d.near_duplicate_floor,
                    d.cue_target_class,
                    d.before,
                    d.after,
                    verdict,
                    d.distiller_model,
                    SURPRISE_DISTILLER_VERSION,
                    d.raw_response_excerpt,
                    d.prompt_chars,
                    d.response_chars,
                    d.latency_ms,
                    _now(),
                )
                inserted = True
    except Exception as exc:
        logger.warning(
            "surprise_distill persist_failed candidate_id=%d outcome=%s"
            " entry_id=%s exc=%s",
            c.id,
            d.outcome,
            d.entry_id,
            type(exc).__name__,
        )
        raise
    if not inserted:
        logger.warning("surprise_distill tripwire=double_distill candidate_id=%d", c.id)
        counts["double_distill"] += 1
        return
    counts[d.outcome] += 1
    cue = d.cue or {}
    if (
        d.outcome == "merged"
        and d.before == 1
        and cue.get("tool") == "Bash"
        and " " in cue.get("target_class", "")
    ):
        counts["promoted"] += 1
    logger.info(
        "surprise_distill decision candidate_id=%d shape=%d session_id=%s"
        " project=%s outcome=%s reason=%s match_kind=%s matched_entry_id=%s"
        " entry_id=%s similarity=%s observed_sessions=%s->%s cue=%s llm_ms=%d",
        c.id,
        c.shape,
        c.session_id,
        c.project,
        d.outcome,
        d.reason,
        d.match_kind,
        d.matched_id,
        d.entry_id,
        d.similarity,
        d.before,
        d.after,
        d.cue_target_class,
        d.latency_ms or 0,
    )
    if d.outcome in _DISTILL_FAILED_OUTCOMES:
        logger.warning(
            "surprise_distill distill_failed candidate_id=%d shape=%d outcome=%s"
            " reason=%s",
            c.id,
            c.shape,
            d.outcome,
            d.reason,
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


async def _fetch_candidates(pool: DbPool, ids: list[int]) -> list[SurpriseCandidate]:
    if not ids:
        return []
    placeholders = ", ".join(f"${i}" for i in range(1, len(ids) + 1))
    rows = await pool.fetch(
        f"SELECT {_CANDIDATE_COLUMNS} FROM surprise_candidates"  # noqa: S608
        f" WHERE id IN ({placeholders}) ORDER BY id",
        *ids,
    )
    return [candidate_from_row(r) for r in rows]


async def _insert_records(
    pool: DbPool,
    cur: TurnDigest,
    records: list[DetectionRecord],
    mode: SurpriseCaptureMode,
    flags: dict[str, bool],
    now: str,
) -> list[tuple[int, int]]:
    """Insert one digest's rows in one transaction; return (id, shape) pairs."""
    inserted: list[tuple[int, int]] = []
    async with pool.acquire() as conn, conn.transaction():
        for record in records:
            candidate_id: int | None = None
            if record.candidate is not None:
                cand = record.candidate
                row = await conn.fetchrow(
                    _INSERT_CANDIDATE_SQL,
                    cand.shape,
                    cur.session_id,
                    cur.project,
                    json.dumps(cand.turn_event_ids),
                    cand.detector_model,
                    json.dumps(cand.detector_output),
                    "pending" if mode == "on" else "shadow",
                    now,
                )
                candidate_id = int(row["id"])
                inserted.append((candidate_id, cand.shape))
            await conn.execute(
                _INSERT_DETECTION_SQL,
                cur.event_id,
                cur.session_id,
                cur.project,
                record.shape,
                mode,
                record.outcome,
                record.reason,
                record.detector_model,
                SURPRISE_DETECTOR_VERSION,
                record.confidence,
                candidate_id,
                json.dumps({**record.details, **flags}),
                record.raw_response_excerpt,
                record.prompt_chars,
                record.response_chars,
                record.latency_ms,
                now,
            )
    return inserted


async def drain_once(pool: DbPool, kb: Any) -> DrainResult:
    """Detect, claim and record every pending digest; distill in mode 'on'.

    Mode 'off' returns zeros without touching the DB. Otherwise: prune old
    digests first, then for each pending digest (by session, turn) run the
    detectors, claim it via ``mark_turn_digests_processed`` (the atomic
    per-row claim, deliberately outside the insert transaction) and insert
    its rows. No model call runs while a connection is held.
    """
    mode = surprise_capture_mode()
    if mode == "off":
        return DrainResult(0, [], [], [])

    pruned = await prune_turn_events(pool)
    floors = detector_min_confidence_by_shape()
    floor2, floor3 = floors[2], floors[3]
    llm = get_detector_llm()
    pending = await list_pending_turn_digests(pool)
    pending = sorted(pending, key=lambda d: (d.session_id, d.turn_index))

    c: Counter[str] = Counter()
    call_outcomes: list[str] = []
    any_no_llm = False
    inserted_ids: list[int] = []
    sessions: dict[str, list[Any]] = {}

    for p in pending:
        if p.session_id not in sessions:
            sessions[p.session_id] = await get_session_turn_digests(pool, p.session_id)
        stored = sessions[p.session_id]
        cur = digest_from_row(p.model_dump())
        session_digests = [
            digest_from_row(s.model_dump())
            for s in stored
            if s.turn_index <= p.turn_index
        ]
        flags = {
            "truncated": cur.truncated,
            "out_of_order": any(
                s.processed_at is not None and s.turn_index > p.turn_index
                for s in stored
            ),
            "turn_gap": p.turn_index > 0
            and not any(s.turn_index == p.turn_index - 1 for s in stored),
        }
        records = await detect_digest(
            llm, cur, session_digests, min_confidence_by_shape=floors
        )
        for rec in records:
            if rec.shape in (2, 3) and rec.outcome not in (
                "not_applicable",
                "no_llm",
            ):
                call_outcomes.append(rec.outcome)
                c["llm_ms"] += rec.latency_ms or 0
                c["prompt_chars"] += rec.prompt_chars or 0
                c["response_chars"] += rec.response_chars or 0
            if rec.outcome == "no_llm":
                any_no_llm = True
            if rec.shape in (2, 3) or rec.outcome != "no_surprise":
                c[rec.outcome] += 1
            if rec.outcome in _FAILED_OUTCOMES:
                logger.warning(
                    "surprise_drain detector_failed event_id=%s shape=%d"
                    " outcome=%s reason=%s",
                    cur.event_id,
                    rec.shape,
                    rec.outcome,
                    rec.reason,
                )

        now = _now()
        if await mark_turn_digests_processed(pool, [cur.event_id], now) != 1:
            logger.warning(
                "surprise_drain tripwire=double_detect event_id=%s", cur.event_id
            )
            c["double_detect"] += 1
            continue
        c["digests"] += 1
        for name, flag in (
            ("truncated", flags["truncated"]),
            ("out_of_order", flags["out_of_order"]),
            ("turn_gaps", flags["turn_gap"]),
        ):
            if flag:
                c[name] += 1
        try:
            inserted = await _insert_records(pool, cur, records, mode, flags, now)
        except Exception as exc:
            logger.error(
                "surprise_drain lost_after_claim event_id=%s exc=%s",
                cur.event_id,
                type(exc).__name__,
            )
            raise
        for cand_id, shape in inserted:
            inserted_ids.append(cand_id)
            c["new_candidates"] += 1
            c[f"shape{shape}"] += 1

    if llm is None and any_no_llm:
        logger.warning("surprise_drain no detector LLM; shapes 2-3 skipped")
    if call_outcomes and all(o == "llm_error" for o in call_outcomes):
        logger.warning(
            "surprise_drain detector unavailable llm_calls=%d", len(call_outcomes)
        )

    distilled_ids: list[int] = []
    dres = DistillResult()
    if mode == "on":
        rows = await pool.fetch(
            f"SELECT {_CANDIDATE_COLUMNS} FROM surprise_candidates"  # noqa: S608
            " WHERE status = 'pending' ORDER BY id"
        )
        to_distill = [candidate_from_row(r) for r in rows]
        if to_distill:
            distilled_ids = [cand.id for cand in to_distill]
            dres = await distill_candidates(pool, kb, to_distill, mode)

    ids = sorted(set(inserted_ids) | set(distilled_ids))
    candidates = await _fetch_candidates(pool, ids)

    logger.info(
        "surprise_drain mode=%s detector_version=%d min_confidence_s2=%.2f"
        " min_confidence_s3=%.2f pruned=%d"
        " pending=%d digests=%d double_detect=%d new_candidates=%d shape1=%d"
        " shape2=%d shape3=%d llm_calls=%d llm_errors=%d unparseable=%d"
        " invalid=%d no_surprise=%d low_confidence=%d ungrounded=%d no_llm=%d"
        " truncated=%d out_of_order=%d turn_gaps=%d llm_ms=%d prompt_chars=%d"
        " response_chars=%d distill_input=%d written=%d merged=%d",
        mode,
        SURPRISE_DETECTOR_VERSION,
        floor2,
        floor3,
        pruned,
        len(pending),
        c["digests"],
        c["double_detect"],
        c["new_candidates"],
        c["shape1"],
        c["shape2"],
        c["shape3"],
        len(call_outcomes),
        c["llm_error"],
        c["unparseable"],
        c["invalid_fields"],
        c["no_surprise"],
        c["low_confidence"],
        c["ungrounded"],
        c["no_llm"],
        c["truncated"],
        c["out_of_order"],
        c["turn_gaps"],
        c["llm_ms"],
        c["prompt_chars"],
        c["response_chars"],
        len(distilled_ids),
        len(dres.entries_written),
        len(dres.entries_merged),
    )
    return DrainResult(
        digests_processed=c["digests"],
        candidates=candidates,
        entries_written=list(dres.entries_written),
        entries_merged=list(dres.entries_merged),
    )


# --- background worker -------------------------------------------------------


def should_start_surprise_worker(
    mode: SurpriseCaptureMode, database_url: str | None
) -> bool:
    """True iff capture is shadow/on AND the kb-core backend is Postgres."""
    return mode in ("shadow", "on") and bool(database_url)


class SurpriseCaptureWorker:
    """Runs ``drain_once`` every poll interval under the shared drain lock."""

    def __init__(
        self,
        kb: Any,
        lock: asyncio.Lock,
        *,
        poll_interval_seconds: float = SURPRISE_WORKER_POLL_SECONDS,
    ) -> None:
        """Bind the KB and the in-process drain lock (shared with the route)."""
        self._kb = kb
        self._lock = lock
        self._poll = poll_interval_seconds
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        """True while the loop task exists and has not finished."""
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start the loop once (idempotent)."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run_forever())
        logger.info(
            "surprise_worker started mode=%s poll=%ss",
            surprise_capture_mode(),
            self._poll,
        )

    async def stop(self) -> None:
        """Cancel and await the loop; safe when never started."""
        task = self._task
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run_forever(self) -> None:
        while True:
            try:
                async with self._lock:
                    await drain_once(await get_db(), self._kb)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("surprise_worker drain failed")
            await asyncio.sleep(self._poll)
