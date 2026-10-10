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

from kb_core.llm.provider import LLMProvider

from kb_service.config import build_anthropic_config
from kb_service.database import get_db
from kb_service.db_types import DbPool
from kb_service.models import SurpriseCaptureMode
from kb_service.surprise import (
    DETECTOR_MIN_CONFIDENCE,
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
    shape3_skip_reason,
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


def detector_min_confidence() -> float:
    """Read ``KB_SURPRISE_MIN_CONFIDENCE`` (0..1); invalid values give 0.7."""
    raw = os.environ.get("KB_SURPRISE_MIN_CONFIDENCE", "").strip()
    if raw == "":
        return DETECTOR_MIN_CONFIDENCE
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if math.isfinite(value) and 0.0 <= value <= 1.0:
        return value
    logger.warning("surprise_drain bad_min_confidence value=%r fallback=0.7", raw)
    return DETECTOR_MIN_CONFIDENCE


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
) -> DetectionRecord:
    model = detector_model_name(llm)
    details: dict[str, Any] = {"min_confidence": threshold}
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
) -> list[DetectionRecord]:
    """Run shapes 1, 2 and 3 on *cur*; return its detection records in order.

    *session_digests* holds the session's digests with ``turn_index`` up to
    the current one. Model calls run sequentially (shape 2, then shape 3).
    """
    threshold = (
        min_confidence if min_confidence is not None else detector_min_confidence()
    )
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
                threshold,
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
                llm, 3, build_shape3_prompt(cur), sources, [cur.event_id], threshold
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
    """Distill pending candidates into KB entries (hook; a no-op here).

    Contract for the implementation: ``drain_once`` calls this ONLY when mode
    is 'on', with every status='pending' candidate in id order. It is
    responsible for moving each row to rejected / written / merged and setting
    entry_id per the status semantics: pending = detected while
    KB_SURPRISE_CAPTURE was 'on', awaiting distillation; shadow = detected
    while KB_SURPRISE_CAPTURE was 'shadow', terminal, never passed here and
    never written to the KB (entry_id NULL); rejected = the distiller judged
    it not a durable correction (entry_id NULL); written = a new KB entry was
    created (entry_id = that entry's id); merged = folded into an existing
    entry (entry_id = that entry's id). There is no 'distilled' status.

    For any mode other than 'on' it returns ``DistillResult()`` with no DB
    write, no KB write and no LLM call (defence in depth, since drain_once
    never makes such a call). Candidates detected in shadow have status
    'shadow' and are never passed, so flipping shadow to on does not replay
    them. The distiller uses its own model getter, not ``get_detector_llm``.
    Shape-1 cue data is recoverable as tool 'Bash', target_class =
    ``kb_core.cues.target_class('Bash', detector_output['wrong_belief'])``,
    provenance.event_id = ``turn_event_ids[-1]``; shape-2/3 candidates carry
    no tool or target.
    """
    del pool, kb, candidates, mode
    return DistillResult()


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
    min_confidence = detector_min_confidence()
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
            llm, cur, session_digests, min_confidence=min_confidence
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
        "surprise_drain mode=%s detector_version=%d min_confidence=%.2f pruned=%d"
        " pending=%d digests=%d double_detect=%d new_candidates=%d shape1=%d"
        " shape2=%d shape3=%d llm_calls=%d llm_errors=%d unparseable=%d"
        " invalid=%d no_surprise=%d low_confidence=%d ungrounded=%d no_llm=%d"
        " truncated=%d out_of_order=%d turn_gaps=%d llm_ms=%d prompt_chars=%d"
        " response_chars=%d distill_input=%d written=%d merged=%d",
        mode,
        SURPRISE_DETECTOR_VERSION,
        min_confidence,
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
