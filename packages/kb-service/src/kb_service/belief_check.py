"""Mid-thought belief check: rank recorded wrong beliefs against agent text.

The KB half of inject point D (``proposals/inject-points.md``): a harness
sends the agent's latest text and reasoning, and this module scores every
trusted resolution's ``wrong_belief`` against it by embedding cosine
similarity, max over short chunks of the text. ``POST /api/kb/belief-check``
(``routes/belief_check_routes.py``) wraps :func:`check_beliefs`.

No model or LLM call is made here; the only external call is the configured
embedder, through ``kb.embed_batch``. wrong_belief vectors are cached in
process, keyed by ``(entry_id, updated_at, wrong_belief)``, and filled by a
single background warm task so a request embeds only its own chunks.

Every constant below is read from module globals at call time, so tests can
monkeypatch it.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from kb_service.models import BeliefCheckMatch, BeliefCheckReason
from kb_service.prevention import (
    Resolution,
    _gate_trusted,
    load_resolutions,
    provenance_label,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Calibrated on qwen3-embedding:0.6b, no instruction prefix: 14 agent sentences
# restating a wrong belief scored 0.66-0.90 (median about 0.83), and 0.85 keeps 4
# of them (0.853-0.901); the hardest of 13 hard negatives (the agent doing the
# corrected thing) scored 0.819; unrelated text at most 0.53. 0.85 is the lowest
# round value above every measured hard negative: precision first. Negated
# restatements of a wrong belief were not measured.
BELIEF_CHECK_THRESHOLD = 0.85
# MUST be bumped on any change to BELIEF_CHECK_THRESHOLD, the chunking rule, the
# candidate rule or the scoring rule. It is reported per response, so such a
# change needs no CONTRACT_VERSIONS bump.
BELIEF_CHECK_MATCHER_VERSION = 1
BELIEF_CHECK_TOP_K = 3
BELIEF_CHECK_MAX_CHUNKS = 16
BELIEF_CHECK_MIN_CHUNK_CHARS = 16
BELIEF_CHECK_MAX_CHUNK_CHARS = 300
BELIEF_CHECK_MAX_INPUT_CHARS = 4096
BELIEF_CHECK_MAX_CANDIDATES = 500
BELIEF_CHECK_DEADLINE_S = 0.9
BELIEF_CHECK_WARM_BATCH = 32

_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*\n\s*")

Vector = npt.NDArray[np.float64]
WarmState = Literal["none", "started", "joined", "busy"]


# --- chunking ---------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One sentence or short window of the request text."""

    source: Literal["reasoning", "text"]
    text: str


def _split_one(s: str) -> list[str]:
    s = s[-BELIEF_CHECK_MAX_INPUT_CHARS:]
    out: list[str] = []
    for raw in _SPLIT.split(s):
        p = " ".join(raw.split())
        while len(p) > BELIEF_CHECK_MAX_CHUNK_CHARS:
            k = p.rfind(" ", 0, BELIEF_CHECK_MAX_CHUNK_CHARS + 1)
            k = BELIEF_CHECK_MAX_CHUNK_CHARS if k <= 0 else k
            w, p = p[:k].rstrip(), p[k:].lstrip()
            if len(w) >= BELIEF_CHECK_MIN_CHUNK_CHARS:
                out.append(w)
        if len(p) >= BELIEF_CHECK_MIN_CHUNK_CHARS:
            out.append(p)
    return out


def chunk_text(text: str, reasoning: str | None) -> list[Chunk]:
    """Split reasoning then text into sentence chunks.

    Exact duplicates keep their last occurrence, and only the last
    ``BELIEF_CHECK_MAX_CHUNKS`` chunks are kept.
    """
    chunks: list[Chunk] = []
    if reasoning is not None:
        chunks.extend(Chunk("reasoning", p) for p in _split_one(reasoning))
    chunks.extend(Chunk("text", p) for p in _split_one(text))
    last: dict[str, int] = {c.text: i for i, c in enumerate(chunks)}
    deduped = [c for i, c in enumerate(chunks) if last[c.text] == i]
    return deduped[-BELIEF_CHECK_MAX_CHUNKS:] if BELIEF_CHECK_MAX_CHUNKS > 0 else []


# --- candidates -------------------------------------------------------------


def select_candidates(resolutions: list[Resolution]) -> tuple[list[Resolution], int]:
    """Gate-trusted resolutions with a wrong_belief; ``(kept, dropped_by_cap)``."""
    eligible = [r for r in resolutions if r.wrong_belief.strip() and _gate_trusted(r)]
    kept = eligible[:BELIEF_CHECK_MAX_CANDIDATES]
    return kept, len(eligible) - len(kept)


# --- vectors and scoring ----------------------------------------------------


def unit_vector(v: Sequence[float]) -> Vector | None:
    """``v`` normalised to unit length; ``None`` when it is unusable."""
    try:
        arr = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or arr.size == 0 or not bool(np.all(np.isfinite(arr))):
        return None
    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or not math.isfinite(norm):
        return None
    out: Vector = arr / norm
    return out


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate's max cosine over the chunks and the chunk that gave it."""

    resolution: Resolution
    score: float
    chunk_index: int


def score_candidates(
    chunk_vectors: list[Vector], candidates: list[tuple[Resolution, Vector]]
) -> list[ScoredCandidate]:
    """Score each candidate by its max cosine over chunks, best first."""
    if not chunk_vectors or not candidates:
        return []
    chunks = np.vstack(chunk_vectors)
    beliefs = np.vstack([v for _, v in candidates])
    sims = beliefs @ chunks.T
    best = np.argmax(sims, axis=1)
    scored = [
        ScoredCandidate(r, float(sims[i, int(best[i])]), int(best[i]))
        for i, (r, _) in enumerate(candidates)
    ]
    scored.sort(key=lambda s: (-s.score, s.resolution.entry_id))
    return scored


# --- wrong_belief vector cache and the warm task ----------------------------

_VECTOR_CACHE: dict[str, tuple[str, str, Vector]] = {}
_warm_task: asyncio.Task[None] | None = None
_warm_ids: frozenset[str] = frozenset()


def reset_state() -> None:
    """Empty the vector cache and forget the warm task (tests, restarts)."""
    global _warm_task, _warm_ids
    _VECTOR_CACHE.clear()
    _warm_task = None
    _warm_ids = frozenset()


def cached_vector(r: Resolution) -> Vector | None:
    """The cached unit vector for *r*'s current wrong_belief, if any."""
    hit = _VECTOR_CACHE.get(r.entry_id)
    if hit is None:
        return None
    updated_at, belief, vec = hit
    if updated_at != r.updated_at or belief != r.wrong_belief.strip():
        return None
    return vec


def _store_vector(r: Resolution, raw: Sequence[float]) -> bool:
    vec = unit_vector(raw)
    if vec is None:
        return False
    _VECTOR_CACHE[r.entry_id] = (r.updated_at, r.wrong_belief.strip(), vec)
    return True


async def _warm(kb: Any, project: str, todo: list[Resolution]) -> None:
    """Embed *todo*'s wrong beliefs in batches, caching each batch as it lands."""
    started = time.monotonic()
    outcome = "ok"
    exc_name = ""
    stored = unusable = batches = 0
    try:
        i = 0
        while i < len(todo):
            size = max(1, BELIEF_CHECK_WARM_BATCH)
            batch = todo[i : i + size]
            i += len(batch)
            batches += 1
            vectors = await kb.embed_batch([r.wrong_belief.strip() for r in batch])
            if vectors is None:
                outcome = "none"
                break
            if len(vectors) != len(batch):
                outcome = "length-mismatch"
                break
            for r, raw in zip(batch, vectors, strict=True):
                if _store_vector(r, raw):
                    stored += 1
                else:
                    unusable += 1
    except asyncio.CancelledError:
        outcome = "cancelled"
        _log_warm(outcome, project, todo, stored, unusable, batches, started, "")
        raise
    except Exception as exc:
        outcome = "error"
        exc_name = type(exc).__name__
    _log_warm(outcome, project, todo, stored, unusable, batches, started, exc_name)


def _log_warm(
    outcome: str,
    project: str,
    todo: list[Resolution],
    stored: int,
    unusable: int,
    batches: int,
    started: float,
    exc_name: str,
) -> None:
    level = logging.INFO if outcome == "ok" else logging.WARNING
    logger.log(
        level,
        "belief_check_warm outcome=%s project=%s requested=%d stored=%d"
        " unusable=%d batches=%d ms=%d exc=%s",
        outcome,
        project,
        len(todo),
        stored,
        unusable,
        batches,
        int((time.monotonic() - started) * 1000),
        exc_name,
    )


def _ensure_warm(kb: Any, project: str, uncached: list[Resolution]) -> WarmState:
    """Start, join or defer to the single warm task for *uncached*."""
    global _warm_task, _warm_ids
    loop = asyncio.get_running_loop()
    if _warm_task is not None and (
        _warm_task.done() or _warm_task.get_loop() is not loop
    ):
        _warm_task = None
        _warm_ids = frozenset()
    if not uncached:
        return "none"
    ids = frozenset(r.entry_id for r in uncached)
    if _warm_task is not None:
        return "joined" if _warm_ids & ids else "busy"
    _warm_task = loop.create_task(_warm(kb, project, list(uncached)))
    _warm_ids = ids
    return "started"


# --- check_beliefs ----------------------------------------------------------


@dataclass(frozen=True)
class BeliefCheckOutcome:
    """What :func:`check_beliefs` found, plus the counters the route logs."""

    reason: BeliefCheckReason
    matches: tuple[BeliefCheckMatch, ...] = ()
    detail: str = ""
    warm: str = "none"
    candidates: int = 0
    candidates_truncated: int = 0
    uncached: int = 0
    scored: int = 0
    chunks: int = 0
    top_chunk_idx: int | None = None
    top_chunk_src: str = ""
    dim: int | None = None
    load_ms: int = 0
    embed_ms: int = 0


def _ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


def _tripwire_filter(project: str, candidates: list[Resolution]) -> list[Resolution]:
    """Drop (and report) candidates the trust rule should never have admitted."""
    once = [r.entry_id for r in candidates if r.observed_once]
    if once:
        logger.warning(
            "belief_check tripwire=observed_once_in_candidates project=%s ids=%s",
            project,
            once,
        )
    asserted = [
        r.entry_id
        for r in candidates
        if r.capture == "autonomous" and r.grounding != "observed"
    ]
    if asserted:
        logger.warning(
            "belief_check tripwire=asserted_in_candidates project=%s ids=%s",
            project,
            asserted,
        )
    bad = set(once) | set(asserted)
    return [r for r in candidates if r.entry_id not in bad]


async def check_beliefs(
    kb: Any, project: str, text: str, reasoning: str | None
) -> BeliefCheckOutcome:
    """Rank the project's recorded wrong beliefs against *text* and *reasoning*."""
    if kb.embedder is None:
        return BeliefCheckOutcome(reason="no-embedder")
    chunks = chunk_text(text, reasoning)
    if not chunks:
        return BeliefCheckOutcome(reason="no-chunks")
    n_chunks = len(chunks)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + BELIEF_CHECK_DEADLINE_S

    # (3) load and select candidates
    load_start = time.monotonic()
    try:
        async with asyncio.timeout_at(deadline) as cm:
            resolutions, _stats = await load_resolutions(
                kb.db, project, include_observed_once=False
            )
            candidates, truncated = select_candidates(resolutions)
    except TimeoutError:
        if not cm.expired():
            raise
        return BeliefCheckOutcome(
            reason="timeout", detail="load", chunks=n_chunks, load_ms=_ms(load_start)
        )
    load_ms = _ms(load_start)
    candidates = _tripwire_filter(project, candidates)
    n_cand = len(candidates)
    if not candidates:
        return BeliefCheckOutcome(
            reason="no-candidates",
            candidates_truncated=truncated,
            chunks=n_chunks,
            load_ms=load_ms,
        )

    # (4) warm task, then the chunk embed
    uncached = [r for r in candidates if cached_vector(r) is None]
    warm = _ensure_warm(kb, project, uncached)
    task = _warm_task

    def _fail(reason: BeliefCheckReason, detail: str) -> BeliefCheckOutcome:
        return BeliefCheckOutcome(
            reason=reason,
            detail=detail,
            warm=warm,
            candidates=n_cand,
            candidates_truncated=truncated,
            uncached=len(uncached),
            chunks=n_chunks,
            load_ms=load_ms,
            embed_ms=_ms(embed_start),
        )

    embed_start = time.monotonic()
    try:
        async with asyncio.timeout_at(deadline) as cm2:
            raw = await kb.embed_batch([c.text for c in chunks])
    except TimeoutError:
        if cm2.expired():
            return _fail("timeout", "chunks")
        return _fail("embed-failed", "chunks-raised")
    except Exception:
        return _fail("embed-failed", "chunks-raised")

    # (5) chunk vectors
    if raw is None:
        return _fail("embed-failed", "chunks-none")
    if len(raw) != len(chunks):
        return _fail("embed-failed", "chunks-length")
    dim: int | None = None
    chunk_vecs: list[Vector] = []
    chunk_idx: list[int] = []
    for i, v in enumerate(raw):
        u = unit_vector(v)
        if u is None:
            continue
        if dim is None:
            dim = int(u.shape[0])
        if u.shape[0] != dim:
            continue
        chunk_vecs.append(u)
        chunk_idx.append(i)
    if dim is None:
        return _fail("embed-failed", "chunks-unusable")

    # (6) wait for the warm task, never cancelling it
    if warm in ("started", "joined") and task is not None:
        await asyncio.wait({task}, timeout=max(0.0, deadline - loop.time()))

    # (7) score what is cached
    pairs: list[tuple[Resolution, Vector]] = []
    mismatched: list[str] = []
    cached_dim = 0
    for r in candidates:
        vec = cached_vector(r)
        if vec is None:
            continue
        if vec.shape[0] != dim:
            mismatched.append(r.entry_id)
            cached_dim = int(vec.shape[0])
            _VECTOR_CACHE.pop(r.entry_id, None)
            continue
        pairs.append((r, vec))
    if mismatched:
        logger.warning(
            "belief_check tripwire=dim_mismatch project=%s ids=%s"
            " cached_dim=%d chunk_dim=%d",
            project,
            mismatched,
            cached_dim,
            dim,
        )
    scored = score_candidates(chunk_vecs, pairs)
    embed_ms = _ms(embed_start)
    common: dict[str, Any] = {
        "warm": warm,
        "candidates": n_cand,
        "candidates_truncated": truncated,
        "uncached": len(uncached),
        "scored": len(scored),
        "chunks": n_chunks,
        "dim": dim,
        "load_ms": load_ms,
        "embed_ms": embed_ms,
    }
    if not scored:
        if warm in ("started", "joined") and task is not None and not task.done():
            return BeliefCheckOutcome(reason="warming", detail="own", **common)
        if warm == "busy":
            return BeliefCheckOutcome(reason="warming", detail="busy", **common)
        return BeliefCheckOutcome(
            reason="embed-failed", detail="no-cached-candidates", **common
        )

    # (8) top matches, including below-threshold ones
    top = scored[: max(0, BELIEF_CHECK_TOP_K)]
    matches = tuple(
        BeliefCheckMatch(
            entry_id=s.resolution.entry_id,
            score=s.score,
            wrong_belief=s.resolution.wrong_belief,
            corrected_fact=s.resolution.corrected_fact,
            provenance_label=provenance_label(s.resolution),
        )
        for s in top
    )
    first_chunk = chunk_idx[top[0].chunk_index] if top else None
    return BeliefCheckOutcome(
        reason="checked",
        matches=matches,
        top_chunk_idx=first_chunk,
        top_chunk_src=chunks[first_chunk].source if first_chunk is not None else "",
        **common,
    )
