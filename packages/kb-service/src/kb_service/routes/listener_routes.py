"""Listener gate: POST /api/kb/listener.

Surfaces a single KB mental-map pointer when an AI agent's working context
matches an orientation map with high confidence (unanimous-3 Sonnet votes,
rules A + B candidate filtering).

Kill switch: returns ``{"pointer": null}`` immediately when the env var
``KB_LISTENER_ENABLED`` is not set to ``'TRUE'`` (default ``'FALSE'`` — opt-in
pilot only; flip on the dev server when piloting).

Every request also writes a best-effort ``listener_decisions`` row (SERVICE
DB) recording WHY the request ended the way it did — including declines,
which previously left no durable trace at all (GTD 65b308de). The write is
fire-and-forget from the caller's perspective: any DB failure is logged and
swallowed so the listener response is never affected.
"""

import asyncio
import collections
import logging
import os
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.models.search import SearchQuery

from kb_service.auth import get_current_user
from kb_service.database import get_db
from kb_service.models import (
    ListenerDecisionReason,
    ListenerPointer,
    ListenerRequest,
    ListenerResponse,
    User,
)

router = APIRouter(prefix="/api/kb", tags=["kb"])

logger = logging.getLogger(__name__)

_DECISION_INSERT_SQL = (
    "INSERT INTO listener_decisions ("
    "session_id, cwd_project, source_kb, decided_ts, candidates_considered,"
    " decision, reason"
    ") VALUES ($1, $2, $3, $4, $5, $6, $7)"
)


async def _record_listener_decision(
    *,
    session_id: str | None,
    cwd_project: str | None,
    source_kb: str,
    candidates_considered: int,
    reason: ListenerDecisionReason,
) -> None:
    """Best-effort insert of one ``listener_decisions`` row. Never raises.

    ``decision`` is derived from ``reason`` — "whisper" iff reason ==
    "whispered", "declined" otherwise. A failure here (bad DSN, pool
    exhaustion, whatever) is logged at DEBUG and swallowed: this write must
    never fail or slow the listener response.
    """
    decision = "whisper" if reason == "whispered" else "declined"
    try:
        pool = await get_db()
        await pool.execute(
            _DECISION_INSERT_SQL,
            session_id,
            cwd_project,
            source_kb,
            datetime.now(UTC).isoformat(),
            candidates_considered,
            decision,
            reason,
        )
    except Exception as exc:  # best-effort telemetry: must never raise
        logger.debug("listener decision write failed: %s", exc)


def _basic_blocks(entries: list[KnowledgeEntry]) -> str:
    """Format KnowledgeEntry objects as candidate blocks.

    Mirrors ``run_gate15.py::basic_blocks`` (lines 81-82).  Each candidate
    renders as ``'### {id}: {short_title}'`` + newline + ``knowledge_details``
    truncated to 400 chars; candidates are joined by blank lines.
    """
    return "\n\n".join(
        f"### {e.id}: {e.short_title}\n{(e.knowledge_details or '')[:400]}"
        for e in entries
    )


def _build_prompt(text: str, source: str, entries: list[KnowledgeEntry]) -> str:
    """Build the gate prompt (verbatim text of ``prompt_base``, run_gate15.py:102-113).

    Uses the proven Gate-1.5 config A prompt.  Do NOT substitute
    ``prompt_contra`` (lines 116-127) — its richer-evidence variant backfired
    per kb-01725 v6.

    String literals are split across source lines for PEP-8 width compliance;
    the concatenated value is byte-identical to the original f-string.
    """
    # Sentence-wrapped for line-length compliance (E501); runtime value is
    # identical to prompt_base in run_gate15.py:102-113.
    header = (
        "You are a strict relevance gate for a knowledge-surfacing system."
        " A false positive (surfacing an irrelevant map) is far worse than"
        " a false negative (staying silent)."
    )
    context = f'An AI coding agent working in the project "{source}" wrote:'
    quote = f'"""{text}"""'
    instruction = (
        "Candidate orientation maps follow."
        " Pick a map ONLY IF the agent is ASSERTING OR ASSUMING SPECIFIC FACTS"
        " about that map's domain"
        " — facts the map's domain documentation would confirm or correct"
        " (network topology, which machine runs what, how a specific system is wired)."
        " Shared vocabulary, tooling mentions, or topical adjacency are NOT sufficient."
        " Ordinary in-project coding/debugging needs NO map."
    )
    reply = (
        "Reply with AT MOST ONE candidate id (the single most load-bearing),"
        " or exactly NONE. When in doubt: NONE. No other text."
    )
    blocks = _basic_blocks(entries)
    return f"{header}\n\n{context}\n\n{quote}\n\n{instruction}\n\n{blocks}\n\n{reply}"


def _parse_vote(text: str | None, valid_ids: set[str]) -> str | None:
    """Parse a single ``generate()`` response into a valid candidate id or None.

    Port of ``run_gate15.py::sonnet`` lines 158-166.  A ``generate()``
    returning ``None`` (provider failure — anthropic.py swallows exceptions)
    counts as a None vote.
    """
    if text is None:
        return None
    stripped = text.strip()
    if stripped.upper().startswith("NONE"):
        return None
    tokens = [t.strip() for t in stripped.replace("\n", ",").split(",")]
    ids = [t for t in tokens if t.startswith("kb-")]
    valid = [i for i in ids if i in valid_ids]
    return valid[0] if valid else None


@router.post("/listener", response_model=ListenerResponse)
async def listener(
    body: ListenerRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> ListenerResponse:
    """Listener gate: surface a single KB map pointer or return null.

    Steps:
    1. Kill switch — return ``{pointer: null}`` if ``KB_LISTENER_ENABLED != 'TRUE'``.
    2. Search — ``EntryType.MENTAL_MAP``, ``limit=5``, query = ``body.text``.
    3. Rule A — drop candidates whose ``project_ref == cwd_project``.
    4. Rule B — drop candidates whose ``operated_via`` hint is in ``body.operating``.
    5. Short-circuit — return null if no candidates remain or no LLM available.
    6. LLM gate — fire 3 concurrent ``kb.synthesis_llm.generate(prompt)`` calls.
    7. Verdict — all 3 votes identical and non-None → return pointer; else null.

    Every return path also fires a best-effort ``listener_decisions`` write
    (see :func:`_record_listener_decision`) so declines are no longer
    invisible — never affects the response, even if the write fails.
    """
    # source_kb identifies which KB context this decision is attributed to
    # (this service is single-KB-per-stack; mirrors the {source} substitution
    # already used in the gate prompt below).
    source_kb = body.source_label or body.cwd_project or "unknown"

    # ── Kill switch (default OFF) ─────────────────────────────────────────────
    # Mirrors config.py::is_agentic_ingest (config.py:159-161): read per-request.
    if os.environ.get("KB_LISTENER_ENABLED", "FALSE").upper() != "TRUE":
        await _record_listener_decision(
            session_id=body.session_id,
            cwd_project=body.cwd_project,
            source_kb=source_kb,
            candidates_considered=0,
            reason="kill-switch",
        )
        return ListenerResponse(
            pointer=None,
            reason="kill-switch: KB_LISTENER_ENABLED!=TRUE",
        )

    kb = request.app.state.kb

    # ── Candidate retrieval ───────────────────────────────────────────────────
    results, _ = await kb.search(
        SearchQuery(query=body.text, entry_type=EntryType.MENTAL_MAP, limit=5)
    )
    n_retrieved = len(results)

    # ── Rule A: cross-project filter ──────────────────────────────────────────
    # Drop every candidate whose project_ref equals cwd_project.
    # When cwd_project is null, rule A drops nothing.
    candidates = list(results)
    if body.cwd_project is not None:
        candidates = [r for r in candidates if r.entry.project_ref != body.cwd_project]
    n_after_a = len(candidates)

    # ── Rule B: operating context filter ─────────────────────────────────────
    # Drop candidates whose operated_via hint (string) is in body.operating.
    # Maps with no operated_via hint are never dropped by rule B.
    operating_set = set(body.operating)
    candidates = [
        r
        for r in candidates
        if not (
            isinstance(r.entry.hints.get("operated_via"), str)
            and r.entry.hints["operated_via"] in operating_set
        )
    ]
    n_after_b = len(candidates)  # noqa: F841 — per-stage counter pinned by AC; reads as 0 below

    # ── Short-circuit (split): no candidates ─────────────────────────────────
    # Attribute the drop using the per-stage counters so the debug `reason`
    # distinguishes no-retrieval / rule-A-emptied / rule-B-emptied.
    if not candidates:
        if n_retrieved == 0:
            reason = "no-injection: no candidates from retrieval"
            decision_reason: ListenerDecisionReason = "no-candidates"
        elif n_after_a == 0:
            reason = (
                f"no-injection: {n_retrieved} candidate(s), "
                f"all dropped by rule-A (cwd-project)"
            )
            decision_reason = "rule-a"
        else:
            reason = (
                f"no-injection: {n_after_a} candidate(s), "
                f"all dropped by rule-B (operating-manifest)"
            )
            decision_reason = "rule-b"
        await _record_listener_decision(
            session_id=body.session_id,
            cwd_project=body.cwd_project,
            source_kb=source_kb,
            candidates_considered=0,
            reason=decision_reason,
        )
        return ListenerResponse(pointer=None, reason=reason)

    # ── Short-circuit (split): candidates exist but no LLM ───────────────────
    if kb.synthesis_llm is None:
        await _record_listener_decision(
            session_id=body.session_id,
            cwd_project=body.cwd_project,
            source_kb=source_kb,
            candidates_considered=len(candidates),
            reason="no-llm",
        )
        return ListenerResponse(pointer=None, reason="no-injection: LLM unavailable")

    # ── LLM gate: 3 concurrent votes ─────────────────────────────────────────
    entries = [r.entry for r in candidates]
    prompt = _build_prompt(body.text, source_kb, entries)
    valid_ids = {e.id for e in entries}

    raw_votes = await asyncio.gather(
        kb.synthesis_llm.generate(prompt),
        kb.synthesis_llm.generate(prompt),
        kb.synthesis_llm.generate(prompt),
    )

    votes = [_parse_vote(v, valid_ids) for v in raw_votes]

    # ── Unanimous verdict ─────────────────────────────────────────────────────
    # All 3 votes must be identical AND non-None.
    # Retrieve-and-cite invariant: winner_id can only come from valid_ids.
    if votes[0] is not None and all(v == votes[0] for v in votes):
        winner_id = votes[0]
        winner_entry = next(e for e in entries if e.id == winner_id)
        await _record_listener_decision(
            session_id=body.session_id,
            cwd_project=body.cwd_project,
            source_kb=source_kb,
            candidates_considered=len(candidates),
            reason="whispered",
        )
        return ListenerResponse(
            pointer=ListenerPointer(id=winner_id, short_title=winner_entry.short_title),
            reason=f"matched {winner_id} (unanimous 3/3)",
        )

    # ── Non-unanimous fall-through ───────────────────────────────────────────
    # Derive (best_id, best_count) from non-None votes; covers BOTH the 2/3
    # split AND the 1-1-1 three-distinct-votes case (tie-break by Counter
    # insertion order — acceptable for a debug-only string). When all three
    # votes are None, no candidate received a vote at all.
    non_none = [v for v in votes if v is not None]
    if non_none:
        best_id, best_count = collections.Counter(non_none).most_common(1)[0]
        reason = (
            f"no-injection: best candidate {best_id} not unanimous ({best_count}/3)"
        )
        decision_reason = "vote-split"
    else:
        reason = "no-injection: no candidate received a vote (0/3)"
        decision_reason = "vote-none"
    await _record_listener_decision(
        session_id=body.session_id,
        cwd_project=body.cwd_project,
        source_kb=source_kb,
        candidates_considered=len(candidates),
        reason=decision_reason,
    )
    return ListenerResponse(pointer=None, reason=reason)
