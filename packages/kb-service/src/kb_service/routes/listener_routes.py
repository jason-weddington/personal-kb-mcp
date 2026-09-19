"""Listener gate: POST /api/kb/listener.

Surfaces a single KB mental-map pointer when an AI agent's working context
matches an orientation map with high confidence (unanimous-3 Sonnet votes,
rules A + B candidate filtering).

Kill switch: returns ``{"pointer": null}`` immediately when the env var
``KB_LISTENER_ENABLED`` is not set to ``'TRUE'`` (default ``'FALSE'`` — opt-in
pilot only; flip on the dev server when piloting).

Candidate retrieval (GTD bf40d4f1): mental maps are DELIBERATELY fact-free
(the map-purity lint rejects file paths, ENV_VAR tokens, dotted identifiers
and slash-joined paths from map bodies), so searching maps directly against a
real prompt cannot discriminate — the tokens a prompt contains are exactly
the tokens maps do not have. Retrieval instead searches the chunky DETAIL
entries (which are full of paths/identifiers/numbers by design), then
resolves each detail hit to its owning mental map(s) via the deterministic
``references`` graph edge (see :func:`_retrieve_candidate_maps`). When that
path yields zero candidate maps (e.g. a map with no outbound pointer edges),
the route falls back to today's direct mental_map search so behaviour cannot
regress below current — see :func:`_retrieve_candidate_maps` for detail.

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
from typing import TYPE_CHECKING, Annotated, Any

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

if TYPE_CHECKING:
    from kb_core import KnowledgeBase

router = APIRouter(prefix="/api/kb", tags=["kb"])

logger = logging.getLogger(__name__)

_DECISION_INSERT_SQL = (
    "INSERT INTO listener_decisions ("
    "session_id, cwd_project, source_kb, decided_ts, candidates_considered,"
    " decision, reason"
    ") VALUES ($1, $2, $3, $4, $5, $6, $7)"
)

# ── Candidate retrieval tuning (all pinned literals per GTD bf40d4f1) ───────
# Over-fetch bound for the detail-corpus search: the SearchQuery.limit max
# (le=50). We search with NO entry_type filter (the full corpus) then drop
# mental_map hits client-side, since SearchQuery only supports an entry_type
# EQUALITY filter, not an exclusion.
_DETAIL_SEARCH_FETCH_LIMIT = 50
# Pinned literal N: the top N non-map detail hits considered for owning-map
# resolution, after excluding entry_type='mental_map' from the fetch above.
_DETAIL_TOP_N = 20
# Pinned literal cap on the deduplicated candidate MAP list, applied before
# Rule A/B and the vote — keeps the gate prompt the size it is today.
_MAP_CANDIDATE_CAP = 5
# The fallback path mirrors today's pre-fix retrieval verbatim: direct
# mental_map search, limit=5.
_FALLBACK_MAP_SEARCH_LIMIT = 5


async def _owning_active_maps(db: Any, detail_ids: list[str]) -> list[tuple[str, str]]:
    """Reverse-resolve detail entry ids to their owning ACTIVE mental_map ids.

    One indexed query: ``graph_edges`` (``edge_type='references'``) joined to
    ``knowledge_entries`` so only ACTIVE ``mental_map`` sources are returned
    (deactivated maps, or edges whose source is some other entry type, never
    surface). Portable across SQLite/Postgres — ``?`` placeholders only (the
    Postgres backend's translator rewrites ``?`` -> ``$N`` and nothing else;
    see :mod:`kb_core.graph.queries` for the established convention).

    Returns a list of ``(detail_id, owning_map_id)`` pairs — i.e.
    ``(target, source)`` in graph-edge terms. A detail id with no matching
    row contributes nothing (caller treats that as "no owning map").
    """
    if not detail_ids:
        return []
    # The only interpolated text is a run of literal '?' placeholders (one per
    # detail id) -- the ids themselves are bound as query parameters below,
    # never spliced into the SQL string, so this is not an injection vector.
    placeholders = ",".join("?" for _ in detail_ids)
    sql = (
        "SELECT ge.target, ge.source FROM graph_edges ge"  # noqa: S608
        " JOIN knowledge_entries ke ON ke.id = ge.source"
        " WHERE ge.edge_type = 'references' AND ge.target IN ("
        + placeholders
        + ") AND ke.entry_type = 'mental_map' AND ke.is_active = 1"
    )
    cursor = await db.execute(sql, list(detail_ids))
    rows = await cursor.fetchall()
    return [(str(row[0]), str(row[1])) for row in rows]


async def _retrieve_candidate_maps(
    kb: "KnowledgeBase", text: str
) -> tuple[list[KnowledgeEntry], bool]:
    """Detail-match retrieval: find candidate mental maps via chunky details.

    1. Search the NON-map corpus for ``text`` (no entry_type restriction on
       the query itself — filtered client-side, see
       ``_DETAIL_SEARCH_FETCH_LIMIT``), take the top ``_DETAIL_TOP_N`` hits.
    2. Resolve each hit to its owning ACTIVE mental_map(s) via the
       ``references`` graph edge (:func:`_owning_active_maps`) — genuinely
       many-to-many: one detail can be owned by several maps, one map is
       typically hit via several details.
    3. Aggregate to a DEDUPLICATED set of candidate maps, ranked by the BEST
       (numerically smallest) rank of any detail that resolves to them —
       rank position, not the fused search score (which is a
       ``1/(60+rank)`` RRF artefact with no discriminating power). Ties
       break by map id ascending for reproducibility.
    4. Cap at ``_MAP_CANDIDATE_CAP``, then fetch the full entries.

    When step 3 yields zero candidate maps (e.g. one of the two active maps
    with no outbound ``references`` edge — unreachable by detail-matching by
    construction), falls back to today's direct ``mental_map`` search so
    behaviour cannot regress below current.

    Returns ``(candidate_maps, used_fallback)``, in candidate-rank order.
    """
    raw_results, _ = await kb.search(
        SearchQuery(query=text, limit=_DETAIL_SEARCH_FETCH_LIMIT)
    )
    detail_hits = [
        r.entry for r in raw_results if r.entry.entry_type != EntryType.MENTAL_MAP
    ][:_DETAIL_TOP_N]

    if detail_hits:
        detail_rank = {entry.id: i + 1 for i, entry in enumerate(detail_hits)}
        owning = await _owning_active_maps(kb.db, list(detail_rank))
        best_rank: dict[str, int] = {}
        for target, source in owning:
            rank = detail_rank.get(target)
            if rank is None:
                continue
            if source not in best_rank or rank < best_rank[source]:
                best_rank[source] = rank

        if best_rank:
            ranked_map_ids = sorted(best_rank, key=lambda mid: (best_rank[mid], mid))
            candidates: list[KnowledgeEntry] = []
            for map_id in ranked_map_ids[:_MAP_CANDIDATE_CAP]:
                entry = await kb.get(map_id)
                # Defense-in-depth: _owning_active_maps already filters to
                # active mental_map sources at the SQL level; re-check here
                # in case of a race (map deactivated between the two calls).
                if (
                    entry is not None
                    and entry.is_active
                    and entry.entry_type == EntryType.MENTAL_MAP
                ):
                    candidates.append(entry)
            if candidates:
                return candidates, False

    # ── Fallback: detail-matching yielded zero candidate maps ───────────────
    # Mirrors the pre-fix retrieval verbatim so behaviour cannot regress.
    fallback_results, _ = await kb.search(
        SearchQuery(
            query=text,
            entry_type=EntryType.MENTAL_MAP,
            limit=_FALLBACK_MAP_SEARCH_LIMIT,
        )
    )
    return [r.entry for r in fallback_results], True


async def _record_listener_decision(
    *,
    session_id: str | None,
    cwd_project: str | None,
    source_kb: str,
    candidates_considered: int,
    reason: ListenerDecisionReason,
    fallback: bool = False,
) -> None:
    """Best-effort insert of one ``listener_decisions`` row. Never raises.

    ``decision`` is derived from ``reason`` — "whisper" iff reason ==
    "whispered", "declined" otherwise — BEFORE the ``fallback`` override
    below, so it stays correct regardless of which retrieval path produced
    the decision.

    ``fallback=True`` means this decision was reached via the direct-search
    fallback path (see :func:`_retrieve_candidate_maps`): the stored
    ``reason`` is overridden to ``"fallback-direct"`` so the fallback is
    distinguishable in telemetry, at the cost of the granular per-branch
    reason for that subset of requests (decision itself is unaffected).

    A failure here (bad DSN, pool exhaustion, whatever) is logged at DEBUG
    and swallowed: this write must never fail or slow the listener response.
    """
    decision = "whisper" if reason == "whispered" else "declined"
    stored_reason: ListenerDecisionReason = "fallback-direct" if fallback else reason
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
            stored_reason,
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
    2. Retrieval — detail-match + owning-map resolution, ranked/deduped/capped
       at 5, falling back to a direct mental_map search on zero candidates
       (see :func:`_retrieve_candidate_maps`).
    3. Rule A — drop candidates whose ``project_ref == cwd_project``.
    4. Rule B — drop candidates whose ``operated_via`` hint is in ``body.operating``.
    5. Short-circuit — return null if no candidates remain or no LLM available.
    6. LLM gate — fire 3 concurrent ``kb.synthesis_llm.generate(prompt)`` calls.
    7. Verdict — all 3 votes identical and non-None → return pointer; else null.

    Every return path also fires a best-effort ``listener_decisions`` write
    (see :func:`_record_listener_decision`) so declines are no longer
    invisible — never affects the response, even if the write fails. Writes
    from the fallback retrieval path are recorded with the distinguishing
    ``reason="fallback-direct"``.
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

    # ── Candidate retrieval (detail-match + owning-map resolution) ───────────
    candidates, used_fallback = await _retrieve_candidate_maps(kb, body.text)
    n_retrieved = len(candidates)

    # ── Rule A: cross-project filter ──────────────────────────────────────────
    # Drop every candidate whose project_ref equals cwd_project.
    # When cwd_project is null, rule A drops nothing.
    if body.cwd_project is not None:
        candidates = [e for e in candidates if e.project_ref != body.cwd_project]
    n_after_a = len(candidates)

    # ── Rule B: operating context filter ─────────────────────────────────────
    # Drop candidates whose operated_via hint (string) is in body.operating.
    # Maps with no operated_via hint are never dropped by rule B.
    operating_set = set(body.operating)
    candidates = [
        e
        for e in candidates
        if not (
            isinstance(e.hints.get("operated_via"), str)
            and e.hints["operated_via"] in operating_set
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
            fallback=used_fallback,
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
            fallback=used_fallback,
        )
        return ListenerResponse(pointer=None, reason="no-injection: LLM unavailable")

    # ── LLM gate: 3 concurrent votes ─────────────────────────────────────────
    entries = candidates
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
            fallback=used_fallback,
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
        fallback=used_fallback,
    )
    return ListenerResponse(pointer=None, reason=reason)
