"""Listener gate: POST /api/kb/listener.

Surfaces up to :data:`kb_service.models.MAX_POINTERS_PER_RESPONSE` KB
mental-map pointers when an AI agent's working context matches one or more
orientation maps with sufficient confidence (majority-of-3 Sonnet votes over
a SET of candidate ids, rules A + B candidate filtering, an evidence bar for
any pointer beyond the first).

THE REFRAME (GTD 66ea1fe4, Jason 2026-09-19): a map is a SUBJECT AREA; detail
entries are the facts inside it. When several high-quality detail hits
resolve to more than one owning map, more than one map is relevant and both
should surface — capped at
:data:`kb_service.models.MAX_POINTERS_PER_RESPONSE` (2) so the one-line
whisper doesn't regrow into a roster. A second (or later) pointer is only
ever emitted when it clears its OWN evidence bar (:func:`_meets_second_slot_bar`)
— it does not ride along just because a stronger candidate matched.

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
import json
import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple

from fastapi import APIRouter, Depends, Request
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.models.search import SearchQuery

from kb_service.auth import get_current_user
from kb_service.database import get_db
from kb_service.models import (
    MAX_POINTERS_PER_RESPONSE,
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
    " decision, reason, vote_shape"
    ") VALUES ($1, $2, $3, $4, $5, $6, $7, $8)"
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

# ── Plural-pointer output contract (all pinned literals per GTD 66ea1fe4) ──
# Majority threshold for the reframed set-returning vote: a candidate is
# emitted only when it appears in AT LEAST this many of the 3 concurrent
# voters' returned sets. (Unanimity across free-form SETS would abstain even
# more than the old single-id unanimous-3 gate; majority is the intentional
# loosening that also targets the abstain problem — see GTD 66ea1fe4.)
_VOTE_MAJORITY_THRESHOLD = 2
# Max ids a single voter's reply may contribute to its own set (mirrors the
# "at most 2" instruction in the gate prompt; enforced defensively here too
# in case a voter ignores the instruction).
_VOTE_MAX_IDS_PER_VOTER = MAX_POINTERS_PER_RESPONSE
# Evidence bar for any pointer BEYOND the first (best-evidence) one: it must
# independently earn its slot rather than simply riding along with a
# stronger candidate. A candidate clears the bar if EITHER:
#   (a) at least this many DISTINCT detail-entry hits resolve to it, or
#   (b) its single best detail hit ranked at or above (numerically <=) this
#       position in the detail search.
_SECOND_SLOT_MIN_HIT_COUNT = 2
_SECOND_SLOT_MAX_RANK = 3


class _CandidateEvidence(NamedTuple):
    """Per-candidate-map evidence, used only to gate non-primary pointers.

    ``hit_count`` — number of DISTINCT top-N detail hits that resolved to
    this map via the ``references`` edge (primary path), or ``1`` for a map
    surfaced by the direct-map-search fallback (it wasn't detail-hit-owned
    at all, so it stands alone). ``best_rank`` — the best (smallest) rank
    among those hits (primary path), or the map's 1-based position in the
    fallback search results (fallback path).
    """

    hit_count: int
    best_rank: int


def _meets_second_slot_bar(evidence: "_CandidateEvidence") -> bool:
    """True if a non-primary candidate has independent support (see AC)."""
    return (
        evidence.hit_count >= _SECOND_SLOT_MIN_HIT_COUNT
        or evidence.best_rank <= _SECOND_SLOT_MAX_RANK
    )


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
) -> tuple[list[KnowledgeEntry], bool, dict[str, _CandidateEvidence]]:
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

    Returns ``(candidate_maps, used_fallback, evidence)``, in candidate-rank
    order. ``evidence`` (GTD 66ea1fe4) maps each candidate's id to its
    :class:`_CandidateEvidence` — the hit-count/best-rank pair the caller
    uses to decide whether a NON-primary pointer earns its slot (see
    :func:`_meets_second_slot_bar`). For the fallback path, each returned
    map gets ``hit_count=1`` (it wasn't detail-hit-owned — it stands alone)
    and ``best_rank`` equal to its 1-based position in the fallback results.
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
        hit_details: dict[str, set[str]] = collections.defaultdict(set)
        for target, source in owning:
            rank = detail_rank.get(target)
            if rank is None:
                continue
            hit_details[source].add(target)
            if source not in best_rank or rank < best_rank[source]:
                best_rank[source] = rank

        if best_rank:
            ranked_map_ids = sorted(best_rank, key=lambda mid: (best_rank[mid], mid))
            candidates: list[KnowledgeEntry] = []
            evidence: dict[str, _CandidateEvidence] = {}
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
                    evidence[map_id] = _CandidateEvidence(
                        hit_count=len(hit_details[map_id]),
                        best_rank=best_rank[map_id],
                    )
            if candidates:
                return candidates, False, evidence

    # ── Fallback: detail-matching yielded zero candidate maps ───────────────
    # Mirrors the pre-fix retrieval verbatim so behaviour cannot regress.
    fallback_results, _ = await kb.search(
        SearchQuery(
            query=text,
            entry_type=EntryType.MENTAL_MAP,
            limit=_FALLBACK_MAP_SEARCH_LIMIT,
        )
    )
    fallback_entries = [r.entry for r in fallback_results]
    fallback_evidence = {
        e.id: _CandidateEvidence(hit_count=1, best_rank=i + 1)
        for i, e in enumerate(fallback_entries)
    }
    return fallback_entries, True, fallback_evidence


async def _record_listener_decision(
    *,
    session_id: str | None,
    cwd_project: str | None,
    source_kb: str,
    candidates_considered: int,
    reason: ListenerDecisionReason,
    fallback: bool = False,
    vote_shape: str = "",
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

    ``vote_shape`` (GTD 66ea1fe4) is a ``json.dumps``-encoded list of the
    three voters' raw candidate-id sets (e.g. ``'[["kb-1"],["kb-1","kb-2"],[]]'``),
    empty string ``""`` on every branch that never reached the LLM gate
    (kill-switch, no-candidates, rule-A/B, no-llm) — recorded so the
    reframed set-returning vote's effect on whisper rate is measurable
    directly from ``listener_decisions`` without re-deriving it from logs.

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
            vote_shape,
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
        "Reply with the SET of candidate subject areas implicated by what the"
        " agent wrote (comma-separated ids, AT MOST 2), or exactly NONE if"
        " none qualify. When in doubt about any one candidate: leave it out."
        " No other text."
    )
    blocks = _basic_blocks(entries)
    return f"{header}\n\n{context}\n\n{quote}\n\n{instruction}\n\n{blocks}\n\n{reply}"


def _parse_vote_set(text: str | None, valid_ids: set[str]) -> tuple[str, ...]:
    """Parse a single ``generate()`` response into a SET of candidate ids.

    Reframed (GTD 66ea1fe4) from the single-id-or-None ``run_gate15.py`` port:
    a voter now answers "which subset of candidates is implicated?" rather
    than "which single id, if any?". A ``generate()`` returning ``None``
    (provider failure — anthropic.py swallows exceptions) counts as an empty
    set, same as an explicit "NONE" reply. Ids not in ``valid_ids`` (the
    retrieve-and-cite invariant) are dropped. The result is capped at
    ``_VOTE_MAX_IDS_PER_VOTER``, keeping the FIRST-mentioned valid ids in
    case a voter ignores the "at most 2" instruction.

    Returns an ordered ``tuple`` (first-mention order), NOT a ``frozenset``:
    a real Python set/frozenset of strings iterates in an order that depends
    on ``PYTHONHASHSEED``, which would make the debug tie-break in the
    non-majority branch below silently non-reproducible across processes.
    Membership (``in``) and counting both work identically on a tuple.
    """
    if text is None:
        return ()
    stripped = text.strip()
    if stripped.upper().startswith("NONE"):
        return ()
    tokens = [t.strip() for t in stripped.replace("\n", ",").split(",")]
    ids = [t for t in tokens if t.startswith("kb-")]
    valid_in_order: list[str] = []
    for i in ids:
        if i in valid_ids and i not in valid_in_order:
            valid_in_order.append(i)
    return tuple(valid_in_order[:_VOTE_MAX_IDS_PER_VOTER])


@router.post("/listener", response_model=ListenerResponse)
async def listener(
    body: ListenerRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> ListenerResponse:
    """Listener gate: surface 0..MAX_POINTERS_PER_RESPONSE KB map pointers.

    Steps:
    1. Kill switch — return ``{pointer: null, pointers: []}`` if
       ``KB_LISTENER_ENABLED != 'TRUE'``.
    2. Retrieval — detail-match + owning-map resolution, ranked/deduped/capped
       at 5, falling back to a direct mental_map search on zero candidates
       (see :func:`_retrieve_candidate_maps`).
    3. Rule A — drop candidates whose ``project_ref == cwd_project``.
    4. Rule B — drop candidates whose ``operated_via`` hint is in ``body.operating``.
    5. Short-circuit — return null if no candidates remain or no LLM available.
    6. LLM gate — fire 3 concurrent ``kb.synthesis_llm.generate(prompt)`` calls,
       each returning a SET of candidate ids (GTD 66ea1fe4 reframe).
    7. Verdict — a candidate is emitted iff it appears in >= 2 of the 3
       returned sets (majority); ordered by evidence rank, any pointer past
       the first must also clear :func:`_meets_second_slot_bar`; result is
       capped at ``MAX_POINTERS_PER_RESPONSE``.

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
            pointers=[],
            reason="kill-switch: KB_LISTENER_ENABLED!=TRUE",
        )

    kb = request.app.state.kb

    # ── Candidate retrieval (detail-match + owning-map resolution) ───────────
    candidates, used_fallback, evidence = await _retrieve_candidate_maps(kb, body.text)
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
        return ListenerResponse(pointer=None, pointers=[], reason=reason)

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
        return ListenerResponse(
            pointer=None, pointers=[], reason="no-injection: LLM unavailable"
        )

    # ── LLM gate: 3 concurrent votes ─────────────────────────────────────────
    entries = candidates
    prompt = _build_prompt(body.text, source_kb, entries)
    valid_ids = {e.id for e in entries}

    raw_votes = await asyncio.gather(
        kb.synthesis_llm.generate(prompt),
        kb.synthesis_llm.generate(prompt),
        kb.synthesis_llm.generate(prompt),
    )

    vote_sets = [_parse_vote_set(v, valid_ids) for v in raw_votes]
    # Recorded verbatim (as sorted lists for reproducible JSON) so the
    # reframed vote's effect is measurable straight off listener_decisions.
    vote_shape = json.dumps([sorted(s) for s in vote_sets])

    # ── Majority verdict ──────────────────────────────────────────────────────
    # A candidate is emitted iff it appears in >= _VOTE_MAJORITY_THRESHOLD of
    # the 3 returned sets (a materially looser bar than the old "all 3
    # identical AND non-None" unanimity, by design — see module docstring).
    # Retrieve-and-cite invariant: every id here already came from valid_ids
    # (enforced inside _parse_vote_set).
    vote_counts = collections.Counter(mid for s in vote_sets for mid in s)
    majority_ids = {
        mid for mid, count in vote_counts.items() if count >= _VOTE_MAJORITY_THRESHOLD
    }

    # Evidence order: `entries` is already ranked best-evidence-first by
    # _retrieve_candidate_maps, so filtering it (rather than sorting
    # majority_ids some other way) preserves that order for free.
    ranked_majority = [e for e in entries if e.id in majority_ids]

    selected: list[KnowledgeEntry] = []
    for i, e in enumerate(ranked_majority):
        if len(selected) >= MAX_POINTERS_PER_RESPONSE:
            break
        if i == 0:
            # The single best-evidence majority winner is never second-guessed
            # by the evidence bar — it already cleared retrieval, Rules A/B,
            # and a real vote majority.
            selected.append(e)
            continue
        ev = evidence.get(e.id)
        if ev is not None and _meets_second_slot_bar(ev):
            selected.append(e)

    if selected:
        await _record_listener_decision(
            session_id=body.session_id,
            cwd_project=body.cwd_project,
            source_kb=source_kb,
            candidates_considered=len(candidates),
            reason="whispered",
            fallback=used_fallback,
            vote_shape=vote_shape,
        )
        reason_parts = [
            f"{e.id} ({'unanimous 3/3' if vote_counts[e.id] == 3 else 'majority 2/3'})"
            for e in selected
        ]
        return ListenerResponse(
            pointer=None,
            pointers=[
                ListenerPointer(id=e.id, short_title=e.short_title) for e in selected
            ],
            reason=f"matched {', '.join(reason_parts)}",
        )

    # ── No candidate reached majority ────────────────────────────────────────
    if vote_counts:
        best_id, best_count = vote_counts.most_common(1)[0]
        reason = (
            f"no-injection: no candidate reached majority "
            f"(best: {best_id} {best_count}/3)"
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
        vote_shape=vote_shape,
    )
    return ListenerResponse(pointer=None, pointers=[], reason=reason)
