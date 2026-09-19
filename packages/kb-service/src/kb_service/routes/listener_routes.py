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

Lexical candidate path (GTD be964e94): detail-matching cannot fire for any
map that owns no chunky detail entry pointing at it — measured on the live
service, only 3.8% of active entries are pointed at by any map. A project
name or map title is a distinctive, low-cardinality token; a prompt that
literally names one is strong, free, deterministic evidence that its
subject area is implicated, no LLM required. :func:`_retrieve_lexical_candidates`
runs ALONGSIDE detail-matching (never instead of it) and
:func:`_retrieve_candidates` merges the two signals into one deduplicated,
ranked candidate list before Rules A/B and the vote ever see it — see that
function's docstring for the merge/precedence rules.

Every request also writes a best-effort ``listener_decisions`` row (SERVICE
DB) recording WHY the request ended the way it did — including declines,
which previously left no durable trace at all (GTD 65b308de). The write is
fire-and-forget from the caller's perspective: any DB failure is logged and
swallowed so the listener response is never affected. The row also records
(GTD be964e94) WHICH candidate signal (lexical / detail / fallback) produced
a whispered pointer, via ``candidate_signal`` — see
:func:`_record_listener_decision`.
"""

import asyncio
import collections
import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple

from fastapi import APIRouter, Depends, Request
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.models.search import SearchQuery

from kb_service.auth import get_current_user
from kb_service.database import get_db
from kb_service.models import (
    MAX_POINTERS_PER_RESPONSE,
    ListenerCandidateSignal,
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
    " decision, reason, vote_shape, candidate_signal, candidate_ids,"
    " whispered_ids, n_retrieved, n_after_a, n_after_b, retrieval_path"
    ") VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,"
    " $14, $15)"
)

# retrieval_path values (GTD 268e2af3): whether the direct-mental_map-search
# FALLBACK ran (see _retrieve_candidate_maps), split out of `reason` so the
# granular decline cause is never masked by "fallback-direct" again.
_RETRIEVAL_PATH_PRIMARY = "primary"
_RETRIEVAL_PATH_FALLBACK = "fallback-direct"

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

# ── Lexical candidate path tuning (all pinned literals per GTD be964e94) ───
# Minimum length, in characters, a project_ref or a single title token must
# have to be eligible for lexical substring matching. Guards short/incidental
# tokens (a 2-3 char ref, a short common word buried in a longer title) from
# driving a false-positive match. Measured against the live map roster (5
# map-bearing projects, 27 active maps): every real project_ref clears this
# comfortably.
MIN_LEXICAL_TOKEN_LEN = 6
# Lead fix at merge (review of be964e94): a SINGLE non-stopword title token
# is not evidence. Probed against the three real agent-gtd map titles the
# branch's own eval uses as decoys, 4 of 4 ordinary sentences produced a
# candidate — "let me dispatch the next wave and check the worker logs",
# "reviewing file attachments on the PR", and so on. Every false candidate
# converts a request that would have SHORT-CIRCUITED with zero candidates
# into three concurrent Sonnet calls, so the cost lands on exactly the
# orchestration chatter Jason writes all day. A project_ref match stays
# single-signal (refs are distinctive by construction); a title-only match
# now needs two distinct non-stopword tokens, which "Dispatch Service &
# Worker" requires both of rather than either.
MIN_LEXICAL_TITLE_TOKEN_MATCHES = 2
# Small, pinned stopword set: generic title words that must NOT drive a
# lexical match on their own. A match still fires on a map's project_ref (no
# stopword check there — refs are already distinctive by construction) or on
# any OTHER, non-stopword title token; only a stopword-only title cannot,
# alone, surface a map lexically.
_LEXICAL_TITLE_STOPWORDS = frozenset(
    {
        "project",
        "system",
        "server",
        "service",
        "config",
        "general",
        "status",
        "update",
        "details",
        "overview",
        "network",
        "machine",
        "database",
        "cluster",
        "runbook",
        "summary",
        "orientation",
    }
)
# Collapses any run of hyphen/underscore/whitespace to a single space, so
# 'camera-profiles', 'camera_profiles' and 'camera profiles' all normalize to
# the identical 'camera profiles' — the AC's hyphen/underscore/space-variant
# requirement.
#
# Lead fix at merge (review of be964e94): splitting on ONLY [-_\s] meant a
# project named inside a real path missed — '~/git/camera-profiles-data/' and
# 'work on camera-profiles.' both failed, and that path form is the item's own
# motivating example. Every non-alphanumeric run is a separator, so slashes,
# dots, quotes, parens and backticks all normalize away. Applied after
# lowercasing, so the class only needs a-z0-9.
_LEXICAL_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")

# Merge precedence tiers (GTD be964e94 AC3): a map found by BOTH the lexical
# and detail signals ranks above one found by either alone; a lexical
# project_ref match outranks a detail-only match (higher precision); a
# lexical TITLE-token match sits between the two (still lexical, but a
# title-word mention is weaker evidence than a project-name mention);
# detail-only candidates rank last, in their existing best-detail-rank order.
# Lower value == ranks first.
_TIER_BOTH_SIGNALS = 0
_TIER_LEXICAL_PROJECT_REF = 1
_TIER_LEXICAL_TITLE = 2
_TIER_DETAIL_ONLY = 3

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


# ── Lexical candidate path (GTD be964e94) ────────────────────────────────────

_ACTIVE_MAPS_SQL = (
    "SELECT id, project_ref, short_title FROM knowledge_entries"
    " WHERE is_active = 1 AND entry_type = 'mental_map'"
)


async def _active_mental_maps(db: Any) -> list[tuple[str, str | None, str]]:
    """Return ``(id, project_ref, short_title)`` for every ACTIVE mental map.

    Used only by :func:`_retrieve_lexical_candidates` to scan a project's
    small, low-cardinality map roster (GTD be964e94 evidence: "5 map-bearing
    projects, 27 active maps") for a literal project-name / map-title mention
    in the request text — independent of, and unaffected by, whatever
    :func:`_retrieve_candidate_maps` (detail-match retrieval) did or didn't
    find. No params to bind — this is a flat scan with two literal filters,
    same portable-SQL convention (``?`` placeholders where params exist) as
    :func:`_owning_active_maps`.
    """
    cursor = await db.execute(_ACTIVE_MAPS_SQL)
    rows = await cursor.fetchall()
    return [
        (str(row[0]), (str(row[1]) if row[1] is not None else None), str(row[2]))
        for row in rows
    ]


def _normalize_lexical(text: str) -> str:
    """Lowercase, then collapse hyphen/underscore/whitespace runs to one space.

    Makes 'camera-profiles', 'camera_profiles' and 'camera profiles' (and any
    mix thereof) normalize identically, and turns a hyphen-joined path token
    like 'camera-profiles-data' into the space-separated 'camera profiles
    data' — which is what lets :func:`_lexical_word_match` treat 'camera
    profiles' as a whole-word-bounded match inside it.
    """
    return _LEXICAL_SEPARATOR_RE.sub(" ", text.strip().lower()).strip()


def _lexical_word_match(haystack_normalized: str, needle_normalized: str) -> bool:
    """True if ``needle`` occurs in ``haystack``, bounded by start/end/space.

    Both arguments must already be run through :func:`_normalize_lexical`.
    Bounding the match on whitespace (rather than matching "anywhere") is
    what makes a hyphen-joined path token like 'camera-profiles-data' —
    normalized to 'camera profiles data' — match the needle 'camera
    profiles' (immediately followed by a space, then 'data'), while still
    refusing to match an unrelated word that merely contains the needle's
    letters run together with no separator (e.g. 'megacamera' normalizes to
    a single space-free word, so it is never bounded by a space immediately
    before 'camera' and cannot match the needle 'camera').
    """
    if not needle_normalized:
        return False
    pattern = r"(?:^|(?<=\s))" + re.escape(needle_normalized) + r"(?:$|(?=\s))"
    return re.search(pattern, haystack_normalized) is not None


async def _retrieve_lexical_candidates(
    db: Any, text: str, cwd_project: str | None
) -> dict[str, bool]:
    """Lexical project_ref / map-title candidate path (GTD be964e94).

    Case-insensitively matches ``text`` against every ACTIVE mental map's
    ``project_ref`` and ``short_title`` (:func:`_active_mental_maps`) — a
    cheap, deterministic, high-precision complement to detail-match
    retrieval (:func:`_retrieve_candidate_maps`); see the module docstring.
    Runs ALONGSIDE detail matching, never in place of it — the caller
    (:func:`_retrieve_candidates`) merges both signals into one list.

    Guards (all pinned literals — see the module-level constants above):

    * ``MIN_LEXICAL_TOKEN_LEN`` — a project_ref, or an individual title
      token, shorter than this is NOT eligible for substring matching.
    * A map whose ``project_ref`` equals ``cwd_project`` is excluded BEFORE
      matching (Rule A, later in the route, would drop it anyway on any
      signal — excluding it here keeps this signal's own candidate list,
      and the telemetry attribution derived from it, honest).
    * ``_LEXICAL_TITLE_STOPWORDS`` — a generic title token cannot drive a
      match on its own; only the project_ref, or a non-stopword title
      token, may.

    No LLM call, no network — pure string work over at most a few dozen rows
    (the pinned literal AC's latency bar).

    Returns ``{map_id: matched_via_project_ref}``: ``True`` when the match
    came from the (higher-precision) ``project_ref``, ``False`` when it came
    only from a ``short_title`` token. A map that matches via both still
    reports ``True`` — the caller ranks by the higher-precision tier.
    """
    normalized_text = _normalize_lexical(text)
    if not normalized_text:
        return {}

    hits: dict[str, bool] = {}
    for map_id, project_ref, short_title in await _active_mental_maps(db):
        if project_ref is not None and project_ref == cwd_project:
            continue

        if (
            project_ref
            and len(project_ref) >= MIN_LEXICAL_TOKEN_LEN
            and _lexical_word_match(normalized_text, _normalize_lexical(project_ref))
        ):
            hits[map_id] = True
            continue

        if short_title:
            matched_title_tokens = 0
            for token in _normalize_lexical(short_title).split(" "):
                if len(token) < MIN_LEXICAL_TOKEN_LEN:
                    continue
                if token in _LEXICAL_TITLE_STOPWORDS:
                    continue
                if _lexical_word_match(normalized_text, token):
                    matched_title_tokens += 1
            if matched_title_tokens >= MIN_LEXICAL_TITLE_TOKEN_MATCHES:
                hits[map_id] = False
    return hits


async def _retrieve_candidates(
    kb: "KnowledgeBase", text: str, cwd_project: str | None
) -> tuple[
    list[KnowledgeEntry],
    bool,
    dict[str, _CandidateEvidence],
    dict[str, ListenerCandidateSignal],
]:
    """Merge the lexical and detail-match candidate signals (GTD be964e94 AC3).

    Runs both :func:`_retrieve_candidate_maps` (detail-match, GTD bf40d4f1 —
    left entirely untouched, including its own internal fallback) and
    :func:`_retrieve_lexical_candidates` (lexical, GTD be964e94), then merges
    into ONE deduplicated candidate list:

    * A map found by BOTH signals ranks above one found by either alone
      (``_TIER_BOTH_SIGNALS``).
    * A lexical project_ref match outranks a detail-only match — it is the
      higher-precision signal (``_TIER_LEXICAL_PROJECT_REF``).
    * A lexical title-token-only match also outranks a detail-only match,
      but sits below a project_ref match (``_TIER_LEXICAL_TITLE``).
    * Detail-only candidates keep their existing best-detail-rank ordering,
      below both lexical tiers (``_TIER_DETAIL_ONLY``).
    * Ties within a tier break by map id ascending.

    The merged list is capped at ``_MAP_CANDIDATE_CAP`` — same pinned cap as
    the detail-only path, applied AFTER the merge.

    Returns ``(candidates, used_fallback, evidence, signal_source)``:

    * ``candidates`` / ``used_fallback`` / ``evidence`` are exactly the
      shapes :func:`_retrieve_candidate_maps` returned alone (Rule A, Rule B
      and the vote consume them identically — GTD be964e94 AC5). A
      lexical-only candidate is synthesized evidence
      ``_CandidateEvidence(hit_count=1, best_rank=1)`` so it automatically
      clears :func:`_meets_second_slot_bar` if it lands in a non-primary
      slot — deliberate: a literal project-name/title mention is treated as
      at least as strong as a rank-1 detail hit. A map found by both signals
      keeps its detail ``hit_count`` but its ``best_rank`` is tightened to
      ``min(existing, 1)`` for the same reason.
    * ``signal_source`` maps each candidate id to which signal is credited
      for it — ``"lexical"`` (present in the lexical hits, regardless of
      whether detail-matching also found it — lexical is the higher-
      precision signal so it wins attribution), ``"fallback"`` (detail-only,
      via the detail-match retrieval's own direct-search fallback), or
      ``"detail"`` (detail-only, via detail-match's primary path). Consumed
      by :func:`_record_listener_decision` (``candidate_signal`` column) so
      "which signal produced a whisper" is answerable straight off
      ``listener_decisions`` — see that column's comment in
      ``kb_service.database`` for the exact query.
    """
    detail_candidates, used_fallback, detail_evidence = await _retrieve_candidate_maps(
        kb, text
    )
    lexical_hits = await _retrieve_lexical_candidates(kb.db, text, cwd_project)

    detail_by_id = {entry.id: entry for entry in detail_candidates}

    lexical_only_ids = [mid for mid in lexical_hits if mid not in detail_by_id]
    lexical_only_entries: dict[str, KnowledgeEntry] = {}
    for map_id in lexical_only_ids:
        entry = await kb.get(map_id)
        # Defense-in-depth, mirrors _retrieve_candidate_maps: re-check
        # active/mental_map in case of a race with deactivation between the
        # active-maps scan and this fetch.
        if (
            entry is not None
            and entry.is_active
            and entry.entry_type == EntryType.MENTAL_MAP
        ):
            lexical_only_entries[map_id] = entry

    entries_by_id: dict[str, KnowledgeEntry] = {}
    tier: dict[str, int] = {}
    merged_evidence: dict[str, _CandidateEvidence] = {}
    signal_source: dict[str, ListenerCandidateSignal] = {}

    for map_id, entry in detail_by_id.items():
        entries_by_id[map_id] = entry
        ev = detail_evidence[map_id]
        if map_id in lexical_hits:
            tier[map_id] = _TIER_BOTH_SIGNALS
            merged_evidence[map_id] = _CandidateEvidence(
                hit_count=ev.hit_count, best_rank=min(ev.best_rank, 1)
            )
            signal_source[map_id] = "lexical"
        else:
            tier[map_id] = _TIER_DETAIL_ONLY
            merged_evidence[map_id] = ev
            signal_source[map_id] = "fallback" if used_fallback else "detail"

    for map_id, entry in lexical_only_entries.items():
        entries_by_id[map_id] = entry
        matched_project_ref = lexical_hits[map_id]
        tier[map_id] = (
            _TIER_LEXICAL_PROJECT_REF if matched_project_ref else _TIER_LEXICAL_TITLE
        )
        merged_evidence[map_id] = _CandidateEvidence(hit_count=1, best_rank=1)
        signal_source[map_id] = "lexical"

    ordered_ids = sorted(
        entries_by_id,
        key=lambda mid: (tier[mid], merged_evidence[mid].best_rank, mid),
    )[:_MAP_CANDIDATE_CAP]

    merged_candidates = [entries_by_id[mid] for mid in ordered_ids]
    return merged_candidates, used_fallback, merged_evidence, signal_source


async def _record_listener_decision(
    *,
    session_id: str | None,
    cwd_project: str | None,
    source_kb: str,
    candidates_considered: int,
    reason: ListenerDecisionReason,
    fallback: bool = False,
    vote_shape: str = "",
    candidate_signal: ListenerCandidateSignal = "",
    candidate_ids: list[str] | None = None,
    whispered_ids: list[str] | None = None,
    n_retrieved: int = 0,
    n_after_a: int = 0,
    n_after_b: int = 0,
) -> None:
    """Best-effort insert of one ``listener_decisions`` row. Never raises.

    ``decision`` is derived from ``reason`` — "whisper" iff reason ==
    "whispered", "declined" otherwise.

    ``reason`` (GTD 268e2af3) is ALWAYS stored verbatim — it is no longer
    masked to ``"fallback-direct"`` when the fallback retrieval path ran.
    Whether the fallback ran now lives in the separate ``retrieval_path``
    column (derived from ``fallback`` below), so both the granular decline
    cause AND the retrieval path are recoverable from the same row.

    ``candidate_ids`` (GTD 268e2af3) is the RETRIEVED candidate pool,
    captured by the caller BEFORE rule A / rule B filter it — a decision
    declined by rule A or rule B therefore still records a non-empty list.
    ``whispered_ids`` is the (possibly empty) subset actually surfaced to
    the caller. Contrast with the existing ``vote_shape`` column, which
    stores each voter's raw CHOSEN set — "the pool we voted on", not this.

    ``n_retrieved`` / ``n_after_a`` / ``n_after_b`` (GTD 268e2af3) are the
    per-stage candidate counts already computed by the route, so rule-A and
    rule-B attrition are measurable independently.

    ``vote_shape`` (GTD 66ea1fe4) is a ``json.dumps``-encoded list of the
    three voters' raw candidate-id sets (e.g. ``'[["kb-1"],["kb-1","kb-2"],[]]'``),
    empty string ``""`` on every branch that never reached the LLM gate
    (kill-switch, no-candidates, rule-A/B, no-llm) — recorded so the
    reframed set-returning vote's effect on whisper rate is measurable
    directly from ``listener_decisions`` without re-deriving it from logs.

    ``candidate_signal`` (GTD be964e94) is which candidate-retrieval signal
    produced the surfaced pointer — ``"lexical"`` / ``"detail"`` /
    ``"fallback"`` — populated by the caller ONLY on the "whispered" branch
    (default ``""`` everywhere else, including every decline branch — see
    the column comment in ``kb_service.database`` for the full vocabulary
    and the query it makes answerable).

    A failure here (bad DSN, pool exhaustion, whatever) is logged at DEBUG
    and swallowed: this write must never fail or slow the listener response.
    """
    decision = "whisper" if reason == "whispered" else "declined"
    retrieval_path = _RETRIEVAL_PATH_FALLBACK if fallback else _RETRIEVAL_PATH_PRIMARY
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
            vote_shape,
            candidate_signal,
            json.dumps(candidate_ids or []),
            json.dumps(whispered_ids or []),
            n_retrieved,
            n_after_a,
            n_after_b,
            retrieval_path,
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
    2. Retrieval — detail-match + owning-map resolution MERGED with the
       lexical project_ref/title path, ranked/deduped/capped at 5 (see
       :func:`_retrieve_candidates`, GTD be964e94).
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
    ``retrieval_path="fallback-direct"``; ``reason`` always stays the
    granular per-branch value (GTD 268e2af3).
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

    # ── Candidate retrieval (lexical + detail-match, merged) ─────────────────
    candidates, used_fallback, evidence, signal_source = await _retrieve_candidates(
        kb, body.text, body.cwd_project
    )
    n_retrieved = len(candidates)
    # PRE-RULE-A capture (GTD 268e2af3): the retrieved pool, before rule A/B
    # filter it. This is what makes a rule-a/rule-b decline's candidate_ids
    # non-empty — captured here because `candidates` is reassigned below.
    candidate_ids = [e.id for e in candidates]

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
    n_after_b = len(candidates)

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
            candidate_ids=candidate_ids,
            n_retrieved=n_retrieved,
            n_after_a=n_after_a,
            n_after_b=n_after_b,
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
            candidate_ids=candidate_ids,
            n_retrieved=n_retrieved,
            n_after_a=n_after_a,
            n_after_b=n_after_b,
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
    # _retrieve_candidates (lexical/both-signal tiers first, then detail-only
    # in best-detail-rank order), so filtering it (rather than sorting
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
        # Attribution (GTD be964e94): credit the PRIMARY (best-evidence,
        # first-slot) selected pointer's signal — the query this feeds
        # ("how many whispers came from lexical vs detail vs fallback") is
        # about the surfaced whisper as a whole, and the primary pointer is
        # the one that earned the slot without needing the evidence bar.
        candidate_signal = signal_source.get(selected[0].id, "")
        await _record_listener_decision(
            session_id=body.session_id,
            cwd_project=body.cwd_project,
            source_kb=source_kb,
            candidates_considered=len(candidates),
            reason="whispered",
            fallback=used_fallback,
            vote_shape=vote_shape,
            candidate_signal=candidate_signal,
            candidate_ids=candidate_ids,
            whispered_ids=[e.id for e in selected],
            n_retrieved=n_retrieved,
            n_after_a=n_after_a,
            n_after_b=n_after_b,
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
        candidate_ids=candidate_ids,
        n_retrieved=n_retrieved,
        n_after_a=n_after_a,
        n_after_b=n_after_b,
    )
    return ListenerResponse(pointer=None, pointers=[], reason=reason)
