"""Nightly map-maintenance loop endpoints (Rung 0b/1 of the design docs).

``GET /api/kb/map-loop-input?project_ref=<ref>`` — everything the nightly
map-maintenance loop needs for ONE project, in ONE
authed read-only call: the project's mappable entries (titles, tags, a bounded
excerpt, the unpointed flag), its existing active maps with their FULL bodies
and authorship fields, and the unpointed dense pockets as candidate
member-id lists with pairwise similarity evidence. The response is the
pre-fetch payload Rung 1 of docs/nightly-map-maintenance-design.md injects as
synthetic tool-call/result events, so its field names are the contract of
record for docs/somnus-functional-spec.md and must not be renamed or
re-nested.

This route is deliberately read-only: no telemetry rows, no graph mutations,
no entry updates, no decay-anchor touch, and no per-entry round trips —
everything comes from the three statements below plus ``kb.maps_for_project``.
It is modelled on its read-only precedent ``nudge_routes.py`` (positional row
access, awaited ``db.execute``) but deliberately does NOT copy that route's
per-candidate-map read through the facade's per-entry getter: the loop sweeps
every eligible project nightly, and the per-entry read path is the sole
writer of the pull-through read signal.

READ-ONLY COST FLOOR. The admission gate below is the endpoint's fixed cost:
``kb.map_eligibility()`` (kb_core/knowledge_base.py) delegates to
``resolve_eligibility`` (kb_core/map_eligibility.py), which runs
``compute_evidence`` — the corpus-wide eligibility counts scan of
``knowledge_entries`` plus an ``ingested_files`` unnest, zero bind parameters
— and ``list_overrides``; and ``kb.maps_for_project()`` issues its own
statement. None of those appear in the hermetic fakes' recorded statements,
because the fakes return canned data without touching the fake DB — nobody
should later make the fakes faithful and break every test in this suite.

WHAT RUNS WHERE — deliberate supersession of the design's own table.
docs/nightly-map-maintenance-design.md's "what runs where" table routes
pocket detection "off the Pi — desktop or r7", reasoning "No ANN index,
exact brute force, 4-core aarch64 Pi 4 serving live requests", and the
pocket-detection paragraph says the kNN "must not run in the service event
loop". This endpoint puts that computation in a kb-service request handler
on the Pi, deliberately, and the table line is superseded: all distance
arithmetic executes in pgvector on the data-DB host — "the Pis are 4-core
aarch64 Pi 4s, the data DB is not even on them" — there is no Python-side
vector math, ``kb.db.execute`` is awaited so the event loop is never blocked,
and the same table's first row already sanctions an in-request per-project
kNN for the capture-time nudge ("kb-service, in request path"). What runs on
the Pi is row assembly over at most ``POCKET_MAX_UNPOINTED`` ids and at most
``POCKET_MAX_PAIRS`` rows. Mutual-top-K also reintroduces a small ``k``,
contradicting the design's stated virtue "It needs no ``k``" — defensible
because ``k`` here caps a neighbour list rather than partitioning the
project, but recorded rather than discovered.

LEDGER FILTER — pockets are returned UNFILTERED by the cluster/decline
ledger, and the caller owns the filter. This endpoint does not read the
ledger (the sibling item owns that table, keyed on member-set overlap with
the member-set-doubling reopen); somnus applies the decline filter after
fetching, so it never re-proposes a declined cluster. This is a deliberate
placement choice, and the statement ceiling encoded in the tests (at most
three statements issued directly, exactly two when pockets are omitted) is
the test to change if server-side suppression is ever decided instead.

NO MACHINE-AUTHORED FLAG. The route does not compute a machine principal
boolean: the machine principal is service config (``kb_service.attribution``
reads an app_config key from the SERVICE DB) while this route touches only
the DATA DB. Somnus derives the authorship tier itself from
``contributor``/``updated_by`` against its own identity.

THE WORKLIST. This module's second endpoint, ``GET /api/kb/map-worklist``,
is Rung 0b of docs/somnus-functional-spec.md: the ranked, non-admin
enumeration of map-eligible projects that somnus's ``nightly`` subcommand
picks its three projects a night from, taking the first three WITHOUT
re-sorting. It exists because ``GET /api/kb/map-eligibility`` returns every
verdict but is ``require_admin`` and the machine principal is deliberately
non-admin — the enumeration the loop is built on did not exist for the one
caller that needs it. Auth is ``get_current_user`` (ANY authenticated user),
the eligibility predicate is kb-core's own (``map_eligibility()`` plus a
drop of every verdict whose ``effective_eligible`` is False, so the override
table is respected exactly as everywhere else — never reimplemented here),
and the per-project map-write facts come from
``kb_core.map_caps.map_write_summary``. The RANKING lives in this route, not
in somnus: it needs each project's map-write history, which the server holds
and the loop does not, and a copy of the ordering rule in the Rust binary
would be one more thing that can drift from this one. The endpoint takes NO
query parameters, no limit, no truncation — the three-projects-per-night
cap is somnus's worklist policy, and a server-side limit would silently
hide projects from any other reader of this list.
"""

import hashlib
import logging
import time
from typing import Annotated, Any, NamedTuple

from fastapi import APIRouter, Depends, HTTPException, Request
from kb_core.db.backend import MAPPABLE_ENTRY_WHERE_SQL
from kb_core.db.sqlite_backend import SQLiteBackend
from kb_core.map_caps import map_write_summary

from kb_service.auth import get_current_user
from kb_service.models import (
    MapLoopEntry,
    MapLoopInputResponse,
    MapLoopMap,
    MapLoopPocket,
    MapLoopPocketEdge,
    MapWorklistProject,
    MapWorklistResponse,
    PocketsOmittedReason,
    User,
)

logger = logging.getLogger(__name__)

# Greppable log marker — mirrors MAP_ELIGIBILITY_ROUTE_MARKER.
MAP_LOOP_INPUT_MARKER = "map-loop-input"

# Greppable log marker for the worklist endpoint, same convention.
MAP_WORKLIST_MARKER = "map-worklist"

router = APIRouter(prefix="/api/kb", tags=["kb"])

# ── Pinned constants ─────────────────────────────────────────────────────────
# The request has exactly ONE input, project_ref; no excerpt length,
# threshold, k or cap below is client-suppliable.
#
# Titles plus tags plus a 600-character excerpt across all 25 eligible
# projects is 186,170 tokens, so the largest single project (personal-kb,
# 243 mappable entries) packs to roughly 43k — the design's packing figure,
# docs/nightly-map-maintenance-design.md.
LOOP_INPUT_EXCERPT_CHARS = 600

# HYPOTHESIS, not a measurement. The design pins only what it EXCLUDES
# (thresholded connected components collapse at cosine >= 0.62 — cleanr to
# [172, 6, 2] of 180, harness-design to one component of all 67; and k-means
# because choosing k is the entire problem) and gives four similarity values
# from one worked home-network group (0.795/0.635/0.599/0.577) — not the
# full pair matrix, and not the project-wide neighbour ranks that
# mutual-top-K actually depends on, so whether these values admit that group
# is UNVERIFIED. 0.55 sits below the 0.62 collapse point and below the
# worked group's lowest cited value. Retune after the design's own
# falsifying experiment (the by-hand three-project backfill, design doc
# Phase 2) and treat a first run that returns zero pockets on home-network
# as the signal to lower 0.55, not as a bug in the algorithm.
# POCKET_OBSERVATION_FLOOR exists precisely so that retune needs no redeploy
# to DIAGNOSE: the SQL floor is 0.40 and the threshold is applied in Python,
# so the log always reports what a lower threshold would have admitted.
POCKET_OBSERVATION_FLOOR = 0.40
POCKET_MIN_SIMILARITY = 0.55
POCKET_TOP_K = 3
POCKET_MIN_SIZE = 2
POCKET_MAX_COUNT = 10

# Real headroom over the largest eligible project (personal-kb, 243
# mappable), so this branch cannot fire on today's corpus and is a safety
# valve exercised only by test.
POCKET_MAX_UNPOINTED = 300
POCKET_MAX_PAIRS = 20000

# A pair statement slower than this is the precursor warning that makes
# POCKET_MAX_UNPOINTED measurable instead of invented.
SLOW_PAIR_QUERY_MS = 5000

# ── SQL ──────────────────────────────────────────────────────────────────────

# One definition, one call site: this anti-join fragment is interpolated into
# the entries statement below and nowhere else. Edge direction matches
# nudge_routes._has_owning_map — source is the map, target the pointed-at
# entry. The pair statement deliberately takes an explicit id list built in
# Python instead of a second copy of this fragment: a second copy inside a
# CTE would have to correlate against a knowledge_entries the fragment itself
# re-introduces as mm, and the hermetic fakes never parse SQL, so a
# mis-correlated subquery would pass every test and then return "every entry
# unpointed" (or none) in production.
_UNPOINTED_NOT_EXISTS_SQL = (
    "SELECT 1 FROM graph_edges ge JOIN knowledge_entries mm ON mm.id = ge.source"
    " WHERE ge.target = {target} AND ge.edge_type = 'references'"
    " AND mm.entry_type = 'mental_map' AND mm.is_active = 1"
)

# The only interpolated text is literal SQL fragments (the kb-core mappable
# predicate constant, the in-module anti-join template above, and module
# int constants); every value from the request crosses as a ? bind parameter,
# so this is not an injection vector. The excerpt length is interpolated as
# an int LITERAL, deliberately, so the statement has exactly one ? and no
# param-order hazard (the Postgres ?->$N translator counts placeholders in
# strict textual order and does nothing else).
_ENTRIES_SQL = f"""WITH mappable AS (
    SELECT id, short_title, long_title, entry_type, tags, knowledge_details
    FROM knowledge_entries
    WHERE {MAPPABLE_ENTRY_WHERE_SQL} AND project_ref = ?
)
SELECT m.id, m.short_title, m.long_title, m.entry_type, m.tags,
       substr(m.knowledge_details, 1, {LOOP_INPUT_EXCERPT_CHARS}) AS excerpt,
       length(m.knowledge_details) AS details_length,
       CASE WHEN EXISTS ({_UNPOINTED_NOT_EXISTS_SQL.format(target="m.id")})
            THEN 0 ELSE 1 END AS unpointed
FROM mappable m
"""  # noqa: S608

# A plain literal, no interpolation — therefore NO noqa (RUF100 is selected,
# so an unnecessary noqa fails the gate). contributor/updated_by exist on
# knowledge_entries (backfilled multi-user columns).
_MAP_BODIES_SQL = (
    "SELECT id, knowledge_details, contributor, updated_by FROM knowledge_entries"
    " WHERE is_active = 1 AND entry_type = 'mental_map' AND project_ref = ?"
)

# All distance arithmetic runs in pgvector: <=> is pgvector cosine DISTANCE,
# so 1 - distance is the similarity reported. The restriction lives in the
# JOIN TREE, not a WHERE semi-join: knowledge_vec holds every embedded entry
# in the corpus with no index of any kind beyond its primary key, and whether
# a planner pushes a semi-join below the distance expression is not a
# guarantee, so the CTE materialises the unpointed id set first.
# a.entry_id < b.entry_id yields each unordered pair exactly once — a TEXT
# range comparison needs no COLLATE "C": any consistent total order
# suffices, because the comparison only de-duplicates pairs and its order
# never crosses the wire, unlike an ORDER BY whose output order does.
# An entry with no vector row drops out of every pair structurally and lands
# in no pocket — correct behaviour, not an error, since a just-stored entry
# is normally unembedded. The sub-threshold band is deliberately NOT
# discarded here (no row_number window): LIMIT bounds the wire, Python
# top-K filters the rest, and the observation band is what lets a retune
# diagnose itself from the log without a redeploy.
_PAIRS_SQL_TEMPLATE = f"""WITH unpointed AS (
    SELECT entry_id, embedding FROM knowledge_vec WHERE entry_id IN ({{placeholders}})
)
SELECT a.entry_id, b.entry_id, 1 - (a.embedding <=> b.embedding) AS similarity
FROM unpointed a JOIN unpointed b ON a.entry_id < b.entry_id
WHERE 1 - (a.embedding <=> b.embedding) >= {POCKET_OBSERVATION_FLOOR}
ORDER BY similarity DESC LIMIT {POCKET_MAX_PAIRS}
"""  # noqa: S608


class _PocketStats(NamedTuple):
    """Counters the pocket builder reports to the telemetry line."""

    pairs_returned: int
    pairs_below_threshold: int
    max_similarity_below_threshold: float
    pairs_mutual: int
    components_total: int
    components_below_min_size: int
    pockets_truncated: int


def _build_pockets(
    pairs: list[tuple[str, str, float]],
) -> tuple[list[MapLoopPocket], _PocketStats]:
    """Mutual-top-K connected components over the pair list.

    NOT thresholded components and NOT k-means — both are excluded on
    measured evidence in the design. Order of operations, pinned: (0) drop
    every pair below ``POCKET_MIN_SIMILARITY`` FIRST, so a sub-threshold pair
    can never influence a top-K list; (1) per-entry surviving neighbours
    sorted by similarity descending, tie-broken by neighbour id ascending,
    truncated to ``POCKET_TOP_K``; (2) keep a pair as a MUTUAL edge iff each
    endpoint is in the other's top-K; (3) pockets are the connected
    components of the mutual-edge graph with at least ``POCKET_MIN_SIZE``
    members; (4) mean/min/max similarity are computed over that pocket's
    MUTUAL EDGES only; (5) at most ``POCKET_MAX_COUNT`` pockets, keeping the
    highest-mean ones.
    """
    below = [(a, b, s) for a, b, s in pairs if s < POCKET_MIN_SIMILARITY]
    kept = [(a, b, s) for a, b, s in pairs if s >= POCKET_MIN_SIMILARITY]
    max_below = max((s for _, _, s in below), default=0.0)

    neighbours: dict[str, list[tuple[str, float]]] = {}
    for a, b, sim in kept:
        neighbours.setdefault(a, []).append((b, sim))
        neighbours.setdefault(b, []).append((a, sim))
    topk: dict[str, list[str]] = {}
    for node, nbs in neighbours.items():
        ordered = sorted(nbs, key=lambda edge: (-edge[1], edge[0]))
        topk[node] = [nb for nb, _ in ordered[:POCKET_TOP_K]]

    mutual: list[tuple[str, str, float]] = []
    for a, b, sim in kept:
        if b in topk.get(a, []) and a in topk.get(b, []):
            mutual.append((a, b, sim))

    # Connected components of the mutual-edge graph (union-find).
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for a, b, _ in mutual:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    members: dict[str, set[str]] = {}
    edges_by_root: dict[str, list[tuple[str, str, float]]] = {}
    for a, b, sim in mutual:
        root = find(a)
        members.setdefault(root, set()).update((a, b))
        edges_by_root.setdefault(root, []).append((a, b, sim))

    pockets: list[MapLoopPocket] = []
    for root, member_set in members.items():
        if len(member_set) < POCKET_MIN_SIZE:
            continue
        edges = edges_by_root[root]
        sims = [sim for _, _, sim in edges]
        pockets.append(
            MapLoopPocket(
                member_entry_ids=sorted(member_set),
                mean_similarity=sum(sims) / len(sims),
                min_similarity=min(sims),
                max_similarity=max(sims),
                edges=[
                    MapLoopPocketEdge(a=a, b=b, similarity=sim)
                    for a, b, sim in sorted(edges, key=lambda e: (e[0], e[1]))
                ],
            )
        )
    pockets.sort(
        key=lambda pocket: (-pocket.mean_similarity, pocket.member_entry_ids[0])
    )
    pockets_truncated = max(0, len(pockets) - POCKET_MAX_COUNT)
    pockets = pockets[:POCKET_MAX_COUNT]
    stats = _PocketStats(
        pairs_returned=len(pairs),
        pairs_below_threshold=len(below),
        max_similarity_below_threshold=max_below,
        pairs_mutual=len(mutual),
        components_total=len(members),
        components_below_min_size=sum(
            1 for member_set in members.values() if len(member_set) < POCKET_MIN_SIZE
        ),
        pockets_truncated=pockets_truncated,
    )
    return pockets, stats


def _optional_str(value: Any) -> str | None:
    """Pass a nullable text column through as ``str | None``."""
    return None if value is None else str(value)


@router.get("/map-loop-input", response_model=MapLoopInputResponse)
async def map_loop_input(
    project_ref: str,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> MapLoopInputResponse:
    """Return everything the nightly loop needs for one project, read-only.

    Args:
        project_ref: The single required query parameter.
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user (JWT or API key via Bearer header) — same
            dependency as the other ``/api/kb/*`` routes, NOT admin:
            somnus runs as a plain non-admin user by design, and an
            admin-only gate would force the loop to hold admin credentials
            it should not have.

    Returns:
        ``MapLoopInputResponse``. 404 when no verdict carries the ref; 409
        when the effective verdict is not eligible (a human override is
        honoured in BOTH directions). The route issues none of its own SQL
        before the gate passes.

    Raises:
        HTTPException: 404 unknown ref, 409 not map-eligible.
    """
    kb = request.app.state.kb
    started = time.perf_counter()

    # ── Admission gate: NO own SQL before this passes ────────────────────────
    verdicts = await kb.map_eligibility()
    verdict = next((v for v in verdicts if v.evidence.project_ref == project_ref), None)
    if verdict is None:
        logger.info(
            "%s outcome=not_found project_ref=%s", MAP_LOOP_INPUT_MARKER, project_ref
        )
        raise HTTPException(status_code=404, detail="project_ref not found")
    evidence = verdict.evidence
    if verdict.effective_eligible is False:
        logger.info(
            "%s outcome=ineligible project_ref=%s decided_by=%s"
            " computed_eligible=%s mappable=%s hand_authored=%s"
            " is_journal=%s is_too_thin=%s",
            MAP_LOOP_INPUT_MARKER,
            project_ref,
            verdict.decided_by,
            evidence.computed_eligible,
            evidence.mappable,
            evidence.hand_authored,
            evidence.is_journal,
            evidence.is_too_thin,
        )
        raise HTTPException(status_code=409, detail="project_ref not map-eligible")

    # ── Statement 1: entries ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    cursor = await kb.db.execute(_ENTRIES_SQL, (project_ref,))
    entry_rows = await cursor.fetchall()
    entries_query_ms = int((time.perf_counter() - t0) * 1000)

    entries: list[MapLoopEntry] = []
    for row in entry_rows:
        # Positional row access (row[0], row[1], ...) — matches the hermetic
        # fakes, which hand back plain tuples.
        entries.append(
            MapLoopEntry(
                id=str(row[0]),
                short_title=str(row[1] or ""),
                long_title=str(row[2] or ""),
                # Raw stored string (factual_reference | decision |
                # pattern_convention | lesson_learned); mental_map is
                # impossible because the mappable predicate excludes it.
                entry_type=str(row[3] or ""),
                # The column default is the literal two-character string
                # '[]' — space-joined text, NOT JSON, so it splits to
                # ["[]"]; never json.loads it. split() with no argument
                # collapses whitespace runs and yields [] for empty/blank.
                tags=str(row[4] or "").split(),
                excerpt=str(row[5] or ""),
                details_length=int(row[6] or 0),
                unpointed=bool(row[7]),
            )
        )
    # Deterministic order: the payload becomes a prompt prefix, and an
    # unstable order silently destroys prompt-cache hits.
    entries.sort(key=lambda entry: entry.id)
    unpointed_ids = [entry.id for entry in entries if entry.unpointed]

    # ── Maps: pointers from the facade, bodies from statement 2 ─────────────
    map_refs = await kb.maps_for_project(project_ref)
    cursor = await kb.db.execute(_MAP_BODIES_SQL, (project_ref,))
    body_rows = await cursor.fetchall()
    bodies: dict[str, tuple[str, str | None, str | None]] = {
        str(row[0]): (str(row[1] or ""), _optional_str(row[2]), _optional_str(row[3]))
        for row in body_rows
    }
    maps: list[MapLoopMap] = []
    for ref in map_refs:
        map_id = str(ref.get("id", ""))
        body_row = bodies.get(map_id)
        maps.append(
            MapLoopMap(
                id=map_id,
                short_title=str(ref.get("short_title") or ""),
                long_title=str(ref.get("long_title") or ""),
                pointers=[str(pointer) for pointer in (ref.get("pointers") or [])],
                body=body_row[0] if body_row else "",
                contributor=body_row[1] if body_row else None,
                updated_by=body_row[2] if body_row else None,
            )
        )
    # maps_for_project returns created_at DESC (newest first); the payload
    # becomes a prompt prefix, so it is deliberately re-sorted by id
    # ascending — an unstable order silently destroys prompt-cache hits.
    maps.sort(key=lambda map_ref: map_ref.id)

    # ── Runtime audit 1: the two "unpointed" derivations ─────────────────────
    # Advisory, never a 500, repairs nothing (the repair pass stays out of
    # scope). A divergence would have somnus propose a duplicate add_pointer
    # every night while the pointer count still moves and the gate stays
    # green — the design's most-feared quiet degradation.
    already_pointed = {p for m in maps for p in m.pointers} & {
        entry.id for entry in entries if entry.unpointed
    }
    if already_pointed:
        first20 = sorted(already_pointed)[:20]
        logger.warning(
            "%s edge_regex_divergence=%s project_ref=%s count=%d",
            MAP_LOOP_INPUT_MARKER,
            ",".join(first20),
            project_ref,
            len(already_pointed),
        )

    # ── Pockets: three skip reasons, then statement 3 ────────────────────────
    omitted: PocketsOmittedReason | None
    pockets: list[MapLoopPocket] = []
    stats = _PocketStats(0, 0, 0.0, 0, 0, 0, 0)
    pair_query_ms = 0
    if len(unpointed_ids) < POCKET_MIN_SIZE:
        omitted = "too-few-unpointed-entries"
    elif len(unpointed_ids) > POCKET_MAX_UNPOINTED:
        omitted = "unpointed-set-too-large"
    elif isinstance(kb.db, SQLiteBackend):
        # <=> is pgvector-only; a SQLite KB can serve every other half of
        # this payload. Never a normal state on the hosted Pi — KB_DATABASE_URL
        # unset or empty is the likely cause — so this is a WARNING, not info.
        omitted = "non-postgres-backend"
        logger.warning(
            "%s pockets_omitted_reason=non-postgres-backend project_ref=%s"
            " (KB_DATABASE_URL unset or empty is the likely cause: the vector"
            " pair statement needs pgvector)",
            MAP_LOOP_INPUT_MARKER,
            project_ref,
        )
    else:
        omitted = None
        placeholders = ",".join("?" for _ in unpointed_ids)
        t1 = time.perf_counter()
        cursor = await kb.db.execute(
            _PAIRS_SQL_TEMPLATE.format(placeholders=placeholders),
            tuple(unpointed_ids),
        )
        pair_rows = await cursor.fetchall()
        pair_query_ms = int((time.perf_counter() - t1) * 1000)
        pockets, stats = _build_pockets(
            [(str(row[0]), str(row[1]), float(row[2])) for row in pair_rows]
        )
        # ── Runtime audit 2: pocket-member containment ───────────────────────
        # Every pocket member must appear in entries with unpointed true; on
        # violation DROP the offending pocket rather than serving it.
        unpointed_set = set(unpointed_ids)
        filtered: list[MapLoopPocket] = []
        for pocket in pockets:
            bad = [
                member_id
                for member_id in pocket.member_entry_ids
                if member_id not in unpointed_set
            ]
            if bad:
                logger.warning(
                    "%s pocket_member_not_unpointed=%s project_ref=%s",
                    MAP_LOOP_INPUT_MARKER,
                    ",".join(bad),
                    project_ref,
                )
            else:
                filtered.append(pocket)
        pockets = filtered
        if pair_query_ms > SLOW_PAIR_QUERY_MS:
            logger.warning(
                "%s pair_query_ms=%d unpointed_count=%d",
                MAP_LOOP_INPUT_MARKER,
                pair_query_ms,
                len(unpointed_ids),
            )

    payload = MapLoopInputResponse(
        project_ref=project_ref,
        excerpt_chars=LOOP_INPUT_EXCERPT_CHARS,
        entries=entries,
        maps=maps,
        pockets=pockets,
        pockets_omitted_reason=omitted,
    )
    dump = payload.model_dump_json()
    dump_bytes = dump.encode("utf-8")

    # ── Telemetry: ONE info line per successful call ─────────────────────────
    # Strict single-space key=value pairs after the marker, no prose, no
    # spaces inside any value, floats %.4f, integers bare. This route makes
    # no LLM call, so token/cost accounting belongs to somnus; payload_bytes
    # is the server-side half.
    total_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        " ".join(
            [
                MAP_LOOP_INPUT_MARKER,
                f"project_ref={project_ref}",
                f"entries={len(entries)}",
                f"unpointed={len(unpointed_ids)}",
                f"pockets={len(pockets)}",
                f"pairs_returned={stats.pairs_returned}",
                f"pairs_below_threshold={stats.pairs_below_threshold}",
                f"max_similarity_below_threshold="
                f"{stats.max_similarity_below_threshold:.4f}",
                f"pairs_mutual={stats.pairs_mutual}",
                f"components_total={stats.components_total}",
                f"components_below_min_size={stats.components_below_min_size}",
                f"pockets_truncated={stats.pockets_truncated}",
                # The server cannot observe what LIMIT discarded; 0 is the
                # honest "no rows were withheld on this side of the wire".
                "pairs_truncated=0",
                f"entries_query_ms={entries_query_ms}",
                f"pair_query_ms={pair_query_ms}",
                f"total_ms={total_ms}",
                f"payload_bytes={len(dump_bytes)}",
                f"payload_digest={hashlib.sha256(dump_bytes).hexdigest()[:12]}",
                f"pockets_omitted_reason={omitted or 'none'}",
                f"min_similarity={POCKET_MIN_SIMILARITY:.4f}",
                f"observation_floor={POCKET_OBSERVATION_FLOOR:.4f}",
                f"top_k={POCKET_TOP_K}",
                f"min_size={POCKET_MIN_SIZE}",
                f"max_count={POCKET_MAX_COUNT}",
                f"max_unpointed={POCKET_MAX_UNPOINTED}",
                f"max_pairs={POCKET_MAX_PAIRS}",
            ]
        )
    )
    return payload


@router.get("/map-worklist", response_model=MapWorklistResponse)
async def map_worklist(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> MapWorklistResponse:
    """Return the RANKED worklist of map-eligible projects (Rung 0b).

    Every eligible project, left-joined against its map-write summary and
    ranked for somnus's ``nightly`` to read head-first. No limit, no
    truncation, no query parameters: the three-projects-per-night cap is
    somnus's worklist policy, and a server-side limit would silently hide
    projects from any other reader of this list. Empty is a legitimate
    quiet night and renders ``{"projects": []}`` — never a 404.

    Args:
        request: FastAPI request (provides ``app.state.kb``).
        user: Authenticated user (JWT or API key via Bearer header) — ANY
            authenticated user, deliberately NOT admin: the machine principal
            somnus runs as is non-admin by design, and ``map-eligibility``
            being admin-only is exactly why this endpoint exists.

    Returns:
        ``MapWorklistResponse``: only the verdicts whose
        ``effective_eligible`` is true, so the override table is honoured
        exactly as everywhere else; each row's ``mappable`` is the
        verdict's own evidence count and its ``map_count`` /
        ``latest_map_written_at`` come from kb-core's summary, which omits
        never-mapped projects — hence the left-join in Python: the eligible
        verdicts drive the iteration, never the summary rows.
    """
    kb = request.app.state.kb
    verdicts = await kb.map_eligibility()

    summary_by_ref = {row.project_ref: row for row in await map_write_summary(kb.db)}
    projects: list[MapWorklistProject] = []
    for verdict in verdicts:
        # Truthiness rather than `is False`: this is an eligibility GATE, and
        # the two spellings differ in the direction that matters. `is False`
        # admits anything that is not the literal False — so the day
        # `effective_eligible` becomes `bool | None`, every unresolved verdict
        # silently becomes eligible and the loop starts working projects the
        # human excluded. It is `bool` today; the point is that the gate should
        # not fail open if that ever loosens.
        if not verdict.effective_eligible:
            continue
        project_ref = verdict.evidence.project_ref
        summary = summary_by_ref.get(project_ref)
        projects.append(
            MapWorklistProject(
                project_ref=project_ref,
                mappable=verdict.evidence.mappable,
                map_count=0 if summary is None else summary.map_count,
                latest_map_written_at=(
                    None if summary is None else summary.latest_map_written_at
                ),
            )
        )

    # One explicit sort key with three components: (1) never-mapped first —
    # map_count == 0 sorts ahead of every mapped project; (2) oldest
    # latest_map_written_at first; (3) project_ref ascending, so two runs
    # against identical state pick the same three. A never-mapped row's null
    # timestamp needs no sentinel: component (1) has already separated the
    # never-mapped rows from the mapped ones, so component (2) only ever
    # compares null against null (equal — tuple comparison falls through to
    # the ref) or one aware datetime against another. NOT
    # most-unpointed-first, which starves: a project whose unpointed tail
    # sits entirely in declined clusters would top that list forever and
    # block every other project, and staleness-first is starvation-free by
    # construction.
    projects.sort(
        key=lambda project: (
            project.map_count != 0,
            project.latest_map_written_at,
            project.project_ref,
        )
    )

    never_mapped = sum(1 for project in projects if project.map_count == 0)
    logger.info(
        " ".join(
            [
                MAP_WORKLIST_MARKER,
                f"verdicts={len(verdicts)}",
                f"eligible={len(projects)}",
                f"never_mapped={never_mapped}",
            ]
        )
    )
    return MapWorklistResponse(projects=projects)
