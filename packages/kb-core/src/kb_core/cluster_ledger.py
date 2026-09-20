"""Cluster/decline ledger — the nightly map-maintenance loop's only durable state.

Somnus's counterpart of ``map_eligibility_override``: the override table
holds the human verdict on WHICH PROJECTS deserve maintenance, this table
holds the loop's proposed subject-area clusters and the human declines over
them. Identity is **member-set overlap, never the label** — labels drift
between runs, so ``cluster_key`` is the hash of the member set the cluster
was born with and ``last_label`` is bookkeeping, not identity. A cluster
the loop actually maps stops recurring on its own (its entries become
pointed and drop out of the unpointed set), which is why a proposal is
recorded at all: the row is the digest line, the audit trail and the
``sightings`` counter.

Wire contract of record — the HTTP models in the web-service repo and the
somnus crate cite these names verbatim; nothing here renames, re-nests or
rounds anything. ``ClusterLedgerRow`` serializes via
``dataclasses.asdict()`` to exactly ``{"cluster_key", "project_ref",
"member_entry_ids", "last_label", "status", "sightings", "first_seen_at",
"last_seen_at", "declined_member_count", "declined_reason", "declined_by",
"declined_at"}``; ``ClusterLedgerMatchResult`` to
``{"project_ref", "jaccard_threshold", "near_miss_floor",
"marginal_match_ceiling", "reopen_growth_factor", "rows_considered",
"verdicts"}`` with each verdict as ``{"member_entry_ids", "jaccard",
"matched", "suppressed", "reopened", "near_miss", "marginal_match",
"oversized"}``. Every Jaccard crosses the wire exactly as Python computed
it — the same discipline ``map_eligibility``'s ``top_prefix_share``
already follows.

Invariants this module enforces, each of which a later edit could silently
break:

* ``status = "proposed"`` does NOT suppress. Suppression is the human's
  decline alone; a proposal exists for the digest, the audit trail and the
  ``sightings`` counter, and a cluster that recurs as ``proposed`` with a
  rising ``sightings`` count is a SIGNAL (``PROPOSED_SIGHTINGS_WARN``) —
  a suppressing ``proposed`` would hide it.
* A sighting NEVER clears a decline, and a reopened cluster stays
  ``status = "declined"`` until a fresh ``decline_cluster`` call re-arms
  suppression at the new member size.
* ``cluster_key`` is a BIRTH hash, never recomputed on update — an
  identity token, not a content hash. The UPDATE branch of
  ``record_cluster`` changes ``member_entry_ids`` and leaves
  ``cluster_key`` alone, which makes the INSERT branch reachable (the
  member set drifts away from the set that minted the key and the
  original set can return), which is why the INSERT is key-safe
  (``ON CONFLICT(cluster_key) DO UPDATE``).
* ``match_clusters`` writes ``audit_events`` rows — one per near miss,
  marginal match, suppression and reopen — and NEVER a
  ``map_cluster_ledger`` row.
* The nightly loop must never write a decline; ``decline_cluster`` is the
  human's verdict and the HTTP route behind it is admin-gated with a
  machine-principal refusal. This module cannot enforce that, it can only
  not invite it: there is no batch decline, no auto-decline and no
  threshold that produces one.

The tuning extraction is pinned here and in the reader script
(``scripts/cluster_near_miss_report.py`` in the web-service repo):
``jaccard=`` is the FIRST numeric key in every band row's audit detail,
formatted ``%.4f``, and the Postgres query that will do the tuning reads
``SELECT created_at, event_type, detail, substring(detail from
'jaccard=([0-9.]+)')::float AS j FROM audit_events WHERE event_type IN
('map_cluster_near_miss','map_cluster_marginal_match') AND created_at >= $1
ORDER BY j DESC`` — cheap on both backends because ``idx_audit_type`` and
``idx_audit_created`` exist on both.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from kb_core.db.backend import Database, Row

logger = logging.getLogger(__name__)

# Greppable marker — mirrors kb_core.map_eligibility.MAP_ELIGIBILITY_MARKER.
CLUSTER_LEDGER_MARKER = "cluster-ledger"

# HYPOTHESIS, not a measurement. `docs/nightly-map-maintenance-design.md:317`
# states it outright: the overlap SHAPE is verified (17 detail entries with
# two owning maps, 4 with three or four, kb-03215 and kb-03217 genuinely in
# both VPN and DNS) but 0.5 is INVENTED — too high thrashes, too low
# permanently suppresses real clusters. The design's own resolution is
# instrumentation, not argument: log near-misses in the 0.3-0.5 band from
# night one and tune from real data. That is why the band floor is a named
# constant, why every near miss writes both a log line and a durable
# `audit_events` row (journald rotates and the tuning question outlives
# it), and why `scripts/cluster_near_miss_report.py` ships in the same
# commit — an emitter with no reader tunes nothing.
CLUSTER_MATCH_JACCARD = 0.5

# The floor of the 0.3-0.5 near-miss band named above — a named constant so
# the tuning band can move without touching the match threshold.
CLUSTER_NEAR_MISS_FLOOR = 0.3

# The design asked only for the 0.3-0.5 near-miss band, which can only ever
# show 0.5 is too HIGH. `:317` names BOTH failure directions ('too high
# thrashes; too low permanently suppresses'), so the 0.5-0.7 marginal-match
# band is recorded too — a match at 0.52 that merges two genuinely different
# subject areas is the thrash failure, and without this band it leaves no
# trace. 0.7 is invented and carries no evidence; it is a starting value
# chosen to make the instrument two-sided.
CLUSTER_MARGINAL_MATCH_CEILING = 0.7

# docs/nightly-map-maintenance-design.md:293 — "permanent, with one escape —
# reopen when the cluster's member set doubles... prevents a cluster
# declined at 20 entries from being suppressed forever at 200".
REOPEN_GROWTH_FACTOR = 2

# A collision-risk judgement, not a measurement. 16 hex chars is 64 bits of
# sha256, negligible at the few-hundred-row scale this table reaches (25
# eligible projects x 10-20 clusters). Truncation is deliberate: a full
# 64-char digest makes every log line and audit detail unreadable. Widen it
# if the table ever passes ~10,000 rows, at which point the `project_ref`
# index of AC-7 is due too.
CLUSTER_KEY_HEX_CHARS = 16

# A runaway guard, NOT a policy. Candidates are Rung-1 clusters over ALL
# mappable entries of one project (not pockets), and the measured corpus has
# 1,051 mappable entries total across 25 eligible projects with the largest
# single project at 243, so 1200 exceeds the whole corpus and cannot fire on
# legitimate input. It is deliberately NOT derived from the sibling item's
# `POCKET_MAX_UNPOINTED = 300`: pockets are unpointed-only and a Rung-1
# cluster is not, and a ceiling below a legitimate cluster size would
# silently bypass suppression (AC-13 returns `oversized` rather than raising
# for exactly that reason). Exercised only by test.
CLUSTER_MEMBER_MAX = 1200

# Above the theoretical maximum number of disjoint 2-member clusters in the
# largest eligible project (243 mappable / 2 = 121), so it cannot fire on
# today's corpus; it bounds one request, and the caller controls batching,
# so exceeding it is an honest 422.
CLUSTER_MATCH_MAX_CANDIDATES = 200

# Invented. Three nights is the point at which per-night caps and deferral
# stop explaining recurrence, so a still-proposed cluster means nothing is
# landing.
PROPOSED_SIGHTINGS_WARN = 3

ClusterLedgerStatus = Literal["proposed", "declined"]
"""The row's state. Exactly two members, no ``accepted``: a proposal is
bookkeeping and an acceptance is indistinguishable from a proposal nobody
declined, so a third member would have no reader."""


@dataclasses.dataclass(frozen=True)
class ClusterLedgerRow:
    """One ``map_cluster_ledger`` row.

    ``member_entry_ids`` matches the loop-input contract and the
    functional spec's cluster shape verbatim, and it is a tuple because
    the dataclass is frozen. ``last_label``, not ``label``: the design's
    central decision is that the label is NOT the identity, and the name
    encodes that.
    """

    cluster_key: str
    project_ref: str
    member_entry_ids: tuple[str, ...]
    last_label: str
    status: ClusterLedgerStatus
    sightings: int
    first_seen_at: str
    last_seen_at: str
    declined_member_count: int | None
    declined_reason: str | None
    declined_by: str | None
    declined_at: str | None


@dataclasses.dataclass(frozen=True)
class ClusterLedgerVerdict:
    """One candidate's match outcome.

    ``member_entry_ids`` is the NORMALIZED candidate, echoed so a batch
    caller can align results positionally AND by value. ``suppressed`` is
    the single boolean the loop keys on; every other field is evidence
    for it. ``oversized`` means the candidate exceeded
    ``CLUSTER_MEMBER_MAX`` and was not matched — it is therefore NOT
    evidence of non-suppression.
    """

    member_entry_ids: tuple[str, ...]
    jaccard: float
    matched: ClusterLedgerRow | None
    suppressed: bool
    reopened: bool
    near_miss: bool
    marginal_match: bool
    oversized: bool


@dataclasses.dataclass(frozen=True)
class ClusterLedgerMatchResult:
    """A whole-batch match response.

    The four pinned constants travel WITH the result so (a) a consumer six
    weeks after a retune can read what was in force, and (b) the HTTP route
    can echo them without importing a kb-core symbol.
    """

    project_ref: str
    jaccard_threshold: float
    near_miss_floor: float
    marginal_match_ceiling: float
    reopen_growth_factor: int
    rows_considered: int
    verdicts: tuple[ClusterLedgerVerdict, ...]


# ---------------------------------------------------------------------------
# Pure helpers (no DB)
# ---------------------------------------------------------------------------


def normalize_members(member_entry_ids: Sequence[str]) -> tuple[str, ...]:
    """Dedupe then sort ascending, so the stored value and the hash input are canonical."""
    return tuple(sorted(set(member_entry_ids)))


def cluster_key_for(project_ref: str, member_entry_ids: Sequence[str]) -> str:
    """Mint the ``cluster_key`` for one (project, member set) pair.

    BIRTH-HASH INVARIANT: ``cluster_key`` is the hash of the member set
    the cluster was BORN with and is NEVER recomputed on update — the
    UPDATE branch of :func:`record_cluster` changes ``member_entry_ids``
    and leaves ``cluster_key`` alone, so it is an identity token, not a
    content hash. This function exists to MINT one, never to verify one.
    """
    payload = project_ref + "\n" + " ".join(normalize_members(member_entry_ids))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:CLUSTER_KEY_HEX_CHARS]


def jaccard(a: Collection[str], b: Collection[str]) -> float:
    """Set Jaccard similarity; ``0.0`` on an empty union so an empty candidate matches nothing."""
    sa = set(a)
    sb = set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _best_match(
    rows: Sequence[ClusterLedgerRow],
    members: tuple[str, ...],
) -> tuple[ClusterLedgerRow | None, float]:
    """Pick the best row for one candidate and the max Jaccard over every row.

    ``max_jaccard`` is the HIGHEST Jaccard over EVERY row regardless of
    status. ``matched`` is selected from the rows at or above
    ``CLUSTER_MATCH_JACCARD``, preferring ``declined`` over ``proposed``
    (suppression is the safety property and a proposed row must never
    out-score a decline out of existence), then the highest Jaccard, then
    the lexicographically smaller ``cluster_key`` so the result is
    deterministic. NOTE: the caller's reported ``jaccard`` MAY therefore
    exceed ``jaccard(candidate, matched.member_entry_ids)`` when a
    lower-scoring declined row was preferred — a reader will otherwise
    assume the two are equal.
    """
    if not rows:
        return None, 0.0
    scored = [(jaccard(members, row.member_entry_ids), row) for row in rows]
    max_jaccard = max(score for score, _ in scored)
    eligible = [(score, row) for score, row in scored if score >= CLUSTER_MATCH_JACCARD]
    if not eligible:
        return None, max_jaccard
    best = min(
        eligible,
        key=lambda pair: (
            pair[1].status != "declined",
            -pair[0],
            pair[1].cluster_key,
        ),
    )
    return best[1], max_jaccard


# ---------------------------------------------------------------------------
# Timestamp helper (mirrors map_eligibility; module-private there too)
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    """Render an aware ``datetime`` as a UTC ISO-8601 string (house convention)."""
    return dt.astimezone(UTC).isoformat()


def _require_aware(now: datetime) -> None:
    """Raise ``ValueError`` if ``now`` is naive — a naive value would corrupt ordering."""
    if now.tzinfo is None:
        msg = "cluster_ledger: `now` must be timezone-aware"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Audit trail (fire-and-forget, mirrors map_eligibility._record_audit_event)
# ---------------------------------------------------------------------------


async def _record_audit_event(
    db: Database,
    event_type: str,
    entry_id: str | None,
    *,
    detail: str | None = None,
) -> None:
    """Record an audit event. Fire-and-forget — failures never break the caller.

    ``entry_id`` is nullable on both backends, so a cluster-scoped event
    passes ``None`` and needs no DDL change. ``entry_id`` is ``None`` on
    every event this module writes: there is no entry to attribute.
    """
    try:
        created_at = datetime.now(UTC).isoformat()
        await db.execute(
            "INSERT INTO audit_events (event_type, entry_id, contributor, detail, created_at)"
            " VALUES (?, ?, NULL, ?, ?)",
            (event_type, entry_id, detail, created_at),
        )
        await db.commit()
    except Exception:
        logger.warning(
            "%s: failed to record audit event %s for %s",
            CLUSTER_LEDGER_MARKER,
            event_type,
            entry_id,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Corrupt-row parsing (fail-open, mypy-strict clean)
# ---------------------------------------------------------------------------


def _parse_status(raw: object) -> ClusterLedgerStatus:
    """Coerce a raw status column into the Literal — unknown reads as ``proposed``.

    The annotation is what satisfies ``mypy --strict`` on the ``Literal``
    field; the fail direction is deliberate: an unrecognised status must
    never silently suppress work.
    """
    if str(raw) == "declined":
        return "declined"
    return "proposed"


def _parse_members(cluster_key: str, raw: object) -> tuple[str, ...]:
    """Decode the member-id JSON array, or fail open to ``()`` with one warning.

    The column stores ``json.dumps(list(normalize_members(ids)))`` — a
    JSON array of strings, mirroring ``ingested_files.entry_ids``. An
    unreadable member set yields Jaccard 0.0 and therefore never
    suppresses, rather than crashing the 3am loop.
    """
    try:
        decoded = json.loads(str(raw))
        if not isinstance(decoded, list):
            raise TypeError("member_entry_ids is not a JSON array")
        return normalize_members([str(item) for item in decoded])
    except Exception:
        logger.warning(
            "%s: unparseable member_entry_ids on cluster_key=%s — reading as empty",
            CLUSTER_LEDGER_MARKER,
            cluster_key,
        )
        return ()


def _parse_row(row: Row) -> ClusterLedgerRow:
    """Build a ClusterLedgerRow from a raw DB row (named access, house style)."""
    cluster_key = str(row["cluster_key"])
    return ClusterLedgerRow(
        cluster_key=cluster_key,
        project_ref=str(row["project_ref"]),
        member_entry_ids=_parse_members(cluster_key, row["member_entry_ids"]),
        last_label=str(row["last_label"]),
        status=_parse_status(row["status"]),
        sightings=int(row["sightings"]),
        first_seen_at=str(row["first_seen_at"]),
        last_seen_at=str(row["last_seen_at"]),
        declined_member_count=(
            int(row["declined_member_count"]) if row["declined_member_count"] is not None else None
        ),
        declined_reason=(
            str(row["declined_reason"]) if row["declined_reason"] is not None else None
        ),
        declined_by=str(row["declined_by"]) if row["declined_by"] is not None else None,
        declined_at=str(row["declined_at"]) if row["declined_at"] is not None else None,
    )


# ---------------------------------------------------------------------------
# Read CRUD
# ---------------------------------------------------------------------------


_SELECT_COLUMNS_SQL = (
    "SELECT cluster_key, project_ref, member_entry_ids, last_label, status,"
    " sightings, first_seen_at, last_seen_at, declined_member_count,"
    " declined_reason, declined_by, declined_at FROM map_cluster_ledger"
)


async def get_cluster(db: Database, cluster_key: str) -> ClusterLedgerRow | None:
    """Return the ledger row for ``cluster_key``, or ``None`` (PK-equality read)."""
    cursor = await db.execute(
        _SELECT_COLUMNS_SQL + " WHERE cluster_key = ?",
        (cluster_key,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return _parse_row(row)


async def list_clusters(db: Database, project_ref: str) -> list[ClusterLedgerRow]:
    """Return every ledger row for ``project_ref``, sorted by ``cluster_key`` ascending.

    One statement, params exactly ``(project_ref,)``. Sorted in PYTHON so
    no SQL ``ORDER BY`` means no collation dependency — the same reason
    ``map_eligibility.list_overrides`` sorts in Python.
    """
    cursor = await db.execute(
        _SELECT_COLUMNS_SQL + " WHERE project_ref = ?",
        (project_ref,),
    )
    rows = await cursor.fetchall()
    parsed = [_parse_row(row) for row in rows]
    return sorted(parsed, key=lambda row: row.cluster_key)


# ---------------------------------------------------------------------------
# Match (the nightly loop's read)
# ---------------------------------------------------------------------------


async def match_clusters(
    db: Database,
    project_ref: str,
    candidates: Sequence[Sequence[str]],
) -> ClusterLedgerMatchResult:
    """Match a batch of candidate clusters against one project's ledger rows.

    ONE ``list_clusters`` read for the whole batch, then pure Python per
    candidate — never one read per candidate. Verdict order matches
    candidate order exactly. A row in project A NEVER matches a candidate
    submitted under project B: only the submitted project's rows are
    loaded.

    This function writes ``audit_events`` rows (one per near miss,
    marginal match, suppression and reopen — the design's only named
    tuning instrument) and NEVER a ``map_cluster_ledger`` row.

    An EMPTY candidate raises ``ValueError`` (a caller bug, not a shape)
    and so does a batch longer than ``CLUSTER_MATCH_MAX_CANDIDATES`` (the
    caller controls batching). An OVERSIZED candidate — more than
    ``CLUSTER_MEMBER_MAX`` normalized members — is never batch-fatal: the
    design rejected thresholded connected components because they
    collapse, so a model returning one all-in cluster on a homogeneous
    project is a known shape, and raising would cost suppression for
    every other cluster in that project that night. It returns a normal
    verdict with ``oversized=True`` plus one warning.
    """
    if len(candidates) > CLUSTER_MATCH_MAX_CANDIDATES:
        msg = (
            f"cluster_ledger: at most {CLUSTER_MATCH_MAX_CANDIDATES} candidates"
            f" per match call, got {len(candidates)}"
        )
        raise ValueError(msg)

    rows = await list_clusters(db, project_ref)
    verdicts: list[ClusterLedgerVerdict] = []
    for candidate in candidates:
        members = normalize_members(candidate)
        if not members:
            msg = "cluster_ledger: a candidate's member set is empty"
            raise ValueError(msg)
        if len(members) > CLUSTER_MEMBER_MAX:
            logger.warning(
                "%s: outcome=oversized-candidate project_ref=%s candidate_members=%d",
                CLUSTER_LEDGER_MARKER,
                project_ref,
                len(members),
            )
            verdicts.append(
                ClusterLedgerVerdict(
                    member_entry_ids=members,
                    jaccard=0.0,
                    matched=None,
                    suppressed=False,
                    reopened=False,
                    near_miss=False,
                    marginal_match=False,
                    oversized=True,
                )
            )
            continue

        matched, max_jaccard = _best_match(rows, members)
        near_miss = CLUSTER_NEAR_MISS_FLOOR <= max_jaccard < CLUSTER_MATCH_JACCARD
        marginal_match = CLUSTER_MATCH_JACCARD <= max_jaccard < CLUSTER_MARGINAL_MATCH_CEILING
        reopened = bool(
            matched is not None
            and matched.status == "declined"
            and matched.declined_member_count is not None
            and len(members) >= REOPEN_GROWTH_FACTOR * matched.declined_member_count
        )
        suppressed = bool(matched is not None and matched.status == "declined" and not reopened)
        verdicts.append(
            ClusterLedgerVerdict(
                member_entry_ids=members,
                jaccard=max_jaccard,
                matched=matched,
                suppressed=suppressed,
                reopened=reopened,
                near_miss=near_miss,
                marginal_match=marginal_match,
                oversized=False,
            )
        )

        matched_key = "none" if matched is None else matched.cluster_key
        matched_members = 0 if matched is None else len(matched.member_entry_ids)
        if near_miss:
            logger.info(
                "%s: outcome=near-miss project_ref=%s jaccard=%.4f candidate_members=%d"
                " matched_cluster_key=%s matched_members=%d threshold=%.4f"
                " near_miss_floor=%.4f",
                CLUSTER_LEDGER_MARKER,
                project_ref,
                max_jaccard,
                len(members),
                matched_key,
                matched_members,
                CLUSTER_MATCH_JACCARD,
                CLUSTER_NEAR_MISS_FLOOR,
            )
            await _record_audit_event(
                db,
                "map_cluster_near_miss",
                None,
                detail=(
                    f"project_ref={project_ref};jaccard={max_jaccard:.4f};"
                    f"candidate_members={len(members)};matched_cluster_key={matched_key};"
                    f"matched_members={matched_members};"
                    f"threshold={CLUSTER_MATCH_JACCARD:.4f};"
                    f"near_miss_floor={CLUSTER_NEAR_MISS_FLOOR:.4f}"
                ),
            )
        if marginal_match:
            logger.info(
                "%s: outcome=marginal-match project_ref=%s jaccard=%.4f"
                " candidate_members=%d matched_cluster_key=%s matched_members=%d"
                " threshold=%.4f marginal_ceiling=%.4f",
                CLUSTER_LEDGER_MARKER,
                project_ref,
                max_jaccard,
                len(members),
                matched_key,
                matched_members,
                CLUSTER_MATCH_JACCARD,
                CLUSTER_MARGINAL_MATCH_CEILING,
            )
            await _record_audit_event(
                db,
                "map_cluster_marginal_match",
                None,
                detail=(
                    f"project_ref={project_ref};jaccard={max_jaccard:.4f};"
                    f"candidate_members={len(members)};matched_cluster_key={matched_key};"
                    f"matched_members={matched_members};"
                    f"threshold={CLUSTER_MATCH_JACCARD:.4f};"
                    f"marginal_ceiling={CLUSTER_MARGINAL_MATCH_CEILING:.4f}"
                ),
            )
        if suppressed and matched is not None:
            # declined_member_count is None only on a hand-written row
            # (unreachable through the API); render it as `none` rather
            # than crash the batch. reopen_at is the member count the
            # candidate must reach to escape.
            declined_count = matched.declined_member_count
            declined_count_str = "none" if declined_count is None else str(declined_count)
            reopen_at_str = (
                "none" if declined_count is None else str(REOPEN_GROWTH_FACTOR * declined_count)
            )
            logger.info(
                "%s: outcome=suppressed project_ref=%s jaccard=%.4f candidate_members=%d"
                " matched_cluster_key=%s matched_members=%d declined_member_count=%s"
                " declined_at=%s reopen_at=%s",
                CLUSTER_LEDGER_MARKER,
                project_ref,
                max_jaccard,
                len(members),
                matched_key,
                matched_members,
                declined_count_str,
                matched.declined_at or "none",
                reopen_at_str,
            )
            await _record_audit_event(
                db,
                "map_cluster_suppressed",
                None,
                detail=(
                    f"project_ref={project_ref};jaccard={max_jaccard:.4f};"
                    f"candidate_members={len(members)};matched_cluster_key={matched_key};"
                    f"matched_members={matched_members};"
                    f"declined_member_count={declined_count_str};"
                    f"declined_at={matched.declined_at or 'none'};reopen_at={reopen_at_str}"
                ),
            )
        if reopened and matched is not None:
            logger.info(
                "%s: outcome=reopened project_ref=%s jaccard=%.4f candidate_members=%d"
                " matched_cluster_key=%s matched_members=%d declined_member_count=%d"
                " declined_at=%s reopen_at=%d declined_reason=%s",
                CLUSTER_LEDGER_MARKER,
                project_ref,
                max_jaccard,
                len(members),
                matched_key,
                matched_members,
                matched.declined_member_count or 0,
                matched.declined_at or "none",
                REOPEN_GROWTH_FACTOR * (matched.declined_member_count or 0),
                (matched.declined_reason or "")[:200],
            )
            await _record_audit_event(
                db,
                "map_cluster_reopened",
                None,
                detail=(
                    f"project_ref={project_ref};jaccard={max_jaccard:.4f};"
                    f"candidate_members={len(members)};matched_cluster_key={matched_key};"
                    f"matched_members={matched_members};"
                    f"declined_member_count={matched.declined_member_count or 0};"
                    f"declined_at={matched.declined_at or 'none'};"
                    f"reopen_at={REOPEN_GROWTH_FACTOR * (matched.declined_member_count or 0)};"
                    f"declined_reason={(matched.declined_reason or '')[:200]}"
                ),
            )

    return ClusterLedgerMatchResult(
        project_ref=project_ref,
        jaccard_threshold=CLUSTER_MATCH_JACCARD,
        near_miss_floor=CLUSTER_NEAR_MISS_FLOOR,
        marginal_match_ceiling=CLUSTER_MARGINAL_MATCH_CEILING,
        reopen_growth_factor=REOPEN_GROWTH_FACTOR,
        rows_considered=len(rows),
        verdicts=tuple(verdicts),
    )


# ---------------------------------------------------------------------------
# Write CRUD
# ---------------------------------------------------------------------------


_UPDATE_SIGHTING_SQL = (
    "UPDATE map_cluster_ledger SET member_entry_ids = ?, last_label = ?,"
    " last_seen_at = ?, sightings = sightings + 1 WHERE cluster_key = ?"
)

_INSERT_SIGHTING_SQL = (
    "INSERT INTO map_cluster_ledger"
    " (cluster_key, project_ref, member_entry_ids, last_label, status, sightings,"
    " first_seen_at, last_seen_at, declined_member_count, declined_reason,"
    " declined_by, declined_at)"
    " VALUES (?, ?, ?, ?, 'proposed', 1, ?, ?, NULL, NULL, NULL, NULL)"
    " ON CONFLICT(cluster_key) DO UPDATE SET"
    " member_entry_ids = excluded.member_entry_ids, last_label = excluded.last_label,"
    " last_seen_at = excluded.last_seen_at,"
    " sightings = map_cluster_ledger.sightings + 1"
)


async def record_cluster(
    db: Database,
    project_ref: str,
    member_entry_ids: Sequence[str],
    label: str,
    *,
    now: datetime,
) -> ClusterLedgerRow:
    """Record one sighting of a candidate cluster and return the observed row.

    Matches first. IF a row matches, UPDATE it by ``cluster_key`` —
    ``member_entry_ids``, ``last_label``, ``last_seen_at`` and
    ``sightings + 1`` only, leaving ``cluster_key``, ``project_ref``,
    ``first_seen_at``, ``status`` and all four ``declined_*`` columns
    UNTOUCHED. The declined-over-proposed precedence of ``_best_match``
    applies here too, so a sighting can attach to a declined row in
    preference to a better-overlapping proposed one — which is correct,
    because re-arming a decline at the new member size is the behaviour
    the reopen rule pins, and a sighting never clears a decline. ELSE
    INSERT with ``status = 'proposed'``.

    The INSERT is KEY-SAFE and the old 'unreachable' claim is FALSE. The
    UPDATE branch mutates ``member_entry_ids`` while leaving
    ``cluster_key`` alone (the birth-hash invariant), so a row's stored
    members drift away from the set that minted its key and a later
    recurrence of the ORIGINAL set can recompute the SAME key. Guarded
    TWICE: the key already present in the in-memory row list routes to the
    UPDATE branch, and the INSERT itself is ``ON CONFLICT(cluster_key)
    DO UPDATE``.

    ``ValueError`` on an empty set or one longer than
    ``CLUSTER_MEMBER_MAX`` — here it DOES raise, because a single
    candidate is the whole request, so 422 is the honest answer.
    """
    _require_aware(now)
    members = normalize_members(member_entry_ids)
    if not members:
        msg = "cluster_ledger: a cluster's member set is empty"
        raise ValueError(msg)
    if len(members) > CLUSTER_MEMBER_MAX:
        msg = (
            f"cluster_ledger: at most {CLUSTER_MEMBER_MAX} members per cluster, got {len(members)}"
        )
        raise ValueError(msg)
    ts = _iso(now)

    rows = await list_clusters(db, project_ref)
    matched, max_jaccard = _best_match(rows, members)

    prior: ClusterLedgerRow | None
    if matched is not None:
        prior = matched
    else:
        new_key = cluster_key_for(project_ref, members)
        colliding = next((r for r in rows if r.cluster_key == new_key), None)
        prior = colliding

    if prior is not None:
        await db.execute(
            _UPDATE_SIGHTING_SQL,
            (json.dumps(list(members)), label, ts, prior.cluster_key),
        )
        await db.commit()
        observed = await get_cluster(db, prior.cluster_key)
        if observed is None:
            logger.error(
                "%s: invariant=row_vanished cluster_key=%s",
                CLUSTER_LEDGER_MARKER,
                prior.cluster_key,
            )
            msg = f"cluster_ledger: row {prior.cluster_key} vanished mid-write"
            raise RuntimeError(msg)
        # Decline-clobber tripwire: the pre-UPDATE values are already in
        # hand from list_clusters, so one extra PK-equality read turns the
        # returned row into an observation of the DB rather than a
        # Python-side claim about it. If a later edit to the SET clause
        # ever touches one of these columns, every declined cluster
        # silently un-declines and somnus re-proposes everything the
        # human declined — advisory, never an exception.
        clobbered = (
            observed.status != prior.status
            or observed.first_seen_at != prior.first_seen_at
            or observed.declined_member_count != prior.declined_member_count
            or observed.declined_reason != prior.declined_reason
            or observed.declined_by != prior.declined_by
            or observed.declined_at != prior.declined_at
        )
        if clobbered:
            logger.error(
                "%s: invariant=decline_clobbered cluster_key=%s was_status=%s"
                " now_status=%s was_declined_at=%s now_declined_at=%s",
                CLUSTER_LEDGER_MARKER,
                prior.cluster_key,
                prior.status,
                observed.status,
                prior.declined_at or "none",
                observed.declined_at or "none",
            )
            await _record_audit_event(
                db,
                "map_cluster_decline_clobbered",
                None,
                detail=(
                    f"cluster_key={prior.cluster_key};was_status={prior.status};"
                    f"now_status={observed.status};"
                    f"was_declined_at={prior.declined_at or 'none'};"
                    f"now_declined_at={observed.declined_at or 'none'}"
                ),
            )
        row = observed
    else:
        key = cluster_key_for(project_ref, members)
        await db.execute(
            _INSERT_SIGHTING_SQL,
            (key, project_ref, json.dumps(list(members)), label, ts, ts),
        )
        await db.commit()
        inserted = await get_cluster(db, key)
        if inserted is None:
            logger.error("%s: invariant=row_vanished cluster_key=%s", CLUSTER_LEDGER_MARKER, key)
            msg = f"cluster_ledger: row {key} vanished mid-write"
            raise RuntimeError(msg)
        row = inserted

    await _record_audit_event(
        db,
        "map_cluster_ledger_recorded",
        None,
        detail=(
            f"project_ref={project_ref};cluster_key={row.cluster_key};"
            f"members={len(row.member_entry_ids)};sightings={row.sightings};"
            f"status={row.status};jaccard={max_jaccard:.4f};label={label[:200]}"
        ),
    )
    if row.status == "proposed" and row.sightings >= PROPOSED_SIGHTINGS_WARN:
        logger.warning(
            "%s: outcome=proposed-stalled project_ref=%s cluster_key=%s sightings=%d"
            " first_seen_at=%s last_label=%s",
            CLUSTER_LEDGER_MARKER,
            project_ref,
            row.cluster_key,
            row.sightings,
            row.first_seen_at,
            row.last_label[:200],
        )
        await _record_audit_event(
            db,
            "map_cluster_proposed_stalled",
            None,
            detail=(
                f"project_ref={project_ref};cluster_key={row.cluster_key};"
                f"sightings={row.sightings};first_seen_at={row.first_seen_at};"
                f"last_label={row.last_label[:200]}"
            ),
        )
    return row


async def decline_cluster(
    db: Database,
    cluster_key: str,
    *,
    reason: str,
    declined_by: str | None = None,
    now: datetime,
) -> ClusterLedgerRow | None:
    """Decline one ledger row — the human's verdict. Returns ``None`` if absent.

    ``declined_member_count`` is read FROM THE ROW, never caller-supplied:
    a caller-supplied count could lie the reopen rule into never firing.
    """
    _require_aware(now)
    row = await get_cluster(db, cluster_key)
    if row is None:
        return None
    was = row.status
    await db.execute(
        "UPDATE map_cluster_ledger SET status = 'declined',"
        " declined_member_count = ?, declined_reason = ?, declined_by = ?,"
        " declined_at = ? WHERE cluster_key = ?",
        (len(row.member_entry_ids), reason, declined_by, _iso(now), cluster_key),
    )
    await db.commit()
    await _record_audit_event(
        db,
        "map_cluster_ledger_declined",
        None,
        detail=(
            f"cluster_key={cluster_key};project_ref={row.project_ref};"
            f"declined_member_count={len(row.member_entry_ids)};was={was};"
            f"reason={reason[:200]}"
        ),
    )
    declined = await get_cluster(db, cluster_key)
    if declined is None:
        logger.error(
            "%s: invariant=row_vanished cluster_key=%s",
            CLUSTER_LEDGER_MARKER,
            cluster_key,
        )
        msg = f"cluster_ledger: row {cluster_key} vanished mid-write"
        raise RuntimeError(msg)
    return declined


async def clear_cluster(db: Database, cluster_key: str) -> bool:
    """Delete one ledger row.

    Returns ``True`` iff a row was actually deleted, and audits the
    cleared row only in that case.
    """
    prior = await get_cluster(db, cluster_key)
    cursor = await db.execute(
        "DELETE FROM map_cluster_ledger WHERE cluster_key = ?", (cluster_key,)
    )
    await db.commit()
    if cursor.rowcount == 0:
        return False
    if prior is not None:
        await _record_audit_event(
            db,
            "map_cluster_ledger_cleared",
            None,
            detail=(
                f"cluster_key={cluster_key};project_ref={prior.project_ref};"
                f"was={prior.status};members={len(prior.member_entry_ids)}"
            ),
        )
    return True
