"""Map-eligibility predicate + human override table (nightly map maintenance).

Decides which projects deserve ``mental_map`` maintenance, and records the
human verdicts that override that computation. The predicate is computed from
per-project counts (``Database.map_eligibility_counts``); a
``map_eligibility_override`` row (one per project_ref) is a human decision
record that wins over the computed value in BOTH directions. The audit trail
(``map_eligibility_override_set`` / ``map_eligibility_override_cleared``
events) is why the table can be current-state-only: prior verdicts live in
``audit_events``, not in the table.

Wire contract of record — the API and MCP consumer items cite these names
verbatim; nothing in this module serializes anything. ``MapEligibilityVerdict``
serializes with ``dataclasses.asdict()`` to exactly ``{"evidence":
{"project_ref", "mappable", "ingested", "hand_authored", "maps",
"top_prefix", "top_prefix_share", "is_ingest_corpus", "is_too_thin",
"is_journal", "computed_eligible"}, "override": {"project_ref", "eligible",
"reason", "set_by", "set_at"} | null, "effective_eligible": bool,
"decided_by": "computed" | "override", "orphaned": bool}``. Consumers MUST
NOT rename, re-nest, flatten or round any field; ``top_prefix_share`` crosses
the wire as an unrounded float and ``set_at`` as the stored ISO-8601 string.
``maps`` is in the contract because the review question is "which projects
deserve maps", i.e. eligible AND under-mapped: measured live, 21 of the 25
eligible project_refs have ZERO ``mental_map`` entries, and without this
field the consumer would either loop ``KnowledgeBase.maps_for_project`` once
per project or write its own SQL.

Calibration (as measured 2026-09-19; the live corpus grows daily — these
counts are documentation, not assertions): 47 project_refs carry at least
one mappable entry; 25 are computed-eligible covering 1,053 mappable entries;
21 are excluded by ``is_too_thin`` — 5 ingest corpora (harness-design-
research 962/962 ingested, pdf-examples 364/364, ml-papers 41/41, history
33/33, karabiner_pro_ui 9/9) plus 16 thin projects with 1-4 hand-authored
entries and zero ingested; and exactly 1, ``dispatch-performance-log``, is
excluded as a journal (624 mappable, top prefix ``Run`` on 611 of them,
share 0.9792); 25 + 21 + 1 = 47. The durable structural facts:
``dispatch-performance-log`` is the ONLY journal — the next-highest prefix
share among projects with ``mappable >= 20`` is ``sample-project`` at 0.1739, so
the 0.60 threshold has roughly a 5x margin on both sides; ``threat-intel``
sits exactly at the ``hand_authored = 5`` floor (45 mappable, 40 ingested);
and ``harness-design`` (71 mappable, 0 ingested, 3 as its top prefix count)
passes every gate although it is a dated session journal — ``threat-intel``
and ``harness-design`` are what the override table exists to settle.

The title prefix is everything before the first ``:``. The separator lives
in each backend's SQL (``sqlite_backend`` / ``postgres_backend``) because
kb-core SQL is not string-built from values. A project_ref that has mental
maps but zero mappable entries is absent from the counts query entirely and
reports ``maps = 0`` in a synthesized override-only row — flagged
``orphaned`` on the verdict. No nightly loop lives here; it is a later item,
and ``resolve_eligibility`` returns every verdict with the caller left to
filter.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from kb_core.db.backend import Database

logger = logging.getLogger(__name__)

# Greppable marker — mirrors kb_core.embedding_retry.EMBEDDING_RETRY_MARKER.
MAP_ELIGIBILITY_MARKER = "map-eligibility"

# A project needs at least this many hand-authored (non-ingested) mappable
# entries to be worth partitioning into maps.
MIN_HAND_AUTHORED = 5

# A project this large whose dominant title prefix claims at least
# JOURNAL_MIN_PREFIX_SHARE of its mappable entries is a dated session
# journal, not a topical project.
JOURNAL_MIN_MAPPABLE = 20
JOURNAL_MIN_PREFIX_SHARE = 0.60


@dataclasses.dataclass(frozen=True)
class MapEligibilityEvidence:
    """Computed evidence for one project's map eligibility."""

    project_ref: str
    mappable: int
    ingested: int
    hand_authored: int
    maps: int
    top_prefix: str
    top_prefix_share: float
    is_ingest_corpus: bool
    """Reporting-only: every mappable entry was ingested (hand_authored == 0).

    Appears in NO verdict term — every live corpus also fails the thinness
    test, so gating on it would never fire.
    """

    is_too_thin: bool
    """Fewer than MIN_HAND_AUTHORED hand-authored entries — the verdict gate.

    Covers BOTH an ingested reference corpus (hand_authored 0) and a project
    too thin to partition (1-4 hand-authored entries). As measured 2026-09-19,
    21 live project_refs trip this flag, of which 5 are actual ingest corpora
    — the other 16 are one-to-four-entry projects.
    """

    is_journal: bool
    """A dated session journal: >= JOURNAL_MIN_MAPPABLE mappable entries and
    one title prefix claiming >= JOURNAL_MIN_PREFIX_SHARE of them."""

    computed_eligible: bool
    """The computed verdict: not too thin and not a journal."""


@dataclasses.dataclass(frozen=True)
class MapEligibilityOverride:
    """One human verdict row from ``map_eligibility_override``."""

    project_ref: str
    eligible: bool
    reason: str
    set_by: str | None
    set_at: str


@dataclasses.dataclass(frozen=True)
class MapEligibilityVerdict:
    """Resolved eligibility for one project: evidence + override + outcome."""

    evidence: MapEligibilityEvidence
    override: MapEligibilityOverride | None
    effective_eligible: bool
    decided_by: Literal["computed", "override"]
    orphaned: bool


# ---------------------------------------------------------------------------
# Pure classifier
# ---------------------------------------------------------------------------


def classify(
    project_ref: str,
    *,
    mappable: int,
    ingested: int,
    top_prefix_count: int,
    top_prefix: str,
    maps: int,
) -> MapEligibilityEvidence:
    """Compute the eligibility evidence for one project from its counts.

    Pure — no database, so it is separately unit-testable. No rounding
    anywhere. The ``mappable == 0`` guard on the share is mandatory:
    override-only rows synthesized by :func:`resolve_eligibility` arrive
    with zero mappable entries and must not raise.
    """
    hand_authored = mappable - ingested
    top_prefix_share = top_prefix_count / mappable if mappable else 0.0
    is_ingest_corpus = mappable > 0 and ingested == mappable
    is_too_thin = hand_authored < MIN_HAND_AUTHORED
    is_journal = mappable >= JOURNAL_MIN_MAPPABLE and top_prefix_share >= JOURNAL_MIN_PREFIX_SHARE
    computed_eligible = not is_too_thin and not is_journal
    return MapEligibilityEvidence(
        project_ref=project_ref,
        mappable=mappable,
        ingested=ingested,
        hand_authored=hand_authored,
        maps=maps,
        top_prefix=top_prefix,
        top_prefix_share=top_prefix_share,
        is_ingest_corpus=is_ingest_corpus,
        is_too_thin=is_too_thin,
        is_journal=is_journal,
        computed_eligible=computed_eligible,
    )


async def compute_evidence(db: Database) -> list[MapEligibilityEvidence]:
    """Classify every project that has at least one mappable entry.

    Sorted by ``project_ref`` ascending (Python code-point order, so the
    ordering is backend-independent even though the counts query itself
    orders in SQL).
    """
    counts = await db.map_eligibility_counts()
    evidence = [
        classify(
            c.project_ref,
            mappable=c.mappable,
            ingested=c.ingested,
            top_prefix_count=c.top_prefix_count,
            top_prefix=c.top_prefix,
            maps=c.maps,
        )
        for c in counts
    ]
    return sorted(evidence, key=lambda e: e.project_ref)


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    """Render an aware ``datetime`` as a UTC ISO-8601 string (house convention)."""
    return dt.astimezone(UTC).isoformat()


def _require_aware(now: datetime) -> None:
    """Raise ``ValueError`` if ``now`` is naive — a naive value would corrupt ordering."""
    if now.tzinfo is None:
        msg = "map_eligibility: `now` must be timezone-aware"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Audit trail (fire-and-forget, mirrors embedding_retry._record_audit_event)
# ---------------------------------------------------------------------------


async def _record_audit_event(
    db: Database,
    event_type: str,
    entry_id: str | None,
    *,
    detail: str | None = None,
) -> None:
    """Record an audit event. Fire-and-forget — failures never break the caller.

    ``entry_id`` is nullable on both backends, so a project-scoped event
    passes ``None`` and needs no DDL change.
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
            MAP_ELIGIBILITY_MARKER,
            event_type,
            entry_id,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Override CRUD
# ---------------------------------------------------------------------------


async def get_override(db: Database, project_ref: str) -> MapEligibilityOverride | None:
    """Return the override row for ``project_ref``, or ``None`` if there is none."""
    cursor = await db.execute(
        "SELECT project_ref, eligible, reason, set_by, set_at"
        " FROM map_eligibility_override WHERE project_ref = ?",
        (project_ref,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return MapEligibilityOverride(
        project_ref=str(row["project_ref"]),
        eligible=bool(row["eligible"]),
        reason=str(row["reason"]),
        set_by=row["set_by"],
        set_at=str(row["set_at"]),
    )


async def list_overrides(db: Database) -> list[MapEligibilityOverride]:
    """Return every override row, sorted by ``project_ref`` ascending."""
    cursor = await db.execute(
        "SELECT project_ref, eligible, reason, set_by, set_at FROM map_eligibility_override"
    )
    rows = await cursor.fetchall()
    overrides = [
        MapEligibilityOverride(
            project_ref=str(row["project_ref"]),
            eligible=bool(row["eligible"]),
            reason=str(row["reason"]),
            set_by=row["set_by"],
            set_at=str(row["set_at"]),
        )
        for row in rows
    ]
    return sorted(overrides, key=lambda o: o.project_ref)


async def set_override(
    db: Database,
    project_ref: str,
    *,
    eligible: bool,
    reason: str,
    set_by: str | None = None,
    now: datetime,
) -> MapEligibilityOverride:
    """Upsert the human override for ``project_ref`` and audit the transition.

    Reads the prior row first so the audit detail records the old verdict —
    the audit trail is why the table can be current-state-only.
    """
    _require_aware(now)
    prior = await get_override(db, project_ref)
    was = "absent" if prior is None else str(int(prior.eligible))
    await db.execute(
        "INSERT INTO map_eligibility_override (project_ref, eligible, reason, set_by, set_at)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(project_ref) DO UPDATE SET"
        " eligible = excluded.eligible, reason = excluded.reason,"
        " set_by = excluded.set_by, set_at = excluded.set_at",
        (project_ref, int(eligible), reason, set_by, _iso(now)),
    )
    await db.commit()
    await _record_audit_event(
        db,
        "map_eligibility_override_set",
        None,
        detail=(
            f"project_ref={project_ref};eligible={int(eligible)};was={was};reason={reason[:200]}"
        ),
    )
    return MapEligibilityOverride(
        project_ref=project_ref,
        eligible=eligible,
        reason=reason,
        set_by=set_by,
        set_at=_iso(now),
    )


async def clear_override(db: Database, project_ref: str) -> bool:
    """Delete the override row for ``project_ref``.

    Returns ``True`` iff a row was actually deleted, and audits the
    cleared verdict only in that case.
    """
    prior = await get_override(db, project_ref)
    cursor = await db.execute(
        "DELETE FROM map_eligibility_override WHERE project_ref = ?", (project_ref,)
    )
    await db.commit()
    if cursor.rowcount == 0:
        return False
    if prior is not None:
        await _record_audit_event(
            db,
            "map_eligibility_override_cleared",
            None,
            detail=(
                f"project_ref={project_ref};was={int(prior.eligible)};reason={prior.reason[:200]}"
            ),
        )
    return True


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


async def resolve_eligibility(db: Database) -> list[MapEligibilityVerdict]:
    """Resolve a verdict for the union of computed projects and override rows.

    One verdict per project_ref that has mappable entries OR an override
    row, sorted by ``project_ref`` ascending. Precedence: the override wins
    over the computed value in BOTH directions; a force-ineligible override
    does NOT hide the row — the verdict always carries its full evidence so
    the review surface can see what it is overriding. A project_ref with an
    override row but zero mappable entries gets a synthesized zero-evidence
    row with ``orphaned = True`` (a misspelled or renamed ref would
    otherwise look like a successful write that does nothing forever).
    """
    evidence_rows = await compute_evidence(db)
    by_ref = {e.project_ref: e for e in evidence_rows}
    overrides = await list_overrides(db)
    override_by_ref = {o.project_ref: o for o in overrides}

    verdicts: list[MapEligibilityVerdict] = []
    for project_ref in sorted(set(by_ref) | set(override_by_ref)):
        evidence = by_ref.get(project_ref)
        orphaned = evidence is None
        if evidence is None:
            evidence = classify(
                project_ref,
                mappable=0,
                ingested=0,
                top_prefix_count=0,
                top_prefix="",
                maps=0,
            )
        override = override_by_ref.get(project_ref)
        decided_by: Literal["computed", "override"] = (
            "override" if override is not None else "computed"
        )
        effective = override.eligible if override is not None else evidence.computed_eligible
        verdicts.append(
            MapEligibilityVerdict(
                evidence=evidence,
                override=override,
                effective_eligible=effective,
                decided_by=decided_by,
                orphaned=orphaned,
            )
        )
        logger.debug(
            "%s: project_ref=%s effective_eligible=%s decided_by=%s mappable=%d "
            "hand_authored=%d top_prefix=%r top_prefix_share=%s is_ingest_corpus=%s "
            "is_too_thin=%s is_journal=%s",
            MAP_ELIGIBILITY_MARKER,
            project_ref,
            effective,
            decided_by,
            evidence.mappable,
            evidence.hand_authored,
            evidence.top_prefix,
            evidence.top_prefix_share,
            evidence.is_ingest_corpus,
            evidence.is_too_thin,
            evidence.is_journal,
        )

    for project_ref in sorted(set(override_by_ref) - set(by_ref)):
        logger.warning(
            "%s: override for project_ref %r has 0 mappable entries "
            "(invariant breach - stale, renamed or misspelled ref)",
            MAP_ELIGIBILITY_MARKER,
            project_ref,
        )

    eligible = sum(1 for v in verdicts if v.effective_eligible)
    logger.info(
        "%s: %d project_refs, %d eligible (%d computed, %d override), "
        "%d excluded as too thin, %d as journal, %d ingest corpora, %d active overrides",
        MAP_ELIGIBILITY_MARKER,
        len(verdicts),
        eligible,
        sum(1 for v in verdicts if v.effective_eligible and v.decided_by == "computed"),
        sum(1 for v in verdicts if v.effective_eligible and v.decided_by == "override"),
        sum(1 for v in verdicts if v.evidence.is_too_thin),
        sum(1 for v in verdicts if v.evidence.is_journal),
        sum(1 for v in verdicts if v.evidence.is_ingest_corpus),
        len(overrides),
    )
    return verdicts
