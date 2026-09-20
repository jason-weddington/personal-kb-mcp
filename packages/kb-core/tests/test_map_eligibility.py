"""Hermetic SQLite suite for the map-eligibility predicate + override table.

No Postgres, no Ollama, no network: everything runs against a real SQLite
KB created via ``create_sqlite(tmp_path / "kb.db")``. The eight-project
corpus and the expected counts live in ``map_eligibility_counts_fixture.py``
— the SAME lists the Postgres counts test seeds from, so a dialect drift
fails wherever Postgres runs.

Seeding uses direct ``kb.db.execute`` INSERTs, never ``kb.store()`` — the
store touches the embedder, and these tests are about SQL arithmetic.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from kb_core import create_sqlite
from kb_core.map_eligibility import (
    MIN_HAND_AUTHORED,
    classify,
    clear_override,
    compute_evidence,
    get_override,
    list_overrides,
    resolve_eligibility,
    set_override,
)
from map_eligibility_counts_fixture import EXPECTED_COUNTS, SEED_ENTRIES, SEED_INGESTED_FILES

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from kb_core.db.backend import Database
    from kb_core.knowledge_base import KnowledgeBase
    from kb_core.map_eligibility import MapEligibilityEvidence

_ENTRY_INSERT_SQL = (
    "INSERT INTO knowledge_entries"
    " (id, project_ref, short_title, long_title, knowledge_details, entry_type,"
    " created_at, updated_at, is_active)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_FILE_INSERT_SQL = (
    "INSERT INTO ingested_files"
    " (relative_path, content_hash, note_node_id, entry_ids, summary, file_size,"
    " file_extension, ingested_at, updated_at, is_active)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


async def _seed(db: Database) -> None:
    """Insert the shared eight-project corpus via direct SQL."""
    for row in SEED_ENTRIES:
        await db.execute(_ENTRY_INSERT_SQL, row)
    for row in SEED_INGESTED_FILES:
        await db.execute(_FILE_INSERT_SQL, row)
    await db.commit()


@pytest.fixture
async def seeded_kb(tmp_path: Path) -> AsyncIterator[KnowledgeBase]:
    """A SQLite KB with the shared eight-project corpus seeded."""
    kb = await create_sqlite(tmp_path / "kb.db")
    await _seed(kb.db)
    try:
        yield kb
    finally:
        await kb.close()


async def test_map_eligibility_counts_matches_expected(seeded_kb: KnowledgeBase) -> None:
    """The SQLite counts query returns the shared EXPECTED_COUNTS exactly."""
    assert await seeded_kb.db.map_eligibility_counts() == list(EXPECTED_COUNTS)


async def test_compute_evidence_per_project(seeded_kb: KnowledgeBase) -> None:
    """Eight rows, ascending project_ref order, with the pinned per-project evidence."""
    evidence = await compute_evidence(seeded_kb.db)
    by_ref = {e.project_ref: e for e in evidence}

    assert [e.project_ref for e in evidence] == [
        "p-corpus",
        "p-dup-ingest",
        "p-eligible-mixed",
        "p-inactive-file",
        "p-journal",
        "p-journal-sub20",
        "p-leading-colon",
        "p-thin",
    ]
    # No row may carry a NULL project_ref (three such entries were seeded).
    assert all(e.project_ref for e in evidence)

    corpus = by_ref["p-corpus"]
    assert (corpus.mappable, corpus.ingested, corpus.hand_authored) == (6, 6, 0)
    assert corpus.is_ingest_corpus is True
    assert corpus.is_too_thin is True
    assert corpus.is_journal is False
    assert corpus.computed_eligible is False

    # Without the DISTINCT in `ing`, the duplicated file row inflates BOTH
    # mappable (5 -> 6) and ingested (1 -> 2).
    dup = by_ref["p-dup-ingest"]
    assert (dup.mappable, dup.ingested) == (5, 1)
    assert dup.hand_authored == 4
    assert dup.is_too_thin is True
    assert dup.computed_eligible is False

    # Ingested entries COUNT toward an eligible project; the mental_map and
    # is_active=0 entries count nowhere.
    mixed = by_ref["p-eligible-mixed"]
    assert (mixed.mappable, mixed.ingested, mixed.hand_authored, mixed.maps) == (10, 5, 5, 1)
    assert mixed.top_prefix == "Mixed note 01"
    assert mixed.top_prefix_share == pytest.approx(0.1)
    assert mixed.is_ingest_corpus is False
    assert mixed.is_too_thin is False
    assert mixed.computed_eligible is True

    # Without the f.is_active filter on ingested_files this flips to ineligible.
    inactive = by_ref["p-inactive-file"]
    assert (inactive.mappable, inactive.ingested, inactive.hand_authored) == (5, 0, 5)
    assert inactive.computed_eligible is True

    journal = by_ref["p-journal"]
    assert (journal.mappable, journal.top_prefix) == (20, "Run")
    assert journal.top_prefix_share == 1.0
    assert journal.is_journal is True
    assert journal.computed_eligible is False

    # 19 entries pins the `mappable >= 20` journal guard.
    sub = by_ref["p-journal-sub20"]
    assert (sub.mappable, sub.top_prefix_share) == (19, 1.0)
    assert sub.is_journal is False
    assert sub.computed_eligible is True

    # The leading-colon prefix is the empty string and the byte-ordered
    # tie-break selects it deterministically over "zzz other".
    colon = by_ref["p-leading-colon"]
    assert colon.top_prefix == ""
    assert colon.top_prefix_share == pytest.approx(0.5)
    assert colon.is_too_thin is True
    assert colon.computed_eligible is False

    thin = by_ref["p-thin"]
    assert (thin.mappable, thin.ingested, thin.hand_authored) == (9, 5, 4)
    assert thin.is_ingest_corpus is False
    assert thin.is_too_thin is True
    assert thin.computed_eligible is False

    assert sum(1 for e in evidence if e.computed_eligible) == 3
    assert sum(1 for e in evidence if e.is_too_thin) == 4
    assert sum(1 for e in evidence if e.is_journal) == 1
    assert sum(1 for e in evidence if e.is_ingest_corpus) == 1
    # Every project is eligible, too thin, or a journal — no fourth state.
    assert (
        sum(1 for e in evidence if e.computed_eligible or e.is_too_thin or e.is_journal)
        == len(evidence)
        == 8
    )


# ---------------------------------------------------------------------------
# Pure classifier boundaries
# ---------------------------------------------------------------------------


def _classify(mappable: int, ingested: int, top_prefix_count: int) -> MapEligibilityEvidence:
    """Call classify with neutral filler for the non-boundary kwargs."""
    return classify(
        "p-x",
        mappable=mappable,
        ingested=ingested,
        top_prefix_count=top_prefix_count,
        top_prefix="Run",
        maps=0,
    )


def test_classify_boundaries() -> None:
    """The pinned boundary table for the three gates."""
    # 12/20 is exactly the same double as the literal 0.6 -> journal.
    at_threshold = _classify(20, 0, 12)
    assert at_threshold.top_prefix_share == 0.6
    assert at_threshold.is_journal is True

    below_share = _classify(20, 0, 11)
    assert below_share.top_prefix_share == 0.55
    assert below_share.is_journal is False

    below_mappable = _classify(19, 0, 19)
    assert below_mappable.is_journal is False

    at_hand_floor = _classify(9, 4, 1)
    assert at_hand_floor.hand_authored == 5
    assert at_hand_floor.is_too_thin is False

    below_hand_floor = _classify(9, 5, 1)
    assert below_hand_floor.hand_authored == 4
    assert below_hand_floor.is_too_thin is True

    assert _classify(6, 6, 1).is_ingest_corpus is True
    assert _classify(10, 5, 1).is_ingest_corpus is False

    zero = _classify(0, 0, 0)
    assert zero.top_prefix_share == 0.0
    assert zero.is_ingest_corpus is False
    assert zero.is_too_thin is True
    assert MIN_HAND_AUTHORED == 5


# ---------------------------------------------------------------------------
# Override / resolution matrix
# ---------------------------------------------------------------------------


async def test_no_override_decided_by_computed(seeded_kb: KnowledgeBase) -> None:
    """With an empty override table every verdict is computed and not orphaned."""
    verdicts = await resolve_eligibility(seeded_kb.db)
    assert len(verdicts) == 8
    assert all(v.decided_by == "computed" for v in verdicts)
    assert all(v.override is None for v in verdicts)
    assert all(v.orphaned is False for v in verdicts)
    assert all(v.effective_eligible == v.evidence.computed_eligible for v in verdicts)


async def test_override_forces_both_directions(seeded_kb: KnowledgeBase) -> None:
    """An override wins over the computed value when forcing IN and OUT."""
    db = seeded_kb.db
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)

    # Force OUT an eligible project.
    stored = await set_override(
        db,
        "p-eligible-mixed",
        eligible=False,
        reason="dated session journal, no topical partition",
        now=now,
    )
    assert stored.set_at == "2026-03-01T12:00:00+00:00"

    # Force IN an ineligible (too-thin ingest corpus) project.
    await set_override(db, "p-corpus", eligible=True, reason="reference corpus", now=now)

    verdicts = {v.evidence.project_ref: v for v in await resolve_eligibility(db)}
    forced_out = verdicts["p-eligible-mixed"]
    assert forced_out.effective_eligible is False
    assert forced_out.decided_by == "override"
    assert forced_out.override is not None
    assert forced_out.override.reason == "dated session journal, no topical partition"
    assert forced_out.override.set_at == "2026-03-01T12:00:00+00:00"

    forced_in = verdicts["p-corpus"]
    assert forced_in.effective_eligible is True
    assert forced_in.decided_by == "override"
    # The overridden row still carries its full computed evidence.
    assert forced_in.evidence.is_ingest_corpus is True


async def test_override_upsert_and_set_by_roundtrip(seeded_kb: KnowledgeBase) -> None:
    """A second set_override upserts in place; set_by round-trips both ways."""
    db = seeded_kb.db
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    later = datetime(2026, 3, 2, 12, 0, tzinfo=UTC)

    await set_override(db, "p-thin", eligible=True, reason="first", set_by="a@b.c", now=now)
    await set_override(db, "p-thin", eligible=True, reason="second", now=later)

    overrides = await list_overrides(db)
    assert len(overrides) == 1
    assert overrides[0].reason == "second"
    assert overrides[0].set_at == "2026-03-02T12:00:00+00:00"

    # set_by=None stores SQL NULL and reads back as None (not "").
    await set_override(db, "p-journal", eligible=True, reason="r", set_by=None, now=now)
    row = await get_override(db, "p-journal")
    assert row is not None
    assert row.set_by is None

    await set_override(db, "p-thin", eligible=True, reason="third", set_by="a@b.c", now=later)
    row = await get_override(db, "p-thin")
    assert row is not None
    assert row.set_by == "a@b.c"

    # No row -> None.
    assert await get_override(db, "p-ghost") is None


async def test_clear_override_reverts_to_computed(seeded_kb: KnowledgeBase) -> None:
    """clear_override deletes once, reports False the second time, and reverts."""
    db = seeded_kb.db
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    await set_override(db, "p-eligible-mixed", eligible=False, reason="r", now=now)

    assert await clear_override(db, "p-eligible-mixed") is True
    assert await clear_override(db, "p-eligible-mixed") is False

    verdicts = {v.evidence.project_ref: v for v in await resolve_eligibility(db)}
    assert verdicts["p-eligible-mixed"].decided_by == "computed"
    assert verdicts["p-eligible-mixed"].effective_eligible is True
    assert await list_overrides(db) == []


async def test_orphaned_override_synthesizes_zero_evidence(seeded_kb: KnowledgeBase) -> None:
    """An override on a ref with zero entries yields a 9th, orphaned verdict."""
    db = seeded_kb.db
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    await set_override(db, "p-ghost", eligible=True, reason="alias of p-journal", now=now)

    verdicts = {v.evidence.project_ref: v for v in await resolve_eligibility(db)}
    assert len(verdicts) == 9

    ghost = verdicts["p-ghost"]
    assert ghost.orphaned is True
    assert ghost.effective_eligible is True  # the override is honoured
    assert ghost.decided_by == "override"
    assert (ghost.evidence.mappable, ghost.evidence.ingested, ghost.evidence.maps) == (0, 0, 0)
    assert ghost.evidence.top_prefix == ""
    assert ghost.evidence.top_prefix_share == 0.0
    assert ghost.evidence.is_ingest_corpus is False
    assert ghost.evidence.is_too_thin is True
    assert ghost.evidence.is_journal is False
    assert ghost.evidence.computed_eligible is False

    assert sum(1 for v in verdicts.values() if v.orphaned) == 1


async def test_set_override_requires_aware_now(seeded_kb: KnowledgeBase) -> None:
    """A naive `now` is rejected before any write lands."""
    with pytest.raises(ValueError, match="map_eligibility"):
        await set_override(
            seeded_kb.db,
            "p-corpus",
            eligible=True,
            reason="r",
            now=datetime(2026, 1, 1),  # deliberately naive
        )
    assert await get_override(seeded_kb.db, "p-corpus") is None


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


async def _audit_details(db: Database, event_type: str) -> list[str | None]:
    cursor = await db.execute(
        "SELECT detail FROM audit_events WHERE event_type = ? ORDER BY id", (event_type,)
    )
    rows = await cursor.fetchall()
    return [row["detail"] for row in rows]


async def test_override_set_writes_audit_trail(seeded_kb: KnowledgeBase) -> None:
    """Every set_override audits the transition, including the prior verdict."""
    db = seeded_kb.db
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)

    await set_override(db, "p-corpus", eligible=True, reason="keep", now=now)
    await set_override(db, "p-corpus", eligible=False, reason="actually drop", now=now)

    details = await _audit_details(db, "map_eligibility_override_set")
    assert len(details) == 2
    assert details[0] is not None and "was=absent" in details[0]
    assert details[1] is not None and "was=1" in details[1]
    assert details[1] is not None and "eligible=0" in details[1]


async def test_override_clear_writes_audit_trail_once(seeded_kb: KnowledgeBase) -> None:
    """A cleared override audits exactly once; the no-op clear audits nothing."""
    db = seeded_kb.db
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    await set_override(db, "p-thin", eligible=True, reason="keep", now=now)

    assert await clear_override(db, "p-thin") is True
    assert await clear_override(db, "p-thin") is False  # no-op

    details = await _audit_details(db, "map_eligibility_override_cleared")
    assert len(details) == 1
    assert details[0] is not None and "was=1" in details[0]


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


async def test_facade_override_round_trip(seeded_kb: KnowledgeBase) -> None:
    """The KnowledgeBase facade wires the three operations without an attribution fallback."""
    kb = seeded_kb

    stored = await kb.set_map_eligibility_override("p-corpus", eligible=True, reason="r")
    assert stored.set_by is None  # default Attribution() must NOT backfill one
    parsed = datetime.fromisoformat(stored.set_at)
    assert parsed.tzinfo is not None

    explicit = await kb.set_map_eligibility_override(
        "p-thin", eligible=True, reason="r", set_by="agent-7"
    )
    assert explicit.set_by == "agent-7"

    assert await kb.map_eligibility() == await resolve_eligibility(kb.db)

    assert await kb.clear_map_eligibility_override("p-corpus") is True
    assert await kb.clear_map_eligibility_override("p-corpus") is False
    verdicts = {v.evidence.project_ref: v for v in await kb.map_eligibility()}
    assert verdicts["p-corpus"].decided_by == "computed"


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


async def test_resolve_emits_one_info_summary(
    seeded_kb: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    """One INFO record carrying the corpus-level counts."""
    caplog.set_level(logging.INFO, logger="kb_core.map_eligibility")
    await resolve_eligibility(seeded_kb.db)

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info_records) == 1
    assert "8 project_refs" in info_records[0].getMessage()
    assert "3 eligible" in info_records[0].getMessage()


async def test_orphaned_override_emits_warning(
    seeded_kb: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    """An orphaned override trips the invariant-breach WARNING."""
    caplog.set_level(logging.INFO, logger="kb_core.map_eligibility")
    await set_override(
        seeded_kb.db,
        "p-ghost",
        eligible=True,
        reason="r",
        now=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
    )
    await resolve_eligibility(seeded_kb.db)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "invariant breach" in warnings[0].getMessage()
    assert "p-ghost" in warnings[0].getMessage()
