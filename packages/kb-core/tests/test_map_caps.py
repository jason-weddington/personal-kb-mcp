"""Hermetic SQLite suite for the per-night map-creation caps count.

No Postgres, no Ollama, no network: every test runs on a real SQLite KB from ``create_sqlite``.

Rows are seeded via direct ``kb.db.execute`` INSERTs, never ``kb.store()``.

The store touches the embedder; these tests are about one SQL WHERE clause.

The corpus and every expected int live in ``map_caps_fixture.py``.

The Postgres parity test seeds from the SAME lists, so a dialect drift fails somewhere.

The tests are deliberately one-behaviour-per-test rather than one big assertion.

Each exclusion caps a specific failure mode of the nightly loop.

A deactivated map still counting would let the cap creep.

A ``factual_reference`` counting would cap on the wrong table.

A foreign contributor counting would cap someone else's maps.

An exclusive boundary would silently re-count the first map of the previous night.

The same corpus and the same style also cover ``map_caps.map_write_summary``.

That query answers a different question — which project was written to, and when last.

Its tests therefore live at the bottom of this file, on the same seeded KB.

Its sharp edge is the column it reads: the latest write is MAX(updated_at), never MAX(created_at).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from kb_core import create_sqlite
from kb_core.map_caps import MapWriteSummary, count_maps_created_since, map_write_summary
from map_caps_fixture import (
    CONTRIBUTOR,
    EXPECTED_COUNTS,
    EXPECTED_SUMMARY,
    LATEST_MAP_WRITE,
    P_BETA_LATEST_WRITE,
    SEED_ENTRIES,
    SINCE,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from kb_core.db.backend import Database
    from kb_core.knowledge_base import KnowledgeBase

_ENTRY_INSERT_SQL = (
    "INSERT INTO knowledge_entries"
    " (id, project_ref, short_title, long_title, knowledge_details, entry_type,"
    " contributor, created_at, updated_at, is_active)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


async def _seed(db: Database) -> None:
    """Insert the shared boundary corpus via direct SQL."""
    for row in SEED_ENTRIES:
        await db.execute(_ENTRY_INSERT_SQL, row)
    await db.commit()


@pytest.fixture
async def seeded_kb(tmp_path: Path) -> AsyncIterator[KnowledgeBase]:
    """A SQLite KB with the shared boundary corpus seeded."""
    kb = await create_sqlite(tmp_path / "kb.db")
    await _seed(kb.db)
    try:
        yield kb
    finally:
        await kb.close()


@pytest.fixture
async def empty_kb(tmp_path: Path) -> AsyncIterator[KnowledgeBase]:
    """A SQLite KB with the schema applied and zero entries."""
    kb = await create_sqlite(tmp_path / "kb.db")
    try:
        yield kb
    finally:
        await kb.close()


async def test_empty_store_counts_zero(empty_kb: KnowledgeBase) -> None:
    assert await count_maps_created_since(empty_kb.db, contributor=CONTRIBUTOR, since=SINCE) == 0


async def test_matches_expected_counts_on_sqlite(seeded_kb: KnowledgeBase) -> None:
    """The SQLite count query returns the shared EXPECTED_COUNTS exactly, one call per row."""
    for (contributor, project_ref), expected in EXPECTED_COUNTS.items():
        assert (
            await count_maps_created_since(
                seeded_kb.db,
                contributor=contributor,
                since=SINCE,
                project_ref=project_ref,
            )
            == expected
        )


async def test_map_created_exactly_at_since_is_counted(seeded_kb: KnowledgeBase) -> None:
    """The >= boundary is INCLUSIVE — kb-90001 was created at exactly `since` and counts."""
    assert (
        await count_maps_created_since(
            seeded_kb.db, contributor=CONTRIBUTOR, since=SINCE, project_ref="p-alpha"
        )
        == 1
    )


async def test_map_created_strictly_before_since_is_not_counted(seeded_kb: KnowledgeBase) -> None:
    """kb-90002 sits one microsecond before SINCE and is the p-alpha row that must drop."""
    assert (
        await count_maps_created_since(
            seeded_kb.db,
            contributor=CONTRIBUTOR,
            since=SINCE,
            project_ref="p-alpha",
        )
        == 1
    )
    # Moving the boundary back ONTO kb-90002 makes it the inclusive row, so
    # p-alpha counts two — proving the exclusion above was the boundary and
    # not some other term of the WHERE clause.
    assert (
        await count_maps_created_since(
            seeded_kb.db,
            contributor=CONTRIBUTOR,
            since=SINCE - timedelta(microseconds=1),
            project_ref="p-alpha",
        )
        == 2
    )


async def test_factual_reference_by_same_contributor_is_not_counted(
    seeded_kb: KnowledgeBase,
) -> None:
    """kb-90004 is a factual_reference by the same contributor at the same instant."""
    assert (
        await count_maps_created_since(
            seeded_kb.db, contributor=CONTRIBUTOR, since=SINCE, project_ref="p-beta"
        )
        == 1
    )


async def test_entry_by_a_different_contributor_is_not_counted(
    seeded_kb: KnowledgeBase,
) -> None:
    """kb-90005 is the human's map at the same instant — it is not in somnus's count."""
    assert await count_maps_created_since(seeded_kb.db, contributor=CONTRIBUTOR, since=SINCE) == 3
    # The human's own maps count only for the human, and the contributor
    # match is exact — not a prefix or substring match.
    # Two rows are the human's: kb-90005 (p-beta) and kb-90008 (empty ref).
    assert await count_maps_created_since(seeded_kb.db, contributor="jason", since=SINCE) == 2
    assert await count_maps_created_since(seeded_kb.db, contributor="somn", since=SINCE) == 0


async def test_deactivated_mental_map_is_not_counted(seeded_kb: KnowledgeBase) -> None:
    assert (
        await count_maps_created_since(
            seeded_kb.db, contributor=CONTRIBUTOR, since=SINCE, project_ref="p-gamma"
        )
        == 0
    )


async def test_unscoped_counts_across_projects_and_scoped_does_not(
    seeded_kb: KnowledgeBase,
) -> None:
    """project_ref=None sweeps every project; a supplied ref narrows to it."""
    assert await count_maps_created_since(seeded_kb.db, contributor=CONTRIBUTOR, since=SINCE) == 3
    assert (
        await count_maps_created_since(
            seeded_kb.db, contributor=CONTRIBUTOR, since=SINCE, project_ref="p-beta"
        )
        == 1
    )
    # A project_ref with no qualifying maps counts zero rather than raising.
    assert (
        await count_maps_created_since(
            seeded_kb.db, contributor=CONTRIBUTOR, since=SINCE, project_ref="p-absent"
        )
        == 0
    )


async def test_naive_since_is_rejected(seeded_kb: KnowledgeBase) -> None:
    """A naive `since` would compare against a differently-shaped ISO string."""
    with pytest.raises(ValueError, match="timezone-aware"):
        await count_maps_created_since(
            seeded_kb.db, contributor=CONTRIBUTOR, since=SINCE.replace(tzinfo=None)
        )


# ---------------------------------------------------------------------------
# map_write_summary — the per-project facts the somnus worklist ranking eats.
# The ranking itself (never-mapped first, then oldest write first) is the
# consumer's; these tests pin the FACTS, in the same one-behaviour-per-test
# style as the caps count above.
# ---------------------------------------------------------------------------


def _by_project(summary: list[MapWriteSummary], project_ref: str) -> MapWriteSummary:
    """Return the one summary row for ``project_ref`` — exactly one, or fail loudly."""
    matches = [row for row in summary if row.project_ref == project_ref]
    assert len(matches) == 1, f"expected exactly one row for {project_ref!r}, got {matches}"
    return matches[0]


async def test_empty_store_returns_empty_list(empty_kb: KnowledgeBase) -> None:
    assert await map_write_summary(empty_kb.db) == []


async def test_matches_expected_summary_on_sqlite(seeded_kb: KnowledgeBase) -> None:
    """The SQLite summary query returns the shared EXPECTED_SUMMARY exactly."""
    summary = await map_write_summary(seeded_kb.db)
    assert summary == list(EXPECTED_SUMMARY)
    # The parse-back contract: every instant is an AWARE datetime, not the
    # TEXT the column stores.
    for row in summary:
        assert row.latest_map_written_at.tzinfo is not None


async def test_rows_are_ordered_by_project_ref_ascending(seeded_kb: KnowledgeBase) -> None:
    refs = [row.project_ref for row in await map_write_summary(seeded_kb.db)]
    assert refs == sorted(refs)
    assert refs == ["p-alpha", "p-beta"]


async def test_project_with_two_active_maps_counts_two(seeded_kb: KnowledgeBase) -> None:
    """p-alpha holds kb-90001 and kb-90002 — two ACTIVE maps, so map_count is 2."""
    assert _by_project(await map_write_summary(seeded_kb.db), "p-alpha").map_count == 2


async def test_project_whose_only_map_is_deactivated_is_absent(
    seeded_kb: KnowledgeBase,
) -> None:
    """p-gamma's only map (kb-90006) is deactivated, so the project has no row at all."""
    summary = await map_write_summary(seeded_kb.db)
    assert all(row.project_ref != "p-gamma" for row in summary)


async def test_latest_write_follows_updated_at_not_created_at(seeded_kb: KnowledgeBase) -> None:
    """kb-90002 was created FIRST but written LAST — the summary follows the write.

    kb-90001 is p-alpha's most recently CREATED map (SINCE), so a query keyed
    on created_at would report SINCE here and send the loop straight back to
    the project it just finished.
    """
    latest = _by_project(await map_write_summary(seeded_kb.db), "p-alpha").latest_map_written_at
    assert latest == LATEST_MAP_WRITE
    assert latest != SINCE


async def test_factual_reference_contributes_neither_field(seeded_kb: KnowledgeBase) -> None:
    """kb-90004 is p-beta's newest write, but it is a factual_reference."""
    row = _by_project(await map_write_summary(seeded_kb.db), "p-beta")
    assert row.map_count == 2  # kb-90003 and kb-90005 — kb-90004 is not a map
    assert row.latest_map_written_at == P_BETA_LATEST_WRITE  # not LATER_THAN_ANY_MAP


async def test_null_and_empty_project_refs_are_excluded(seeded_kb: KnowledgeBase) -> None:
    """kb-90007 (NULL ref) and kb-90008 (empty ref) belong to no rankable project."""
    refs = {row.project_ref for row in await map_write_summary(seeded_kb.db)}
    assert refs == {"p-alpha", "p-beta"}


async def test_maps_by_a_different_contributor_are_counted(seeded_kb: KnowledgeBase) -> None:
    """kb-90005 is the human's map; the summary measures the project, not the principal.

    The caps count excludes it (see ``test_entry_by_a_different_contributor_is_not_counted``);
    this query includes it, because a map written by anyone still makes the
    project's maps fresh.
    """
    assert _by_project(await map_write_summary(seeded_kb.db), "p-beta").map_count == 2


async def test_naive_stored_timestamp_comes_back_aware(empty_kb: KnowledgeBase) -> None:
    """A map whose updated_at was stored WITHOUT an offset still yields an aware instant.

    Not a hypothetical: the live personal KB holds one map project whose
    updated_at reads ``2026-06-15T10:15:16`` with no suffix, alongside 26 maps
    that all carry ``+00:00``. ``db/queries.py`` renders the field with
    ``.isoformat()``, so any instant that reached the store naive is written
    naive.

    Returning it naive would make the result list UNSORTABLE — comparing a
    naive and an aware ``datetime`` raises TypeError — so the worklist ranking
    that consumes this function would crash on production data while every
    fixture-backed test passed. Found by running the generated SQL against the
    live Postgres, not by reading the code.
    """
    await empty_kb.db.execute(
        _ENTRY_INSERT_SQL,
        (
            "kb-90100",
            "p-naive",
            "naive",
            "naive timestamp map",
            "points at kb-00001",
            "mental_map",
            CONTRIBUTOR,
            "2026-06-15T10:15:16",
            "2026-06-15T10:15:16",
            1,
        ),
    )
    await empty_kb.db.commit()

    rows = await map_write_summary(empty_kb.db)

    assert len(rows) == 1
    assert rows[0].latest_map_written_at.tzinfo is not None
    assert rows[0].latest_map_written_at == datetime(2026, 6, 15, 10, 15, 16, tzinfo=UTC)
    # The property that actually matters: every element is mutually comparable,
    # which is what the consumer's sort needs.
    assert rows[0].latest_map_written_at < datetime.now(UTC)
