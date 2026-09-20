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
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from kb_core import create_sqlite
from kb_core.map_caps import count_maps_created_since
from map_caps_fixture import CONTRIBUTOR, EXPECTED_COUNTS, SEED_ENTRIES, SINCE

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
    # The human's own map counts only for the human, and the contributor
    # match is exact — not a prefix or substring match.
    assert await count_maps_created_since(seeded_kb.db, contributor="jason", since=SINCE) == 1
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
