"""Hermetic SQLite suite for the mental_map hard-delete primitive.

No Postgres, no Ollama, no network: every test runs on a real SQLite KB from ``create_sqlite``.
The corpus and every expected value live in ``map_delete_fixture.py``.
The Postgres parity test seeds that same corpus from the same module.
Every behaviour the item's acceptance criteria name has exactly one test here.
The refusals are the distinction Jason drew: ``deactivate`` is for real knowledge entries.
Their CONTENT is recoverable by design; a map is a directory card, so it is deleted outright.
A factual entry therefore cannot be deleted through this path at all.
``audit_events`` is the one table the deletion must NOT touch.
It is the only thing that survives the map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kb_core import create_sqlite
from kb_core.map_delete import delete_map
from kb_core.map_lint import map_pointer_ids
from map_delete_fixture import (
    BARE_MAP_ID,
    EXPECTED_MAP_AUDIT_EVENT_TYPES,
    EXPECTED_RECORD,
    EXPECTED_SURVIVING_EDGES,
    EXPECTED_SURVIVING_ENTRY_IDS,
    EXPECTED_SURVIVING_VERSIONS,
    FACTUAL_ID,
    MAP_ID,
    POINTER_IDS,
    audit_event_types_for,
    edge_rows,
    entry_ids,
    seed_corpus,
    version_rows,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from kb_core.knowledge_base import KnowledgeBase


@pytest.fixture
async def seeded_kb(tmp_path: Path) -> AsyncIterator[KnowledgeBase]:
    """A SQLite KB with the shared hard-delete corpus seeded."""
    kb = await create_sqlite(tmp_path / "kb.db")
    await seed_corpus(kb.db)
    try:
        yield kb
    finally:
        await kb.close()


async def test_deleting_a_map_returns_the_full_record(seeded_kb: KnowledgeBase) -> None:
    """Three outbound edges, one inbound edge: the record carries the right counts and ids."""
    record = await delete_map(seeded_kb.db, MAP_ID)

    assert record == EXPECTED_RECORD
    assert record.outbound_edges_deleted == 3
    assert record.inbound_edges_deleted == 1
    assert record.inbound_referrer_ids == ["kb-70002"]


async def test_pointer_ids_come_from_map_pointer_ids(seeded_kb: KnowledgeBase) -> None:
    """The record's pointer set is exactly map_lint's, over the same body — no second regex."""
    record = await delete_map(seeded_kb.db, MAP_ID)

    assert record.pointer_ids == sorted(map_pointer_ids(record.knowledge_details))
    assert record.pointer_ids == sorted(POINTER_IDS)


async def test_knowledge_details_is_the_full_body(seeded_kb: KnowledgeBase) -> None:
    """The record carries the body verbatim, not a summary of it."""
    record = await delete_map(seeded_kb.db, MAP_ID)

    assert record.knowledge_details == EXPECTED_RECORD.knowledge_details
    assert "kb-70011" in record.knowledge_details


async def test_entry_row_is_gone_after_delete(seeded_kb: KnowledgeBase) -> None:
    """The deleted map's row is gone and every other entry survives."""
    await delete_map(seeded_kb.db, MAP_ID)

    assert await entry_ids(seeded_kb.db) == list(EXPECTED_SURVIVING_ENTRY_IDS)
    assert MAP_ID not in await entry_ids(seeded_kb.db)


async def test_version_rows_are_gone_and_others_survive(seeded_kb: KnowledgeBase) -> None:
    """The map's two version rows go; the survivor's version row stays."""
    await delete_map(seeded_kb.db, MAP_ID)

    assert await version_rows(seeded_kb.db) == list(EXPECTED_SURVIVING_VERSIONS)
    assert all(entry_id != MAP_ID for entry_id, _ in await version_rows(seeded_kb.db))


async def test_audit_events_rows_remain(seeded_kb: KnowledgeBase) -> None:
    """audit_events is the durable record — the deletion must not remove the evidence."""
    await delete_map(seeded_kb.db, MAP_ID)

    assert await audit_event_types_for(seeded_kb.db, MAP_ID) == list(EXPECTED_MAP_AUDIT_EVENT_TYPES)


async def test_edges_are_gone_and_unrelated_edges_survive(seeded_kb: KnowledgeBase) -> None:
    """Both directions go with the map; the edge between two survivors is untouched."""
    await delete_map(seeded_kb.db, MAP_ID)

    assert await edge_rows(seeded_kb.db) == list(EXPECTED_SURVIVING_EDGES)


async def test_factual_reference_cannot_be_deleted(seeded_kb: KnowledgeBase) -> None:
    """A hard delete of a real knowledge entry must stay impossible through this path."""
    with pytest.raises(ValueError, match="factual_reference"):
        await delete_map(seeded_kb.db, FACTUAL_ID)

    assert FACTUAL_ID in await entry_ids(seeded_kb.db)


async def test_unknown_id_raises_rather_than_no_oping(seeded_kb: KnowledgeBase) -> None:
    """A missing id must raise naming the id — a silent success hides a caller's typo."""
    with pytest.raises(ValueError, match="kb-99999"):
        await delete_map(seeded_kb.db, "kb-99999")

    assert await entry_ids(seeded_kb.db) == [
        "kb-70001",
        "kb-70002",
        "kb-70003",
        "kb-70011",
        "kb-70012",
        "kb-70013",
        "kb-70021",
    ]


async def test_map_with_no_edges_deletes_cleanly(seeded_kb: KnowledgeBase) -> None:
    """A map with zero edges in either direction returns zero counts and no referrers."""
    record = await delete_map(seeded_kb.db, BARE_MAP_ID)

    assert record.outbound_edges_deleted == 0
    assert record.inbound_edges_deleted == 0
    assert record.inbound_referrer_ids == []
    assert record.pointer_ids == ["kb-70011"]


async def test_facade_exposes_delete_map(seeded_kb: KnowledgeBase) -> None:
    """The facade is how every other kb-core facility is reached; this one is too."""
    record = await seeded_kb.delete_map(MAP_ID)

    assert record == EXPECTED_RECORD
    assert MAP_ID not in await entry_ids(seeded_kb.db)
