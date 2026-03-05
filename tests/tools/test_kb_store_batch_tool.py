"""Tests for the kb_store_batch MCP tool."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from personal_kb.graph.enricher import GraphEnricher
from personal_kb.tools.kb_store_batch import batch_store_entries
from tests.conftest import FakeEmbedder, FakeLLM


def _lifespan(db, store, graph_builder, embedder=None, graph_enricher=None):
    """Build a lifespan dict for testing."""
    return {
        "db": db,
        "store": store,
        "embedder": embedder,
        "graph_builder": graph_builder,
        "graph_enricher": graph_enricher,
    }


def _entry_dict(**kwargs):
    """Create a minimal valid entry dict with overrides."""
    defaults = {
        "short_title": "Test",
        "long_title": "Test entry",
        "knowledge_details": "Some details",
    }
    defaults.update(kwargs)
    return defaults


@pytest.mark.asyncio
async def test_batch_store_three_entries(db, store, graph_builder):
    """Basic batch creation of 3 entries."""
    embedder = FakeEmbedder(db)
    ls = _lifespan(db, store, graph_builder, embedder)

    entries = [
        _entry_dict(short_title="First", long_title="First entry", knowledge_details="D1"),
        _entry_dict(short_title="Second", long_title="Second entry", knowledge_details="D2"),
        _entry_dict(
            short_title="Third",
            long_title="Third entry",
            knowledge_details="D3",
            entry_type="decision",
        ),
    ]

    result = await batch_store_entries(entries, ls)
    assert "3 entries created" in result
    assert "kb-00001" in result
    assert "kb-00002" in result
    assert "kb-00003" in result
    assert "3 result(s)" in result


@pytest.mark.asyncio
async def test_batch_store_with_enrichment(db, store, graph_builder):
    """Batch store with enrichment uses a single LLM call."""
    embedder = FakeEmbedder(db)
    batch_response = json.dumps(
        {
            "kb-00001": [{"entity": "python", "entity_type": "technology", "relationship": "uses"}],
            "kb-00002": [{"entity": "sqlite", "entity_type": "tool", "relationship": "uses"}],
        }
    )
    llm = FakeLLM(response=batch_response)
    enricher = GraphEnricher(db, llm)
    ls = _lifespan(db, store, graph_builder, embedder, enricher)

    entries = [
        _entry_dict(short_title="Python tips", knowledge_details="Use list comps"),
        _entry_dict(short_title="SQLite tips", knowledge_details="Use WAL mode"),
    ]

    result = await batch_store_entries(entries, ls)
    assert "2 entries created" in result
    # Single LLM call for batch enrichment
    assert llm.generate_count == 1

    # Verify edges were created
    cursor = await db.execute(
        "SELECT COUNT(*) FROM graph_edges WHERE json_extract(properties, '$.source') = 'llm'"
    )
    count = (await cursor.fetchone())[0]
    assert count == 2


@pytest.mark.asyncio
async def test_batch_store_cap_at_10(db, store, graph_builder):
    """Exceeding 10 entries returns an error."""
    embedder = FakeEmbedder(db)
    ls = _lifespan(db, store, graph_builder, embedder)

    entries = [_entry_dict(short_title=f"Entry {i}") for i in range(11)]
    result = await batch_store_entries(entries, ls)
    assert "Maximum 10" in result


@pytest.mark.asyncio
async def test_batch_store_empty(db, store, graph_builder):
    """Empty entries list returns an error."""
    embedder = FakeEmbedder(db)
    ls = _lifespan(db, store, graph_builder, embedder)

    result = await batch_store_entries([], ls)
    assert "empty" in result.lower()


@pytest.mark.asyncio
async def test_batch_store_validation_error(db, store, graph_builder):
    """Missing required fields returns an error."""
    embedder = FakeEmbedder(db)
    ls = _lifespan(db, store, graph_builder, embedder)

    entries = [{"short_title": "Missing fields"}]
    result = await batch_store_entries(entries, ls)
    assert "missing required fields" in result.lower()
    assert "knowledge_details" in result
    assert "long_title" in result


@pytest.mark.asyncio
async def test_batch_store_rejects_invalid_sensitivity(db, store, graph_builder):
    """Batch with invalid sensitivity should return an error."""
    embedder = FakeEmbedder(db)
    ls = _lifespan(db, store, graph_builder, embedder)

    entries = [
        _entry_dict(short_title="Good", sensitivity="internal"),
        _entry_dict(short_title="Bad", sensitivity="banana"),
    ]
    result = await batch_store_entries(entries, ls)
    assert "invalid sensitivity" in result.lower()
    assert '"banana"' in result
    assert "entry 1" in result

    # No entries should have been created
    cursor = await db.execute("SELECT COUNT(*) FROM knowledge_entries")
    count = (await cursor.fetchone())[0]
    assert count == 0


@pytest.mark.asyncio
async def test_batch_store_enrichment_failure_continues(db, store, graph_builder):
    """If enrichment fails, entries are still created successfully."""
    embedder = FakeEmbedder(db)
    llm = FakeLLM(response="not valid json at all {{{")
    enricher = GraphEnricher(db, llm)
    ls = _lifespan(db, store, graph_builder, embedder, enricher)

    entries = [_entry_dict(short_title="Still works")]

    result = await batch_store_entries(entries, ls)
    assert "1 entries created" in result
    assert "kb-00001" in result


@pytest.mark.asyncio
async def test_batch_store_partial_failure(db, store, graph_builder):
    """If one entry fails, others still succeed and failures are reported."""
    embedder = FakeEmbedder(db)
    ls = _lifespan(db, store, graph_builder, embedder)

    original_create = store.create_entry

    call_count = 0

    async def flaky_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("DB write failed")
        return await original_create(**kwargs)

    entries = [
        _entry_dict(short_title="First", knowledge_details="D1"),
        _entry_dict(short_title="Second", knowledge_details="D2"),
        _entry_dict(short_title="Third", knowledge_details="D3"),
    ]

    with patch.object(store, "create_entry", side_effect=flaky_create):
        result = await batch_store_entries(entries, ls)

    assert "2 entries created" in result
    assert "1 failed" in result
    assert "Failed entries (retry these):" in result
    assert "Second" in result
    assert "DB write failed" in result

    # Verify only 2 entries in DB
    cursor = await db.execute("SELECT COUNT(*) FROM knowledge_entries")
    count = (await cursor.fetchone())[0]
    assert count == 2


@pytest.mark.asyncio
async def test_batch_store_all_fail(db, store, graph_builder):
    """If all entries fail, return a clear error with details."""
    embedder = FakeEmbedder(db)
    ls = _lifespan(db, store, graph_builder, embedder)

    entries = [
        _entry_dict(short_title="A", knowledge_details="D1"),
        _entry_dict(short_title="B", knowledge_details="D2"),
    ]

    with patch.object(
        store, "create_entry", new=AsyncMock(side_effect=RuntimeError("connection lost"))
    ):
        result = await batch_store_entries(entries, ls)

    assert "all 2 entries failed" in result.lower()
    assert "Entry 0 (A)" in result
    assert "Entry 1 (B)" in result
    assert "connection lost" in result

    # Nothing in DB
    cursor = await db.execute("SELECT COUNT(*) FROM knowledge_entries")
    count = (await cursor.fetchone())[0]
    assert count == 0
