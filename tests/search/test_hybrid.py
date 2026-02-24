"""Tests for hybrid search."""

import pytest

from personal_kb.models.entry import EntryType
from personal_kb.models.search import SearchQuery
from personal_kb.search.hybrid import hybrid_search


@pytest.mark.asyncio
async def test_hybrid_fts_only(db, store):
    """Hybrid search without embedder falls back to FTS."""
    await store.create_entry(
        short_title="Python testing",
        long_title="Python testing patterns",
        knowledge_details="Use pytest for testing Python applications. Fixtures are powerful.",
        entry_type=EntryType.PATTERN_CONVENTION,
        tags=["python", "testing"],
    )

    query = SearchQuery(query="pytest testing")
    results = await hybrid_search(db, None, query)
    assert len(results) >= 1
    assert results[0].entry.id == "kb-00001"
    assert results[0].match_source == "fts"


@pytest.mark.asyncio
async def test_hybrid_with_embedder(db, store, fake_embedder):
    """Hybrid search with embedder uses both FTS and vector."""
    entry = await store.create_entry(
        short_title="Docker networking",
        long_title="Docker container networking guide",
        knowledge_details="Docker containers communicate via bridge networks by default.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["docker"],
    )

    # Embed the entry
    text = f"{entry.short_title} {entry.long_title} {entry.knowledge_details}"
    embedding = await fake_embedder.embed(text)
    await fake_embedder.store_embedding(entry.id, embedding)
    await store.mark_embedding(entry.id, True)

    query = SearchQuery(query="Docker networking")
    results = await hybrid_search(db, fake_embedder, query)
    assert len(results) >= 1
    assert results[0].entry.id == "kb-00001"
    assert results[0].match_source == "hybrid"


@pytest.mark.asyncio
async def test_hybrid_confidence_decay(db, store):
    """Search results include confidence decay."""
    await store.create_entry(
        short_title="API version",
        long_title="Current API version",
        knowledge_details="API is at version 2.3.1",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    query = SearchQuery(query="API version")
    results = await hybrid_search(db, None, query)
    assert len(results) >= 1
    # Just created, so effective confidence should be close to base
    assert results[0].effective_confidence > 0.8


@pytest.mark.asyncio
async def test_hybrid_project_filter(db, store):
    await store.create_entry(
        short_title="Config A",
        long_title="Project A configuration",
        knowledge_details="Config details for project A",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="project-a",
    )
    await store.create_entry(
        short_title="Config B",
        long_title="Project B configuration",
        knowledge_details="Config details for project B",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="project-b",
    )

    query = SearchQuery(query="Config", project_ref="project-a")
    results = await hybrid_search(db, None, query)
    assert len(results) >= 1
    assert all(r.entry.project_ref == "project-a" for r in results)


@pytest.mark.asyncio
async def test_hybrid_no_results(db, store):
    query = SearchQuery(query="nonexistent topic xyzzy")
    results = await hybrid_search(db, None, query)
    assert results == []
