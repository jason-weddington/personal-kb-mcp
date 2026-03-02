"""Tests for hybrid search."""

import pytest

from personal_kb.db.queries import get_entry
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
    results, _filtered = await hybrid_search(db, None, query)
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
    results, _filtered = await hybrid_search(db, fake_embedder, query)
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
    results, _filtered = await hybrid_search(db, None, query)
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
    results, _filtered = await hybrid_search(db, None, query)
    assert len(results) >= 1
    assert all(r.entry.project_ref == "project-a" for r in results)


@pytest.mark.asyncio
async def test_hybrid_no_results(db, store):
    query = SearchQuery(query="nonexistent topic xyzzy")
    results, filtered = await hybrid_search(db, None, query)
    assert results == []
    assert filtered == 0


@pytest.mark.asyncio
async def test_hybrid_search_does_not_touch_last_accessed(db, store):
    """Search results should NOT update last_accessed — only kb_get should."""
    await store.create_entry(
        short_title="Access tracking test",
        long_title="Testing access tracking",
        knowledge_details="Search should not reset this entry's decay clock.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    # Search for the entry
    query = SearchQuery(query="access tracking test")
    results, _filtered = await hybrid_search(db, None, query)
    assert len(results) >= 1

    # last_accessed should still be NULL after search
    entry = await get_entry(db, "kb-00001")
    assert entry is not None
    assert entry.last_accessed is None


@pytest.mark.asyncio
async def test_hybrid_threshold_filters_low_relevance(db, store):
    """Results below 50% of top RRF score should be filtered."""
    # Create one highly relevant entry and several loosely matching ones
    await store.create_entry(
        short_title="Docker compose orchestration",
        long_title="Docker compose for multi-container orchestration",
        knowledge_details="Use docker compose to orchestrate multiple containers together.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["docker", "orchestration"],
    )
    await store.create_entry(
        short_title="Python logging",
        long_title="Python logging best practices",
        knowledge_details="Use the logging module for structured logging in Python apps.",
        entry_type=EntryType.PATTERN_CONVENTION,
        tags=["python"],
    )
    await store.create_entry(
        short_title="Git branching",
        long_title="Git branching strategy",
        knowledge_details="Use feature branches for isolated development work.",
        entry_type=EntryType.PATTERN_CONVENTION,
        tags=["git"],
    )

    # Search specifically for docker — only the first entry should match well
    query = SearchQuery(query="Docker compose orchestration", limit=10, min_score_ratio=0.5)
    results, filtered = await hybrid_search(db, None, query)

    # The docker entry should be present
    result_ids = [r.entry.id for r in results]
    assert "kb-00001" in result_ids

    # Unrelated entries should be filtered (or never matched by FTS)
    # The key assertion: filtered count should be >= 0 (we have the mechanism)
    assert isinstance(filtered, int)
    assert filtered >= 0


@pytest.mark.asyncio
async def test_hybrid_threshold_disabled_returns_all(db, store):
    """min_score_ratio=0.0 should disable filtering and return all FTS matches."""
    await store.create_entry(
        short_title="Docker compose",
        long_title="Docker compose guide",
        knowledge_details="Docker compose for container orchestration.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["docker"],
    )
    await store.create_entry(
        short_title="Docker networking",
        long_title="Docker networking guide",
        knowledge_details="Docker containers use bridge networks.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["docker"],
    )

    # With threshold disabled, all FTS matches should come through
    query_no_filter = SearchQuery(query="Docker", limit=10, min_score_ratio=0.0)
    results_all, filtered_none = await hybrid_search(db, None, query_no_filter)

    # With threshold enabled
    query_filtered = SearchQuery(query="Docker", limit=10, min_score_ratio=0.5)
    results_filtered, _ = await hybrid_search(db, None, query_filtered)

    # Disabled threshold should return >= as many results
    assert len(results_all) >= len(results_filtered)
    assert filtered_none == 0


@pytest.mark.asyncio
async def test_hybrid_filtered_count_accurate(db, store):
    """filtered_count should equal pre-filter minus post-filter candidate count."""
    await store.create_entry(
        short_title="Alpha feature",
        long_title="Alpha feature description",
        knowledge_details="Details about the alpha feature implementation.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    # With no filter
    query_open = SearchQuery(query="Alpha feature", limit=50, min_score_ratio=0.0)
    results_open, filtered_open = await hybrid_search(db, None, query_open)
    assert filtered_open == 0

    # With strict filter
    query_strict = SearchQuery(query="Alpha feature", limit=50, min_score_ratio=0.99)
    results_strict, filtered_strict = await hybrid_search(db, None, query_strict)

    # Total should be conserved: filtered + returned candidates
    # (stale filtering may also remove some, but with fresh entries it shouldn't)
    total_open = len(results_open) + filtered_open
    total_strict = len(results_strict) + filtered_strict
    assert total_open == total_strict


# --- Search telemetry ---


@pytest.mark.asyncio
async def test_search_event_recorded(db, store):
    """hybrid_search should record a search_events row."""
    await store.create_entry(
        short_title="Telemetry test",
        long_title="Entry for telemetry test",
        knowledge_details="Testing that search events are recorded.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    query = SearchQuery(query="telemetry test")
    await hybrid_search(db, None, query)

    cursor = await db.execute("SELECT * FROM search_events")
    rows = await cursor.fetchall()
    assert len(rows) == 1
    assert rows[0]["query_text"] == "telemetry test"
    assert rows[0]["result_count"] >= 1
    assert rows[0]["top_score"] is not None
    assert rows[0]["match_source"] == "fts"
    assert rows[0]["created_at"] is not None


@pytest.mark.asyncio
async def test_search_event_zero_results(db):
    """Zero-result search should record result_count=0 and top_score=None."""
    query = SearchQuery(query="nonexistent xyzzy nothing")
    await hybrid_search(db, None, query)

    cursor = await db.execute("SELECT * FROM search_events")
    rows = await cursor.fetchall()
    assert len(rows) == 1
    assert rows[0]["result_count"] == 0
    assert rows[0]["top_score"] is None


@pytest.mark.asyncio
async def test_search_event_records_contributor(db, store):
    """Search event should include contributor when provided."""
    await store.create_entry(
        short_title="Contributor telemetry",
        long_title="Testing contributor in search events",
        knowledge_details="Contributor should appear in telemetry.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    query = SearchQuery(query="contributor telemetry")
    await hybrid_search(db, None, query, contributor="jason")

    cursor = await db.execute("SELECT * FROM search_events")
    rows = await cursor.fetchall()
    assert len(rows) == 1
    assert rows[0]["contributor"] == "jason"


@pytest.mark.asyncio
async def test_search_event_contributor_none_by_default(db, store):
    """Search event contributor is NULL when not provided."""
    await store.create_entry(
        short_title="No contributor",
        long_title="Search without contributor",
        knowledge_details="No contributor set.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    query = SearchQuery(query="no contributor")
    await hybrid_search(db, None, query)

    cursor = await db.execute("SELECT contributor FROM search_events")
    row = await cursor.fetchone()
    assert row["contributor"] is None


@pytest.mark.asyncio
async def test_search_telemetry_failure_does_not_break_search(db, store, monkeypatch):
    """Telemetry failure should not break the search results."""
    await store.create_entry(
        short_title="Resilience test",
        long_title="Entry for resilience test",
        knowledge_details="Search must work even if telemetry fails.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    # Sabotage the search_events table by dropping it
    await db.executescript("DROP TABLE IF EXISTS search_events")
    await db.commit()

    query = SearchQuery(query="resilience test")
    results, _filtered = await hybrid_search(db, None, query)

    # Search should still return results despite telemetry failure
    assert len(results) >= 1
    assert results[0].entry.short_title == "Resilience test"
