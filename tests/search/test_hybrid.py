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
async def test_hybrid_filters_deactivated_entries(db, store, fake_embedder):
    """Deactivated entries should not appear in search results."""
    entry = await store.create_entry(
        short_title="Deprecated fact",
        long_title="A deprecated factual entry",
        knowledge_details="This fact is no longer accurate and was deactivated.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    # Embed the entry so it can be found by vector search too
    embedding = await fake_embedder.embed(entry.embedding_text)
    if embedding:
        await fake_embedder.store_embedding(entry.id, embedding)
        await store.mark_embedding(entry.id, True)

    # Verify it appears before deactivation
    query = SearchQuery(query="deprecated fact")
    results_before, _ = await hybrid_search(db, fake_embedder, query)
    assert any(r.entry.id == entry.id for r in results_before)

    # Deactivate
    await store.deactivate_entry(entry.id)

    # Should NOT appear after deactivation
    results_after, _ = await hybrid_search(db, fake_embedder, query)
    assert not any(r.entry.id == entry.id for r in results_after)


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
    results, _filtered = await hybrid_search(db, None, query)

    # The docker entry should be present
    result_ids = [r.entry.id for r in results]
    assert "kb-00001" in result_ids

    # Unrelated entries should not appear in results
    assert "kb-00002" not in result_ids  # Python logging — unrelated
    assert "kb-00003" not in result_ids  # Git branching — unrelated


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
async def test_hybrid_contributor_filter(db, store):
    """Search with contributor filter should only return matching entries."""
    await store.create_entry(
        short_title="Jason config",
        long_title="Jason configuration details",
        knowledge_details="Config details from Jason.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="jason",
    )
    await store.create_entry(
        short_title="Alice config",
        long_title="Alice configuration details",
        knowledge_details="Config details from Alice.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="alice",
    )

    query = SearchQuery(query="config details", contributor="jason")
    results, _filtered = await hybrid_search(db, None, query)
    assert len(results) >= 1
    assert all(r.entry.contributor == "jason" for r in results)


@pytest.mark.asyncio
async def test_hybrid_team_filter(db, store):
    """Search with team filter should only return matching entries."""
    await store.create_entry(
        short_title="Platform config",
        long_title="Platform team configuration",
        knowledge_details="Config for platform team.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="jason",
        team="platform",
    )
    await store.create_entry(
        short_title="Infra config",
        long_title="Infra team configuration",
        knowledge_details="Config for infra team.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="alice",
        team="infra",
    )

    query = SearchQuery(query="config", team="platform")
    results, _filtered = await hybrid_search(db, None, query)
    assert len(results) >= 1
    assert all(r.entry.team == "platform" for r in results)


@pytest.mark.asyncio
async def test_filter_only_by_project(db, store):
    """Empty query with project_ref returns all entries for that project."""
    await store.create_entry(
        short_title="Alpha entry",
        long_title="Alpha project entry",
        knowledge_details="Part of alpha project.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="alpha",
    )
    await store.create_entry(
        short_title="Beta entry",
        long_title="Beta project entry",
        knowledge_details="Part of beta project.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="beta",
    )
    await store.create_entry(
        short_title="Alpha second",
        long_title="Another alpha entry",
        knowledge_details="Also alpha.",
        entry_type=EntryType.DECISION,
        project_ref="alpha",
    )

    query = SearchQuery(project_ref="alpha")
    results, filtered = await hybrid_search(db, None, query)
    assert len(results) == 2
    assert all(r.entry.project_ref == "alpha" for r in results)
    assert filtered == 0
    assert results[0].match_source == "filter"


@pytest.mark.asyncio
async def test_filter_only_by_tags(db, store):
    """Empty query with tags filter returns matching entries."""
    await store.create_entry(
        short_title="Tagged entry",
        long_title="Entry with python tag",
        knowledge_details="Has python tag.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["python", "testing"],
    )
    await store.create_entry(
        short_title="Other entry",
        long_title="Entry with docker tag",
        knowledge_details="Has docker tag.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["docker"],
    )

    query = SearchQuery(tags=["python"])
    results, _ = await hybrid_search(db, None, query)
    assert len(results) == 1
    assert results[0].entry.id == "kb-00001"


@pytest.mark.asyncio
async def test_filter_only_star_query(db, store):
    """Star query with project_ref works as filter-only."""
    await store.create_entry(
        short_title="Star test",
        long_title="Star query test",
        knowledge_details="Testing star query.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="myproj",
    )

    query = SearchQuery(query="*", project_ref="myproj")
    results, _ = await hybrid_search(db, None, query)
    assert len(results) == 1
    assert results[0].entry.project_ref == "myproj"


@pytest.mark.asyncio
async def test_filter_only_empty_no_filters(db, store):
    """Empty query with no filters returns nothing."""
    await store.create_entry(
        short_title="Orphan",
        long_title="No filter match",
        knowledge_details="Should not appear.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    query = SearchQuery()
    results, filtered = await hybrid_search(db, None, query)
    assert results == []
    assert filtered == 0


@pytest.mark.asyncio
async def test_filter_only_records_telemetry(db, store):
    """Filter-only search records a search event with match_source=filter."""
    await store.create_entry(
        short_title="Telemetry filter",
        long_title="Filter telemetry test",
        knowledge_details="Testing telemetry for filter path.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="tel-proj",
    )

    query = SearchQuery(project_ref="tel-proj")
    await hybrid_search(db, None, query)

    cursor = await db.execute("SELECT * FROM search_events")
    rows = await cursor.fetchall()
    assert len(rows) == 1
    assert rows[0]["match_source"] == "filter"
    assert rows[0]["top_score"] is None


# --- Hybrid-path filter regression tests ---
#
# These tests exercise the HYBRID path (FTS + vector via RRF). The bug being
# fixed: hybrid_search applied entry_type / project_ref / tags / contributor /
# team filters to the FTS leg only, leaving the vector leg unfiltered. When a
# wrong-type entry was a strong vector match it sailed into the fused result
# set and the final results "ignored" the requested filter. Each test below
# seeds the vector index for entries OUTSIDE the requested filter and asserts
# they don't leak through.


async def _embed_and_index(store, fake_embedder, entry):
    """Embed an entry's text and store the vector — so hybrid_search sees
    the entry in BOTH the FTS and vector legs."""
    text = entry.embedding_text
    embedding = await fake_embedder.embed(text)
    assert embedding is not None
    await fake_embedder.store_embedding(entry.id, embedding)
    await store.mark_embedding(entry.id, True)


@pytest.mark.asyncio
async def test_hybrid_entry_type_filter_excludes_other_types(db, store, fake_embedder):
    """Hybrid search with entry_type filter must NOT return other types.

    Regression for: vector leg ignored entry_type, so a strong vector match
    of the wrong entry_type would smuggle into the final results past the
    FTS leg's filter.
    """
    # Entry of the REQUESTED type
    target = await store.create_entry(
        short_title="Project layout map",
        long_title="Mental map of project layout",
        knowledge_details="The mental map shows the project's directory layout.",
        entry_type=EntryType.MENTAL_MAP,
    )
    # Entry of a DIFFERENT type that shares strong query terms — will be a
    # strong match on BOTH FTS and vector legs.
    other = await store.create_entry(
        short_title="Project layout convention",
        long_title="Mental map of project layout",
        knowledge_details="The mental map shows the project's directory layout.",
        entry_type=EntryType.PATTERN_CONVENTION,
    )

    await _embed_and_index(store, fake_embedder, target)
    await _embed_and_index(store, fake_embedder, other)

    query = SearchQuery(
        query="mental map project layout",
        entry_type=EntryType.MENTAL_MAP,
        limit=10,
        min_score_ratio=0.0,
    )
    results, _ = await hybrid_search(db, fake_embedder, query)

    assert len(results) >= 1, "expected the mental_map entry to be returned"
    assert all(r.entry.entry_type == EntryType.MENTAL_MAP for r in results), (
        f"hybrid search returned wrong entry_type(s): "
        f"{[(r.entry.id, r.entry.entry_type) for r in results]}"
    )
    # The mental_map entry should be present; the pattern_convention must not be
    returned_ids = {r.entry.id for r in results}
    assert target.id in returned_ids
    assert other.id not in returned_ids


@pytest.mark.asyncio
async def test_hybrid_project_ref_filter_excludes_other_projects(db, store, fake_embedder):
    """Hybrid search with project_ref filter must NOT return other projects.

    Regression for: vector leg ignored project_ref. (The previous behavior
    happened to apply a project_ref post-fusion filter, but that path
    silently shrinks the result count below `limit`; the proper fix is to
    filter both legs at the SQL level.)
    """
    target = await store.create_entry(
        short_title="Deployment runbook",
        long_title="Production deployment runbook",
        knowledge_details="Steps to deploy the production service.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="project-a",
    )
    other = await store.create_entry(
        short_title="Deployment runbook",
        long_title="Production deployment runbook",
        knowledge_details="Steps to deploy the production service.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="project-b",
    )

    await _embed_and_index(store, fake_embedder, target)
    await _embed_and_index(store, fake_embedder, other)

    query = SearchQuery(
        query="deployment runbook production",
        project_ref="project-a",
        limit=10,
        min_score_ratio=0.0,
    )
    results, _ = await hybrid_search(db, fake_embedder, query)

    assert len(results) >= 1
    assert all(r.entry.project_ref == "project-a" for r in results), (
        f"hybrid search returned wrong project_ref(s): "
        f"{[(r.entry.id, r.entry.project_ref) for r in results]}"
    )
    returned_ids = {r.entry.id for r in results}
    assert target.id in returned_ids
    assert other.id not in returned_ids


@pytest.mark.asyncio
async def test_hybrid_contributor_filter_in_vector_leg(db, store, fake_embedder):
    """Hybrid search with contributor filter must NOT return other contributors."""
    target = await store.create_entry(
        short_title="Build pipeline notes",
        long_title="Build pipeline configuration",
        knowledge_details="Notes about the build pipeline configuration.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="jason",
    )
    other = await store.create_entry(
        short_title="Build pipeline notes",
        long_title="Build pipeline configuration",
        knowledge_details="Notes about the build pipeline configuration.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="alice",
    )

    await _embed_and_index(store, fake_embedder, target)
    await _embed_and_index(store, fake_embedder, other)

    query = SearchQuery(
        query="build pipeline configuration",
        contributor="jason",
        limit=10,
        min_score_ratio=0.0,
    )
    results, _ = await hybrid_search(db, fake_embedder, query)

    assert len(results) >= 1
    assert all(r.entry.contributor == "jason" for r in results)
    returned_ids = {r.entry.id for r in results}
    assert target.id in returned_ids
    assert other.id not in returned_ids


@pytest.mark.asyncio
async def test_hybrid_team_filter_in_vector_leg(db, store, fake_embedder):
    """Hybrid search with team filter must NOT return other teams."""
    target = await store.create_entry(
        short_title="On-call playbook",
        long_title="Platform team on-call playbook",
        knowledge_details="The on-call playbook for incidents.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="jason",
        team="platform",
    )
    other = await store.create_entry(
        short_title="On-call playbook",
        long_title="Platform team on-call playbook",
        knowledge_details="The on-call playbook for incidents.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="alice",
        team="infra",
    )

    await _embed_and_index(store, fake_embedder, target)
    await _embed_and_index(store, fake_embedder, other)

    query = SearchQuery(
        query="on-call playbook incidents",
        team="platform",
        limit=10,
        min_score_ratio=0.0,
    )
    results, _ = await hybrid_search(db, fake_embedder, query)

    assert len(results) >= 1
    assert all(r.entry.team == "platform" for r in results)
    returned_ids = {r.entry.id for r in results}
    assert target.id in returned_ids
    assert other.id not in returned_ids


@pytest.mark.asyncio
async def test_hybrid_tags_filter_in_vector_leg(db, store, fake_embedder):
    """Hybrid search with tags filter must NOT return entries missing those tags."""
    target = await store.create_entry(
        short_title="Caching strategy",
        long_title="Caching strategy notes",
        knowledge_details="Notes about the caching strategy and TTLs.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["caching", "performance"],
    )
    other = await store.create_entry(
        short_title="Caching strategy",
        long_title="Caching strategy notes",
        knowledge_details="Notes about the caching strategy and TTLs.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["database"],
    )

    await _embed_and_index(store, fake_embedder, target)
    await _embed_and_index(store, fake_embedder, other)

    query = SearchQuery(
        query="caching strategy TTL",
        tags=["caching"],
        limit=10,
        min_score_ratio=0.0,
    )
    results, _ = await hybrid_search(db, fake_embedder, query)

    assert len(results) >= 1
    assert all("caching" in (r.entry.tags or []) for r in results)
    returned_ids = {r.entry.id for r in results}
    assert target.id in returned_ids
    assert other.id not in returned_ids


@pytest.mark.asyncio
async def test_vector_search_backend_applies_entry_type_filter(db, store, fake_embedder):
    """Backend-level: db.vector_search must honor entry_type at SQL level.

    The hybrid filter regression is rooted in this layer — the vector
    backend used to ignore filters entirely. Probe it directly so a
    backend regression is caught even if the hybrid wiring above silently
    stops passing the filter through.
    """
    target = await store.create_entry(
        short_title="Map A",
        long_title="Mental map A",
        knowledge_details="Map A content.",
        entry_type=EntryType.MENTAL_MAP,
    )
    other = await store.create_entry(
        short_title="Pattern A",
        long_title="Pattern A",
        knowledge_details="Map A content.",
        entry_type=EntryType.PATTERN_CONVENTION,
    )
    await _embed_and_index(store, fake_embedder, target)
    await _embed_and_index(store, fake_embedder, other)

    query_emb = await fake_embedder.embed("Map A content")
    assert query_emb is not None

    # No filter: both come back
    unfiltered = await db.vector_search(query_emb, limit=10)
    unfiltered_ids = {row[0] for row in unfiltered}
    assert target.id in unfiltered_ids
    assert other.id in unfiltered_ids

    # Filter by entry_type: only mental_map
    filtered = await db.vector_search(query_emb, limit=10, entry_type=EntryType.MENTAL_MAP.value)
    filtered_ids = {row[0] for row in filtered}
    assert target.id in filtered_ids
    assert other.id not in filtered_ids


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


# ---------------------------------------------------------------------------
# Supersession-aware reads
# ---------------------------------------------------------------------------

_SHARED = "zebrafish quokka distinctive supersession phrase"


async def _seed_superseded(db, old_id: str, new_id: str) -> None:
    await db.execute(
        "UPDATE knowledge_entries SET superseded_by = ? WHERE id = ?", (new_id, old_id)
    )
    await db.commit()


async def _pair(store, project_ref=None, b_text=_SHARED):
    a = await store.create_entry(
        short_title="Old way",
        long_title="Old way long",
        knowledge_details=_SHARED,
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref=project_ref,
    )
    b = await store.create_entry(
        short_title="New way",
        long_title="New way long",
        knowledge_details=b_text,
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref=project_ref,
    )
    return a, b


async def test_fused_hides_superseded_by_default(db, store):
    a, b = await _pair(store)
    await _seed_superseded(db, a.id, b.id)
    results, _ = await hybrid_search(db, None, SearchQuery(query=_SHARED))
    ids = [r.entry.id for r in results]
    assert b.id in ids
    assert a.id not in ids


async def test_fused_include_superseded(db, store):
    a, b = await _pair(store)
    await _seed_superseded(db, a.id, b.id)
    results, _ = await hybrid_search(db, None, SearchQuery(query=_SHARED, include_superseded=True))
    assert {r.entry.id for r in results} == {a.id, b.id}


async def test_fused_limit_backfills_superseded_slot(db, store):
    a, b = await _pair(store)
    await _seed_superseded(db, a.id, b.id)
    results, _ = await hybrid_search(db, None, SearchQuery(query=_SHARED, limit=1))
    assert [r.entry.id for r in results] == [b.id]


async def test_fused_supersession_log_superseder_in_results(db, store, caplog):
    a, b = await _pair(store)
    await _seed_superseded(db, a.id, b.id)
    with caplog.at_level("INFO", logger="kb_core.search.hybrid"):
        await hybrid_search(db, None, SearchQuery(query=_SHARED))
    text = caplog.text
    assert "supersession-read op=search path=fused" in text
    assert f"hidden=[('{a.id}', '{b.id}')]" in text
    assert "superseder_in_results=[True]" in text


async def test_fused_supersession_log_superseder_not_in_results(db, store, caplog):
    a, b = await _pair(store, b_text="completely unrelated gardening content")
    await _seed_superseded(db, a.id, b.id)
    with caplog.at_level("INFO", logger="kb_core.search.hybrid"):
        results, _ = await hybrid_search(db, None, SearchQuery(query=_SHARED))
    assert results == []
    assert "superseder_in_results=[False]" in caplog.text


async def test_fused_no_log_when_nothing_hidden(db, store, caplog):
    await _pair(store)
    with caplog.at_level("INFO", logger="kb_core.search.hybrid"):
        await hybrid_search(db, None, SearchQuery(query=_SHARED))
    assert "supersession-read" not in caplog.text


async def test_filter_only_hides_superseded(db, store):
    a, b = await _pair(store, project_ref="p")
    await _seed_superseded(db, a.id, b.id)
    results, _ = await hybrid_search(db, None, SearchQuery(project_ref="p"))
    assert [r.entry.id for r in results] == [b.id]
    results, _ = await hybrid_search(
        db, None, SearchQuery(project_ref="p", include_superseded=True)
    )
    assert {r.entry.id for r in results} == {a.id, b.id}
