"""Tests for GraphEnricher — LLM-based entity extraction."""

import json

import pytest

from personal_kb.graph.enricher import GraphEnricher
from personal_kb.models.entry import EntryType, KnowledgeEntry
from tests.conftest import FakeLLM


def _make_entry(**kwargs) -> KnowledgeEntry:
    defaults = {
        "id": "kb-00001",
        "short_title": "Test Entry",
        "long_title": "A test knowledge entry",
        "knowledge_details": "We use aiosqlite for async database access with SQLite.",
        "entry_type": EntryType.DECISION,
        "tags": ["python", "database"],
        "project_ref": "personal-kb",
    }
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


@pytest.mark.asyncio
async def test_enrich_adds_edges(db):
    llm = FakeLLM(
        response=json.dumps(
            [
                {"entity": "aiosqlite", "entity_type": "tool", "relationship": "uses"},
                {"entity": "async-io", "entity_type": "concept", "relationship": "implements"},
            ]
        )
    )
    enricher = GraphEnricher(db, llm)
    entry = _make_entry()

    # Ensure entry node exists
    from personal_kb.graph.builder import GraphBuilder

    await GraphBuilder(db).build_for_entry(entry)

    added = await enricher.enrich_entry(entry)
    assert added == 2

    # Verify edges have LLM source marker
    cursor = await db.execute(
        "SELECT source, target, edge_type, properties FROM graph_edges "
        "WHERE json_extract(properties, '$.source') = 'llm'"
    )
    rows = await cursor.fetchall()
    assert len(rows) == 2

    targets = {row[1] for row in rows}
    assert "tool:aiosqlite" in targets
    assert "concept:async-io" in targets

    edge_types = {row[2] for row in rows}
    assert "uses" in edge_types
    assert "implements" in edge_types


@pytest.mark.asyncio
async def test_enrich_creates_nodes(db):
    llm = FakeLLM(
        response=json.dumps(
            [
                {"entity": "postgresql", "entity_type": "technology", "relationship": "replaces"},
            ]
        )
    )
    enricher = GraphEnricher(db, llm)
    entry = _make_entry()

    await enricher.enrich_entry(entry)

    cursor = await db.execute(
        "SELECT node_id, node_type FROM graph_nodes WHERE node_id = 'technology:postgresql'"
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row[1] == "technology"


@pytest.mark.asyncio
async def test_enrich_llm_unavailable(db):
    llm = FakeLLM(available=False)
    enricher = GraphEnricher(db, llm)
    entry = _make_entry()

    added = await enricher.enrich_entry(entry)
    assert added == 0


@pytest.mark.asyncio
async def test_enrich_llm_returns_none(db):
    llm = FakeLLM(response=None)
    enricher = GraphEnricher(db, llm)
    entry = _make_entry()

    added = await enricher.enrich_entry(entry)
    assert added == 0


@pytest.mark.asyncio
async def test_enrich_empty_array(db):
    llm = FakeLLM(response="[]")
    enricher = GraphEnricher(db, llm)
    entry = _make_entry()

    added = await enricher.enrich_entry(entry)
    assert added == 0


@pytest.mark.asyncio
async def test_enrich_malformed_json(db):
    llm = FakeLLM(response="this is not json at all")
    enricher = GraphEnricher(db, llm)
    entry = _make_entry()

    added = await enricher.enrich_entry(entry)
    assert added == 0


@pytest.mark.asyncio
async def test_enrich_markdown_fenced_json(db):
    response = '```json\n[{"entity": "redis", "entity_type": "tool", "relationship": "uses"}]\n```'
    llm = FakeLLM(response=response)
    enricher = GraphEnricher(db, llm)
    entry = _make_entry()

    added = await enricher.enrich_entry(entry)
    assert added == 1

    cursor = await db.execute(
        "SELECT target FROM graph_edges WHERE json_extract(properties, '$.source') = 'llm'"
    )
    rows = await cursor.fetchall()
    assert rows[0][0] == "tool:redis"


@pytest.mark.asyncio
async def test_enrich_missing_fields(db):
    """Items missing required fields are skipped, valid ones kept."""
    response = json.dumps(
        [
            {"entity": "valid-tool", "entity_type": "tool", "relationship": "uses"},
            {"entity": "no-type", "relationship": "uses"},  # missing entity_type
            {"entity_type": "tool", "relationship": "uses"},  # missing entity
            {"entity": "no-rel", "entity_type": "tool"},  # missing relationship
        ]
    )
    llm = FakeLLM(response=response)
    enricher = GraphEnricher(db, llm)
    entry = _make_entry()

    added = await enricher.enrich_entry(entry)
    assert added == 1


@pytest.mark.asyncio
async def test_enrich_invalid_entity_type(db):
    """Invalid entity_type values are rejected."""
    response = json.dumps(
        [
            {"entity": "something", "entity_type": "invalid_type", "relationship": "uses"},
            {"entity": "valid", "entity_type": "concept", "relationship": "uses"},
        ]
    )
    llm = FakeLLM(response=response)
    enricher = GraphEnricher(db, llm)
    entry = _make_entry()

    added = await enricher.enrich_entry(entry)
    assert added == 1


@pytest.mark.asyncio
async def test_enrich_max_relationships(db):
    """Relationships are capped at 8."""
    rels = [
        {"entity": f"item-{i}", "entity_type": "concept", "relationship": "related_to"}
        for i in range(12)
    ]
    llm = FakeLLM(response=json.dumps(rels))
    enricher = GraphEnricher(db, llm)
    entry = _make_entry()

    added = await enricher.enrich_entry(entry)
    assert added == 8


@pytest.mark.asyncio
async def test_enrich_clears_previous_llm_edges(db):
    """Re-enrichment replaces old LLM edges."""
    llm = FakeLLM(
        response=json.dumps(
            [
                {"entity": "old-tool", "entity_type": "tool", "relationship": "uses"},
            ]
        )
    )
    enricher = GraphEnricher(db, llm)
    entry = _make_entry()

    await enricher.enrich_entry(entry)

    # Now re-enrich with different relationships
    llm.response = json.dumps(
        [
            {"entity": "new-tool", "entity_type": "tool", "relationship": "depends_on"},
        ]
    )
    await enricher.enrich_entry(entry)

    cursor = await db.execute(
        "SELECT target FROM graph_edges "
        "WHERE source = 'kb-00001' AND json_extract(properties, '$.source') = 'llm'"
    )
    rows = await cursor.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "tool:new-tool"


@pytest.mark.asyncio
async def test_enrich_preserves_deterministic_edges(db):
    """Clearing LLM edges does not remove deterministic edges."""
    from personal_kb.graph.builder import GraphBuilder

    entry = _make_entry(tags=["python"])
    builder = GraphBuilder(db)
    await builder.build_for_entry(entry)

    # Add LLM edges
    llm = FakeLLM(
        response=json.dumps(
            [
                {"entity": "some-tool", "entity_type": "tool", "relationship": "uses"},
            ]
        )
    )
    enricher = GraphEnricher(db, llm)
    await enricher.enrich_entry(entry)

    # Verify both types exist
    cursor = await db.execute("SELECT COUNT(*) FROM graph_edges WHERE source = 'kb-00001'")
    total_before = (await cursor.fetchone())[0]
    assert total_before > 1  # deterministic + LLM edges

    # Re-enrich with empty results (clears LLM, keeps deterministic)
    llm.response = "[]"
    await enricher.enrich_entry(entry)

    cursor = await db.execute(
        "SELECT COUNT(*) FROM graph_edges "
        "WHERE source = 'kb-00001' AND json_extract(properties, '$.source') = 'llm'"
    )
    llm_count = (await cursor.fetchone())[0]
    assert llm_count == 0

    # Deterministic edges still there
    cursor = await db.execute("SELECT COUNT(*) FROM graph_edges WHERE source = 'kb-00001'")
    det_count = (await cursor.fetchone())[0]
    assert det_count > 0


@pytest.mark.asyncio
async def test_enrich_deduplication(db):
    """INSERT OR IGNORE prevents duplicate edges."""
    llm = FakeLLM(
        response=json.dumps(
            [
                {"entity": "sqlite", "entity_type": "tool", "relationship": "uses"},
            ]
        )
    )
    enricher = GraphEnricher(db, llm)
    entry = _make_entry()

    # Enrich twice — should not create duplicate edges
    await enricher.enrich_entry(entry)
    # Second enrichment clears+re-adds, so still 1
    await enricher.enrich_entry(entry)

    cursor = await db.execute(
        "SELECT COUNT(*) FROM graph_edges "
        "WHERE source = 'kb-00001' AND json_extract(properties, '$.source') = 'llm'"
    )
    count = (await cursor.fetchone())[0]
    assert count == 1


@pytest.mark.asyncio
async def test_prompt_includes_entry_content(db):
    llm = FakeLLM(response="[]")
    enricher = GraphEnricher(db, llm)
    entry = _make_entry(
        short_title="My Title",
        knowledge_details="Specific content about async patterns.",
        entry_type=EntryType.LESSON_LEARNED,
    )

    await enricher.enrich_entry(entry)

    assert "My Title" in llm.last_prompt
    assert "Specific content about async patterns." in llm.last_prompt
    assert "lesson_learned" in llm.last_prompt


@pytest.mark.asyncio
async def test_system_prompt_passed(db):
    llm = FakeLLM(response="[]")
    enricher = GraphEnricher(db, llm)
    entry = _make_entry()

    await enricher.enrich_entry(entry)

    assert llm.last_system is not None
    assert "knowledge graph" in llm.last_system.lower()


@pytest.mark.asyncio
async def test_enrich_all(db):
    """Batch enrichment via enrich_all."""
    llm = FakeLLM(
        response=json.dumps(
            [
                {"entity": "batch-tool", "entity_type": "tool", "relationship": "uses"},
            ]
        )
    )
    enricher = GraphEnricher(db, llm)
    entries = [_make_entry(id=f"kb-{i:05d}") for i in range(1, 4)]

    succeeded, failed = await enricher.enrich_all(entries)
    assert succeeded == 3
    assert failed == 0
    assert llm.generate_count == 3


# --- enrich_batch ---


@pytest.mark.asyncio
async def test_enrich_batch_single_call(db):
    """enrich_batch uses a single LLM call for all entries."""
    batch_response = json.dumps(
        {
            "kb-00001": [
                {"entity": "tool-a", "entity_type": "tool", "relationship": "uses"},
            ],
            "kb-00002": [
                {"entity": "tool-b", "entity_type": "tool", "relationship": "depends_on"},
            ],
            "kb-00003": [],
        }
    )
    llm = FakeLLM(response=batch_response)
    enricher = GraphEnricher(db, llm)
    entries = [_make_entry(id=f"kb-{i:05d}") for i in range(1, 4)]

    added = await enricher.enrich_batch(entries)
    assert added == 2
    assert llm.generate_count == 1  # single call

    # Verify edges exist
    cursor = await db.execute(
        "SELECT source, target FROM graph_edges WHERE json_extract(properties, '$.source') = 'llm'"
    )
    rows = await cursor.fetchall()
    assert len(rows) == 2
    sources = {row[0] for row in rows}
    assert sources == {"kb-00001", "kb-00002"}


@pytest.mark.asyncio
async def test_enrich_batch_fallback_on_parse_failure(db):
    """If batch JSON parse fails, falls back to per-entry enrichment."""
    # First call returns garbage (batch), subsequent calls return valid per-entry
    llm = FakeLLM(response="not valid json")
    enricher = GraphEnricher(db, llm)
    entries = [_make_entry(id=f"kb-{i:05d}") for i in range(1, 3)]

    # After batch fails, fallback calls enrich_entry per entry
    # Those also get "not valid json" so add 0 edges, but no crash
    added = await enricher.enrich_batch(entries)
    assert added == 0
    # 1 batch call + 2 per-entry fallback calls = 3
    assert llm.generate_count == 3


@pytest.mark.asyncio
async def test_enrich_batch_empty_list(db):
    """enrich_batch with empty list returns 0."""
    llm = FakeLLM(response="[]")
    enricher = GraphEnricher(db, llm)

    added = await enricher.enrich_batch([])
    assert added == 0
    assert llm.generate_count == 0


@pytest.mark.asyncio
async def test_enrich_batch_llm_unavailable(db):
    """enrich_batch returns 0 when LLM is unavailable."""
    llm = FakeLLM(available=False)
    enricher = GraphEnricher(db, llm)
    entries = [_make_entry(id="kb-00001")]

    added = await enricher.enrich_batch(entries)
    assert added == 0
