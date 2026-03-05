"""Tests for graph data extraction."""

import pytest

from personal_kb.explorer.graph_data import extract_graph_data
from personal_kb.graph.builder import GraphBuilder
from personal_kb.models.entry import EntryType


@pytest.mark.asyncio
async def test_empty_db(db):
    """Empty DB returns empty nodes/edges and zero stats."""
    data = await extract_graph_data(db)
    assert data["nodes"] == []
    assert data["edges"] == []
    assert data["stats"] == {"node_count": 0, "edge_count": 0}


@pytest.mark.asyncio
async def test_nodes_and_edges_from_entries(db, store):
    """Entries with tags produce correct graph nodes and edges."""
    builder = GraphBuilder(db)

    e1 = await store.create_entry(
        short_title="Python Tips",
        long_title="Tips for Python development",
        knowledge_details="Use type hints.",
        entry_type=EntryType.PATTERN_CONVENTION,
        tags=["python"],
        project_ref="my-proj",
    )
    await builder.build_for_entry(e1)
    await db.commit()

    data = await extract_graph_data(db)
    nodes = data["nodes"]
    edges = data["edges"]
    stats = data["stats"]

    # Should have entry node, tag node, project node
    node_ids = {n["id"] for n in nodes}
    assert e1.id in node_ids
    assert "tag:python" in node_ids
    assert "project:my-proj" in node_ids

    # Correct node types
    type_by_id = {n["id"]: n["type"] for n in nodes}
    assert type_by_id[e1.id] == "entry"
    assert type_by_id["tag:python"] == "tag"
    assert type_by_id["project:my-proj"] == "project"

    # Stats match
    assert stats["node_count"] == len(nodes)
    assert stats["edge_count"] == len(edges)

    # Edges: has_tag and in_project
    edge_types = {(e["source"], e["type"]) for e in edges}
    assert (e1.id, "has_tag") in edge_types
    assert (e1.id, "in_project") in edge_types


@pytest.mark.asyncio
async def test_entry_node_enriched_with_metadata(db, store):
    """Entry nodes include short_title, entry_type from knowledge_entries."""
    builder = GraphBuilder(db)

    entry = await store.create_entry(
        short_title="My Decision",
        long_title="Why we chose SQLite",
        knowledge_details="SQLite is good enough.",
        entry_type=EntryType.DECISION,
        tags=["sqlite"],
    )
    await builder.build_for_entry(entry)
    await db.commit()

    data = await extract_graph_data(db)
    entry_node = next(n for n in data["nodes"] if n["id"] == entry.id)

    assert entry_node["label"] == "My Decision"
    assert entry_node["type"] == "entry"
    assert entry_node["properties"]["short_title"] == "My Decision"
    assert entry_node["properties"]["entry_type"] == "decision"
    assert entry_node["properties"]["long_title"] == "Why we chose SQLite"


@pytest.mark.asyncio
async def test_connection_count_as_val(db, store):
    """Node val reflects actual connection count."""
    builder = GraphBuilder(db)

    # Entry with 2 tags + project = 3 edges
    entry = await store.create_entry(
        short_title="Multi-tag",
        long_title="Entry with multiple tags",
        knowledge_details="Has several tags.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["python", "sqlite"],
        project_ref="test-proj",
    )
    await builder.build_for_entry(entry)
    await db.commit()

    data = await extract_graph_data(db)
    entry_node = next(n for n in data["nodes"] if n["id"] == entry.id)

    # Entry has 3 outgoing edges (has_tag x2, in_project x1)
    assert entry_node["val"] == 3


@pytest.mark.asyncio
async def test_entity_node_labels_stripped(db, store):
    """Entity nodes have labels with type prefix stripped."""
    builder = GraphBuilder(db)

    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Just a test.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["mylib"],
    )
    await builder.build_for_entry(entry)
    await db.commit()

    data = await extract_graph_data(db)
    tag_node = next(n for n in data["nodes"] if n["id"] == "tag:mylib")
    assert tag_node["label"] == "mylib"
