"""Tests for the knowledge graph builder."""

import json

import pytest

from personal_kb.models.entry import EntryType, KnowledgeEntry


def _make_entry(
    entry_id: str = "kb-00001",
    short_title: str = "Test Entry",
    long_title: str = "A test entry",
    knowledge_details: str = "Some details",
    entry_type: EntryType = EntryType.FACTUAL_REFERENCE,
    tags: list[str] | None = None,
    project_ref: str | None = None,
    hints: dict | None = None,
    superseded_by: str | None = None,
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        short_title=short_title,
        long_title=long_title,
        knowledge_details=knowledge_details,
        entry_type=entry_type,
        tags=tags or [],
        project_ref=project_ref,
        hints=hints or {},
        superseded_by=superseded_by,
    )


async def _get_nodes(db, node_type: str | None = None) -> list[dict]:
    if node_type:
        cursor = await db.execute(
            "SELECT node_id, node_type, properties FROM graph_nodes WHERE node_type = ?",
            (node_type,),
        )
    else:
        cursor = await db.execute("SELECT node_id, node_type, properties FROM graph_nodes")
    rows = await cursor.fetchall()
    return [{"node_id": r[0], "node_type": r[1], "properties": json.loads(r[2])} for r in rows]


async def _get_edges(db, source: str | None = None, edge_type: str | None = None) -> list[dict]:
    query = "SELECT source, target, edge_type FROM graph_edges WHERE 1=1"
    params: list[str] = []
    if source:
        query += " AND source = ?"
        params.append(source)
    if edge_type:
        query += " AND edge_type = ?"
        params.append(edge_type)
    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()
    return [{"source": r[0], "target": r[1], "edge_type": r[2]} for r in rows]


# --- Schema ---


@pytest.mark.asyncio
async def test_graph_tables_exist(db):
    """Graph tables should be created during connection setup."""
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
        " AND name IN ('graph_nodes', 'graph_edges') ORDER BY name"
    )
    rows = await cursor.fetchall()
    names = [r[0] for r in rows]
    assert names == ["graph_edges", "graph_nodes"]


@pytest.mark.asyncio
async def test_schema_idempotent(db):
    """Applying graph schema twice should not fail."""
    from personal_kb.db.schema import apply_graph_schema

    await apply_graph_schema(db)
    await apply_graph_schema(db)
    # Should not raise


# --- Basic nodes/edges ---


@pytest.mark.asyncio
async def test_entry_node_created(db, graph_builder):
    entry = _make_entry()
    await graph_builder.build_for_entry(entry)

    nodes = await _get_nodes(db, "entry")
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "kb-00001"
    assert nodes[0]["properties"]["short_title"] == "Test Entry"
    assert nodes[0]["properties"]["entry_type"] == "factual_reference"


@pytest.mark.asyncio
async def test_tag_nodes_and_edges(db, graph_builder):
    entry = _make_entry(tags=["python", "sqlite"])
    await graph_builder.build_for_entry(entry)

    tag_nodes = await _get_nodes(db, "tag")
    assert len(tag_nodes) == 2
    tag_ids = {n["node_id"] for n in tag_nodes}
    assert tag_ids == {"tag:python", "tag:sqlite"}

    edges = await _get_edges(db, source="kb-00001", edge_type="has_tag")
    assert len(edges) == 2
    targets = {e["target"] for e in edges}
    assert targets == {"tag:python", "tag:sqlite"}


@pytest.mark.asyncio
async def test_project_node_and_edge(db, graph_builder):
    entry = _make_entry(project_ref="personal-kb")
    await graph_builder.build_for_entry(entry)

    project_nodes = await _get_nodes(db, "project")
    assert len(project_nodes) == 1
    assert project_nodes[0]["node_id"] == "project:personal-kb"

    edges = await _get_edges(db, source="kb-00001", edge_type="in_project")
    assert len(edges) == 1
    assert edges[0]["target"] == "project:personal-kb"


@pytest.mark.asyncio
async def test_no_project_when_none(db, graph_builder):
    entry = _make_entry(project_ref=None)
    await graph_builder.build_for_entry(entry)

    project_nodes = await _get_nodes(db, "project")
    assert len(project_nodes) == 0


# --- Supersedes ---


@pytest.mark.asyncio
async def test_supersedes_from_hints_single(db, graph_builder):
    entry = _make_entry(hints={"supersedes": "kb-00099"})
    await graph_builder.build_for_entry(entry)

    edges = await _get_edges(db, source="kb-00001", edge_type="supersedes")
    assert len(edges) == 1
    assert edges[0]["target"] == "kb-00099"


@pytest.mark.asyncio
async def test_supersedes_from_hints_list(db, graph_builder):
    entry = _make_entry(hints={"supersedes": ["kb-00099", "kb-00098"]})
    await graph_builder.build_for_entry(entry)

    edges = await _get_edges(db, source="kb-00001", edge_type="supersedes")
    assert len(edges) == 2
    targets = {e["target"] for e in edges}
    assert targets == {"kb-00099", "kb-00098"}


@pytest.mark.asyncio
async def test_superseded_by_creates_reversed_edge(db, graph_builder):
    entry = _make_entry(superseded_by="kb-00050")
    await graph_builder.build_for_entry(entry)

    # The edge goes from the superseder TO this entry
    edges = await _get_edges(db, source="kb-00050", edge_type="supersedes")
    assert len(edges) == 1
    assert edges[0]["target"] == "kb-00001"


# --- References ---


@pytest.mark.asyncio
async def test_references_from_text(db, graph_builder):
    entry = _make_entry(knowledge_details="See kb-00010 and also kb-00020 for more.")
    await graph_builder.build_for_entry(entry)

    edges = await _get_edges(db, source="kb-00001", edge_type="references")
    assert len(edges) == 2
    targets = {e["target"] for e in edges}
    assert targets == {"kb-00010", "kb-00020"}


@pytest.mark.asyncio
async def test_references_skip_self(db, graph_builder):
    entry = _make_entry(knowledge_details="This is kb-00001 referring to itself.")
    await graph_builder.build_for_entry(entry)

    edges = await _get_edges(db, source="kb-00001", edge_type="references")
    assert len(edges) == 0


@pytest.mark.asyncio
async def test_references_deduplicated(db, graph_builder):
    entry = _make_entry(knowledge_details="See kb-00010. Again kb-00010.")
    await graph_builder.build_for_entry(entry)

    edges = await _get_edges(db, source="kb-00001", edge_type="references")
    assert len(edges) == 1


# --- Hints: related_entities ---


@pytest.mark.asyncio
async def test_related_entities_with_custom_edge_type(db, graph_builder):
    entry = _make_entry(hints={"related_entities": [{"id": "kb-00005", "edge_type": "depends_on"}]})
    await graph_builder.build_for_entry(entry)

    edges = await _get_edges(db, source="kb-00001", edge_type="depends_on")
    assert len(edges) == 1
    assert edges[0]["target"] == "kb-00005"


@pytest.mark.asyncio
async def test_related_entities_default_edge_type(db, graph_builder):
    entry = _make_entry(hints={"related_entities": [{"id": "kb-00005"}]})
    await graph_builder.build_for_entry(entry)

    edges = await _get_edges(db, source="kb-00001", edge_type="related_to")
    assert len(edges) == 1
    assert edges[0]["target"] == "kb-00005"


@pytest.mark.asyncio
async def test_related_entities_string_form(db, graph_builder):
    entry = _make_entry(hints={"related_entities": ["kb-00005", "kb-00006"]})
    await graph_builder.build_for_entry(entry)

    edges = await _get_edges(db, source="kb-00001", edge_type="related_to")
    assert len(edges) == 2


# --- Person / Tool hints ---


@pytest.mark.asyncio
async def test_person_hint_single(db, graph_builder):
    entry = _make_entry(hints={"person": "Jason"})
    await graph_builder.build_for_entry(entry)

    nodes = await _get_nodes(db, "person")
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "person:jason"

    edges = await _get_edges(db, source="kb-00001", edge_type="mentions_person")
    assert len(edges) == 1


@pytest.mark.asyncio
async def test_person_hint_list(db, graph_builder):
    entry = _make_entry(hints={"person": ["Alice", "Bob"]})
    await graph_builder.build_for_entry(entry)

    nodes = await _get_nodes(db, "person")
    assert len(nodes) == 2
    ids = {n["node_id"] for n in nodes}
    assert ids == {"person:alice", "person:bob"}


@pytest.mark.asyncio
async def test_tool_hint_single(db, graph_builder):
    entry = _make_entry(hints={"tool": "SQLite"})
    await graph_builder.build_for_entry(entry)

    nodes = await _get_nodes(db, "tool")
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "tool:sqlite"

    edges = await _get_edges(db, source="kb-00001", edge_type="uses_tool")
    assert len(edges) == 1


@pytest.mark.asyncio
async def test_tool_hint_list(db, graph_builder):
    entry = _make_entry(hints={"tool": ["SQLite", "Python"]})
    await graph_builder.build_for_entry(entry)

    nodes = await _get_nodes(db, "tool")
    assert len(nodes) == 2
    ids = {n["node_id"] for n in nodes}
    assert ids == {"tool:sqlite", "tool:python"}


# --- Updates ---


@pytest.mark.asyncio
async def test_rebuild_clears_old_edges(db, graph_builder):
    entry = _make_entry(tags=["python", "sqlite"])
    await graph_builder.build_for_entry(entry)

    edges = await _get_edges(db, source="kb-00001", edge_type="has_tag")
    assert len(edges) == 2

    # Rebuild with different tags
    entry2 = _make_entry(tags=["rust"])
    await graph_builder.build_for_entry(entry2)

    edges = await _get_edges(db, source="kb-00001", edge_type="has_tag")
    assert len(edges) == 1
    assert edges[0]["target"] == "tag:rust"


@pytest.mark.asyncio
async def test_rebuild_preserves_incoming_edges(db, graph_builder):
    """Edges from other entries pointing to this one should survive rebuild."""
    # Entry A references entry B
    entry_a = _make_entry(entry_id="kb-00001", knowledge_details="See kb-00002.")
    await graph_builder.build_for_entry(entry_a)

    # Verify edge exists
    edges = await _get_edges(db, source="kb-00001", edge_type="references")
    assert len(edges) == 1
    assert edges[0]["target"] == "kb-00002"

    # Now rebuild entry B — should NOT remove the incoming edge from A
    entry_b = _make_entry(entry_id="kb-00002", tags=["test"])
    await graph_builder.build_for_entry(entry_b)

    edges = await _get_edges(db, source="kb-00001", edge_type="references")
    assert len(edges) == 1
    assert edges[0]["target"] == "kb-00002"


@pytest.mark.asyncio
async def test_idempotent_build(db, graph_builder):
    entry = _make_entry(tags=["python"], project_ref="myproj")
    await graph_builder.build_for_entry(entry)
    await graph_builder.build_for_entry(entry)

    edges = await _get_edges(db, source="kb-00001")
    # has_tag + in_project = 2
    assert len(edges) == 2


# --- Edge cases ---


@pytest.mark.asyncio
async def test_empty_entry(db, graph_builder):
    entry = _make_entry(tags=[], project_ref=None, hints={})
    await graph_builder.build_for_entry(entry)

    # Only the entry node itself
    nodes = await _get_nodes(db)
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "kb-00001"

    edges = await _get_edges(db, source="kb-00001")
    assert len(edges) == 0


@pytest.mark.asyncio
async def test_malformed_hints_ignored(db, graph_builder):
    """Malformed hint values should be silently ignored."""
    entry = _make_entry(
        hints={
            "supersedes": 12345,  # not a string
            "person": {"nested": "dict"},  # not a string/list
            "tool": None,
            "related_entities": "not-a-list-or-dict",
        }
    )
    await graph_builder.build_for_entry(entry)

    # Only entry node, related_to edge for the string
    edges = await _get_edges(db, source="kb-00001")
    # The "related_entities" string gets treated as a single string entity
    assert len(edges) == 1
    assert edges[0]["edge_type"] == "related_to"


@pytest.mark.asyncio
async def test_case_normalization_person_tool(db, graph_builder):
    entry = _make_entry(hints={"person": "ALICE", "tool": "PostgreSQL"})
    await graph_builder.build_for_entry(entry)

    person_nodes = await _get_nodes(db, "person")
    assert person_nodes[0]["node_id"] == "person:alice"

    tool_nodes = await _get_nodes(db, "tool")
    assert tool_nodes[0]["node_id"] == "tool:postgresql"
