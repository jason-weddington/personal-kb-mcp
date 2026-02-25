"""Tests for graph traversal queries."""

import pytest

from personal_kb.graph.queries import (
    bfs_entries,
    entries_for_scope,
    find_path,
    get_graph_vocabulary,
    get_neighbors,
    supersedes_chain,
)
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


# --- get_neighbors ---


@pytest.mark.asyncio
async def test_get_neighbors_basic(db, graph_builder):
    """Should return outgoing and incoming neighbors."""
    entry = _make_entry(tags=["python", "sqlite"])
    await graph_builder.build_for_entry(entry)

    neighbors = await get_neighbors(db, "kb-00001")
    neighbor_ids = {n[0] for n in neighbors}
    assert "tag:python" in neighbor_ids
    assert "tag:sqlite" in neighbor_ids


@pytest.mark.asyncio
async def test_get_neighbors_filtered_by_edge_type(db, graph_builder):
    """Should filter by edge type."""
    entry = _make_entry(tags=["python"], project_ref="myproj")
    await graph_builder.build_for_entry(entry)

    neighbors = await get_neighbors(db, "kb-00001", edge_types=["has_tag"])
    assert len(neighbors) == 1
    assert neighbors[0][0] == "tag:python"
    assert neighbors[0][1] == "has_tag"


@pytest.mark.asyncio
async def test_get_neighbors_outgoing_only(db, graph_builder):
    """Should only return outgoing edges."""
    entry = _make_entry(tags=["python"])
    await graph_builder.build_for_entry(entry)

    outgoing = await get_neighbors(db, "kb-00001", direction="outgoing")
    assert len(outgoing) == 1
    assert outgoing[0][2] == "outgoing"


@pytest.mark.asyncio
async def test_get_neighbors_incoming_only(db, graph_builder):
    """Should only return incoming edges."""
    entry = _make_entry(tags=["python"])
    await graph_builder.build_for_entry(entry)

    incoming = await get_neighbors(db, "tag:python", direction="incoming")
    assert len(incoming) == 1
    assert incoming[0][0] == "kb-00001"
    assert incoming[0][2] == "incoming"


@pytest.mark.asyncio
async def test_get_neighbors_empty(db):
    """No neighbors for a nonexistent node."""
    neighbors = await get_neighbors(db, "nonexistent")
    assert neighbors == []


# --- bfs_entries ---


@pytest.mark.asyncio
async def test_bfs_entries_depth_1(db, graph_builder):
    """BFS at depth 1 should find directly connected entries."""
    e1 = _make_entry(entry_id="kb-00001", tags=["python"])
    e2 = _make_entry(entry_id="kb-00002", tags=["python"])
    await graph_builder.build_for_entry(e1)
    await graph_builder.build_for_entry(e2)

    results = await bfs_entries(db, "kb-00001", max_depth=2)
    entry_ids = [r[0] for r in results]
    # kb-00001 -> tag:python -> kb-00002 (depth 2)
    assert "kb-00002" in entry_ids


@pytest.mark.asyncio
async def test_bfs_entries_respects_max_depth(db, graph_builder):
    """BFS should not go beyond max_depth."""
    e1 = _make_entry(entry_id="kb-00001", tags=["python"])
    e2 = _make_entry(entry_id="kb-00002", tags=["python"])
    await graph_builder.build_for_entry(e1)
    await graph_builder.build_for_entry(e2)

    # Depth 1: kb-00001 -> tag:python only, can't reach kb-00002
    results = await bfs_entries(db, "kb-00001", max_depth=1)
    entry_ids = [r[0] for r in results]
    assert "kb-00002" not in entry_ids


@pytest.mark.asyncio
async def test_bfs_entries_no_cycles(db, graph_builder):
    """BFS should not revisit nodes."""
    e1 = _make_entry(entry_id="kb-00001", knowledge_details="See kb-00002")
    e2 = _make_entry(entry_id="kb-00002", knowledge_details="See kb-00001")
    await graph_builder.build_for_entry(e1)
    await graph_builder.build_for_entry(e2)

    results = await bfs_entries(db, "kb-00001", max_depth=4)
    entry_ids = [r[0] for r in results]
    # Should find kb-00002 exactly once
    assert entry_ids.count("kb-00002") == 1


@pytest.mark.asyncio
async def test_bfs_entries_excludes_start(db, graph_builder):
    """BFS should not include the start node in results."""
    e1 = _make_entry(entry_id="kb-00001", tags=["python"])
    await graph_builder.build_for_entry(e1)

    results = await bfs_entries(db, "kb-00001", max_depth=2)
    entry_ids = [r[0] for r in results]
    assert "kb-00001" not in entry_ids


# --- find_path ---


@pytest.mark.asyncio
async def test_find_path_direct(db, graph_builder):
    """Should find a direct 1-hop connection."""
    entry = _make_entry(entry_id="kb-00001", tags=["python"])
    await graph_builder.build_for_entry(entry)

    path = await find_path(db, "kb-00001", "tag:python")
    assert path is not None
    assert len(path) == 1
    assert path[0] == ("kb-00001", "has_tag", "tag:python")


@pytest.mark.asyncio
async def test_find_path_two_hop(db, graph_builder):
    """Should find a 2-hop path through shared tag."""
    e1 = _make_entry(entry_id="kb-00001", tags=["python"])
    e2 = _make_entry(entry_id="kb-00002", tags=["python"])
    await graph_builder.build_for_entry(e1)
    await graph_builder.build_for_entry(e2)

    path = await find_path(db, "kb-00001", "kb-00002")
    assert path is not None
    assert len(path) == 2


@pytest.mark.asyncio
async def test_find_path_none(db, graph_builder):
    """Should return None when no path exists."""
    e1 = _make_entry(entry_id="kb-00001", tags=["python"])
    e2 = _make_entry(entry_id="kb-00002", tags=["rust"])
    await graph_builder.build_for_entry(e1)
    await graph_builder.build_for_entry(e2)

    path = await find_path(db, "kb-00001", "kb-00002")
    assert path is None


@pytest.mark.asyncio
async def test_find_path_same_node(db, graph_builder):
    """Should return empty path for same source and target."""
    entry = _make_entry(entry_id="kb-00001")
    await graph_builder.build_for_entry(entry)

    path = await find_path(db, "kb-00001", "kb-00001")
    assert path is not None
    assert path == []


@pytest.mark.asyncio
async def test_find_path_max_depth(db, graph_builder):
    """Should respect max_depth limit."""
    # Create a chain: kb-00001 -> tag:a -> kb-00002 -> tag:b -> kb-00003
    e1 = _make_entry(entry_id="kb-00001", tags=["a"])
    e2 = _make_entry(entry_id="kb-00002", tags=["a", "b"])
    e3 = _make_entry(entry_id="kb-00003", tags=["b"])
    await graph_builder.build_for_entry(e1)
    await graph_builder.build_for_entry(e2)
    await graph_builder.build_for_entry(e3)

    # Depth 1 can't reach kb-00003
    path = await find_path(db, "kb-00001", "kb-00003", max_depth=1)
    assert path is None


# --- entries_for_scope ---


@pytest.mark.asyncio
async def test_entries_for_scope_project(db, store, graph_builder):
    """Should find entries by project ref."""
    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="personal-kb",
    )
    await graph_builder.build_for_entry(entry)

    ids = await entries_for_scope(db, "project:personal-kb")
    assert entry.id in ids


@pytest.mark.asyncio
async def test_entries_for_scope_tag(db, store, graph_builder):
    """Should find entries by tag via graph edges."""
    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["python"],
    )
    await graph_builder.build_for_entry(entry)

    ids = await entries_for_scope(db, "tag:python")
    assert entry.id in ids


@pytest.mark.asyncio
async def test_entries_for_scope_entry_type(db, store):
    """Should find entries by type."""
    entry = await store.create_entry(
        short_title="A decision",
        long_title="A decision entry",
        knowledge_details="decided X",
        entry_type=EntryType.DECISION,
    )

    ids = await entries_for_scope(db, "decision")
    assert entry.id in ids


@pytest.mark.asyncio
async def test_entries_for_scope_entry_id(db):
    """Should return just the entry ID for a kb-XXXXX scope."""
    ids = await entries_for_scope(db, "kb-00042")
    assert ids == ["kb-00042"]


@pytest.mark.asyncio
async def test_entries_for_scope_tool(db, store, graph_builder):
    """Should find entries by tool hint via graph edges."""
    entry = await store.create_entry(
        short_title="SQLite tip",
        long_title="SQLite tip",
        knowledge_details="use WAL mode",
        entry_type=EntryType.LESSON_LEARNED,
        hints={"tool": "sqlite"},
    )
    await graph_builder.build_for_entry(entry)

    ids = await entries_for_scope(db, "tool:sqlite")
    assert entry.id in ids


# --- supersedes_chain ---


@pytest.mark.asyncio
async def test_supersedes_chain_simple(db, graph_builder):
    """Should build a simple supersedes chain, oldest first."""
    # kb-00001 was superseded by kb-00002, which was superseded by kb-00003
    e1 = _make_entry(entry_id="kb-00001")
    e2 = _make_entry(entry_id="kb-00002", hints={"supersedes": "kb-00001"})
    e3 = _make_entry(entry_id="kb-00003", hints={"supersedes": "kb-00002"})
    await graph_builder.build_for_entry(e1)
    await graph_builder.build_for_entry(e2)
    await graph_builder.build_for_entry(e3)

    chain = await supersedes_chain(db, "kb-00002")
    assert chain == ["kb-00001", "kb-00002", "kb-00003"]


@pytest.mark.asyncio
async def test_supersedes_chain_single(db, graph_builder):
    """Single entry with no chain should return just itself."""
    e1 = _make_entry(entry_id="kb-00001")
    await graph_builder.build_for_entry(e1)

    chain = await supersedes_chain(db, "kb-00001")
    assert chain == ["kb-00001"]


@pytest.mark.asyncio
async def test_supersedes_chain_from_end(db, graph_builder):
    """Starting from the newest entry should still build the full chain."""
    e1 = _make_entry(entry_id="kb-00001")
    e2 = _make_entry(entry_id="kb-00002", hints={"supersedes": "kb-00001"})
    await graph_builder.build_for_entry(e1)
    await graph_builder.build_for_entry(e2)

    chain = await supersedes_chain(db, "kb-00002")
    assert chain == ["kb-00001", "kb-00002"]


# --- get_graph_vocabulary ---


@pytest.mark.asyncio
async def test_get_graph_vocabulary_basic(db, graph_builder):
    """Should return non-entry nodes grouped by type."""
    e1 = _make_entry(entry_id="kb-00001", tags=["python", "sqlite"], project_ref="myproj")
    await graph_builder.build_for_entry(e1)

    vocab = await get_graph_vocabulary(db)
    assert "tag" in vocab
    assert "python" in vocab["tag"]
    assert "sqlite" in vocab["tag"]
    assert "project" in vocab
    assert "myproj" in vocab["project"]
    # Entry nodes should not appear
    assert "entry" not in vocab


@pytest.mark.asyncio
async def test_get_graph_vocabulary_ordered_by_connections(db, graph_builder):
    """Nodes with more connections should appear first."""
    e1 = _make_entry(entry_id="kb-00001", tags=["popular", "rare"])
    e2 = _make_entry(entry_id="kb-00002", tags=["popular"])
    e3 = _make_entry(entry_id="kb-00003", tags=["popular"])
    await graph_builder.build_for_entry(e1)
    await graph_builder.build_for_entry(e2)
    await graph_builder.build_for_entry(e3)

    vocab = await get_graph_vocabulary(db)
    tags = vocab["tag"]
    # "popular" has 3 connections, "rare" has 1
    assert tags.index("popular") < tags.index("rare")


@pytest.mark.asyncio
async def test_get_graph_vocabulary_empty(db):
    """Should return empty dict when no nodes exist."""
    vocab = await get_graph_vocabulary(db)
    assert vocab == {}


@pytest.mark.asyncio
async def test_get_graph_vocabulary_max_nodes(db, graph_builder):
    """Should respect max_nodes limit."""
    # Create entries with unique tags to generate many tag nodes
    for i in range(10):
        e = _make_entry(entry_id=f"kb-{i:05d}", tags=[f"tag{i}"])
        await graph_builder.build_for_entry(e)

    vocab = await get_graph_vocabulary(db, max_nodes=3)
    total = sum(len(v) for v in vocab.values())
    assert total <= 3
