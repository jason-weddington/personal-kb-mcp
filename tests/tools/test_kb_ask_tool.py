"""Tests for the kb_ask MCP tool formatting and strategies."""

from datetime import UTC, datetime

import pytest

from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.tools.kb_ask import (
    _format_entries,
    _strategy_connection,
    _strategy_related,
    _strategy_timeline,
)


def _make_entry(
    entry_id: str = "kb-00001",
    short_title: str = "Test Entry",
    entry_type: EntryType = EntryType.FACTUAL_REFERENCE,
    knowledge_details: str = "Some details",
    tags: list[str] | None = None,
    project_ref: str | None = None,
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        short_title=short_title,
        long_title="A test entry",
        knowledge_details=knowledge_details,
        entry_type=entry_type,
        tags=tags or [],
        project_ref=project_ref,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


# --- _format_entries ---


def test_format_entries_basic():
    """Should format entries with context strings."""
    entries = [
        (_make_entry(entry_id="kb-00001", short_title="First"), "search match"),
        (_make_entry(entry_id="kb-00002", short_title="Second"), "graph neighbor"),
    ]
    output = _format_entries(entries, "Test results")
    assert "Test results" in output
    assert "kb-00001" in output
    assert "kb-00002" in output
    assert "search match" in output
    assert "graph neighbor" in output
    assert "2 result(s)" in output


def test_format_entries_with_tags():
    """Should include tags in formatted output."""
    entries = [
        (_make_entry(tags=["python", "sqlite"]), "test"),
    ]
    output = _format_entries(entries, "Header")
    assert "python, sqlite" in output


def test_format_entries_decision_type():
    """Should show entry type correctly."""
    entries = [
        (_make_entry(entry_type=EntryType.DECISION, short_title="Chose X"), "current decision"),
    ]
    output = _format_entries(entries, "Decisions")
    assert "decision" in output
    assert "Chose X" in output


def test_format_entries_includes_details():
    """Should include knowledge details."""
    entries = [
        (_make_entry(knowledge_details="Very important fact"), "found"),
    ]
    output = _format_entries(entries, "Header")
    assert "Very important fact" in output


# --- _strategy_timeline ---


@pytest.mark.asyncio
async def test_timeline_requires_scope(db):
    """Timeline should require a scope."""
    result = await _strategy_timeline(db, scope=None, limit=20)
    assert "requires" in result.lower()


@pytest.mark.asyncio
async def test_timeline_empty_scope(db):
    """Timeline should handle missing entries gracefully."""
    result = await _strategy_timeline(db, scope="project:nonexistent", limit=20)
    assert "No entries found" in result


@pytest.mark.asyncio
async def test_timeline_chronological_order(db, store, graph_builder):
    """Timeline should return entries in chronological order."""
    e1 = await store.create_entry(
        short_title="First",
        long_title="First entry",
        knowledge_details="First details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="test-proj",
    )
    await graph_builder.build_for_entry(e1)
    e2 = await store.create_entry(
        short_title="Second",
        long_title="Second entry",
        knowledge_details="Second details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="test-proj",
    )
    await graph_builder.build_for_entry(e2)

    result = await _strategy_timeline(db, scope="project:test-proj", limit=20)
    assert "First" in result
    assert "Second" in result
    # First should appear before Second (chronological)
    assert result.index("First") < result.index("Second")


# --- _strategy_related ---


@pytest.mark.asyncio
async def test_related_requires_scope(db):
    """Related should require a scope."""
    result = await _strategy_related(db, scope=None, limit=20)
    assert "requires" in result.lower()


@pytest.mark.asyncio
async def test_related_finds_connected_entries(db, store, graph_builder):
    """Related should find entries connected via shared tags."""
    e1 = await store.create_entry(
        short_title="Entry 1",
        long_title="Entry 1",
        knowledge_details="details 1",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["python"],
    )
    await graph_builder.build_for_entry(e1)
    e2 = await store.create_entry(
        short_title="Entry 2",
        long_title="Entry 2",
        knowledge_details="details 2",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["python"],
    )
    await graph_builder.build_for_entry(e2)

    result = await _strategy_related(db, scope=e1.id, limit=20)
    assert e2.id in result


# --- _strategy_connection ---


@pytest.mark.asyncio
async def test_connection_requires_both_params(db):
    """Connection should require scope and target."""
    result = await _strategy_connection(db, scope=None, target=None)
    assert "requires" in result.lower()

    result = await _strategy_connection(db, scope="kb-00001", target=None)
    assert "requires" in result.lower()


@pytest.mark.asyncio
async def test_connection_no_path(db, graph_builder):
    """Connection should report no path when nodes are disconnected."""
    e1 = _make_entry(entry_id="kb-00001", tags=["python"])
    e2 = _make_entry(entry_id="kb-00002", tags=["rust"])
    await graph_builder.build_for_entry(e1)
    await graph_builder.build_for_entry(e2)

    result = await _strategy_connection(db, scope="kb-00001", target="kb-00002")
    assert "No connection" in result


@pytest.mark.asyncio
async def test_connection_finds_path(db, graph_builder):
    """Connection should find and display a path."""
    e1 = _make_entry(entry_id="kb-00001", tags=["python"])
    e2 = _make_entry(entry_id="kb-00002", tags=["python"])
    await graph_builder.build_for_entry(e1)
    await graph_builder.build_for_entry(e2)

    result = await _strategy_connection(db, scope="kb-00001", target="kb-00002")
    assert "Connection" in result
    assert "Path:" in result
    assert "has_tag" in result
