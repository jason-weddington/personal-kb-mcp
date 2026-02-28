"""Tests for the kb_search MCP tool formatting and graph hints."""

from datetime import UTC, datetime

import pytest

from personal_kb.db.queries import insert_entry
from personal_kb.graph.builder import GraphBuilder
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.models.search import SearchResult
from personal_kb.tools.formatters import format_graph_hint
from personal_kb.tools.kb_search import collect_graph_hints, format_search_results


def _make_entry(
    entry_id: str = "kb-00001",
    short_title: str = "Test",
    long_title: str = "Test entry",
    knowledge_details: str = "Some details here",
    entry_type: EntryType = EntryType.FACTUAL_REFERENCE,
    tags: list[str] | None = None,
    project_ref: str | None = None,
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        short_title=short_title,
        long_title=long_title,
        knowledge_details=knowledge_details,
        entry_type=entry_type,
        tags=tags or [],
        project_ref=project_ref,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_result(
    entry: KnowledgeEntry,
    score: float = 0.025,
    confidence: float = 0.85,
    staleness: str | None = None,
) -> SearchResult:
    return SearchResult(
        entry=entry,
        score=score,
        effective_confidence=confidence,
        staleness_warning=staleness,
        match_source="hybrid",
    )


# --- format_search_results ---


def test_format_empty_results():
    result = format_search_results([])
    assert result == "No results found."


def test_format_results_with_note():
    result = format_search_results([], match_source_note="FTS only")
    assert "No results found." in result


def test_format_results():
    entry = _make_entry(project_ref="test-proj", tags=["a", "b"])
    results = [_make_result(entry)]
    output = format_search_results(results)
    assert "kb-00001" in output
    assert "Test" in output
    assert "test-proj" in output
    assert "85%" in output
    assert "#a #b" in output
    # Compact format should NOT include knowledge_details
    assert "Some details here" not in output


def test_format_results_with_staleness():
    entry = _make_entry(
        entry_id="kb-00002",
        short_title="Old fact",
        long_title="An old factual reference",
        knowledge_details="Outdated info",
    )
    results = [_make_result(entry, score=0.01, confidence=0.3, staleness="Stale")]
    output = format_search_results(results)
    assert "[STALE]" in output


def test_format_results_with_graph_hints():
    """Graph hints should appear in formatted output."""
    entry = _make_entry()
    results = [_make_result(entry)]
    hints = ["See also: [kb-00042] Related entry (via tag:python)"]
    output = format_search_results(results, graph_hints=hints)
    assert "Related entries via graph:" in output
    assert "kb-00042" in output
    assert "via tag:python" in output


def test_format_results_no_hints_when_none():
    """No graph hints section when hints is None."""
    entry = _make_entry()
    results = [_make_result(entry)]
    output = format_search_results(results, graph_hints=None)
    assert "Related entries via graph:" not in output


def test_format_results_no_hints_when_empty():
    """No graph hints section when hints list is empty."""
    entry = _make_entry()
    results = [_make_result(entry)]
    output = format_search_results(results, graph_hints=[])
    assert "Related entries via graph:" not in output


# --- format_graph_hint ---


def test_format_graph_hint():
    entry = _make_entry(entry_id="kb-00042", short_title="Chose aiosqlite")
    hint = format_graph_hint(entry, "concept:async-io")
    assert hint == "See also: [kb-00042] Chose aiosqlite (via concept:async-io)"


# --- collect_graph_hints ---


@pytest.mark.asyncio
async def test_collect_graph_hints_empty_results(db):
    """No hints when there are no search results."""
    hints = await collect_graph_hints(db, [])
    assert hints == []


@pytest.mark.asyncio
async def test_collect_graph_hints_no_graph_edges(db):
    """No hints when entries have no graph connections."""
    entry = _make_entry()
    await insert_entry(db, entry)
    results = [_make_result(entry)]
    hints = await collect_graph_hints(db, results)
    assert hints == []


@pytest.mark.asyncio
async def test_collect_graph_hints_via_shared_tag(db):
    """Should find related entries via shared tag nodes."""
    # Create two entries that share a tag
    entry1 = _make_entry(entry_id="kb-00001", short_title="First", tags=["python"])
    entry2 = _make_entry(entry_id="kb-00002", short_title="Second", tags=["python"])
    await insert_entry(db, entry1)
    await insert_entry(db, entry2)

    # Build graph edges for both
    builder = GraphBuilder(db)
    await builder.build_for_entry(entry1)
    await builder.build_for_entry(entry2)

    # Search returns only entry1 — should hint at entry2 via tag:python
    results = [_make_result(entry1)]
    hints = await collect_graph_hints(db, results)
    assert len(hints) == 1
    assert "kb-00002" in hints[0]
    assert "tag:python" in hints[0]


@pytest.mark.asyncio
async def test_collect_graph_hints_skips_result_entries(db):
    """Should not hint at entries already in the result set."""
    entry1 = _make_entry(entry_id="kb-00001", short_title="First", tags=["python"])
    entry2 = _make_entry(entry_id="kb-00002", short_title="Second", tags=["python"])
    await insert_entry(db, entry1)
    await insert_entry(db, entry2)

    builder = GraphBuilder(db)
    await builder.build_for_entry(entry1)
    await builder.build_for_entry(entry2)

    # Both entries in results — no hints
    results = [_make_result(entry1), _make_result(entry2)]
    hints = await collect_graph_hints(db, results)
    assert hints == []


@pytest.mark.asyncio
async def test_collect_graph_hints_max_limit(db):
    """Should cap hints at max_hints."""
    # Create entry1 + 5 related entries via shared tag
    entries = [
        _make_entry(entry_id=f"kb-{i:05d}", short_title=f"Entry {i}", tags=["shared"])
        for i in range(1, 7)
    ]
    builder = GraphBuilder(db)
    for e in entries:
        await insert_entry(db, e)
        await builder.build_for_entry(e)

    # Search returns only first entry
    results = [_make_result(entries[0])]
    hints = await collect_graph_hints(db, results, max_hints=3)
    assert len(hints) == 3


@pytest.mark.asyncio
async def test_collect_graph_hints_direct_entry_edge(db):
    """Should find hints via direct entry-to-entry edges (e.g. supersedes)."""
    entry1 = _make_entry(entry_id="kb-00001", short_title="Original decision")
    entry2 = _make_entry(
        entry_id="kb-00002",
        short_title="Updated decision",
        entry_type=EntryType.DECISION,
    )
    # entry2 supersedes entry1 via hints
    entry2.hints = {"supersedes": "kb-00001"}
    await insert_entry(db, entry1)
    await insert_entry(db, entry2)

    # Build graph — builder handles nodes + edges
    builder = GraphBuilder(db)
    await builder.build_for_entry(entry1)
    await builder.build_for_entry(entry2)

    # Search returns only entry1 — should hint at entry2
    results = [_make_result(entry1)]
    hints = await collect_graph_hints(db, results)
    assert len(hints) == 1
    assert "kb-00002" in hints[0]
    assert "supersedes" in hints[0]


@pytest.mark.asyncio
async def test_collect_graph_hints_skips_inactive(db):
    """Should not include hints for inactive entries."""
    entry1 = _make_entry(entry_id="kb-00001", short_title="Active", tags=["python"])
    entry2 = _make_entry(entry_id="kb-00002", short_title="Inactive", tags=["python"])
    entry2.is_active = False
    await insert_entry(db, entry1)
    await insert_entry(db, entry2)

    builder = GraphBuilder(db)
    await builder.build_for_entry(entry1)
    await builder.build_for_entry(entry2)

    # Deactivate entry2 in DB
    await db.execute("UPDATE knowledge_entries SET is_active = 0 WHERE id = ?", ("kb-00002",))
    await db.commit()

    results = [_make_result(entry1)]
    hints = await collect_graph_hints(db, results)
    assert hints == []


@pytest.mark.asyncio
async def test_collect_graph_hints_via_project(db):
    """Should find related entries via shared project node."""
    entry1 = _make_entry(entry_id="kb-00001", short_title="First", project_ref="my-proj")
    entry2 = _make_entry(entry_id="kb-00002", short_title="Second", project_ref="my-proj")
    await insert_entry(db, entry1)
    await insert_entry(db, entry2)

    builder = GraphBuilder(db)
    await builder.build_for_entry(entry1)
    await builder.build_for_entry(entry2)

    results = [_make_result(entry1)]
    hints = await collect_graph_hints(db, results)
    assert len(hints) == 1
    assert "kb-00002" in hints[0]
    assert "project:my-proj" in hints[0]
