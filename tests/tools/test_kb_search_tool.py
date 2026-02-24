"""Tests for the kb_search MCP tool formatting."""

from datetime import UTC, datetime

from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.models.search import SearchResult
from personal_kb.tools.kb_search import format_search_results


def test_format_empty_results():
    result = format_search_results([])
    assert result == "No results found."


def test_format_results_with_note():
    result = format_search_results([], match_source_note="FTS only")
    assert "No results found." in result


def test_format_results():
    entry = KnowledgeEntry(
        id="kb-00001",
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Some details here",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="test-proj",
        tags=["a", "b"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    results = [
        SearchResult(
            entry=entry,
            score=0.025,
            effective_confidence=0.85,
            staleness_warning=None,
            match_source="hybrid",
        )
    ]
    output = format_search_results(results)
    assert "kb-00001" in output
    assert "Test" in output
    assert "test-proj" in output
    assert "85%" in output
    assert "a, b" in output


def test_format_results_with_staleness():
    entry = KnowledgeEntry(
        id="kb-00002",
        short_title="Old fact",
        long_title="An old factual reference",
        knowledge_details="Outdated info",
        entry_type=EntryType.FACTUAL_REFERENCE,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    results = [
        SearchResult(
            entry=entry,
            score=0.01,
            effective_confidence=0.3,
            staleness_warning="Stale factual_reference entry (confidence: 30%).",
            match_source="fts",
        )
    ]
    output = format_search_results(results)
    assert "WARNING" in output
    assert "Stale" in output
