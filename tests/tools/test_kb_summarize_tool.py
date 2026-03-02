"""Tests for the kb_summarize tool."""

import pytest

from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.tools.kb_ask import _strategy_auto
from personal_kb.tools.kb_summarize import (
    _merge_entries,
    _synthesize,
    register_kb_summarize,
    summarize_question,
)
from tests.conftest import FakeLLM


def _make_entry(
    entry_id: str = "kb-00001",
    title: str = "Test",
    details: str = "Details here",
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        short_title=title,
        long_title=title,
        knowledge_details=details,
        entry_type=EntryType.FACTUAL_REFERENCE,
    )


# --- _synthesize ---


@pytest.mark.asyncio
async def test_synthesize_success():
    """LLM should synthesize an answer from structured entries."""
    llm = FakeLLM(response="SQLite uses WAL mode for concurrency [kb-00001].")
    entries = [(_make_entry(details="WAL mode improves reads"), "search match")]
    result = await _synthesize(llm, "How does sqlite handle concurrency?", entries)
    assert result is not None
    assert "WAL" in result
    assert "kb-00001" in result


@pytest.mark.asyncio
async def test_synthesize_includes_question_in_prompt():
    """The question should appear in the prompt sent to LLM."""
    llm = FakeLLM(response="answer")
    entries = [(_make_entry(), "search match")]
    await _synthesize(llm, "my specific question", entries)
    assert llm.last_prompt is not None
    assert "my specific question" in llm.last_prompt


@pytest.mark.asyncio
async def test_synthesize_includes_entry_details_in_prompt():
    """Entry knowledge_details should appear in the prompt."""
    llm = FakeLLM(response="answer")
    entries = [(_make_entry(details="unique detail XYZ"), "search match")]
    await _synthesize(llm, "question", entries)
    assert llm.last_prompt is not None
    assert "unique detail XYZ" in llm.last_prompt


@pytest.mark.asyncio
async def test_synthesize_uses_system_prompt():
    """Synthesis should use the citation system prompt."""
    llm = FakeLLM(response="answer")
    entries = [(_make_entry(), "search match")]
    await _synthesize(llm, "question", entries)
    assert llm.last_system is not None
    assert "kb-XXXXX" in llm.last_system
    assert "cite" in llm.last_system.lower()


@pytest.mark.asyncio
async def test_synthesize_returns_none_on_failure():
    """Should return None when LLM fails."""
    llm = FakeLLM(response=None)
    entries = [(_make_entry(), "search match")]
    result = await _synthesize(llm, "question", entries)
    assert result is None


@pytest.mark.asyncio
async def test_synthesize_returns_none_when_unavailable():
    """Should return None when LLM is unavailable."""
    llm = FakeLLM(available=False)
    entries = [(_make_entry(), "search match")]
    result = await _synthesize(llm, "question", entries)
    assert result is None


# --- _merge_entries ---


def test_merge_deduplicates():
    e1 = _make_entry("kb-00001", "First")
    e2 = _make_entry("kb-00002", "Second")
    e3 = _make_entry("kb-00001", "First duplicate")
    original = [(e1, "ctx1"), (e2, "ctx2")]
    extra = [(e3, "ctx3"), (_make_entry("kb-00003", "Third"), "ctx4")]
    result = _merge_entries(original, extra)
    assert len(result) == 3
    assert [e.id for e, _ in result] == ["kb-00001", "kb-00002", "kb-00003"]


def test_merge_preserves_order():
    e1 = _make_entry("kb-00001")
    e2 = _make_entry("kb-00002")
    result = _merge_entries([(e1, "a")], [(e2, "b")])
    assert [e.id for e, _ in result] == ["kb-00001", "kb-00002"]


# --- Integration via tool-level functions ---


@pytest.mark.asyncio
async def test_kb_summarize_with_llm(db, store, fake_embedder):
    """summarize_question should return synthesized answer when LLM available."""
    await store.create_entry(
        short_title="SQLite WAL",
        long_title="SQLite WAL mode",
        knowledge_details="WAL mode improves concurrent reads",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    llm = FakeLLM(response="SQLite WAL mode allows concurrent reads [kb-00001].")

    result = await summarize_question(
        db,
        fake_embedder,
        llm,
        "sqlite WAL",
        limit=20,
    )
    assert result is not None
    assert "kb-00001" in result


@pytest.mark.asyncio
async def test_kb_summarize_fallback_no_results(db, fake_embedder):
    """Should return clear message when no entries match."""
    raw = await _strategy_auto(
        db,
        fake_embedder,
        "nonexistent topic xyz",
        None,
        True,
        20,
    )
    assert raw == "No results found."


@pytest.mark.asyncio
async def test_kb_summarize_no_llm_fallback(db, store, fake_embedder):
    """Without LLM, should show raw results with fallback message."""
    await store.create_entry(
        short_title="Test Entry",
        long_title="Test Entry for Fallback",
        knowledge_details="Some details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    result = await summarize_question(
        db,
        fake_embedder,
        None,
        "test entry",
        limit=20,
    )
    assert "LLM unavailable" in result
    assert "Test Entry" in result


@pytest.mark.asyncio
async def test_register_kb_summarize():
    """Should register without error."""
    from fastmcp import FastMCP

    mcp = FastMCP("test")
    register_kb_summarize(mcp)
    # Just verifying no exceptions during registration
