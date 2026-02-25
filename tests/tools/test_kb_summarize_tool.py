"""Tests for the kb_summarize tool."""

import pytest

from personal_kb.models.entry import EntryType
from personal_kb.tools.kb_ask import _strategy_auto
from personal_kb.tools.kb_summarize import _synthesize, register_kb_summarize
from tests.conftest import FakeLLM

# --- _synthesize ---


@pytest.mark.asyncio
async def test_synthesize_success():
    """LLM should synthesize an answer from raw results."""
    llm = FakeLLM(response="SQLite uses WAL mode for concurrency [kb-00001].")
    result = await _synthesize(llm, "How does sqlite handle concurrency?", "raw results here")
    assert result is not None
    assert "WAL" in result
    assert "kb-00001" in result


@pytest.mark.asyncio
async def test_synthesize_includes_question_in_prompt():
    """The question should appear in the prompt sent to LLM."""
    llm = FakeLLM(response="answer")
    await _synthesize(llm, "my specific question", "raw results")
    assert llm.last_prompt is not None
    assert "my specific question" in llm.last_prompt


@pytest.mark.asyncio
async def test_synthesize_includes_raw_results_in_prompt():
    """Raw results should appear in the prompt sent to LLM."""
    llm = FakeLLM(response="answer")
    await _synthesize(llm, "question", "entry data here")
    assert llm.last_prompt is not None
    assert "entry data here" in llm.last_prompt


@pytest.mark.asyncio
async def test_synthesize_uses_system_prompt():
    """Synthesis should use the citation system prompt."""
    llm = FakeLLM(response="answer")
    await _synthesize(llm, "question", "results")
    assert llm.last_system is not None
    assert "kb-XXXXX" in llm.last_system
    assert "cite" in llm.last_system.lower()


@pytest.mark.asyncio
async def test_synthesize_returns_none_on_failure():
    """Should return None when LLM fails."""
    llm = FakeLLM(response=None)
    result = await _synthesize(llm, "question", "results")
    assert result is None


@pytest.mark.asyncio
async def test_synthesize_returns_none_when_unavailable():
    """Should return None when LLM is unavailable."""
    llm = FakeLLM(available=False)
    result = await _synthesize(llm, "question", "results")
    assert result is None


# --- Integration via tool-level functions ---


@pytest.mark.asyncio
async def test_kb_summarize_with_llm(db, store, fake_embedder):
    """kb_summarize should return synthesized answer when LLM is available."""
    await store.create_entry(
        short_title="SQLite WAL",
        long_title="SQLite WAL mode",
        knowledge_details="WAL mode improves concurrent reads",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    llm = FakeLLM(response="SQLite WAL mode allows concurrent reads [kb-00001].")

    raw = await _strategy_auto(db, fake_embedder, "sqlite WAL", None, True, 20)
    assert "WAL" in raw

    result = await _synthesize(llm, "How does WAL work?", raw)
    assert result is not None
    assert "kb-00001" in result


@pytest.mark.asyncio
async def test_kb_summarize_fallback_no_results(db, fake_embedder):
    """Should return clear message when no entries match."""
    raw = await _strategy_auto(db, fake_embedder, "nonexistent topic xyz", None, True, 20)
    assert raw == "No results found."


@pytest.mark.asyncio
async def test_register_kb_summarize():
    """Should register without error."""
    from fastmcp import FastMCP

    mcp = FastMCP("test")
    register_kb_summarize(mcp)
    # Just verifying no exceptions during registration
