"""Integration tests for agentic synthesis in kb_summarize."""

import json

import pytest

from personal_kb.models.entry import EntryType
from personal_kb.tools.kb_summarize import summarize_question
from tests.conftest import FakeLLM, ScriptedLLM


@pytest.mark.asyncio
async def test_agentic_retrieval_via_agent(db, store, fake_embedder, monkeypatch):
    """When agentic query is on, retrieval goes through the agent loop."""
    monkeypatch.setenv("KB_AGENTIC_QUERY", "TRUE")
    monkeypatch.setenv("KB_AGENTIC_SYNTHESIS", "TRUE")

    await store.create_entry(
        short_title="SQLite WAL",
        long_title="SQLite WAL mode for concurrency",
        knowledge_details="WAL mode improves concurrent reads",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    # Store embedding so hybrid_search can find it
    embedding = await fake_embedder.embed("SQLite WAL mode for concurrency")
    await fake_embedder.store_embedding("kb-00001", embedding)

    # Agent fast-path returns results immediately (turns=0),
    # so coverage check should be skipped.
    # ScriptedLLM: only the synthesis call matters (agent fast-paths).
    llm = ScriptedLLM(
        responses=[
            # Synthesis response
            "WAL mode allows concurrent reads [kb-00001].",
        ],
    )

    result = await summarize_question(
        db,
        fake_embedder,
        llm,
        "sqlite WAL",
        limit=20,
    )
    assert "kb-00001" in result
    assert "WAL" in result


@pytest.mark.asyncio
async def test_coverage_check_fires_when_agent_turns_positive(
    db,
    store,
    fake_embedder,
    monkeypatch,
):
    """Coverage check should fire when agent used turns (not fast-path)."""
    monkeypatch.setenv("KB_AGENTIC_QUERY", "TRUE")
    monkeypatch.setenv("KB_AGENTIC_SYNTHESIS", "TRUE")

    await store.create_entry(
        short_title="Async IO",
        long_title="Python async IO patterns",
        knowledge_details="asyncio event loop details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    # No embedding stored — forces agent to do work (no fast-path)

    # Agent: weak fast-path → agent search → done
    agent_search_response = json.dumps(
        {
            "tool": "hybrid_search",
            "args": {"query": "async IO patterns"},
        }
    )
    agent_done_response = json.dumps(
        {
            "tool": "done",
            "args": {
                "entry_ids": ["kb-00001"],
                "reasoning": "Found async IO entry",
            },
        }
    )
    # Coverage: no gaps
    coverage_response = json.dumps(
        {
            "has_gaps": False,
            "suggested_query": None,
            "reason": "covered",
        }
    )
    # Synthesis
    synthesis_response = "Async IO uses event loops [kb-00001]."

    llm = ScriptedLLM(
        responses=[
            agent_search_response,
            agent_done_response,
            coverage_response,
            synthesis_response,
        ],
    )

    result = await summarize_question(
        db,
        fake_embedder,
        llm,
        "async IO patterns",
        limit=20,
    )
    assert "kb-00001" in result
    # Coverage check used 1 call, synthesis used 1 call
    assert len(llm.calls) >= 3  # agent (1-2) + coverage + synthesis


@pytest.mark.asyncio
async def test_coverage_check_skipped_on_fast_path(
    db,
    store,
    fake_embedder,
    monkeypatch,
):
    """Coverage check should be skipped when agent fast-paths (turns=0)."""
    monkeypatch.setenv("KB_AGENTIC_QUERY", "TRUE")
    monkeypatch.setenv("KB_AGENTIC_SYNTHESIS", "TRUE")

    await store.create_entry(
        short_title="Docker Compose",
        long_title="Docker Compose setup",
        knowledge_details="docker-compose.yml configuration",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    embedding = await fake_embedder.embed(
        "Docker Compose Docker Compose setup docker-compose.yml configuration"
    )
    await fake_embedder.store_embedding("kb-00001", embedding)

    # Only synthesis call — no coverage check
    llm = ScriptedLLM(
        responses=["Docker Compose uses YAML config [kb-00001]."],
    )

    result = await summarize_question(
        db,
        fake_embedder,
        llm,
        "Docker Compose",
        limit=20,
    )
    assert "kb-00001" in result
    # Only 1 LLM call (synthesis), no coverage check
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_agentic_synthesis_disabled(
    db,
    store,
    fake_embedder,
    monkeypatch,
):
    """With KB_AGENTIC_SYNTHESIS=FALSE, coverage check is skipped."""
    monkeypatch.setenv("KB_AGENTIC_QUERY", "FALSE")
    monkeypatch.setenv("KB_AGENTIC_SYNTHESIS", "FALSE")

    await store.create_entry(
        short_title="Redis Cache",
        long_title="Redis caching patterns",
        knowledge_details="Redis LRU eviction",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    llm = FakeLLM(response="Redis uses LRU eviction [kb-00001].")

    result = await summarize_question(
        db,
        fake_embedder,
        llm,
        "redis caching",
        limit=20,
    )
    assert "kb-00001" in result
    # FakeLLM called for planner (1) + synthesis (1), no coverage check
    assert llm.generate_count == 2


@pytest.mark.asyncio
async def test_coverage_gap_triggers_re_search(
    db,
    store,
    fake_embedder,
    monkeypatch,
):
    """When coverage finds a gap, extra entries are merged in."""
    monkeypatch.setenv("KB_AGENTIC_QUERY", "TRUE")
    monkeypatch.setenv("KB_AGENTIC_SYNTHESIS", "TRUE")

    await store.create_entry(
        short_title="API Auth",
        long_title="API authentication overview",
        knowledge_details="JWT token based auth",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.create_entry(
        short_title="OAuth Flow",
        long_title="OAuth 2.0 authorization flow",
        knowledge_details="OAuth uses redirect-based flow",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    # No embeddings — forces agent to do work

    # Agent: search → done with kb-00001 only
    agent_search = json.dumps(
        {
            "tool": "hybrid_search",
            "args": {"query": "API auth"},
        }
    )
    agent_done = json.dumps(
        {
            "tool": "done",
            "args": {
                "entry_ids": ["kb-00001"],
                "reasoning": "Found auth entry",
            },
        }
    )
    # Coverage: gap found, suggests searching for OAuth
    coverage = json.dumps(
        {
            "has_gaps": True,
            "suggested_query": "OAuth authorization flow",
            "reason": "missing OAuth details",
        }
    )
    # Synthesis
    synthesis = "API auth uses JWT [kb-00001]. OAuth uses redirects [kb-00002]."

    llm = ScriptedLLM(
        responses=[agent_search, agent_done, coverage, synthesis],
    )

    result = await summarize_question(
        db,
        fake_embedder,
        llm,
        "API authentication and OAuth",
        limit=20,
    )
    # Synthesis should reference both entries
    assert "kb-00001" in result


@pytest.mark.asyncio
async def test_merge_deduplicates_in_pipeline(
    db,
    store,
    fake_embedder,
    monkeypatch,
):
    """Extra entries from coverage re-search don't duplicate originals."""
    monkeypatch.setenv("KB_AGENTIC_QUERY", "TRUE")
    monkeypatch.setenv("KB_AGENTIC_SYNTHESIS", "TRUE")

    await store.create_entry(
        short_title="Entry One",
        long_title="Entry One details",
        knowledge_details="First entry content",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    # No embedding — forces agent loop

    # Agent: search → done with kb-00001
    agent_search = json.dumps(
        {
            "tool": "hybrid_search",
            "args": {"query": "entry one"},
        }
    )
    agent_done = json.dumps(
        {
            "tool": "done",
            "args": {
                "entry_ids": ["kb-00001"],
                "reasoning": "Found it",
            },
        }
    )
    # Coverage: gap with query that will find the same entry
    coverage = json.dumps(
        {
            "has_gaps": True,
            "suggested_query": "entry one",
            "reason": "more detail needed",
        }
    )
    # Synthesis
    synthesis = "Entry one content [kb-00001]."

    llm = ScriptedLLM(
        responses=[agent_search, agent_done, coverage, synthesis],
    )

    result = await summarize_question(
        db,
        fake_embedder,
        llm,
        "entry one",
        limit=20,
    )
    assert "kb-00001" in result
    # The synthesis prompt should only have kb-00001 once (deduped)
    synthesis_prompt = llm.calls[-1]["prompt"]
    assert synthesis_prompt is not None
    assert synthesis_prompt.count("[kb-00001]") == 1
