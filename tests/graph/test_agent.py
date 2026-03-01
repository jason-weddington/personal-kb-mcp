"""Tests for the agentic query loop."""

import json

import pytest
import pytest_asyncio

from personal_kb.db.connection import create_connection
from personal_kb.graph.agent import _parse_response, agentic_query
from personal_kb.graph.builder import GraphBuilder
from personal_kb.models.entry import EntryType
from personal_kb.store.knowledge_store import KnowledgeStore
from tests.conftest import FakeEmbedder, ScriptedLLM


def _tool_call(tool: str, **kwargs) -> str:
    return json.dumps({"tool": tool, "args": kwargs})


def _done(entry_ids: list[str], reasoning: str = "Found results") -> str:
    return json.dumps({"tool": "done", "args": {"entry_ids": entry_ids, "reasoning": reasoning}})


@pytest_asyncio.fixture
async def agent_kb():
    """Lightweight KB with a few entries for agent unit tests.

    Yields (db, embedder, entry_ids) where entry_ids maps short names to IDs.
    """
    db = await create_connection(":memory:")
    embedder = FakeEmbedder(db)
    store = KnowledgeStore(db)
    builder = GraphBuilder(db)

    ids: dict[str, str] = {}

    # Create a handful of entries across different types
    for title, etype, project, tags in [
        ("aiosqlite-async", EntryType.FACTUAL_REFERENCE, "personal-kb", ["sqlite", "python"]),
        ("fts5-triggers", EntryType.FACTUAL_REFERENCE, "personal-kb", ["sqlite", "search"]),
        ("fastapi-decision", EntryType.DECISION, "web-service", ["python", "api"]),
        ("cors-config", EntryType.PATTERN_CONVENTION, "web-service", ["api", "security"]),
        ("race-condition", EntryType.LESSON_LEARNED, None, ["asyncio", "debugging"]),
    ]:
        entry = await store.create_entry(
            short_title=title,
            long_title=f"Long title for {title}",
            knowledge_details=f"Details about {title}",
            entry_type=etype,
            project_ref=project,
            tags=list(tags),
        )
        ids[title] = entry.id
        await builder.build_for_entry(entry)

        # Store embedding
        embedding = await embedder.embed(entry.embedding_text)
        if embedding:
            await embedder.store_embedding(entry.id, embedding)
            await store.mark_embedding(entry.id)

    yield db, embedder, ids

    await db.close()


class TestParseResponse:
    def test_parse_tool_call(self):
        raw = '{"tool": "hybrid_search", "args": {"query": "test"}}'
        result = _parse_response(raw)
        assert result is not None
        assert result.tool == "hybrid_search"

    def test_parse_done(self):
        raw = _done(["kb-00001"])
        result = _parse_response(raw)
        assert result is not None
        assert result.entry_ids == ["kb-00001"]

    def test_parse_markdown_fences(self):
        inner = '{"tool": "done", "args": {"entry_ids": ["kb-00001"], "reasoning": "x"}}'
        raw = f"```json\n{inner}\n```"
        result = _parse_response(raw)
        assert result is not None
        assert result.entry_ids == ["kb-00001"]

    def test_parse_garbage_returns_none(self):
        assert _parse_response("not json at all") is None

    def test_parse_empty_object_returns_none(self):
        assert _parse_response("{}") is None


@pytest.mark.asyncio
async def test_agent_fast_path_skips_llm(agent_kb):
    """Strong hybrid_search results → 0 LLM calls."""
    db, embedder, _ids = agent_kb

    llm = ScriptedLLM([])
    # Use a query that matches well via FTS — "aiosqlite async" should hit
    result = await agentic_query(db, embedder, llm, "aiosqlite async sqlite")

    assert result.turns_used == 0
    assert result.reasoning == "Fast-path: strong hybrid search results"
    assert len(result.entries) > 0
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_agent_single_turn_done(agent_kb):
    """Agent calls done immediately in 1 turn."""
    db, embedder, ids = agent_kb
    target_id = ids["aiosqlite-async"]

    llm = ScriptedLLM(
        [
            _done([target_id], "Found the async sqlite entry"),
        ]
    )

    result = await agentic_query(
        db,
        embedder,
        llm,
        "obscure query xyz123 no matches",
        max_tool_calls=4,
    )

    assert result.turns_used == 1
    assert any(eid == target_id for eid, _ in result.entries)
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_agent_refines_on_sparse(agent_kb):
    """Agent searches, gets sparse results, refines, then done."""
    db, embedder, ids = agent_kb
    target_id = ids["aiosqlite-async"]

    llm = ScriptedLLM(
        [
            _tool_call("hybrid_search", query="aiosqlite connection patterns"),
            _done([target_id], "Found after refinement"),
        ]
    )

    result = await agentic_query(
        db,
        embedder,
        llm,
        "obscure query xyz123",
        max_tool_calls=4,
    )

    assert result.turns_used == 2
    assert any(eid == target_id for eid, _ in result.entries)


@pytest.mark.asyncio
async def test_agent_terminates_at_max_calls(agent_kb):
    """Agent stops after max_tool_calls even if LLM keeps trying."""
    db, embedder, _ids = agent_kb

    llm = ScriptedLLM(
        responses=[],
        fallback=_tool_call("hybrid_search", query="test"),
    )

    result = await agentic_query(
        db,
        embedder,
        llm,
        "obscure query xyz123",
        max_tool_calls=3,
    )

    assert result.turns_used == 3
    assert result.reasoning == "Exhausted tool call budget"
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_agent_handles_llm_failure(agent_kb):
    """LLM returns None → graceful fallback."""
    db, embedder, _ids = agent_kb

    llm = ScriptedLLM([None])

    result = await agentic_query(
        db,
        embedder,
        llm,
        "obscure query xyz123",
        max_tool_calls=4,
    )

    assert result.turns_used == 0


@pytest.mark.asyncio
async def test_agent_handles_malformed_json(agent_kb):
    """Bad JSON → error injected, loop continues to next turn."""
    db, embedder, ids = agent_kb
    target_id = ids["aiosqlite-async"]

    llm = ScriptedLLM(
        [
            "This is not valid JSON",
            _done([target_id], "Found it on second try"),
        ]
    )

    result = await agentic_query(
        db,
        embedder,
        llm,
        "obscure query xyz123",
        max_tool_calls=4,
    )

    assert result.turns_used == 2
    assert any(eid == target_id for eid, _ in result.entries)


@pytest.mark.asyncio
async def test_agent_graph_exploration(agent_kb):
    """Agent can explore graph neighbors."""
    db, embedder, ids = agent_kb
    target_id = ids["aiosqlite-async"]

    llm = ScriptedLLM(
        [
            _tool_call("graph_neighbors", node_id=target_id, limit=5),
            _done([target_id], "Found via graph exploration"),
        ]
    )

    result = await agentic_query(
        db,
        embedder,
        llm,
        "obscure query xyz123",
        max_tool_calls=4,
    )

    assert result.turns_used == 2
    assert any(eid == target_id for eid, _ in result.entries)


@pytest.mark.asyncio
async def test_agent_decision_chain(agent_kb):
    """Agent can follow decision chains."""
    db, embedder, ids = agent_kb
    target_id = ids["fastapi-decision"]

    llm = ScriptedLLM(
        [
            _tool_call("decision_chain", entry_id=target_id),
            _done([target_id], "Found decision chain"),
        ]
    )

    result = await agentic_query(
        db,
        embedder,
        llm,
        "obscure query xyz123",
        max_tool_calls=4,
    )

    assert result.turns_used == 2
    assert any(eid == target_id for eid, _ in result.entries)


@pytest.mark.asyncio
async def test_agent_list_graph_nodes(agent_kb):
    """Agent can browse graph vocabulary."""
    db, embedder, ids = agent_kb
    target_id = ids["aiosqlite-async"]

    llm = ScriptedLLM(
        [
            _tool_call("list_graph_nodes"),
            _done([target_id], "Found after checking vocabulary"),
        ]
    )

    result = await agentic_query(
        db,
        embedder,
        llm,
        "obscure query xyz123",
        max_tool_calls=4,
    )

    assert result.turns_used == 2
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_agent_scope_entries(agent_kb):
    """Agent can list entries for a scope."""
    db, embedder, ids = agent_kb
    target_id = ids["aiosqlite-async"]

    llm = ScriptedLLM(
        [
            _tool_call("scope_entries", scope="project:personal-kb", limit=10),
            _done([target_id], "Found via scope"),
        ]
    )

    result = await agentic_query(
        db,
        embedder,
        llm,
        "obscure query xyz123",
        max_tool_calls=4,
    )

    assert result.turns_used == 2


@pytest.mark.asyncio
async def test_agent_unknown_tool(agent_kb):
    """Unknown tool name → error message, loop continues."""
    db, embedder, ids = agent_kb
    target_id = ids["aiosqlite-async"]

    llm = ScriptedLLM(
        [
            _tool_call("nonexistent_tool", foo="bar"),
            _done([target_id], "Found after error"),
        ]
    )

    result = await agentic_query(
        db,
        embedder,
        llm,
        "obscure query xyz123",
        max_tool_calls=4,
    )

    assert result.turns_used == 2


@pytest.mark.asyncio
async def test_toggle_off_uses_single_shot(monkeypatch):
    """KB_AGENTIC_QUERY=FALSE → config returns False."""
    monkeypatch.setenv("KB_AGENTIC_QUERY", "FALSE")

    from personal_kb.config import is_agentic_query

    assert not is_agentic_query()
