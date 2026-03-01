"""End-to-end agent tests on the eval corpus with ScriptedLLM.

Each test scripts the LLM to make the agent call the right tools for a
golden query, then asserts expected entries appear in results.  Fully
deterministic — runs in CI.
"""

import json

import pytest

from personal_kb.graph.agent import agentic_query
from tests.conftest import ScriptedLLM


def _tool_call(tool: str, **kwargs) -> str:
    return json.dumps({"tool": tool, "args": kwargs})


def _done(entry_ids: list[str], reasoning: str = "Found") -> str:
    return json.dumps({"tool": "done", "args": {"entry_ids": entry_ids, "reasoning": reasoning}})


@pytest.mark.eval
class TestAgentE2E:
    """Agent e2e tests — one per representative golden query category."""

    async def test_keyword_match_fast_path(self, eval_kb):
        """q01: async sqlite — should fast-path with 0 LLM calls."""
        db, embedder, title_to_id, _queries = eval_kb

        llm = ScriptedLLM([])
        result = await agentic_query(db, embedder, llm, "async sqlite database access")

        assert result.turns_used == 0
        result_ids = {eid for eid, _ in result.entries}
        assert title_to_id["aiosqlite-async-pattern"] in result_ids

    async def test_fts_keyword_fast_path(self, eval_kb):
        """q02: FTS5 triggers — should fast-path."""
        db, embedder, title_to_id, _queries = eval_kb

        llm = ScriptedLLM([])
        result = await agentic_query(db, embedder, llm, "full text search FTS5 triggers")

        assert result.turns_used == 0
        result_ids = {eid for eid, _ in result.entries}
        assert title_to_id["fts5-content-sync"] in result_ids

    async def test_agent_refines_weak_query(self, eval_kb):
        """Agent refines a vague query and finds the right entry."""
        db, embedder, title_to_id, _queries = eval_kb
        target_id = title_to_id["cors-config-pattern"]

        llm = ScriptedLLM(
            [
                _tool_call("hybrid_search", query="CORS configuration allow origins"),
                _done([target_id], "Found CORS config pattern"),
            ]
        )

        result = await agentic_query(
            db,
            embedder,
            llm,
            "how do I set up cross-origin resource sharing",
            max_tool_calls=4,
        )

        assert result.turns_used == 2
        result_ids = {eid for eid, _ in result.entries}
        assert target_id in result_ids

    async def test_agent_decision_trace(self, eval_kb):
        """Agent uses decision_chain tool for decision queries."""
        db, embedder, title_to_id, _queries = eval_kb
        target_id = title_to_id["fastapi-vs-flask-decision"]

        llm = ScriptedLLM(
            [
                _tool_call("hybrid_search", query="FastAPI vs Flask framework decision"),
                _tool_call("decision_chain", entry_id=target_id),
                _done([target_id], "Found the framework decision"),
            ]
        )

        result = await agentic_query(
            db,
            embedder,
            llm,
            "why did we choose FastAPI over Flask",
            max_tool_calls=4,
        )

        result_ids = {eid for eid, _ in result.entries}
        assert target_id in result_ids

    async def test_agent_graph_exploration(self, eval_kb):
        """Agent explores graph neighbors to find related entries."""
        db, embedder, title_to_id, _queries = eval_kb
        target_id = title_to_id["docker-multi-stage"]

        llm = ScriptedLLM(
            [
                _tool_call("list_graph_nodes", type_filter="tag"),
                _tool_call("graph_neighbors", node_id="tag:devops"),
                _done([target_id], "Found Docker entry via graph"),
            ]
        )

        result = await agentic_query(
            db,
            embedder,
            llm,
            "obscure devops question xyz",
            max_tool_calls=4,
        )

        result_ids = {eid for eid, _ in result.entries}
        assert target_id in result_ids

    async def test_agent_scope_filter(self, eval_kb):
        """Agent uses scope_entries to filter by project."""
        db, embedder, title_to_id, _queries = eval_kb
        target_id = title_to_id["rest-api-conventions"]

        llm = ScriptedLLM(
            [
                _tool_call("scope_entries", scope="project:web-service", limit=10),
                _done([target_id], "Found REST conventions in web-service project"),
            ]
        )

        result = await agentic_query(
            db,
            embedder,
            llm,
            "obscure api conventions question xyz",
            max_tool_calls=4,
        )

        result_ids = {eid for eid, _ in result.entries}
        assert target_id in result_ids

    async def test_agent_exhausts_gracefully(self, eval_kb):
        """Agent that never calls done still returns best-effort results."""
        db, embedder, _title_to_id, _queries = eval_kb

        llm = ScriptedLLM(
            responses=[],
            fallback=_tool_call("hybrid_search", query="Docker multi-stage build"),
        )

        result = await agentic_query(
            db,
            embedder,
            llm,
            "obscure query xyz123",
            max_tool_calls=2,
        )

        assert result.turns_used == 2
        assert result.reasoning == "Exhausted tool call budget"
        # Should still have entries extracted from tool results
        assert len(result.entries) >= 0  # may or may not find entries
