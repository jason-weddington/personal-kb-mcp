"""Tests for the query planner."""

import json

import pytest

from personal_kb.graph.planner import QueryPlanner
from personal_kb.models.entry import EntryType
from tests.conftest import FakeLLM


def _make_plan_response(
    strategy: str = "auto",
    scope: str | None = None,
    target: str | None = None,
    search_query: str | None = None,
    reasoning: str | None = None,
) -> str:
    return json.dumps(
        {
            "strategy": strategy,
            "scope": scope,
            "target": target,
            "search_query": search_query,
            "reasoning": reasoning,
        }
    )


@pytest.mark.asyncio
async def test_plan_auto_strategy(db):
    """Planner should return auto strategy for general questions."""
    llm = FakeLLM(
        response=_make_plan_response(
            strategy="auto",
            search_query="sqlite performance",
            reasoning="General question about sqlite",
        )
    )
    planner = QueryPlanner(db, llm)
    plan = await planner.plan("How do I improve sqlite performance?")
    assert plan is not None
    assert plan.strategy == "auto"
    assert plan.search_query == "sqlite performance"


@pytest.mark.asyncio
async def test_plan_related_strategy(db):
    """Planner should select related strategy with scope."""
    llm = FakeLLM(
        response=_make_plan_response(
            strategy="related",
            scope="tag:python",
            reasoning="Looking for related entries",
        )
    )
    planner = QueryPlanner(db, llm)
    plan = await planner.plan("What relates to python?")
    assert plan is not None
    assert plan.strategy == "related"
    assert plan.scope == "tag:python"


@pytest.mark.asyncio
async def test_plan_decision_trace_strategy(db):
    """Planner should select decision_trace for decision questions."""
    llm = FakeLLM(
        response=_make_plan_response(
            strategy="decision_trace",
            search_query="database choice",
            reasoning="Asking about a decision",
        )
    )
    planner = QueryPlanner(db, llm)
    plan = await planner.plan("Why did we choose sqlite?")
    assert plan is not None
    assert plan.strategy == "decision_trace"


@pytest.mark.asyncio
async def test_plan_connection_strategy(db):
    """Planner should select connection with scope and target."""
    llm = FakeLLM(
        response=_make_plan_response(
            strategy="connection",
            scope="tag:python",
            target="tool:aiosqlite",
            reasoning="Finding connection between two concepts",
        )
    )
    planner = QueryPlanner(db, llm)
    plan = await planner.plan("How are python and aiosqlite connected?")
    assert plan is not None
    assert plan.strategy == "connection"
    assert plan.scope == "tag:python"
    assert plan.target == "tool:aiosqlite"


@pytest.mark.asyncio
async def test_plan_timeline_strategy(db):
    """Planner should select timeline with scope."""
    llm = FakeLLM(
        response=_make_plan_response(
            strategy="timeline",
            scope="project:personal-kb",
            reasoning="Asking about project history",
        )
    )
    planner = QueryPlanner(db, llm)
    plan = await planner.plan("What happened in the personal-kb project?")
    assert plan is not None
    assert plan.strategy == "timeline"
    assert plan.scope == "project:personal-kb"


@pytest.mark.asyncio
async def test_plan_returns_none_when_llm_unavailable(db):
    """Planner returns None when LLM is unavailable."""
    llm = FakeLLM(available=False)
    planner = QueryPlanner(db, llm)
    plan = await planner.plan("test question")
    assert plan is None


@pytest.mark.asyncio
async def test_plan_returns_none_on_malformed_json(db):
    """Planner returns None on malformed LLM response."""
    llm = FakeLLM(response="This is not JSON at all")
    planner = QueryPlanner(db, llm)
    plan = await planner.plan("test question")
    assert plan is None


@pytest.mark.asyncio
async def test_plan_handles_markdown_fences(db):
    """Planner should strip markdown fences from response."""
    inner = _make_plan_response(strategy="related", scope="tag:python")
    llm = FakeLLM(response=f"```json\n{inner}\n```")
    planner = QueryPlanner(db, llm)
    plan = await planner.plan("test")
    assert plan is not None
    assert plan.strategy == "related"
    assert plan.scope == "tag:python"


@pytest.mark.asyncio
async def test_plan_invalid_strategy_falls_back_to_auto(db):
    """Invalid strategy in LLM response falls back to auto."""
    llm = FakeLLM(
        response=json.dumps(
            {
                "strategy": "invalid_strategy",
                "search_query": "test",
            }
        )
    )
    planner = QueryPlanner(db, llm)
    plan = await planner.plan("test")
    assert plan is not None
    assert plan.strategy == "auto"


@pytest.mark.asyncio
async def test_plan_context_includes_vocabulary(db, store, graph_builder):
    """Planner context should include graph vocabulary."""
    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["python", "sqlite"],
    )
    await graph_builder.build_for_entry(entry)

    llm = FakeLLM(response=_make_plan_response())
    planner = QueryPlanner(db, llm)
    await planner.plan("test question")

    assert llm.last_prompt is not None
    assert "python" in llm.last_prompt
    assert "sqlite" in llm.last_prompt
    assert "test question" in llm.last_prompt


@pytest.mark.asyncio
async def test_plan_context_includes_graph_stats(db, store):
    """Planner context should include graph statistics."""
    await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    llm = FakeLLM(response=_make_plan_response())
    planner = QueryPlanner(db, llm)
    await planner.plan("test question")

    assert llm.last_prompt is not None
    assert "Graph stats" in llm.last_prompt
    assert "Active entries" in llm.last_prompt


@pytest.mark.asyncio
async def test_plan_uses_system_prompt(db):
    """Planner should pass system prompt to LLM."""
    llm = FakeLLM(response=_make_plan_response())
    planner = QueryPlanner(db, llm)
    await planner.plan("test question")

    assert llm.last_system is not None
    assert "query planner" in llm.last_system.lower()
    assert "strategy" in llm.last_system.lower()
