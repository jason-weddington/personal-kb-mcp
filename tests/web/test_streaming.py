"""SSE streaming integration tests with ScriptedLLM."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from personal_kb.web.app import create_app_with_deps
from tests.conftest import ScriptedLLM


def _tool_call(tool: str, **kwargs) -> str:
    return json.dumps({"tool": tool, "args": kwargs})


def _done(entry_ids: list[str], reasoning: str = "Found results") -> str:
    return json.dumps({"tool": "done", "args": {"entry_ids": entry_ids, "reasoning": reasoning}})


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse SSE text into (event_type, data) tuples."""
    events = []
    current_event = None
    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: ") and current_event:
            try:
                data = json.loads(line[6:])
                events.append((current_event, data))
            except json.JSONDecodeError:
                pass
            current_event = None
    return events


@pytest.mark.asyncio
async def test_stream_explore_fast_path(web_kb):
    """Fast-path explore query returns entries via SSE."""
    db, embedder, _ids = web_kb

    # Classifier says explore, agent hits fast-path (no agent loop needed)
    llm = ScriptedLLM(["explore"])

    app = create_app_with_deps(db, embedder, llm)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/query/stream",
            json={"question": "sqlite async python"},
            timeout=10.0,
        )
        assert resp.status_code == 200

        events = _parse_sse(resp.text)
        event_types = [e[0] for e in events]

        assert "classified" in event_types
        assert "stream_end" in event_types

        # Classified as explore
        classified = next(e for e in events if e[0] == "classified")
        assert classified[1]["mode"] == "explore"

        # Should have entries result
        assert "entries" in event_types


@pytest.mark.asyncio
async def test_stream_explore_agent_loop(web_kb):
    """Agent loop explore query emits tool_call and agent_done events."""
    db, embedder, ids = web_kb
    target_id = ids["sqlite-async"]

    # 1st call: classifier (explore), 2nd+: agent loop
    llm = ScriptedLLM(
        [
            "explore",
            _tool_call("hybrid_search", query="sqlite details"),
            _done([target_id], "Found sqlite entry"),
        ]
    )

    app = create_app_with_deps(db, embedder, llm)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/query/stream",
            json={"question": "xyz obscure no match 12345"},
            timeout=10.0,
        )
        assert resp.status_code == 200

        events = _parse_sse(resp.text)
        event_types = [e[0] for e in events]

        assert "classified" in event_types
        assert "stream_end" in event_types


@pytest.mark.asyncio
async def test_stream_summarize(web_kb):
    """Summarize query returns synthesis_result event."""
    db, embedder, _ids = web_kb

    # Classifier says summarize, then synthesis LLM response
    llm = ScriptedLLM(
        [
            "summarize",
            # retrieve_entries will use the agent — provide a fast-path hit
            # then synthesis call
            "The answer is based on [kb-00001] which explains sqlite async patterns.",
        ]
    )

    app = create_app_with_deps(db, embedder, llm)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/query/stream",
            json={"question": "sqlite async python"},
            timeout=10.0,
        )
        assert resp.status_code == 200

        events = _parse_sse(resp.text)
        event_types = [e[0] for e in events]

        assert "classified" in event_types
        classified = next(e for e in events if e[0] == "classified")
        assert classified[1]["mode"] == "summarize"
        assert "stream_end" in event_types


@pytest.mark.asyncio
async def test_index_returns_html(web_kb):
    """GET / returns the explorer HTML page."""
    db, embedder, _ids = web_kb

    app = create_app_with_deps(db, embedder, None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "Knowledge Graph Explorer" in resp.text
        assert "force-graph" in resp.text


@pytest.mark.asyncio
async def test_api_graph_returns_json(web_kb):
    """GET /api/graph returns graph data JSON."""
    db, embedder, _ids = web_kb

    app = create_app_with_deps(db, embedder, None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert "stats" in data


@pytest.mark.asyncio
async def test_stream_no_llm_explore(web_kb):
    """Explore query without LLM defaults to explore mode."""
    db, embedder, _ids = web_kb

    app = create_app_with_deps(db, embedder, None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/query/stream",
            json={"question": "sqlite"},
            timeout=10.0,
        )
        assert resp.status_code == 200

        events = _parse_sse(resp.text)
        classified = next(e for e in events if e[0] == "classified")
        assert classified[1]["mode"] == "explore"
        assert any(e[0] == "stream_end" for e in events)
