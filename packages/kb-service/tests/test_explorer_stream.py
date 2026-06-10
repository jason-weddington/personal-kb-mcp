"""Hermetic tests for the P3 SSE query stream, classifier, and sse helpers."""

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kb_service.auth import get_current_user_sse
from kb_service.classifier import classify_query
from kb_service.main import app
from kb_service.sse import event_to_status, sse_event
from tests.conftest import FakeKnowledgeBase, fake_user

# ---------------------------------------------------------------------------
# SSE parsing helper
# ---------------------------------------------------------------------------


def _parse_sse(
    client: TestClient, method: str, url: str, **kwargs: Any
) -> list[dict[str, Any]]:
    """Stream a request and collect parsed SSE events.

    Iterates response lines, grouping them into {type, data} dicts.
    """
    events: list[dict[str, Any]] = []
    current_type: str | None = None
    current_data: str | None = None

    with client.stream(method, url, **kwargs) as resp:
        for line in resp.iter_lines():
            if line.startswith("event: "):
                current_type = line[len("event: ") :]
            elif line.startswith("data: "):
                current_data = line[len("data: ") :]
            elif not line and current_type is not None:
                events.append(
                    {
                        "type": current_type,
                        "data": json.loads(current_data or "{}"),
                    }
                )
                current_type = None
                current_data = None

    # Flush any trailing event (no trailing blank line)
    if current_type is not None:
        events.append(
            {
                "type": current_type,
                "data": json.loads(current_data or "{}"),
            }
        )
    return events


# ---------------------------------------------------------------------------
# POST /api/kb/query/stream — auth gate (no dependency override)
# ---------------------------------------------------------------------------


def test_query_stream_no_token_returns_401(client: TestClient) -> None:
    resp = client.post("/api/kb/query/stream", json={"question": "hello"})
    assert resp.status_code == 401


def test_query_stream_garbage_token_returns_401(client: TestClient) -> None:
    resp = client.post("/api/kb/query/stream?token=garbage", json={"question": "hello"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/kb/query/stream — 422 on missing required field
# ---------------------------------------------------------------------------


def test_query_stream_empty_body_returns_422(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user_sse] = fake_user
    resp = client.post("/api/kb/query/stream", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/kb/query/stream — explore happy path (query_llm is None)
# ---------------------------------------------------------------------------


def test_query_stream_explore_happy_path(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user_sse] = fake_user
    # query_llm is None by default → mode = "explore"
    events = _parse_sse(
        client,
        "POST",
        "/api/kb/query/stream",
        json={"question": "what nodes link to sqlite?"},
    )

    types = [e["type"] for e in events]
    assert types[0] == "classified"
    assert events[0]["data"]["mode"] == "explore"

    entries_events = [e for e in events if e["type"] == "entries"]
    assert len(entries_events) == 1
    entries_data = entries_events[0]["data"]
    assert entries_data["turns_used"] == 3
    first = entries_data["entries"][0]
    assert first["id"] == "kb-00001"
    assert first["context"] == "fake ask context"

    assert events[-1]["type"] == "stream_end"


# ---------------------------------------------------------------------------
# POST /api/kb/query/stream — summarize path with event callbacks
# ---------------------------------------------------------------------------


def test_query_stream_summarize_path(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user_sse] = fake_user

    # Stub LLM that returns "summarize" → triggers summarize path
    class _LLMStub:
        async def generate(self, prompt: str, *, system: str | None = None) -> str:
            return "summarize"

    fake_kb.query_llm = _LLMStub()

    # Replace fake_kb.summarize with a coroutine that emits a fast_path event
    async def _custom_summarize(
        q: str, *, event_callback: Any = None, **_kw: Any
    ) -> str:
        if event_callback is not None:
            await event_callback(
                {"type": "fast_path", "entry_ids": ["kb-00002"], "top_score": 0.9}
            )
        return "fake synthesized answer"

    fake_kb.summarize = _custom_summarize  # type: ignore[method-assign]

    events = _parse_sse(
        client,
        "POST",
        "/api/kb/query/stream",
        json={"question": "why did we choose FastAPI?"},
    )

    types = [e["type"] for e in events]

    # classified first
    assert types[0] == "classified"
    assert events[0]["data"]["mode"] == "summarize"

    # fast_path event followed immediately by status
    fp_idx = types.index("fast_path")
    assert types[fp_idx + 1] == "status"
    assert events[fp_idx + 1]["data"]["message"] == "Found strong matches..."

    # synthesis_result with flattened entry_ids (extend not append)
    sr = next(e for e in events if e["type"] == "synthesis_result")
    assert sr["data"]["answer"] == "fake synthesized answer"
    assert sr["data"]["question"] == "why did we choose FastAPI?"
    assert sr["data"]["entry_ids"] == ["kb-00002"]  # flat list

    assert types[-1] == "stream_end"


# ---------------------------------------------------------------------------
# POST /api/kb/query/stream — error path
# ---------------------------------------------------------------------------


def test_query_stream_error_path(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    app.dependency_overrides[get_current_user_sse] = fake_user
    # query_llm is None → explore mode
    assert fake_kb.query_llm is None

    async def _boom(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("boom")

    fake_kb.ask = _boom  # type: ignore[method-assign]

    events = _parse_sse(
        client,
        "POST",
        "/api/kb/query/stream",
        json={"question": "trigger error"},
    )

    types = [e["type"] for e in events]
    assert "error" in types
    err_evt = next(e for e in events if e["type"] == "error")
    assert err_evt["data"]["message"] == "RuntimeError: boom"
    assert types[-1] == "stream_end"


# ---------------------------------------------------------------------------
# classify_query unit tests
# ---------------------------------------------------------------------------


class _GenStub:
    """Stub LLMProvider.generate that returns a fixed string or None."""

    def __init__(self, response: str | None, raise_exc: bool = False) -> None:
        self._response = response
        self._raise = raise_exc

    async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
        if self._raise:
            raise ValueError("llm error")
        return self._response


async def test_classify_query_explore() -> None:
    assert await classify_query(_GenStub("explore"), "q") == "explore"  # type: ignore[arg-type]


async def test_classify_query_summarize() -> None:
    assert await classify_query(_GenStub("summarize"), "q") == "summarize"  # type: ignore[arg-type]


async def test_classify_query_substring_summarize() -> None:
    # substring fallback: strip/lower of "The answer is summarize." contains "summarize"
    result = await classify_query(_GenStub("The answer is summarize."), "q")  # type: ignore[arg-type]
    assert result == "summarize"


async def test_classify_query_none_response_falls_back() -> None:
    assert await classify_query(_GenStub(None), "q") == "explore"  # type: ignore[arg-type]


async def test_classify_query_exception_falls_back() -> None:
    assert await classify_query(_GenStub(None, raise_exc=True), "q") == "explore"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# sse_event format test
# ---------------------------------------------------------------------------


def test_sse_event_format() -> None:
    result = sse_event("x", {"a": 1})
    assert result == 'event: x\ndata: {"a":1}\n\n'


# ---------------------------------------------------------------------------
# event_to_status — table-driven parametrised test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event,expected",
    [
        # agent_started
        ({"type": "agent_started"}, "Searching knowledge base..."),
        # tool_call variants
        (
            {
                "type": "tool_call",
                "tool": "graph_neighbors",
                "args": {"node_id": "kb-00001"},
            },
            "Exploring neighbors of kb-00001...",
        ),
        (
            {"type": "tool_call", "tool": "hybrid_search", "args": {"query": "python"}},
            "Searching: python...",
        ),
        (
            {
                "type": "tool_call",
                "tool": "decision_chain",
                "args": {"entry_id": "kb-00002"},
            },
            "Following decision chain from kb-00002...",
        ),
        (
            {
                "type": "tool_call",
                "tool": "scope_entries",
                "args": {"scope": "project:X"},
            },
            "Listing entries in project:X...",
        ),
        (
            {"type": "tool_call", "tool": "list_graph_nodes", "args": {}},
            "Browsing graph vocabulary...",
        ),
        (
            {"type": "tool_call", "tool": "unknown_tool", "args": {}},
            "Running unknown_tool...",
        ),
        # thinking
        ({"type": "thinking", "turn": 2}, "Thinking (turn 2)..."),
        # synthesis_started
        (
            {"type": "synthesis_started", "entry_count": 5},
            "Synthesizing answer from 5 entries...",
        ),
        # fast_path
        ({"type": "fast_path"}, "Found strong matches..."),
        # ingest_summarizing
        (
            {"type": "ingest_summarizing", "source": "doc.md"},
            "Summarizing doc.md...",
        ),
        # ingest_start singular
        (
            {"type": "ingest_start", "total_chunks": 1},
            "Extracting entries (1 chunk)...",
        ),
        # ingest_start plural
        (
            {"type": "ingest_start", "total_chunks": 3},
            "Extracting entries (3 chunks)...",
        ),
        # ingest_chunk_start
        (
            {"type": "ingest_chunk_start", "chunk_index": 0, "total_chunks": 3},
            "Extracting chunk 1/3...",
        ),
        # ingest_chunk_done
        (
            {
                "type": "ingest_chunk_done",
                "chunk_index": 1,
                "total_chunks": 3,
                "entries_extracted": 4,
            },
            "Chunk 2/3 done (4 entries)",
        ),
        # ingest_done
        (
            {"type": "ingest_done", "entry_count": 7},
            "Done — 7 entries created",
        ),
        # ingest_error
        (
            {"type": "ingest_error", "error": "Something went wrong"},
            "Something went wrong",
        ),
        # unmapped → None
        ({"type": "totally_unknown"}, None),
    ],
)
def test_event_to_status(event: dict[str, Any], expected: str | None) -> None:
    assert event_to_status(event) == expected
