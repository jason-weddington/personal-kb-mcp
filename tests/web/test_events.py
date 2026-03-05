"""Tests for SSE event formatting and status translation."""

import json

from personal_kb.web.events import event_to_status, sse_event


class TestSseEvent:
    def test_format_basic(self):
        result = sse_event("test", {"key": "value"})
        assert result.startswith("event: test\n")
        assert "data: " in result
        assert result.endswith("\n\n")
        data = json.loads(result.split("data: ")[1].strip())
        assert data == {"key": "value"}

    def test_format_complex_data(self):
        data = {"entry_ids": ["kb-00001", "kb-00002"], "score": 0.95}
        result = sse_event("entries", data)
        parsed = json.loads(result.split("data: ")[1].strip())
        assert parsed["entry_ids"] == ["kb-00001", "kb-00002"]
        assert parsed["score"] == 0.95


class TestEventToStatus:
    def test_agent_started(self):
        assert event_to_status({"type": "agent_started"}) == "Searching knowledge base..."

    def test_tool_call_graph_neighbors(self):
        event = {"type": "tool_call", "tool": "graph_neighbors", "args": {"node_id": "kb-00001"}}
        assert event_to_status(event) == "Exploring neighbors of kb-00001..."

    def test_tool_call_hybrid_search(self):
        event = {"type": "tool_call", "tool": "hybrid_search", "args": {"query": "sqlite async"}}
        assert event_to_status(event) == "Searching: sqlite async..."

    def test_thinking(self):
        assert event_to_status({"type": "thinking", "turn": 2}) == "Thinking (turn 2)..."

    def test_synthesis_started(self):
        event = {"type": "synthesis_started", "entry_count": 5}
        assert event_to_status(event) == "Synthesizing answer from 5 entries..."

    def test_fast_path(self):
        assert event_to_status({"type": "fast_path"}) == "Found strong matches..."

    def test_unknown_returns_none(self):
        assert event_to_status({"type": "stream_end"}) is None

    def test_tool_call_unknown_tool(self):
        event = {"type": "tool_call", "tool": "custom_tool", "args": {}}
        assert event_to_status(event) == "Running custom_tool..."
