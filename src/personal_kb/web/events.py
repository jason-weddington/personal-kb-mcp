"""SSE event formatting and status translation."""

import json
from typing import Any


def sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format a single SSE event as a string with event: and data: lines."""
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event_type}\ndata: {payload}\n\n"


def event_to_status(event: dict[str, Any]) -> str | None:
    """Translate an agent event to a human-readable status message.

    Returns None if the event doesn't map to a user-facing status.
    """
    etype = event.get("type")

    if etype == "agent_started":
        return "Searching knowledge base..."

    if etype == "tool_call":
        tool = event.get("tool", "")
        args = event.get("args", {})
        if tool == "graph_neighbors":
            node_id = args.get("node_id", "")
            return f"Exploring neighbors of {node_id}..."
        if tool == "hybrid_search":
            query = args.get("query", "")
            return f"Searching: {query}..."
        if tool == "decision_chain":
            entry_id = args.get("entry_id", "")
            return f"Following decision chain from {entry_id}..."
        if tool == "scope_entries":
            scope = args.get("scope", "")
            return f"Listing entries in {scope}..."
        if tool == "list_graph_nodes":
            return "Browsing graph vocabulary..."
        return f"Running {tool}..."

    if etype == "thinking":
        turn = event.get("turn", "?")
        return f"Thinking (turn {turn})..."

    if etype == "synthesis_started":
        count = event.get("entry_count", 0)
        return f"Synthesizing answer from {count} entries..."

    if etype == "fast_path":
        return "Found strong matches..."

    return None
