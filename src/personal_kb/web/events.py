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

    if etype == "ingest_summarizing":
        source = event.get("source", "")
        return f"Summarizing {source}..."

    if etype == "ingest_start":
        total = event.get("total_chunks", 1)
        return f"Extracting entries ({total} chunk{'s' if total != 1 else ''})..."

    if etype == "ingest_chunk_start":
        ci = event.get("chunk_index", 0)
        total = event.get("total_chunks", 1)
        return f"Extracting chunk {ci + 1}/{total}..."

    if etype == "ingest_chunk_done":
        ci = event.get("chunk_index", 0)
        total = event.get("total_chunks", 1)
        n = event.get("entries_extracted", 0)
        return f"Chunk {ci + 1}/{total} done ({n} entries)"

    if etype == "ingest_done":
        n = event.get("entry_count", 0)
        return f"Done — {n} entries created"

    if etype == "ingest_error":
        return str(event.get("error", "Ingestion error"))

    return None
