"""Server-Sent Events helpers: formatting and event-to-status mapping.

Provides two utilities consumed by SSE stream endpoints and the sibling
ingest-stream item:

- ``sse_event`` — format a single SSE message string.
- ``event_to_status`` — map a kb-core progress event to a human-readable
  status string (or ``None`` for unmapped event types).
"""

import json
from typing import Any


def sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format a server-sent event string.

    Args:
        event_type: The SSE ``event:`` field value.
        data: Data payload — serialised as compact JSON (no spaces).

    Returns:
        A properly-formatted SSE string ending with two newlines, ready to
        be yielded directly from a ``StreamingResponse`` generator.
    """
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event_type}\ndata: {payload}\n\n"


def event_to_status(event: dict[str, Any]) -> str | None:
    """Map a kb-core progress event to a human-readable status string.

    Covers every event type emitted by kb-core's ask/summarize/ingest
    pipelines.  The ingest_* branches are included here so the sibling
    ingest-stream item can import this module without modification.

    Args:
        event: Event dict from the kb-core ``event_callback``.  Must have at
            least a ``type`` key; other keys are type-specific.

    Returns:
        A status string for display, or ``None`` for unmapped event types.
    """
    event_type = event.get("type", "")

    if event_type == "agent_started":
        return "Searching knowledge base..."

    if event_type == "tool_call":
        tool = event.get("tool", "")
        args: dict[str, Any] = event.get("args", {})
        if tool == "graph_neighbors":
            return f"Exploring neighbors of {args.get('node_id', '')}..."
        if tool == "hybrid_search":
            return f"Searching: {args.get('query', '')}..."
        if tool == "decision_chain":
            return f"Following decision chain from {args.get('entry_id', '')}..."
        if tool == "scope_entries":
            return f"Listing entries in {args.get('scope', '')}..."
        if tool == "list_graph_nodes":
            return "Browsing graph vocabulary..."
        return f"Running {tool}..."

    if event_type == "thinking":
        return f"Thinking (turn {event.get('turn', '?')})..."

    if event_type == "synthesis_started":
        return f"Synthesizing answer from {event.get('entry_count', 0)} entries..."

    if event_type == "fast_path":
        return "Found strong matches..."

    if event_type == "ingest_summarizing":
        return f"Summarizing {event.get('source', '')}..."

    if event_type == "ingest_start":
        total: int = event.get("total_chunks", 1)
        return f"Extracting entries ({total} chunk{'s' if total != 1 else ''})..."

    if event_type == "ingest_chunk_start":
        ci: int = event.get("chunk_index", 0)
        total_c: int = event.get("total_chunks", 1)
        return f"Extracting chunk {ci + 1}/{total_c}..."

    if event_type == "ingest_chunk_done":
        ci2: int = event.get("chunk_index", 0)
        total_d: int = event.get("total_chunks", 1)
        n: int = event.get("entries_extracted", 0)
        return f"Chunk {ci2 + 1}/{total_d} done ({n} entries)"

    if event_type == "ingest_done":
        nd: int = event.get("entry_count", 0)
        return f"Done — {nd} entries created"

    if event_type == "ingest_error":
        return str(event.get("error", "Ingestion error"))

    return None
