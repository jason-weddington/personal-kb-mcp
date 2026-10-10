"""kb_ask MCP tool (HTTP twin of ``personal_kb.tools.kb_ask``, auto only)."""

import logging
from typing import Annotated, Literal

from fastmcp import FastMCP
from kb_core.formatting import format_entry_full, format_result_list
from pydantic import Field

from kb_service.mcp_server import context
from kb_service.mcp_server.errors import BackendHttpError, map_error

logger = logging.getLogger(__name__)

Strategy = Literal["auto", "decision_trace", "timeline", "related", "connection"]

_ASK_DESCRIPTION = (
    "Answer questions by traversing the knowledge graph and combining with search.\n"
    "\n"
    "Best for discovery and exploration — when you need to find connections,\n"
    "trace history, or understand how knowledge relates.\n"
    "\n"
    "Strategies (prefer specific strategies over auto when intent is clear):\n"
    "- auto: Hybrid search + graph expansion. Good default for open-ended queries.\n"
    "- decision_trace: Follow supersedes chains to see how a decision evolved over\n"
    '  time. Use for "why did we switch from X to Y?" or "what was the original\n'
    '  rationale for Z?"\n'
    '- timeline: Chronological view of entries in a scope. Use for "what happened\n'
    '  in project X?" or "recent changes to tag:auth".\n'
    "- related: BFS from a starting node — finds everything connected to a concept.\n"
    '  Use for "what touches tag:python?" or "what depends on kb-00042?"\n'
    '- connection: Find paths between two nodes. Use for "how are X and Y related?"'
)


def register_kb_ask(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_ask tool with the MCP server."""

    @mcp.tool(name=f"{prefix}ask", description=_ASK_DESCRIPTION)
    async def kb_ask(
        question: Annotated[str, Field(description="Natural language or keywords")],
        strategy: Annotated[
            Strategy,
            Field(
                description=(
                    "Query strategy: auto, decision_trace, timeline, related, "
                    "connection"
                ),
            ),
        ] = "auto",
        scope: Annotated[
            str | None,
            Field(
                description=('Filter: "project:X", "tag:Y", entry ID, or node ID'),
            ),
        ] = None,
        target: Annotated[
            str | None,
            Field(description="Second node for 'connection' strategy"),
        ] = None,
        include_graph_context: Annotated[
            bool,
            Field(description="Expand results with graph neighbors"),
        ] = True,
        limit: Annotated[int, Field(description="Max results", ge=1, le=50)] = 20,
    ) -> str:
        """Answer questions by traversing the knowledge graph (auto strategy)."""
        backend = context.backend_for_request()

        if strategy != "auto":
            return (
                f"Error: strategy '{strategy}' requires a local KB"
                " — only 'auto' is supported in HTTP mode."
            )
        try:
            entries_with_context, agent_turns_used = await backend.ask_auto(
                question, scope, include_graph_context, limit
            )
        except Exception as exc:
            if isinstance(exc, BackendHttpError):
                return map_error(exc)
            return f"Error: {exc}"
        if not entries_with_context:
            return f"[Agent: {agent_turns_used} tool calls] No results found."

        header = f"[Agent: {agent_turns_used} tool calls]"
        formatted = [
            format_entry_full(entry, context=c)
            for entry, c in entries_with_context[:limit]
        ]
        return format_result_list(formatted, header=header)
