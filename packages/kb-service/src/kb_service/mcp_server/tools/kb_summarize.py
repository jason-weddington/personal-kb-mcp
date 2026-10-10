"""kb_summarize MCP tool (HTTP twin of ``personal_kb.tools.kb_summarize``)."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from kb_service.mcp_server import context
from kb_service.mcp_server.errors import BackendHttpError, map_error

logger = logging.getLogger(__name__)


def _summarize_description(prefix: str) -> str:
    """Build kb_summarize description with correct tool name cross-references."""
    return (
        "Answer a question with a synthesized natural language response.\n\n"
        "Best for answering user questions directly — produces a final, "
        "readable answer with [kb-XXXXX] citations, not raw search results. "
        "Retrieves relevant entries via graph+search, then synthesizes with an LLM. "
        "Falls back to raw results if LLM is unavailable.\n\n"
        "Use this for user-facing answers. For your own research or exploration, "
        f"prefer {prefix}search or {prefix}ask — they're cheaper (no synthesis LLM "
        "call)."
    )


def register_kb_summarize(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_summarize tool with the MCP server."""

    @mcp.tool(name=f"{prefix}summarize", description=_summarize_description(prefix))
    async def kb_summarize(
        question: Annotated[str, Field(description="Natural language question")],
        scope: Annotated[
            str | None,
            Field(description="Optional filter (project:X, tag:Y, etc.)"),
        ] = None,
        limit: Annotated[
            int, Field(description="Max entries to retrieve", ge=1, le=50)
        ] = 20,
    ) -> str:
        """Answer a question with a synthesized natural language response."""
        backend = context.backend_for_request()

        try:
            return await backend.summarize(question, scope, limit)
        except Exception as exc:
            if isinstance(exc, BackendHttpError):
                return map_error(exc)
            return f"Error: {exc}"
