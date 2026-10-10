"""kb_feedback MCP tool (HTTP twin of ``personal_kb.tools.kb_feedback``)."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from kb_service.mcp_server import context

logger = logging.getLogger(__name__)

_VALID_TYPES = {"missing", "unhelpful", "friction"}


def _feedback_description(prefix: str) -> str:
    """Build kb_feedback description with correct tool name cross-references."""
    return (
        "Report when a KB query failed to help with your task.\n\n"
        "Takes 3 seconds, helps the human prioritize what to add next.\n"
        f"Do NOT use for storing knowledge (use {prefix}store instead).\n\n"
        f"Call this whenever {prefix}search/{prefix}ask/{prefix}summarize returned "
        "poor results — zero hits, irrelevant entries, or missing knowledge you "
        "expected to find."
    )


def register_kb_feedback(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_feedback tool with the MCP server."""

    @mcp.tool(name=f"{prefix}feedback", description=_feedback_description(prefix))
    async def kb_feedback(
        feedback_type: Annotated[
            str,
            Field(
                description=(
                    "Type of feedback: 'missing' (KB lacked needed knowledge), "
                    "'unhelpful' (results existed but didn't help), "
                    "'friction' (tool was awkward or slow to use)"
                ),
            ),
        ],
        tool_name: Annotated[
            str | None,
            Field(description="Which KB tool triggered this (kb_search, kb_ask, etc.)"),
        ] = None,
        query_or_params: Annotated[
            str | None,
            Field(description="Echo of the query/params that produced poor results"),
        ] = None,
        detail: Annotated[
            str | None,
            Field(description="One sentence of context about what went wrong"),
        ] = None,
    ) -> str:
        """Report when a KB query failed to help with your task."""
        if feedback_type not in _VALID_TYPES:
            return (
                f"Invalid feedback_type '{feedback_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_TYPES))}"
            )

        backend = context.backend_for_request()
        await backend.feedback(feedback_type, tool_name, query_or_params, detail)
        return (
            f"Feedback recorded ({feedback_type}). "
            "Thank you — this helps improve the KB."
        )
