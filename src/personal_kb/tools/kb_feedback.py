"""kb_feedback MCP tool — structured agent friction reporting."""

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.db.backend import Database

logger = logging.getLogger(__name__)

_VALID_TYPES = {"missing", "unhelpful", "friction"}


def register_kb_feedback(mcp: FastMCP) -> None:
    """Register the kb_feedback tool with the MCP server."""

    @mcp.tool()
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
        ctx: Context | None = None,
    ) -> str:
        """Report when a KB query failed to help with your task.

        Takes 3 seconds, helps the human prioritize what to add next.
        Do NOT use for storing knowledge (use kb_store instead).
        """
        if ctx is None:
            raise RuntimeError("Context not injected")

        lifespan = ctx.lifespan_context
        db: Database = lifespan["db"]

        return await submit_feedback(db, feedback_type, tool_name, query_or_params, detail)


async def submit_feedback(
    db: Database,
    feedback_type: str,
    tool_name: str | None = None,
    query_or_params: str | None = None,
    detail: str | None = None,
) -> str:
    """Core feedback logic, testable without FastMCP context."""
    if feedback_type not in _VALID_TYPES:
        return (
            f"Invalid feedback_type '{feedback_type}'. "
            f"Must be one of: {', '.join(sorted(_VALID_TYPES))}"
        )

    now = datetime.now(UTC).isoformat()
    await db.execute(
        "INSERT INTO agent_feedback (feedback_type, tool_name, query_or_params, detail, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (feedback_type, tool_name, query_or_params, detail, now),
    )
    await db.commit()

    return f"Feedback recorded ({feedback_type}). Thank you — this helps improve the KB."
