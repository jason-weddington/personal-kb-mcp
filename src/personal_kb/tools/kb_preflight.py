"""kb_preflight MCP tool — project context primer."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

logger = logging.getLogger(__name__)


def _preflight_description(prefix: str) -> str:
    """Build kb_preflight description with correct tool name cross-references."""
    return (
        "Get a project context primer — a compact table-of-contents of "
        "expiring entries, recent decisions/lessons, and active conventions "
        "for a project.\n\n"
        "Call this at session start to get up to speed on a project. "
        "Returns entry IDs and titles — use "
        f"{prefix}get to read the full details of any entry.\n\n"
        "Use the 'since' parameter to narrow decisions/lessons to a time "
        "window (e.g. '7d' for last week, '2w' for last two weeks). "
        "Expiring entries and conventions are always included regardless "
        "of 'since'."
    )


def register_kb_preflight(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_preflight tool with the MCP server."""

    @mcp.tool(name=f"{prefix}preflight", description=_preflight_description(prefix))
    async def kb_preflight(
        project_ref: Annotated[
            str,
            Field(description="Project identifier to get context for"),
        ],
        since: Annotated[
            str | None,
            Field(
                description="Only show decisions/lessons from this time window. "
                "Format: Nh (hours), Nd (days), Nw (weeks). "
                "Examples: 7d, 2w, 24h. Omit for all recent entries."
            ),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Get a project context primer."""
        if ctx is None:
            raise RuntimeError("Context not injected")

        from personal_kb.preflight import build_project_context
        from personal_kb.tools.ttl import parse_ttl

        lifespan = ctx.lifespan_context
        db = lifespan["db"]
        team: str | None = lifespan.get("team")

        since_td = None
        if since is not None:
            try:
                since_td = parse_ttl(since)
            except ValueError as exc:
                return f"Error: {exc}"

        return await build_project_context(db, project_ref, team=team, since=since_td)
