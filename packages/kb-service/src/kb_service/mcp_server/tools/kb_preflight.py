"""kb_preflight MCP tool (HTTP twin of ``personal_kb.tools.kb_preflight``)."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from kb_core.ttl import parse_ttl
from pydantic import Field

from kb_service.mcp_server import context
from kb_service.mcp_server.errors import BackendHttpError, map_error

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
    ) -> str:
        """Get a project context primer."""
        if since is not None:
            try:
                parse_ttl(since)  # validate only
            except ValueError as exc:
                return f"Error: {exc}"

        backend = context.backend_for_request()

        try:
            return await backend.preflight(project_ref, since)
        except Exception as exc:
            if isinstance(exc, BackendHttpError):
                return map_error(exc)
            return f"Error: {exc}"
