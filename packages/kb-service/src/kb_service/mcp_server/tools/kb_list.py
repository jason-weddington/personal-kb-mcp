"""Discovery tools over HTTP: list_projects, list_contributors, list_teams."""

import logging

from fastmcp import FastMCP

from kb_service.mcp_server import context
from kb_service.mcp_server.errors import BackendHttpError, map_error

logger = logging.getLogger(__name__)

_PROJECTS_DESCRIPTION = (
    "List all projects in the knowledge base with entry counts. "
    "Call this BEFORE storing or searching with a project_ref to check "
    "existing project names and avoid duplicates (e.g. 'agent_gtd' vs "
    "'agent-gtd'). Use the exact project_ref values returned here."
)


def register_kb_list_projects(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the list_projects tool."""

    @mcp.tool(
        name=f"{prefix}list_projects",
        description=_PROJECTS_DESCRIPTION,
    )
    async def kb_list_projects() -> str:
        """List projects with entry counts."""
        backend = context.backend_for_request()
        try:
            rows = await backend.list_projects()
        except Exception as exc:
            if isinstance(exc, BackendHttpError):
                return map_error(exc)
            return f"Error: {exc}"
        if not rows:
            return "No projects found."
        return "\n".join(f"{name} ({count} entries)" for name, count in rows)


def register_kb_list_contributors(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the list_contributors tool."""

    @mcp.tool(
        name=f"{prefix}list_contributors",
        description="List all contributors in the knowledge base with entry counts.",
    )
    async def kb_list_contributors() -> str:
        """List contributors with entry counts."""
        backend = context.backend_for_request()
        try:
            rows = await backend.list_contributors()
        except Exception as exc:
            if isinstance(exc, BackendHttpError):
                return map_error(exc)
            return f"Error: {exc}"
        if not rows:
            return "No contributors found."
        return "\n".join(f"{name} ({count} entries)" for name, count in rows)


def register_kb_list_teams(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the list_teams tool."""

    @mcp.tool(
        name=f"{prefix}list_teams",
        description="List all teams in the knowledge base with entry counts.",
    )
    async def kb_list_teams() -> str:
        """List teams with entry counts."""
        backend = context.backend_for_request()
        try:
            rows = await backend.list_teams()
        except Exception as exc:
            if isinstance(exc, BackendHttpError):
                return map_error(exc)
            return f"Error: {exc}"
        if not rows:
            return "No teams found."
        return "\n".join(f"{name} ({count} entries)" for name, count in rows)
