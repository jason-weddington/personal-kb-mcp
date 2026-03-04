"""Discovery tools: list_projects, list_contributors, list_teams."""

import logging

from fastmcp import FastMCP
from fastmcp.server.context import Context

from personal_kb.db.backend import Database

logger = logging.getLogger(__name__)


async def list_projects_logic(db: Database) -> str:
    """Return project_ref values with entry counts."""
    rows = await db.execute(
        "SELECT project_ref, COUNT(*) as cnt FROM knowledge_entries "
        "WHERE is_active = 1 AND project_ref IS NOT NULL "
        "GROUP BY project_ref ORDER BY cnt DESC"
    )
    results = await rows.fetchall()
    if not results:
        return "No projects found."
    return "\n".join(f"{row[0]} ({row[1]} entries)" for row in results)


async def list_contributors_logic(db: Database) -> str:
    """Return contributor values with entry counts."""
    rows = await db.execute(
        "SELECT contributor, COUNT(*) as cnt FROM knowledge_entries "
        "WHERE is_active = 1 AND contributor IS NOT NULL "
        "GROUP BY contributor ORDER BY cnt DESC"
    )
    results = await rows.fetchall()
    if not results:
        return "No contributors found."
    return "\n".join(f"{row[0]} ({row[1]} entries)" for row in results)


async def list_teams_logic(db: Database) -> str:
    """Return team values with entry counts."""
    rows = await db.execute(
        "SELECT team, COUNT(*) as cnt FROM knowledge_entries "
        "WHERE is_active = 1 AND team IS NOT NULL "
        "GROUP BY team ORDER BY cnt DESC"
    )
    results = await rows.fetchall()
    if not results:
        return "No teams found."
    return "\n".join(f"{row[0]} ({row[1]} entries)" for row in results)


def register_kb_list_projects(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the list_projects tool."""

    @mcp.tool(
        name=f"{prefix}list_projects",
        description="List all projects in the knowledge base with entry counts.",
    )
    async def kb_list_projects(ctx: Context) -> str:
        db = ctx.lifespan_context["db"]
        return await list_projects_logic(db)


def register_kb_list_contributors(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the list_contributors tool."""

    @mcp.tool(
        name=f"{prefix}list_contributors",
        description="List all contributors in the knowledge base with entry counts.",
    )
    async def kb_list_contributors(ctx: Context) -> str:
        db = ctx.lifespan_context["db"]
        return await list_contributors_logic(db)


def register_kb_list_teams(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the list_teams tool."""

    @mcp.tool(
        name=f"{prefix}list_teams",
        description="List all teams in the knowledge base with entry counts.",
    )
    async def kb_list_teams(ctx: Context) -> str:
        db = ctx.lifespan_context["db"]
        return await list_teams_logic(db)
