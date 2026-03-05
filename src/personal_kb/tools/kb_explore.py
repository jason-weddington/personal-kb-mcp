"""kb_explore MCP tool — open interactive graph explorer in browser."""

import logging
import tempfile
import webbrowser

from fastmcp import FastMCP
from fastmcp.server.context import Context

from personal_kb.db.backend import Database
from personal_kb.explorer.graph_data import extract_graph_data
from personal_kb.explorer.renderer import render_explorer_html

logger = logging.getLogger(__name__)


async def explore_logic(db: Database) -> tuple[str, str]:
    """Extract graph data, render HTML, write to temp file, open browser.

    Returns (html_content, summary_message).
    """
    data = await extract_graph_data(db)
    html = render_explorer_html(data)

    with tempfile.NamedTemporaryFile(
        suffix=".html", prefix="kb_explore_", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(html)
        path = f.name

    webbrowser.open(f"file://{path}")

    stats = data["stats"]
    summary = (
        f"Explorer opened: {stats['node_count']} nodes, {stats['edge_count']} edges. File: {path}"
    )
    return html, summary


def register_kb_explore(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_explore tool with the MCP server."""

    @mcp.tool(
        name=f"{prefix}explore",
        description=(
            "Open an interactive graph explorer in the browser. "
            "Generates a self-contained HTML file with force-directed "
            "visualization of the full knowledge graph."
        ),
    )
    async def kb_explore(ctx: Context | None = None) -> str:
        """Open interactive graph explorer in the browser."""
        if ctx is None:
            raise RuntimeError("Context not injected")

        lifespan = ctx.lifespan_context
        db = lifespan["db"]

        _, summary = await explore_logic(db)
        return summary
