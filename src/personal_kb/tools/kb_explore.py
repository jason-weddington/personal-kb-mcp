"""kb_explore MCP tool — open interactive graph explorer in browser."""

import asyncio
import contextlib
import logging
import tempfile
import webbrowser
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.context import Context

from personal_kb.db.backend import Database
from personal_kb.explorer.graph_data import extract_graph_data
from personal_kb.explorer.renderer import render_explorer_html
from personal_kb.llm.provider import LLMProvider
from personal_kb.search.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

# Module-level reference to avoid duplicate server starts
_web_server_task: asyncio.Task[Any] | None = None


async def explore_logic(
    db: Database,
    embedder: EmbeddingClient | None = None,
    query_llm: LLMProvider | None = None,
) -> tuple[str, str]:
    """Open the explorer in the browser — web server or temp file fallback.

    Returns (html_content, summary_message).
    """
    global _web_server_task

    # Try web server mode
    try:
        from personal_kb.web.app import create_app_with_deps

        need_start = _web_server_task is None or _web_server_task.done()
        if need_start:
            app = create_app_with_deps(db, embedder, query_llm)
            import uvicorn

            config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="warning")
            server = uvicorn.Server(config)

            async def _safe_serve() -> None:
                with contextlib.suppress(SystemExit):
                    await server.serve()

            _web_server_task = asyncio.create_task(_safe_serve())
            # Brief wait for the server to start; check for early failure
            await asyncio.sleep(0.3)
            if _web_server_task.done():
                # Server failed to start (port in use, etc.) — fall through to temp file
                _web_server_task = None
                raise OSError("Web server failed to start")

        webbrowser.open("http://localhost:8765")

        data = await extract_graph_data(db)
        stats = data["stats"]
        summary = (
            f"Explorer opened: {stats['node_count']} nodes, {stats['edge_count']} edges. "
            f"http://localhost:8765 (query-enabled)"
        )
        return render_explorer_html(data), summary

    except (OSError, SystemExit):
        logger.debug("Web server failed to start, falling back to temp file mode")

    # Fallback: static temp file (no query support)
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
            "Starts a local server with LLM-powered query support and multi-turn chat. "
            "Falls back to a static HTML file if the port is in use."
        ),
    )
    async def kb_explore(ctx: Context | None = None) -> str:
        """Open interactive graph explorer in the browser."""
        if ctx is None:
            raise RuntimeError("Context not injected")

        lifespan = ctx.lifespan_context
        db = lifespan["db"]
        embedder = lifespan.get("embedder")
        query_llm = lifespan.get("query_llm")

        _, summary = await explore_logic(db, embedder, query_llm)
        return summary
