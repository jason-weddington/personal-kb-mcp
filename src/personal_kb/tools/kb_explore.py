"""kb_explore MCP tool — open interactive graph explorer in browser."""

import asyncio
import contextlib
import logging
import signal
import subprocess
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

_EXPLORER_PORT = 8765


def _kill_port_holder(port: int) -> bool:
    """Kill any process listening on the given TCP port. Returns True if killed.

    Skips the current process to avoid self-termination when an in-process
    server (e.g. from a previous kb_explore call) is still binding the port.
    """
    import os as _os

    try:
        result = subprocess.run(  # noqa: S603
            ["lsof", "-ti", f"tcp:{port}"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = result.stdout.strip()
        if not pids:
            return False
        my_pid = _os.getpid()
        killed = False
        for pid_str in pids.splitlines():
            pid = int(pid_str.strip())
            if pid == my_pid:
                continue  # Don't kill ourselves
            logger.info("Killing existing server on port %d (pid %d)", port, pid)
            try:
                _os.kill(pid, signal.SIGTERM)
                killed = True
            except ProcessLookupError:
                pass
        return killed
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return False


async def explore_logic(
    db: Database,
    embedder: EmbeddingClient | None = None,
    query_llm: LLMProvider | None = None,
    synthesis_llm: LLMProvider | None = None,
    *,
    store: Any | None = None,
    graph_builder: Any | None = None,
    graph_enricher: Any | None = None,
    extraction_llm: LLMProvider | None = None,
    contributor: str | None = None,
    team: str | None = None,
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
            # Kill any existing server on the port (e.g. from another MCP instance)
            if _kill_port_holder(_EXPLORER_PORT):
                await asyncio.sleep(0.3)

            app = create_app_with_deps(
                db,
                embedder,
                query_llm,
                synthesis_llm,
                store=store,
                graph_builder=graph_builder,
                graph_enricher=graph_enricher,
                extraction_llm=extraction_llm,
                contributor=contributor,
                team=team,
            )
            import uvicorn

            config = uvicorn.Config(app, host="127.0.0.1", port=_EXPLORER_PORT, log_level="warning")
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

        webbrowser.open(f"http://localhost:{_EXPLORER_PORT}")

        data = await extract_graph_data(db)
        stats = data["stats"]
        summary = (
            f"Explorer opened: {stats['node_count']} nodes, {stats['edge_count']} edges. "
            f"http://localhost:{_EXPLORER_PORT} (query-enabled)"
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
        synthesis_llm = lifespan.get("synthesis_llm")

        _, summary = await explore_logic(
            db,
            embedder,
            query_llm,
            synthesis_llm,
            store=lifespan.get("store"),
            graph_builder=lifespan.get("graph_builder"),
            graph_enricher=lifespan.get("graph_enricher"),
            extraction_llm=lifespan.get("llm_client"),
            contributor=lifespan.get("contributor"),
            team=lifespan.get("team"),
        )
        return summary
