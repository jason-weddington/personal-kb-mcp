"""kb_explore MCP tool — open interactive graph explorer in browser."""

import logging

from fastmcp import FastMCP
from fastmcp.server.context import Context

logger = logging.getLogger(__name__)


def register_kb_explore(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_explore tool with the MCP server."""

    @mcp.tool(
        name=f"{prefix}explore",
        description=(
            "Open the interactive KB graph explorer. "
            "The explorer is hosted — returns the URL to open in a browser."
        ),
    )
    async def kb_explore(ctx: Context | None = None) -> str:
        """Return the hosted KB explorer URL."""
        from personal_kb.config import get_personal_kb_url

        if ctx is None:
            raise RuntimeError("Context not injected")

        url = get_personal_kb_url() or "the KB service"
        return f"KB explorer is hosted at {url} — open it in a browser."
