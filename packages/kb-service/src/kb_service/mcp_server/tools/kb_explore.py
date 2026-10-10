"""kb_explore MCP tool (HTTP twin of ``personal_kb.tools.kb_explore``)."""

import logging

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_request

from kb_service.config import public_base_url

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
    async def kb_explore() -> str:
        """Return the hosted KB explorer URL."""
        url = public_base_url(get_http_request())
        return f"KB explorer is hosted at {url} — open it in a browser."
