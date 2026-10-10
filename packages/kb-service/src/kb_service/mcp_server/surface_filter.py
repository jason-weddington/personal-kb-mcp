"""Per-surface /mcp tool visibility (write policy).

A caller whose effective write-policy surface is headless or autonomous (the
API key's surface after the ``X-KB-Mode`` downgrade, resolved by
``kb_service.write_policy.resolve_write_context``) sees only the tools in
``NON_INTERACTIVE_TOOL_BASES``: the read tools, ``store`` / ``store_batch``
(routed to the candidate pipeline) and ``feedback``. Every other tool is
hidden from ``tools/list`` and refused on ``tools/call`` without running.
The list is an allow-list, so a tool added later is hidden from unattended
surfaces until it is added here (fail closed). Interactive callers see the
full tool set unchanged.
"""

import logging
from collections.abc import Sequence

import mcp.types as mt
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.tool import Tool, ToolResult

from kb_service.write_policy import Surface, resolve_write_context

logger = logging.getLogger(__name__)

MCP_SURFACE_MARKER = "mcp-surface"

# Tool base names (without the instance prefix) a non-interactive surface sees.
NON_INTERACTIVE_TOOL_BASES: frozenset[str] = frozenset(
    {
        "search",
        "get",
        "ask",
        "summarize",
        "preflight",
        "explore",
        "map_eligibility",
        "list_projects",
        "list_contributors",
        "list_teams",
        "store",
        "store_batch",
        "feedback",
    }
)


async def _effective_surface() -> Surface:
    """The caller's write-policy surface.

    Outside an HTTP request (the server's own startup ``list_tools``) there is
    no caller, so nothing is filtered. A request without an authenticated
    principal fails closed as ``headless``.
    """
    try:
        request = get_http_request()
    except RuntimeError:
        return "interactive"
    principal = getattr(request.state, "kb_principal", None)
    if principal is None:
        return "headless"
    wctx = await resolve_write_context(request, principal.user)
    return wctx.surface


class SurfaceToolFilterMiddleware(Middleware):
    """Hide non-allow-listed tools from headless and autonomous surfaces."""

    def __init__(self, prefix: str) -> None:
        """Bind the instance tool prefix (``kb_``, ``personal_kb_``, ...)."""
        self._allowed = frozenset(prefix + base for base in NON_INTERACTIVE_TOOL_BASES)

    async def on_list_tools(
        self,
        context: MiddlewareContext[mt.ListToolsRequest],
        call_next: CallNext[mt.ListToolsRequest, Sequence[Tool]],
    ) -> Sequence[Tool]:
        """Drop hidden tools from the list for a non-interactive surface."""
        tools = await call_next(context)
        if await _effective_surface() == "interactive":
            return tools
        return [t for t in tools if t.name in self._allowed]

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        """Refuse a hidden tool for a non-interactive surface without running it."""
        name = context.message.name
        if name not in self._allowed:
            surface = await _effective_surface()
            if surface != "interactive":
                logger.info(
                    MCP_SURFACE_MARKER + " tool=%s outcome=hidden surface=%s",
                    name,
                    surface,
                )
                raise ToolError(
                    f"write policy: {name} is not available from the {surface}"
                    " surface; use an interactive session."
                )
        return await call_next(context)
