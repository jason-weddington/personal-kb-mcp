"""Per-request backend construction for /mcp tool calls."""

import logging

from fastmcp.server.dependencies import get_http_request

from kb_service.mcp_server.backend import InProcessBackend
from kb_service.mcp_server.observability import MCP_PRINCIPAL_MISSING_MARKER

logger = logging.getLogger(__name__)


def backend_for_request() -> InProcessBackend:
    """Build an ``InProcessBackend`` for the authenticated caller of this call.

    The ``McpEndpoint`` puts ``kb_principal`` and ``kb_app`` into the ASGI
    scope state before delegating to the FastMCP app. Fails closed: a missing
    request or principal raises instead of falling back to any default user.

    Raises:
        RuntimeError: When no authenticated principal reached the tool call.
    """
    has_principal = False
    has_app = False
    try:
        request = get_http_request()
    except RuntimeError:
        request = None
    if request is not None:
        has_principal = hasattr(request.state, "kb_principal")
        has_app = hasattr(request.state, "kb_app")
    if request is None or not has_principal or not has_app:
        logger.error(
            MCP_PRINCIPAL_MISSING_MARKER + " has_principal=%s has_app=%s",
            has_principal,
            has_app,
        )
        raise RuntimeError("MCP tool call reached without an authenticated principal")
    return InProcessBackend(
        app=request.state.kb_app,
        principal=request.state.kb_principal,
        raw_headers=request.headers.raw,
    )
