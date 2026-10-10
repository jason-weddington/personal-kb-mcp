"""Log markers, backend-status tracking and the per-call log middleware.

Every /mcp tool call emits exactly one ``mcp-call`` INFO record on this
module's logger (under ``kb_service``, which ``configure_logging`` routes).
Arguments, tokens and header values other than the User-Agent are never
logged.
"""

import logging
import time
from contextvars import ContextVar
from typing import Any

import mcp.types as mt
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.tool import ToolResult

logger = logging.getLogger(__name__)

MCP_CALL_MARKER = "mcp-call"
MCP_AUTH_MARKER = "mcp-auth"
MCP_ENDPOINT_MARKER = "mcp-endpoint"
MCP_BACKEND_MARKER = "mcp-backend"
MCP_PRINCIPAL_MISSING_MARKER = "mcp-principal-missing"

_backend_statuses: ContextVar[list[int] | None] = ContextVar(
    "kb_mcp_backend_statuses", default=None
)


def record_backend_status(status: int) -> None:
    """Record one backend call's status for the enclosing tool call (if any)."""
    statuses = _backend_statuses.get()
    if statuses is not None:
        statuses.append(status)


def _principal_fields() -> tuple[str, str, str, str]:
    """Return ``(user_id, key_id, auth, ua)`` for the current request."""
    try:
        request = get_http_request()
    except RuntimeError:
        return "none", "none", "none", ""
    ua = (request.headers.get("user-agent") or "")[:80]
    principal: Any = getattr(request.state, "kb_principal", None)
    if principal is None:
        return "none", "none", "none", ua
    return (
        str(principal.user.id),
        str(principal.api_key_id) if principal.api_key_id is not None else "none",
        str(principal.auth_method),
        ua,
    )


class McpCallLogMiddleware(Middleware):
    """Emit one ``mcp-call`` INFO record per tool call."""

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        """Time the call, then log its outcome and worst backend status."""
        statuses: list[int] = []
        token = _backend_statuses.set(statuses)
        start = time.perf_counter()
        outcome = "ok"
        try:
            result = await call_next(context)
        except BaseException:
            outcome = "exception"
            raise
        finally:
            _backend_statuses.reset(token)
            duration_ms = int((time.perf_counter() - start) * 1000)
            worst = max(statuses) if statuses else None
            if outcome != "exception" and worst is not None and worst >= 400:
                outcome = "tool_error"
            user_id, key_id, auth, ua = _principal_fields()
            logger.info(
                MCP_CALL_MARKER + " tool=%s outcome=%s worst_status=%s"
                " duration_ms=%d user_id=%s key_id=%s auth=%s ua=%r",
                context.message.name,
                outcome,
                worst if worst is not None else "none",
                duration_ms,
                user_id,
                key_id,
                auth,
                ua,
            )
        return result
