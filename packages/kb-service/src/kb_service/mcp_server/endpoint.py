"""The authenticated ASGI entry point mounted at ``/mcp``.

Authenticates the bearer token exactly like the REST API (via
``resolve_principal``), then delegates to the stateless FastMCP streamable
HTTP app built per lifespan (``app.state.mcp_http_app``). The principal and
the kb-service app ride in the ASGI scope state so tool calls can build an
``InProcessBackend`` for the caller.
"""

import hashlib
import logging

from fastapi import HTTPException
from fastapi.security.utils import get_authorization_scheme_param
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from kb_service.auth import resolve_principal
from kb_service.mcp_server.observability import (
    MCP_AUTH_MARKER,
    MCP_ENDPOINT_MARKER,
)

logger = logging.getLogger(__name__)

_DENY_REASONS = {
    "Not authenticated": "missing",
    "Invalid API key": "invalid_key",
    "User not found": "user_not_found",
}


class McpEndpoint:
    """ASGI app for ``/mcp``: bearer auth, then the FastMCP HTTP app.

    A class instance (not a function) so Starlette's ``Route`` treats it as a
    raw ASGI app rather than a request handler.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Authenticate the request and hand it to the inner MCP app."""
        headers = Headers(scope=scope)
        header = headers.get("authorization")
        scheme, cred = get_authorization_scheme_param(header)
        token = cred if header and scheme.lower() == "bearer" and cred else None

        try:
            principal = await resolve_principal(token)
        except HTTPException as exc:
            reason = _DENY_REASONS.get(str(exc.detail), "other")
            key_fp = (
                hashlib.sha256(token.encode()).hexdigest()[:8]
                if token is not None
                else "none"
            )
            logger.info(
                MCP_AUTH_MARKER + " denied status=%d reason=%s key_fp=%s ua=%r",
                exc.status_code,
                reason,
                key_fp,
                (headers.get("user-agent") or "")[:80],
            )
            response_headers = (
                {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
            )
            response = JSONResponse(
                {"detail": exc.detail},
                status_code=exc.status_code,
                headers=response_headers,
            )
            await response(scope, receive, send)
            return

        inner = getattr(scope["app"].state, "mcp_http_app", None)
        if inner is None:
            logger.warning(
                MCP_ENDPOINT_MARKER + " not_started method=%s",
                scope.get("method"),
            )
            response = JSONResponse(
                {"detail": "MCP endpoint not started"}, status_code=503
            )
            await response(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        state["kb_principal"] = principal
        state["kb_app"] = scope["app"]
        await inner(scope, receive, send)
