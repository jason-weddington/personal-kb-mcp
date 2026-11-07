"""Read-only mode (``KB_READ_ONLY``): refuse every knowledge-content write.

The HTTP guard is the ``require_kb_writable`` route dependency; the in-process
MCP backend, the chat tools and the surprise worker call the helpers here.
Routes that write only the service DB (auth, admin, settings, chat history,
hook and telemetry routes) are deliberately not guarded.
"""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from kb_service.auth import get_current_user
from kb_service.config import is_read_only, read_only_message
from kb_service.models import User

logger = logging.getLogger("kb_service")

READ_ONLY_DETAIL_PREFIX = "read-only: "

_BASE_DETAIL = READ_ONLY_DETAIL_PREFIX + "this KB is read-only; writes are refused."


def read_only_detail() -> str:
    """The refusal text: base sentence plus the operator message when set."""
    message = read_only_message()
    return f"{_BASE_DETAIL} {message}" if message else _BASE_DETAIL


def log_refusal(
    surface: str,
    op: str,
    user_id: str | None = None,
    key_id: str | None = None,
) -> None:
    """Log one INFO record per refused call (request metadata only)."""
    logger.info(
        "read-only-refused surface=%s op=%s user_id=%s key_id=%s",
        surface,
        op,
        user_id if user_id is not None else "none",
        key_id if key_id is not None else "none",
    )


def ensure_writable() -> None:
    """Raise HTTP 403 when read-only mode is on."""
    if is_read_only():
        raise HTTPException(403, detail=read_only_detail())


async def require_kb_writable(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> None:
    """Route dependency: 403 in read-only mode (401 first if unauthenticated)."""
    if not is_read_only():
        return
    route = request.scope.get("route")
    op = getattr(route, "path", None) or request.url.path
    principal = getattr(request.state, "kb_principal", None)
    key_id = getattr(principal, "api_key_id", None)
    log_refusal("http", op, user.id, key_id)
    ensure_writable()


READ_ONLY_GUARDED_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/api/kb/store"),
        ("POST", "/api/kb/store_batch"),
        ("POST", "/api/kb/entries/{entry_id}/deactivate"),
        ("POST", "/api/kb/entries/{entry_id}/reactivate"),
        ("POST", "/api/kb/bulk_update"),
        ("POST", "/api/kb/admin/reconcile-supersession"),
        ("POST", "/api/kb/ingest/text"),
        ("POST", "/api/kb/ingest/url"),
        ("POST", "/api/kb/ingest/file"),
        ("POST", "/api/kb/map-op"),
        ("DELETE", "/api/kb/maps/{map_id}"),
        ("POST", "/api/kb/map-eligibility/override"),
        ("POST", "/api/kb/map-eligibility/override/clear"),
        ("POST", "/api/kb/cluster-ledger/decline"),
        ("POST", "/api/kb/cluster-ledger/match"),
        ("POST", "/api/kb/surprise/drain"),
    }
)

# write_policy classes these no_entry_write, but they write
# map_cluster_ledger / audit_events.
READ_ONLY_EXTRA_GUARDED: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/api/kb/cluster-ledger/decline"),
        ("POST", "/api/kb/cluster-ledger/match"),
    }
)

READ_ONLY_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        # auth: service DB only
        ("POST", "/api/auth/register"),
        ("POST", "/api/auth/login"),
        ("POST", "/api/auth/logout"),
        ("POST", "/api/auth/password"),
        ("POST", "/api/auth/password-reset"),
        ("POST", "/api/auth/api-keys"),
        ("DELETE", "/api/auth/api-keys/{key_id}"),
        # admin: service DB only
        ("POST", "/api/admin/invites"),
        ("POST", "/api/admin/users/{user_id}/password-reset"),
        ("POST", "/api/admin/users/{user_id}/promote"),
        ("POST", "/api/admin/shutdown"),
        ("DELETE", "/api/admin/invites/{token}"),
        ("DELETE", "/api/admin/users/{user_id}"),
        # settings: service DB only
        ("PUT", "/api/settings"),
        # chat history lives in the service DB
        ("POST", "/api/chat/stream"),
        ("POST", "/api/chat/create"),
        ("DELETE", "/api/chat/{chat_id}"),
    }
)
