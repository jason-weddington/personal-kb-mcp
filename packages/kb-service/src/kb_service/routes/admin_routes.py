"""Admin-only routes: invite management and user management."""

import asyncio
import os
import secrets
import signal
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from kb_service.auth import _auth_mode, require_admin
from kb_service.config import public_base_url
from kb_service.database import get_db
from kb_service.models import (
    CreateInviteRequest,
    InviteListItem,
    InviteResponse,
    PasswordResetIssueResponse,
    User,
    UserResponse,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/invites", response_model=InviteResponse, status_code=201)
async def create_invite(
    request: Request,
    body: CreateInviteRequest,
    admin: Annotated[User, Depends(require_admin)],
) -> InviteResponse:
    """Create a new invite token (admin only)."""
    db = await get_db()
    token = secrets.token_urlsafe(32)
    now = datetime.now(UTC).isoformat()
    base_url = public_base_url(request)
    invite_url = f"{base_url}/register?token={token}"
    await db.execute(
        "INSERT INTO invites (token, issued_by, note, created_at)"
        " VALUES ($1, $2, $3, $4)",
        token,
        admin.id,
        body.note,
        now,
    )
    return InviteResponse(
        token=token,
        url=invite_url,
        note=body.note,
        created_at=datetime.fromisoformat(now),
    )


@router.get("/invites", response_model=list[InviteListItem])
async def list_invites(
    _admin: Annotated[User, Depends(require_admin)],
) -> list[InviteListItem]:
    """List all invites, most recent first (admin only)."""
    db = await get_db()
    rows = await db.fetch(
        "SELECT token, issued_by, note, created_at, used_at, used_by"
        " FROM invites ORDER BY created_at DESC"
    )
    return [
        InviteListItem(
            token=row["token"],
            issued_by=row["issued_by"],
            note=row["note"],
            created_at=datetime.fromisoformat(row["created_at"]),
            used_at=datetime.fromisoformat(row["used_at"]) if row["used_at"] else None,
            used_by=row["used_by"],
        )
        for row in rows
    ]


@router.post(
    "/users/{user_id}/password-reset", response_model=PasswordResetIssueResponse
)
async def issue_password_reset(
    user_id: str,
    request: Request,
    admin: Annotated[User, Depends(require_admin)],
) -> PasswordResetIssueResponse:
    """Generate a one-time password-reset link for a user (admin only)."""
    db = await get_db()
    user_row = await db.fetchrow("SELECT id FROM users WHERE id = $1", user_id)
    if user_row is None:
        raise HTTPException(status_code=404, detail="User not found")
    token = secrets.token_urlsafe(32)
    now = datetime.now(UTC)
    expires_at = now + timedelta(hours=12)
    await db.execute(
        "INSERT INTO password_resets (token, user_id, created_at, expires_at)"
        " VALUES ($1, $2, $3, $4)",
        token,
        user_id,
        now.isoformat(),
        expires_at.isoformat(),
    )
    base_url = public_base_url(request)
    reset_url = f"{base_url}/reset-password?token={token}"
    return PasswordResetIssueResponse(
        token=token,
        url=reset_url,
        expires_at=expires_at,
    )


@router.delete("/invites/{token}", status_code=204)
async def revoke_invite(
    token: str,
    _admin: Annotated[User, Depends(require_admin)],
) -> Response:
    """Revoke an unused invite (admin only). Returns 409 if already used."""
    db = await get_db()
    row = await db.fetchrow("SELECT used_at FROM invites WHERE token = $1", token)
    if row is None:
        raise HTTPException(status_code=404, detail="Invite not found")
    if row["used_at"] is not None:
        raise HTTPException(status_code=409, detail="Invite already used")
    await db.execute("DELETE FROM invites WHERE token = $1", token)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# User management endpoints
# ---------------------------------------------------------------------------


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    _admin: Annotated[User, Depends(require_admin)],
) -> list[UserResponse]:
    """List all users, most-recent-first (admin only)."""
    db = await get_db()
    rows = await db.fetch(
        "SELECT id, email, is_admin, created_at FROM users ORDER BY created_at DESC"
    )
    return [
        UserResponse(
            id=row["id"],
            email=row["email"],
            is_admin=bool(row["is_admin"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
        for row in rows
    ]


@router.post("/users/{user_id}/promote", response_model=UserResponse)
async def promote_user(
    user_id: str,
    _admin: Annotated[User, Depends(require_admin)],
) -> UserResponse:
    """Promote a user to admin (admin only). Idempotent."""
    db = await get_db()
    row = await db.fetchrow(
        "SELECT id, email, is_admin, created_at FROM users WHERE id = $1", user_id
    )
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    await db.execute("UPDATE users SET is_admin = 1 WHERE id = $1", user_id)
    return UserResponse(
        id=row["id"],
        email=row["email"],
        is_admin=True,
        created_at=datetime.fromisoformat(row["created_at"]),
    )


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: str,
    admin: Annotated[User, Depends(require_admin)],
) -> Response:
    """Delete a user (admin only)."""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    db = await get_db()
    row = await db.fetchrow("SELECT id FROM users WHERE id = $1", user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    # Clean up FK references (no ON DELETE CASCADE) in dependency order.
    await db.execute("DELETE FROM password_resets WHERE user_id = $1", user_id)
    await db.execute("DELETE FROM invites WHERE issued_by = $1", user_id)
    await db.execute("DELETE FROM api_keys WHERE user_id = $1", user_id)
    await db.execute("DELETE FROM users WHERE id = $1", user_id)
    return Response(status_code=204)


_LOOPBACK_CLIENTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _request_shutdown() -> None:
    """Ask the serving uvicorn process to exit gracefully (SIGTERM to self)."""
    os.kill(os.getpid(), signal.SIGTERM)


@router.post("/shutdown", status_code=202)
async def shutdown(request: Request) -> dict[str, str]:
    """Gracefully stop this daemon (local no-auth mode + loopback caller only).

    Fail-closed: 404 unless ``KB_AUTH_MODE=none`` and 403 unless the caller is
    on loopback, so a hosted deployment can never be shut down remotely.
    """
    if _auth_mode() != "none":
        raise HTTPException(status_code=404, detail="Not found")
    host = request.client.host if request.client else ""
    if host not in _LOOPBACK_CLIENTS:
        raise HTTPException(status_code=403, detail="Loopback only")
    # Delay slightly so the 202 response is flushed before the signal lands.
    asyncio.get_running_loop().call_later(0.2, _request_shutdown)
    return {"status": "shutting_down"}
