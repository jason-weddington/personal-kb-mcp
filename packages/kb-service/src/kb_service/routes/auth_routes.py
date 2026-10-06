"""Auth API routes: register, login, logout, me, password, API keys."""

import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from kb_service.auth import (
    authenticate_user,
    consume_password_reset,
    create_token,
    generate_api_key,
    get_current_user,
    hash_api_key,
    hash_password,
    register_user_with_invite,
    verify_password,
)
from kb_service.database import get_db
from kb_service.models import (
    ApiKeyInfo,
    ApiKeyListResponse,
    ApiKeyResponse,
    AuthResponse,
    ChangePasswordRequest,
    CreateApiKeyRequest,
    LoginRequest,
    PasswordResetConsumeRequest,
    RegisterRequest,
    User,
    UserResponse,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        is_admin=user.is_admin,
        created_at=user.created_at,
    )


@router.post("/register", response_model=AuthResponse, status_code=201)
async def register(body: RegisterRequest) -> AuthResponse:
    """Create a new user account (requires a valid invite token)."""
    user = await register_user_with_invite(body.email, body.password, body.invite_token)
    token = create_token(user.id)
    return AuthResponse(token=token, user=_user_response(user))


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest) -> AuthResponse:
    """Authenticate and receive a JWT token."""
    user = await authenticate_user(body.email, body.password)
    token = create_token(user.id)
    return AuthResponse(token=token, user=_user_response(user))


@router.post("/logout", status_code=204)
async def logout(
    _user: Annotated[User, Depends(get_current_user)],
) -> None:
    """Logout (client discards token; placeholder for token blocklist)."""


@router.get("/me", response_model=UserResponse)
async def me(
    user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    """Get current user profile."""
    return _user_response(user)


@router.post("/password", status_code=204)
async def change_password(
    body: ChangePasswordRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> Response:
    """Change the authenticated user's password."""
    if not verify_password(body.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    new_hash = hash_password(body.new_password)
    db = await get_db()
    await db.execute(
        "UPDATE users SET hashed_password = $1 WHERE id = $2",
        new_hash,
        user.id,
    )
    return Response(status_code=204)


@router.post("/password-reset", status_code=204)
async def reset_password(body: PasswordResetConsumeRequest) -> Response:
    """Consume a one-time reset token and update the user's password (public)."""
    await consume_password_reset(body.token, body.new_password)
    return Response(status_code=204)


# --- API Key Management ---


@router.post("/api-keys", response_model=ApiKeyResponse, status_code=201)
async def create_api_key(
    body: CreateApiKeyRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> ApiKeyResponse:
    """Generate a new API key. The plaintext key is returned only once."""
    db = await get_db()
    key = generate_api_key()
    h = hash_api_key(key)
    now = datetime.now(UTC).isoformat()
    await db.execute(
        "INSERT INTO api_keys (id, user_id, key_hash, name, created_at)"
        " VALUES ($1, $2, $3, $4, $5)",
        str(uuid.uuid4()),
        user.id,
        h,
        body.name,
        now,
    )
    return ApiKeyResponse(api_key=key, name=body.name)


@router.get("/api-keys", response_model=ApiKeyListResponse)
async def list_api_keys(
    user: Annotated[User, Depends(get_current_user)],
) -> ApiKeyListResponse:
    """List all API keys for the current user (no secret material)."""
    db = await get_db()
    rows = await db.fetch(
        "SELECT id, name, key_hash, created_at FROM api_keys"
        " WHERE user_id = $1 ORDER BY created_at",
        user.id,
    )
    keys = [
        ApiKeyInfo(
            id=row["id"],
            name=row["name"],
            hash_prefix=row["key_hash"][:8],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
        for row in rows
    ]
    return ApiKeyListResponse(keys=keys)


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: str,
    user: Annotated[User, Depends(get_current_user)],
) -> Response:
    """Revoke an API key."""
    db = await get_db()
    row = await db.fetchrow(
        "SELECT id FROM api_keys WHERE id = $1 AND user_id = $2",
        key_id,
        user.id,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="API key not found")
    await db.execute(
        "DELETE FROM api_keys WHERE id = $1 AND user_id = $2",
        key_id,
        user.id,
    )
    return Response(status_code=204)
