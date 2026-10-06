"""Authentication utilities: password hashing, JWT, API keys, and FastAPI dependency."""

import hashlib
import os
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

import bcrypt as _bcrypt
import jwt
from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from kb_service.database import (
    get_db,
    row_to_dict,
)
from kb_service.models import User

SECRET_KEY = os.environ.get("JWT_SECRET", "dev-secret-change-me")
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 72

# auto_error=False so a missing Authorization header yields ``credentials=None``
# (instead of an automatic 401) — the 401-on-missing decision is made in the
# dependency body, which lets no-auth mode bypass it entirely.
_bearer = HTTPBearer(auto_error=False)

_VALID_AUTH_MODES = {"jwt", "none"}


def _auth_mode() -> Literal["jwt", "none"]:
    """Return the active auth mode from ``KB_AUTH_MODE`` (read per-request).

    The env var is read on EVERY call and never cached into a module constant,
    because ``conftest.py`` imports ``kb_service.main`` (hence this module) at
    collection time — a module-level read would freeze the value before any
    test could ``monkeypatch.setenv`` it. Defaults to ``'jwt'`` so all hosted
    behavior is byte-for-byte unchanged.

    Returns:
        Either ``'jwt'`` (default, JWT/API-key auth) or ``'none'`` (single-user
        no-auth local mode).

    Raises:
        ValueError: If ``KB_AUTH_MODE`` is set to an unrecognized value.
    """
    raw = os.environ.get("KB_AUTH_MODE", "jwt").lower()
    if raw in _VALID_AUTH_MODES:
        return "none" if raw == "none" else "jwt"
    choices = ", ".join(sorted(_VALID_AUTH_MODES))
    msg = f"KB_AUTH_MODE={raw!r} is not valid. Choose from: {choices}"
    raise ValueError(msg)


def _synthetic_user() -> User:
    """Return the single synthetic admin user used in no-auth (local) mode.

    ``hashed_password=''`` is required because ``User.hashed_password`` is a
    non-optional ``str`` (models.py); ``is_admin=True`` is set only so that
    ``require_admin`` does not 403 the synthetic user. Note the scope of what
    "admin" buys you here: the service/auth pool (users / api_keys / invites /
    password_resets) is disabled in no-auth mode, so admin-gated writes against
    the **auth DB** are inert. Admin-gated writes against the **data DB** via
    ``app.state.kb`` (e.g. ``kb_store`` and other kb-core entry mutations) are
    *live* — no-auth mode does not gate the data DB.
    """
    return User(
        id="local",
        email="local@localhost",
        is_admin=True,
        hashed_password="",
        created_at=datetime.now(UTC),
    )


def generate_api_key() -> str:
    """Generate a new API key with identifiable prefix."""
    return "kb_" + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    """Hash an API key with SHA-256 for fast lookup."""
    return hashlib.sha256(key.encode()).hexdigest()


def hash_password(password: str) -> str:
    """Hash a password with bcrypt."""
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its bcrypt hash."""
    return _bcrypt.checkpw(password.encode(), hashed.encode())


def create_token(user_id: str) -> str:
    """Create a JWT token for the given user ID."""
    payload = {
        "sub": user_id,
        "exp": datetime.now(UTC) + timedelta(hours=TOKEN_EXPIRY_HOURS),
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> str:
    """Decode a JWT and return the user ID.

    Raises:
        HTTPException: If the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        ) from None
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        ) from None


async def _authenticate_api_key(token: str) -> User:
    """Try to authenticate via API key. Returns User or raises 401."""
    db = await get_db()
    h = hash_api_key(token)
    row = await db.fetchrow("SELECT user_id FROM api_keys WHERE key_hash = $1", h)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    user_row = await db.fetchrow("SELECT * FROM users WHERE id = $1", row["user_id"])
    if user_row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return User(**row_to_dict(user_row))


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    """FastAPI dependency: authenticate via JWT or API key.

    In ``'none'`` (no-auth) mode this returns the synthetic local admin user
    without any DB access. In ``'jwt'`` mode (the default) a missing credential
    raises 401 and a present credential runs the unchanged JWT-then-API-key
    path.
    """
    if _auth_mode() == "none":
        return _synthetic_user()
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    token = credentials.credentials
    try:
        return await get_current_user_from_token(token)
    except HTTPException:
        return await _authenticate_api_key(token)


async def get_current_user_from_token(token: str) -> User:
    """Validate a JWT token string and return the user.

    Reusable helper for contexts where Bearer auth is unavailable
    (e.g. EventSource query params).

    Raises:
        HTTPException: If the token is invalid or user not found.
    """
    user_id = decode_token(token)
    db = await get_db()
    row = await db.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return User(**row_to_dict(row))


async def get_current_user_sse(
    token: Annotated[str | None, Query()] = None,
) -> User:
    """FastAPI dependency: authenticate SSE stream endpoints via JWT query param.

    EventSource clients cannot set ``Authorization`` headers, so the JWT is
    passed as ``?token=<jwt>`` instead.  Raises 401 when the token is absent,
    invalid, or expired.

    NOTE: API keys do NOT work on stream endpoints — JWT only.  Do NOT add a
    Bearer-header fallback or API-key support to this dependency.

    Args:
        token: JWT passed as a URL query parameter (``?token=...``).

    Returns:
        Authenticated ``User``.

    Raises:
        HTTPException: 401 if token is missing, invalid, or expired.
    """
    if _auth_mode() == "none":
        return _synthetic_user()
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return await get_current_user_from_token(token)


async def require_admin(
    user: Annotated[User, Depends(get_current_user)],
) -> User:
    """FastAPI dependency: require admin privileges.

    Raises:
        HTTPException: 403 if user is not an admin.
    """
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    return user


async def register_user(email: str, password: str) -> User:
    """Create a new user account.

    Raises:
        HTTPException: If the email is already registered.
    """
    db = await get_db()

    row = await db.fetchrow("SELECT id FROM users WHERE email = $1", email)
    if row:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = User(
        id=str(uuid.uuid4()),
        email=email,
        hashed_password=hash_password(password),
        created_at=datetime.now(UTC),
    )

    sql = (
        "INSERT INTO users (id, email, hashed_password, created_at)"
        " VALUES ($1, $2, $3, $4)"
    )
    await db.execute(
        sql,
        user.id,
        user.email,
        user.hashed_password,
        user.created_at.isoformat(),
    )

    return user


async def register_user_with_invite(
    email: str, password: str, invite_token: str
) -> User:
    """Create a new user account using a valid invite token.

    Atomically validates the invite and creates the user inside a single
    transaction so neither the user row nor the invite consumption persists
    if anything fails.

    Raises:
        HTTPException: 400 if the invite token is invalid.
        HTTPException: 410 if the invite token has already been used.
        HTTPException: 409 if the email is already registered.
    """
    hashed = hash_password(password)
    user_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    pool = await get_db()
    async with pool.acquire() as conn, conn.transaction():
        invite_row = await conn.fetchrow(
            "SELECT * FROM invites WHERE token = $1 FOR UPDATE", invite_token
        )
        if invite_row is None:
            raise HTTPException(status_code=400, detail="Invalid invite token")
        if invite_row["used_at"] is not None:
            raise HTTPException(status_code=410, detail="Invite already used")
        existing = await conn.fetchrow("SELECT id FROM users WHERE email = $1", email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )
        await conn.execute(
            "INSERT INTO users (id, email, hashed_password, is_admin, created_at)"
            " VALUES ($1, $2, $3, $4, $5)",
            user_id,
            email,
            hashed,
            0,
            now,
        )
        await conn.execute(
            "UPDATE invites SET used_at = $1, used_by = $2 WHERE token = $3",
            now,
            user_id,
            invite_token,
        )

    return User(
        id=user_id,
        email=email,
        hashed_password=hashed,
        is_admin=False,
        created_at=datetime.fromisoformat(now),
    )


async def consume_password_reset(token: str, new_password: str) -> None:
    """Consume a one-time password-reset token and update the user's password.

    Raises:
        HTTPException: 400 if the token does not exist.
        HTTPException: 410 if the token has already been used or is expired.
    """
    new_hash = hash_password(new_password)
    now = datetime.now(UTC).isoformat()

    pool = await get_db()
    async with pool.acquire() as conn, conn.transaction():
        row = await conn.fetchrow(
            "SELECT * FROM password_resets WHERE token = $1 FOR UPDATE", token
        )
        if row is None:
            raise HTTPException(status_code=400, detail="Invalid reset token")
        if row["used_at"] is not None:
            raise HTTPException(status_code=410, detail="Reset link already used")
        if row["expires_at"] < now:
            raise HTTPException(status_code=410, detail="Reset link has expired")
        await conn.execute(
            "UPDATE users SET hashed_password = $1 WHERE id = $2",
            new_hash,
            str(row["user_id"]),
        )
        await conn.execute(
            "UPDATE password_resets SET used_at = $1 WHERE token = $2",
            now,
            token,
        )


async def authenticate_user(email: str, password: str) -> User:
    """Validate credentials and return the user.

    Raises:
        HTTPException: If credentials are invalid.
    """
    db = await get_db()
    row = await db.fetchrow("SELECT * FROM users WHERE email = $1", email)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    user = User(**row_to_dict(row))
    if not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    return user
