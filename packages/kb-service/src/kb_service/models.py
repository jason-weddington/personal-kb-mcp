"""Pydantic models for the KB service: auth, admin, invites, and search."""

from datetime import datetime

from kb_core.models.search import SearchResult
from pydantic import BaseModel, Field

# --- Auth / User Schemas ---


class User(BaseModel):
    """App user account."""

    id: str
    email: str
    hashed_password: str
    is_admin: bool = False
    created_at: datetime


class RegisterRequest(BaseModel):
    """Account registration request (requires a valid invite token)."""

    email: str
    password: str
    invite_token: str


class LoginRequest(BaseModel):
    """Login request."""

    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    """Change own password (authenticated)."""

    current_password: str
    new_password: str


class AuthResponse(BaseModel):
    """Auth response with JWT token."""

    token: str
    user: "UserResponse"


class UserResponse(BaseModel):
    """Public user info (no password hash)."""

    id: str
    email: str
    is_admin: bool
    created_at: datetime


# --- Invite Schemas ---


class CreateInviteRequest(BaseModel):
    """Create a new invite token (admin only)."""

    note: str = ""


class InviteResponse(BaseModel):
    """Invite creation response with one-time-use token."""

    token: str
    url: str
    note: str
    created_at: datetime


class InviteListItem(BaseModel):
    """Single invite entry in the admin invite list."""

    token: str
    issued_by: str
    note: str
    created_at: datetime
    used_at: datetime | None
    used_by: str | None


# --- Password Reset Schemas ---


class PasswordResetIssueResponse(BaseModel):
    """Admin response when issuing a one-time password-reset link."""

    token: str
    url: str
    expires_at: datetime


class PasswordResetConsumeRequest(BaseModel):
    """Public request body for consuming a password-reset token."""

    token: str
    new_password: str


# --- API Key Schemas ---


class CreateApiKeyRequest(BaseModel):
    """Create a new API key."""

    name: str = ""


class ApiKeyResponse(BaseModel):
    """API key creation response — plaintext key shown only once."""

    api_key: str
    name: str


class ApiKeyInfo(BaseModel):
    """API key metadata (no secret material)."""

    id: str
    name: str
    hash_prefix: str
    created_at: datetime


class ApiKeyListResponse(BaseModel):
    """List of API keys for the current user."""

    keys: list[ApiKeyInfo]


# --- KB Search Schemas ---


class SearchRequest(BaseModel):
    """Client-safe search parameters for ``POST /api/kb/search``.

    Deliberately does NOT expose ``contributor`` or ``team`` — scoping is a
    server-side concern (physical DB isolation), never trusted from the client.
    """

    query: str = ""
    project_ref: str | None = None
    entry_type: str | None = None
    tags: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=50)
    include_stale: bool = False
    include_expired: bool = False


class SearchResponse(BaseModel):
    """Response for ``POST /api/kb/search``."""

    results: list[SearchResult]
    filtered_count: int


# --- App-Config / Settings Schemas ---


class SettingsResponse(BaseModel):
    """Response body for GET and PUT /api/settings."""

    team: str | None


class UpdateSettingsRequest(BaseModel):
    """Request body for PUT /api/settings.

    All fields default to None.  An omitted field is treated identically to an
    explicit null — PUT is full-replace, not partial-patch.
    """

    team: str | None = None
