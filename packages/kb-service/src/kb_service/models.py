"""Pydantic models for the KB service: auth, admin, invites, search, and query."""

from datetime import datetime

from kb_core.models.entry import KnowledgeEntry
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


# --- Maps Index Schemas ---


class MapRef(BaseModel):
    """A single mental-map entry reference."""

    id: str
    short_title: str
    long_title: str


class ProjectMaps(BaseModel):
    """Maps for a single project.

    Lossless round-trip contract: ``ProjectMaps.model_dump()`` equals exactly
    one legacy JSONL index record, whose schema is:
    ``{"project_ref": "<str>", "maps": [{"id": "<entry id>",
    "short_title": "<str>", "long_title": "<str>"}, ...]}`` — one such JSON
    object per line of maps_index.<role>.jsonl.  The P5 thin client/hook will
    re-render local files from this response.
    """

    project_ref: str
    maps: list[MapRef]


class MapsIndexResponse(BaseModel):
    """Response for ``GET /api/kb/maps-index``."""

    projects: list[ProjectMaps]


# --- KB Query Schemas ---


class AskRequest(BaseModel):
    """Client-safe parameters for ``POST /api/kb/ask``.

    Scope syntax: ``project:X``, ``tag:Y``, an entry ID (``kb-XXXXX``), or a
    graph node ID.  Agentic knobs (``agentic``, ``max_tool_calls``) are
    deliberately NOT exposed — they are governed by server env config.
    """

    question: str
    scope: str | None = None
    include_graph_context: bool = True
    limit: int = Field(default=20, ge=1, le=50)


class AskEntry(BaseModel):
    """A single entry returned by the ask endpoint with its context string."""

    entry: KnowledgeEntry
    context: str


class AskResponse(BaseModel):
    """Response for ``POST /api/kb/ask``."""

    entries: list[AskEntry]
    agent_turns_used: int


class SummarizeRequest(BaseModel):
    """Client-safe parameters for ``POST /api/kb/summarize``.

    Scope syntax: ``project:X``, ``tag:Y``, an entry ID (``kb-XXXXX``), or a
    graph node ID.  Agentic knobs are governed by server env config only.
    """

    question: str
    scope: str | None = None
    limit: int = Field(default=20, ge=1, le=50)


class SummarizeResponse(BaseModel):
    """Response for ``POST /api/kb/summarize``."""

    answer: str


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
