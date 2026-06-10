"""Pydantic models for the KB service: auth, admin, invites, search, and query."""

from datetime import datetime
from typing import Any, Literal

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


# --- KB Read / Meta Schemas (P2) ---


class GetRequest(BaseModel):
    """Request body for ``POST /api/kb/get``."""

    ids: list[str] = Field(min_length=1, max_length=20)


class PointerRotTarget(BaseModel):
    """A single rotted pointer target found in a mental-map entry."""

    target_id: str
    superseded_by: str | None


class GetEntryResult(BaseModel):
    """Single result slot in a ``GetResponse``, order-preserving."""

    id: str
    found: bool
    entry: KnowledgeEntry | None
    pointer_rot: list[PointerRotTarget]


class GetResponse(BaseModel):
    """Response for ``POST /api/kb/get``."""

    results: list[GetEntryResult]


class GraphNeighbor(BaseModel):
    """One neighbor in a graph-neighbors response."""

    neighbor_id: str
    edge_type: str
    direction: str


class GraphNeighborsResponse(BaseModel):
    """Response for ``GET /api/kb/graph/neighbors``."""

    neighbors: list[GraphNeighbor]


class GraphBfsEntry(BaseModel):
    """One entry in a BFS traversal response."""

    entry_id: str
    depth: int
    path: list[str]


class GraphBfsResponse(BaseModel):
    """Response for ``GET /api/kb/graph/bfs``."""

    entries: list[GraphBfsEntry]


class GraphPathHop(BaseModel):
    """One hop in a graph path response."""

    source: str
    edge_type: str
    target: str


class GraphPathResponse(BaseModel):
    """Response for ``GET /api/kb/graph/path``."""

    found: bool
    hops: list[GraphPathHop]


class SupersedesChainResponse(BaseModel):
    """Response for ``GET /api/kb/graph/supersedes-chain``."""

    chain: list[str]


class ScopeEntriesResponse(BaseModel):
    """Response for ``GET /api/kb/graph/scope-entries``."""

    entry_ids: list[str]


class GraphVocabularyResponse(BaseModel):
    """Response for ``GET /api/kb/graph/vocabulary``."""

    nodes: dict[str, list[str]]


class PreflightResponse(BaseModel):
    """Response for ``GET /api/kb/preflight``."""

    project_ref: str
    context: str


class KbListItem(BaseModel):
    """One row in a KB list response (project, contributor, or team)."""

    name: str
    entry_count: int


class KbListResponse(BaseModel):
    """Response for ``GET /api/kb/projects``, ``/contributors``, and ``/teams``."""

    items: list[KbListItem]


# --- KB Ingest Schemas ---


class IngestTextRequest(BaseModel):
    """Request body for ``POST /api/kb/ingest/text``."""

    content: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    project_ref: str | None = None
    dry_run: bool = False


class IngestUrlRequest(BaseModel):
    """Request body for ``POST /api/kb/ingest/url``.

    When ``content`` is ``None``, the endpoint fetches and extracts the URL.
    When ``content`` is provided, the pre-fetched text is used directly (useful
    for authenticated/SSO/JS-rendered pages that the service cannot fetch).
    """

    url: str = Field(min_length=1)
    content: str | None = None
    project_ref: str | None = None
    dry_run: bool = False


# --- Graph Full Visualisation Schemas (P3) ---


class GraphFullNode(BaseModel):
    """A single node in the full-graph visualisation dump."""

    id: str
    label: str
    type: str
    val: int
    properties: dict[str, Any]


class GraphFullEdge(BaseModel):
    """A single edge in the full-graph visualisation dump."""

    source: str
    target: str
    type: str
    properties: dict[str, Any]


class GraphFullStats(BaseModel):
    """Summary statistics for the full-graph dump."""

    node_count: int
    edge_count: int


class GraphFullResponse(BaseModel):
    """Response for ``GET /api/kb/graph/full``."""

    nodes: list[GraphFullNode]
    edges: list[GraphFullEdge]
    stats: GraphFullStats


# --- SSE Query Stream Schemas (P3) ---


class QueryStreamRequest(BaseModel):
    """Request body for ``POST /api/kb/query/stream``.

    ``question`` is required (no default).  Sending ``{}`` returns 422.
    This is an intentional divergence from the old explorer's
    ``body.get('question', '')`` default — empty questions are not useful.
    """

    question: str


# --- KB Ingest Schemas ---


# --- Chat Schemas ---


class ChatStreamRequest(BaseModel):
    """Request body for ``POST /api/chat/stream``.

    All fields have defaults so the body can be omitted entirely (empty object
    is valid — the route validates the ``token`` query param separately).
    """

    session_id: str | None = None
    message: str = ""
    seed_question: str | None = None
    seed_answer: str | None = None
    seed_entry_ids: list[str] = []
    mode: str = ""


class ChatCreateRequest(BaseModel):
    """Request body for ``POST /api/chat/create``."""

    chat_id: str
    question: str
    answer: str = ""
    mode: str = ""


class ChatOkResponse(BaseModel):
    """Generic ok response for chat mutating endpoints."""

    ok: bool
    id: str | None = None


class ChatListItem(BaseModel):
    """One item in the chat history list."""

    id: str
    title: str
    mode: str
    updated_at: str


class ChatMessageItem(BaseModel):
    """One message in a chat thread."""

    role: str
    content: str


# --- Listener Schemas ---


class ListenerRequest(BaseModel):
    """Request body for ``POST /api/kb/listener``.

    ``text`` is head-truncated to 4000 chars by the hook before sending;
    the service accepts whatever arrives.  ``operating`` lists the MCP-server
    labels that are currently active in the caller's session.  ``source_label``
    is used ONLY for the ``{source}`` substitution in the gate prompt; it falls
    back to ``cwd_project`` then ``'unknown'`` when omitted.
    """

    text: str = Field(min_length=1)
    cwd_project: str | None = None
    operating: list[str] = Field(default_factory=list)
    source_label: str | None = None


class ListenerPointer(BaseModel):
    """A KB entry pointer returned by the listener gate."""

    id: str
    short_title: str


class ListenerResponse(BaseModel):
    """Response for ``POST /api/kb/listener``."""

    pointer: ListenerPointer | None


class IngestFileResult(BaseModel):
    """Lossless mirror of kb-core's ``FileResult`` dataclass for P5 round-trip.

    Field-for-field copy of ``kb_core.ingest.ingester.FileResult`` (source of
    truth: ingester.py:123-135).  Converted via ``dataclasses.asdict(result)``
    in the route — ``FileResult`` is NOT imported into this module.
    """

    path: str
    action: Literal["ingested", "skipped", "flagged", "error", "unchanged", "dry_run"]
    reason: str | None = None
    entry_count: int = 0
    entry_ids: list[str] = []
    summary: str | None = None
    chunks_processed: int = 0
    chunks_skipped: int = 0
    chunks_flagged: int = 0
