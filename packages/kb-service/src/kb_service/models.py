"""Pydantic models for the KB service: auth, admin, invites, search, and query."""

from datetime import datetime
from typing import Annotated, Any, Literal

from kb_core.models.entry import KnowledgeEntry
from kb_core.models.search import SearchResult
from pydantic import BaseModel, Field, field_validator, model_validator

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


# --- Runtime / Meta Schemas ---


class RuntimeResponse(BaseModel):
    """Response for ``GET /api/kb/runtime`` — the active auth mode.

    FROZEN cross-item contract: the body is exactly ``{"auth": "none"|"jwt"}``.
    The frontend SPA item branches on these two string literals; do NOT rename
    or add fields without updating that item.
    """

    auth: Literal["none", "jwt"]


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
    """A single mental-map entry reference.

    ``pointers`` is the list of ``kb-XXXXX`` ids the map's body mentions —
    the DETAIL entries the map points to. Empty by default so pre-pointer
    clients (and any pre-pointer server output that lacks the field) round-
    trip cleanly. Consumed by the personal-kb-hook's whisper telemetry to
    chain-credit map rows when the model fetches one of a map's pointed-to
    detail entries rather than the map itself (see GTD 88441f9c).
    """

    id: str
    short_title: str
    long_title: str
    pointers: list[str] = []


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
    back to ``cwd_project`` then ``'unknown'`` when omitted. ``session_id`` was
    already present on the hook-side wire body (used for the debug excerpt-hash
    cache key) but had no field here, so pydantic silently dropped it; it is
    now captured so the ``listener_decisions`` telemetry row can be attributed
    to a session.
    """

    text: str = Field(min_length=1)
    cwd_project: str | None = None
    operating: list[str] = Field(default_factory=list)
    source_label: str | None = None
    session_id: str | None = None


class ListenerPointer(BaseModel):
    """A KB entry pointer returned by the listener gate."""

    id: str
    short_title: str


# Pinned literal (GTD 66ea1fe4): hard cap on how many map pointers a single
# listener response may carry. A map is a SUBJECT AREA — plural evidence can
# legitimately implicate more than one — but the whisper is a one-line
# surface, and 3+ titles recreates the roster-noise problem this line of
# work exists to fix. Kept here (not in listener_routes.py) so the response
# model and the route agree on the same constant without a circular import.
MAX_POINTERS_PER_RESPONSE = 2


class ListenerResponse(BaseModel):
    """Response for ``POST /api/kb/listener``.

    ``reason`` is an additive, backward-compatible debug field surfaced
    from the gate stages the handler already computes (kill-switch /
    no-retrieval / rule-A-emptied / rule-B-emptied / LLM-unavailable /
    unanimous match / non-unanimous). It is always present in the
    serialized body but defaulted so existing callers that build
    ``ListenerResponse(pointer=...)`` keep working unchanged.

    ``pointers`` (GTD 66ea1fe4) is the widened output contract: a LIST of
    0..``MAX_POINTERS_PER_RESPONSE`` pointers, in evidence-rank order (best
    evidence first). ``pointer`` is now a DEPRECATED single-pointer alias —
    kept because ``personal_kb_hook.listener_worker._post_one_kb`` reads
    ``data["pointer"]`` directly off the raw JSON body (grepped both repos:
    that is the one production reader; see GTD 66ea1fe4 final comment) — it
    is populated with ``pointers[0]`` when non-empty, else ``None``. New
    callers should read ``pointers``; ``pointer`` may be removed once every
    reader has migrated.
    """

    pointer: ListenerPointer | None = None
    pointers: list[ListenerPointer] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def _populate_deprecated_alias(self) -> "ListenerResponse":
        """Keep ``pointer`` in sync with ``pointers[0]`` when not set explicitly.

        Construction sites in this codebase build ``ListenerResponse`` via
        ``pointers=[...]`` only; this validator derives the deprecated
        ``pointer`` alias from it so callers never have to set both. If a
        caller DOES pass an explicit non-None ``pointer``, it is left
        untouched (defensive — no current caller does this).
        """
        if self.pointer is None and self.pointers:
            self.pointer = self.pointers[0]
        return self


# Closed sets for the ``listener_decisions`` telemetry sink (see
# ``kb_service.database`` schema + ``listener_routes._record_listener_decision``).
# ``decision`` is exactly one of these two values.
ListenerDecisionOutcome = Literal["whisper", "declined"]

# ``reason`` enumerates every actual return branch in ``listener_routes.listener()``
# — one member per branch, no catch-all. Keep in 1:1 sync with that function.
#
# ``fallback-direct`` is the one exception to "one member per branch": when
# the detail-matching retrieval path yields zero candidate maps, the route
# falls back to a direct mental_map search (today's pre-fix behavior). ANY
# terminal decision (whisper or decline) reached via that fallback path is
# recorded with reason="fallback-direct" INSTEAD OF the granular branch
# reason below, so the fallback is distinguishable in telemetry even though
# it costs per-branch granularity for that subset of requests. ``decision``
# (whisper/declined) is still derived correctly regardless of the override.
ListenerDecisionReason = Literal[
    "kill-switch",
    "no-candidates",
    "rule-a",
    "rule-b",
    "no-llm",
    "vote-split",
    "vote-none",
    "whispered",
    "fallback-direct",
]

# ``candidate_signal`` (GTD be964e94): records WHICH candidate-retrieval
# signal produced the surfaced candidate on a 'whisper' decision — see the
# ``candidate_signal`` column comment in ``kb_service.database`` for the
# full explanation and the query this makes possible. '' (not-applicable) on
# every decline branch.
ListenerCandidateSignal = Literal["", "lexical", "detail", "fallback"]


class WhisperTelemetryRow(BaseModel):
    """One whisper-efficacy telemetry row (roster shown or listener whisper).

    Mirrors the ``whisper_telemetry`` table.  ``trigger_context`` stays a
    ``dict`` on the wire; the endpoint serializes it to a TEXT json string at
    the DB boundary.  ``consumed`` is a bool on the wire and bound as ``int``
    at the DB boundary.  ``build_engine`` is read defensively from
    ``HEADLESS_BUILD_ENGINE`` on the hook side — ``None`` is the
    interactive/control-plane case.

    ``pointers`` was already sent by the hook on every roster row (the map's
    pointer ids, used hook-side for chain-credit) but had no field here, so
    pydantic's default ``extra='ignore'`` silently dropped it at the wire
    boundary and the server could never recompute attribution. It is now
    captured and folded into ``trigger_context`` (as a JSON key, NOT a new
    column) at the route boundary.
    """

    session_id: str
    host: str
    surface: Literal["roster", "listener"]
    map_id: str
    source_kb: str
    cwd_project: str | None = None
    trigger_context: dict[str, Any] = Field(default_factory=dict)
    emitted_ts: str
    consumed: bool = False
    consumed_ts: str | None = None
    build_engine: str | None = None
    pointers: list[str] | None = None


class WhisperTelemetryFlushRequest(BaseModel):
    """Request body for ``POST /api/kb/telemetry/whispers``.

    One Stop-flush carries one session's accrued roster + listener rows;
    ``max_length=500`` mirrors the literal style used by :class:`GetRequest`.
    """

    rows: list[WhisperTelemetryRow] = Field(default_factory=list, max_length=500)


class WhisperTelemetryFlushResponse(BaseModel):
    """Response for ``POST /api/kb/telemetry/whispers``.

    ``upserted`` reports rows accepted (== ``len(body.rows)``), NOT a
    DB-affected rowcount — the composite-key ON CONFLICT path may update or
    insert per row, but the route reports acceptance.
    """

    upserted: int


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


class EmbeddingQueueStatusResponse(BaseModel):
    """Response for ``GET /api/kb/embedding-queue`` (admin-only).

    Field-for-field mirror of ``kb_core.embedding_retry.EmbeddingQueueStats``
    plus ``worker_enabled``/``worker_running``, which the route composes from
    ``kb_service.config.is_embed_worker_enabled()`` and
    ``KnowledgeBase.embedding_worker_running``.
    """

    pending: int
    exhausted: int
    oldest_pending_age_seconds: float | None
    next_due_at: str | None
    vectorless_unqueued: int
    worker_enabled: bool
    worker_running: bool


# --- Map eligibility (review + override surface) ---


class MapEligibilityEvidenceModel(BaseModel):
    """Lossless mirror of kb-core's ``MapEligibilityEvidence`` dataclass.

    Field-for-field copy of ``kb_core.map_eligibility.MapEligibilityEvidence``
    (source of truth: kb_core/map_eligibility.py).  Converted via
    ``dataclasses.asdict()`` in the route — the dataclass is NOT imported
    into this module.
    """

    project_ref: str
    mappable: int
    ingested: int
    hand_authored: int
    maps: int
    top_prefix: str
    top_prefix_share: float
    is_ingest_corpus: bool
    is_too_thin: bool
    is_journal: bool
    computed_eligible: bool


class MapEligibilityOverrideModel(BaseModel):
    """Lossless mirror of kb-core's ``MapEligibilityOverride`` dataclass.

    Field-for-field copy of ``kb_core.map_eligibility.MapEligibilityOverride``
    (source of truth: kb_core/map_eligibility.py).  Converted via
    ``dataclasses.asdict()`` in the route — the dataclass is NOT imported
    into this module.
    """

    project_ref: str
    eligible: bool
    reason: str
    set_by: str | None
    set_at: str


class MapEligibilityVerdictModel(BaseModel):
    """Lossless mirror of kb-core's ``MapEligibilityVerdict`` dataclass.

    Field-for-field copy of ``kb_core.map_eligibility.MapEligibilityVerdict``
    (source of truth: kb_core/map_eligibility.py).  Converted via
    ``dataclasses.asdict()`` in the route — the dataclass is NOT imported
    into this module.
    """

    evidence: MapEligibilityEvidenceModel
    override: MapEligibilityOverrideModel | None
    effective_eligible: bool
    decided_by: Literal["computed", "override"]
    orphaned: bool


class MapEligibilityResponse(BaseModel):
    """Response for ``GET /api/kb/map-eligibility`` (admin-only).

    Exactly ``{"projects": [...]}`` — every verdict, eligible and
    ineligible alike, in kb-core's own (``project_ref``-ascending) order,
    with ``top_prefix_share`` unrounded.
    """

    projects: list[MapEligibilityVerdictModel]


class MapEligibilityOverrideSetRequest(BaseModel):
    """Request body for ``POST /api/kb/map-eligibility/override``."""

    project_ref: str = Field(..., min_length=1)
    eligible: bool
    reason: str = Field(..., min_length=1, max_length=2000)

    @field_validator("project_ref", "reason")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Reject a blank value and return the stripped form."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class MapEligibilityOverrideClearRequest(BaseModel):
    """Request body for ``POST /api/kb/map-eligibility/override/clear``."""

    project_ref: str = Field(..., min_length=1)

    @field_validator("project_ref")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Reject a blank value and return the stripped form."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class MapEligibilityOverrideResponse(BaseModel):
    """Response for both override POST endpoints.

    ``changed`` is unconditionally ``True`` on the set endpoint and means
    "the upsert executed", NOT "the effective verdict moved" — whether it
    moved is recoverable from the ``was=`` field of the engine's
    ``map_eligibility_override_set`` audit row.  On the clear endpoint it is
    the engine's own return: ``False`` when no override row existed.

    ``verdict`` is the post-write resolved verdict for the ref; it is null
    exactly when the ref is absent from the post-write list, which is
    reachable only on the clear path (clearing an override on a ref with
    zero mappable entries removes it from both the counts query and the
    override table).
    """

    changed: bool
    verdict: MapEligibilityVerdictModel | None


# --- Pointer-candidates (capture-time map nudge, server half) ---


class PointerCandidatesRequest(BaseModel):
    """Request body for ``POST /api/kb/pointer-candidates``."""

    entry_id: str


class PointerCandidate(BaseModel):
    """A single candidate mental_map that might want to point at an entry."""

    map_id: str
    short_title: str
    project_ref: str | None
    distance: float


class PointerCandidatesResponse(BaseModel):
    """Response for ``POST /api/kb/pointer-candidates``.

    ``has_owning_map`` is true when an ACTIVE mental_map already has a
    ``references`` edge targeting the entry — the client uses this to stay
    silent, and ``candidates`` MAY be empty in that case (the route does not
    bother computing candidates once an owning map is found). Every
    degenerate case (no embedding yet, no project_ref, no maps in the
    project) reports ``has_owning_map=False, candidates=[]`` rather than an
    error — the one exception is a genuinely-missing ``entry_id``, which is
    a 404.
    """

    has_owning_map: bool
    candidates: list[PointerCandidate]


# --- Map loop input (nightly loop pre-fetch, server half) ---


PocketsOmittedReason = Literal[
    "too-few-unpointed-entries",
    "unpointed-set-too-large",
    "non-postgres-backend",
]


class MapLoopDirectoryToken(BaseModel):
    """One coarse directory token mined from an entry's FULL details.

    ``token`` is the lint-safe shape the spec's Rung 3 verified against the purity lint.

    One or two segments, no dot, no leading slash or tilde, so each line stays clean.

    ``hits`` counts the original path-like substrings that normalised to the token.

    Counts, not a single modal token: a bare mode re-introduces the guess.

    The compose refusal existed to prevent exactly that guess.

    The counts let the consumer demand a real plurality and refuse otherwise.
    """

    token: str
    hits: int


class MapLoopEntry(BaseModel):
    """One mappable entry as the loop's prompt prefix sees it.

    ``excerpt`` is the RAW first ``excerpt_chars`` characters of
    ``knowledge_details`` — no ellipsis appended, no whitespace stripping,
    no word-boundary adjustment. ``details_length`` is the FULL stored
    length, which is how a consumer detects truncation
    (``details_length > excerpt_chars``). ``entry_type`` is passed through
    as the raw stored string. ``unpointed`` is the SQL anti-join's verdict:
    no ACTIVE mental_map's pointer set currently includes this entry.

    ``directory_tokens`` is Rung 3's per-ENTRY ``Lives in`` source.

    Extraction reads the FULL ``knowledge_details``; the excerpt would miss most paths.

    Per entry, never per cluster — rung 1 invents the clusters after the fetch.

    No cluster exists to key anything by at assembly time.

    Ranked by ``hits`` descending then ``token`` ascending; empty is allowed.

    The loop aggregates across a cluster's members and owns the plurality refusal.
    """

    id: str
    short_title: str
    long_title: str
    entry_type: str
    tags: list[str]
    excerpt: str
    details_length: int
    unpointed: bool
    directory_tokens: list[MapLoopDirectoryToken] = []


class MapLoopMap(BaseModel):
    """One existing active map, with the fields the loop's MATERIALIZER needs.

    Deliberately NOT a reuse of ``MapRef``: ``body``, ``contributor`` and
    ``updated_by`` are load-bearing. ``body`` is the FULL
    ``knowledge_details``, never excerpted, and it is for the MATERIALIZER,
    not for the prompt — Rung 1 injects titles/tags/pointers and withholds
    the body, because withholding it is what makes "the model cannot rewrite
    prose" structural rather than a prompt rule (there is no rewrite_body
    and no write_map). Every automated write must be a read-modify-write
    carrying the full body — we diff the body we wrote against the body we
    read and reject anything but an append — and ``strike_gap`` needs the
    existing "Not yet documented:" text, which lives only in the body.
    ``contributor``/``updated_by`` are what make the authorship tier rule
    (a machine-authored map is the loop's to edit; a human-authored map is
    append-only, permanently) enforceable. The route deliberately does NOT
    compute a machine_authored boolean: the machine principal is service
    config (read from the SERVICE DB) while this payload comes from the
    DATA DB, so somnus derives the tier itself from these two fields
    against its own identity.
    """

    id: str
    short_title: str
    long_title: str
    pointers: list[str] = []
    body: str
    contributor: str | None = None
    updated_by: str | None = None


class MapLoopPocketEdge(BaseModel):
    """One mutual edge inside a pocket, with its raw pgvector similarity."""

    a: str
    b: str
    similarity: float


class MapLoopPocket(BaseModel):
    """One unpointed dense pocket: a candidate member set plus its evidence.

    ``member_entry_ids`` matches the sibling cluster shape's key so no name
    translation happens anywhere in the wave. A pocket object deliberately
    carries NO label, name, title or topic field — naming is the model's
    job and geometry provably cannot do it. ``mean``/``min``/``max``
    similarity are computed over the pocket's MUTUAL EDGES only, never over
    all member pairs. Every similarity crosses the wire unrounded.
    """

    member_entry_ids: list[str]
    mean_similarity: float
    min_similarity: float
    max_similarity: float
    edges: list[MapLoopPocketEdge]


class MapLoopInputResponse(BaseModel):
    """Response for ``GET /api/kb/map-loop-input?project_ref=<ref>``.

    ``pockets_omitted_reason`` distinguishes "no pockets found" from
    "pockets not computed": ``None`` means the pair statement ran and
    ``pockets`` is the computed result (possibly ``[]`` when no mutual
    component reached the minimum size); the three literal values each mean
    the pair statement was skipped and ``pockets`` is empty.
    """

    project_ref: str
    excerpt_chars: int
    entries: list[MapLoopEntry]
    maps: list[MapLoopMap]
    pockets: list[MapLoopPocket]
    pockets_omitted_reason: PocketsOmittedReason | None = None


# --- Map worklist (nightly loop's ranked eligible-project enumeration) ---


class MapWorklistProject(BaseModel):
    """One ranked worklist row: a project somnus's ``nightly`` may work.

    ``mappable`` is the eligibility verdict's own count of mappable entries,
    ``map_count`` and ``latest_map_written_at`` come from
    ``kb_core.map_caps.map_write_summary`` — a project with no active maps is
    ABSENT from that summary, so the route left-joins in Python and renders
    ``map_count=0`` with a null timestamp rather than dropping the row: a
    never-mapped project is the loop's FIRST priority, not a missing one.
    ``latest_map_written_at`` serialises as an ISO-8601 string and is null
    exactly when ``map_count`` is 0.
    """

    project_ref: str
    mappable: int
    map_count: int
    latest_map_written_at: datetime | None


class MapWorklistResponse(BaseModel):
    """Response for ``GET /api/kb/map-worklist``.

    Exactly ``{"projects": [...]}`` — every ELIGIBLE project, ranked, with NO
    limit and NO truncation: the three-projects-per-night cap is somnus's
    worklist policy, and a server-side limit would silently hide projects
    from any other reader of this list. Empty (no eligible projects) is a
    legitimate quiet night and renders ``{"projects": []}``, never a 404.
    """

    projects: list[MapWorklistProject]


# --- Cluster/decline ledger (nightly loop's only durable state, server half) ---


class ClusterLedgerMatchCandidate(BaseModel):
    """One batch candidate: a proposed cluster's member-id list.

    The list must be non-empty — kb-core raises ``ValueError`` on an empty
    member set, and a 500 is never the honest answer to bad input. There is
    deliberately NO similarity/threshold field here: the overlap constant
    lives in kb-core and nowhere else.
    """

    member_entry_ids: list[str] = Field(..., min_length=1)


class ClusterLedgerMatchRequest(BaseModel):
    """Request body for ``POST /api/kb/cluster-ledger/match``.

    ``candidates`` is a BATCH — a project yields N pockets and N round
    trips at 3am is waste. ``index`` in the response maps results back to
    positions in this array.
    """

    project_ref: str = Field(..., min_length=1)
    candidates: list[ClusterLedgerMatchCandidate] = []

    @field_validator("project_ref")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Reject a blank value and return the stripped form."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class ClusterLedgerMatchItem(BaseModel):
    """One batch candidate's verdict, positionally aligned by ``index``.

    ``ledger_id`` is kb-core's ``cluster_key`` (the row's identity token)
    and is null exactly when ``matched`` is false. ``status`` is the
    matched row's ledger status ("proposed" | "declined"); an unmatched
    candidate carries "none" — the pinned contract makes ``status`` a
    plain string, so the no-match case needs a member, and "none" is the
    value of record. ``jaccard`` is the facade's own unrounded float (it
    is the max overlap over ALL rows, which MAY exceed the overlap with
    the matched row when a lower-scoring declined row was preferred —
    see kb-core ``_best_match``). ``reopen_eligible`` is the
    member-set-doubling escape verdict computed server-side against the
    stored member set: a decline is permanent with exactly one escape.
    """

    index: int
    matched: bool
    ledger_id: str | None
    status: str
    jaccard: float | None
    reopen_eligible: bool


class ClusterLedgerMatchResponse(BaseModel):
    """Response for ``POST /api/kb/cluster-ledger/match``.

    Exactly ``{"results": [...]}`` — one item per request candidate, in
    request order. An empty batch returns ``{"results": []}`` and 200.
    """

    results: list[ClusterLedgerMatchItem]


class ClusterLedgerDeclineRequest(BaseModel):
    """Request body for ``POST /api/kb/cluster-ledger/decline``.

    ``reason`` is non-empty after stripping — the reason IS the audit
    trail, and an empty one makes the ledger row useless. There is no
    ``cluster_key`` field: identity is member-set overlap and the key is a
    server-side birth hash, so the caller submits the member set and the
    server resolves the row.
    """

    project_ref: str = Field(..., min_length=1)
    member_entry_ids: list[str] = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)

    @field_validator("project_ref", "reason")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Reject a blank value and return the stripped form."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class ClusterLedgerDeclineResponse(BaseModel):
    """Response for ``POST /api/kb/cluster-ledger/decline`` (201).

    ``ledger_id`` is the declined row's ``cluster_key`` — the id a later
    ``clear_cluster`` (admin surface) or audit lookup needs.
    """

    ledger_id: str


# --- Map-op (the nightly loop's map write path, machine principal only) ---


class MapOpCreateMapRequest(BaseModel):
    """``create_map`` — store a brand-new mental_map for one subject area.

    ``body`` is the WHOLE composed map body, exactly as somnus's Rung 3 rendered it.

    The server never parses it; it only counts and compares the ``kb-`` refs inside.
    """

    op: Literal["create_map"]
    project_ref: str = Field(..., min_length=1)
    short_title: str = Field(..., min_length=1)
    long_title: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


class MapOpAddPointerRequest(BaseModel):
    """``add_pointer`` — append exactly one pointer to an existing map.

    ``added_entry_id`` is the single id the loop claims to have added.

    The server verifies that claim against the two ref sets; it is never trusted.

    ``base_version`` is the optional cheap guard against a run racing the timer.

    Supplied, it must equal the stored entry's ``version`` or the op is a 409.
    """

    op: Literal["add_pointer"]
    map_id: str = Field(..., min_length=1)
    added_entry_id: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    base_version: int | None = None


class MapOpStrikeGapRequest(BaseModel):
    """``strike_gap`` — remove one already-recorded gap from a map's body.

    ``closing_entry_id`` is the entry that closed the gap.

    It must already be a pointer of the stored map: the spec requires citing it,
    and the entry that closed a gap is by definition already pointed at.
    """

    op: Literal["strike_gap"]
    map_id: str = Field(..., min_length=1)
    gap_text: str = Field(..., min_length=1)
    closing_entry_id: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    base_version: int | None = None


class MapOpProposeGapRequest(BaseModel):
    """``propose_gap`` — record a new gap in a map's body.

    Deliberately terminal for the loop: a proposed gap is ``blocked`` evidence.

    Nothing on this path can fix the missing authoring run upstream.
    """

    op: Literal["propose_gap"]
    map_id: str = Field(..., min_length=1)
    gap_text: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    base_version: int | None = None


# The closed op vocabulary of POST /api/kb/map-op, discriminated on ``op``.
# ``no_change`` is deliberately NOT a member, because it never reaches HTTP.
# A request carrying it fails Pydantic validation with 422, never a silent no-op.
# There is no ``rewrite_body`` and no ``write_map``.
MapOpRequest = Annotated[
    MapOpCreateMapRequest
    | MapOpAddPointerRequest
    | MapOpStrikeGapRequest
    | MapOpProposeGapRequest,
    Field(discriminator="op"),
]


class MapOpResponse(BaseModel):
    """The one uniform success envelope for all four ops.

    ``pointer_count`` and ``budget`` are kb-core's own functions over the
    SUBMITTED body, so the loop gets back the numbers the lint already uses.

    ``version`` is the stored entry's version after the write.
    """

    map_id: str
    version: int
    pointer_count: int
    budget: int
