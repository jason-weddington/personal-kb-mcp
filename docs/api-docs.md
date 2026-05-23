# personal-kb — MCP Tool Reference

## Overview

personal-kb exposes its functionality as MCP (Model Context Protocol) tools via a FastMCP
server with stdio transport. There is no REST API or HTTP authentication — all access is through
MCP tool calls from an authorized MCP client (e.g., Claude Code).

All tools are asynchronous. Tool names are optionally prefixed based on `KB_INSTANCE_ROLE`:

| KB_INSTANCE_ROLE | Tool name prefix | Example |
|---|---|---|
| (unset) | none | `kb_search` |
| `personal` | `personal_kb_` | `personal_kb_search` |
| `team` | `team_kb_` | `team_kb_search` |

Two tools are gated behind `KB_MANAGER=TRUE`: `kb_maintain` and `kb_bulk_update`. They are
not registered unless the server is started with that env var set.

## Authentication

The MCP server uses stdio transport. The client (Claude Code or compatible agent runtime)
spawns the server process. No API key or HTTP authentication is required — access control is
at the process level (who can spawn the server).

For team deployments using the PostgreSQL backend, database credentials are in `KB_DATABASE_URL`
or injected via `KB_PG_IAM_AUTH=TRUE` (AWS RDS IAM authentication via boto3).

## API Endpoints (MCP Tools)

---

### `kb_search`

**Purpose**: Quick keyword and filter-based lookup. Returns compact summaries — use `kb_get`
to retrieve full content.

**When to use**: Duplicate checking, finding entries by keyword, filtering by tags/project/type.
Not for exploratory queries — use `kb_ask` for those.

**Parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Search text (BM25 + vector) |
| `project_ref` | `str \| None` | `None` | Scope to a specific project |
| `entry_type` | `str \| None` | `None` | Filter: factual_reference / decision / pattern_convention / lesson_learned |
| `tags` | `list[str] \| None` | `None` | All tags must match (AND logic) |
| `contributor` | `str \| None` | `None` | Filter by contributor name |
| `team` | `str \| None` | `None` | Filter by team name |
| `limit` | `int` | `10` | Max results (1–50) |
| `include_stale` | `bool` | `False` | Include entries with effective_confidence < 0.5 |
| `include_expired` | `bool` | `False` | Include entries past expires_at |

**Returns**: Compact formatted table with entry ID, type badge, short title, long title, score,
and effective confidence. Staleness warnings shown inline. If sparse results, graph hints may
be included.

---

### `kb_get`

**Purpose**: Retrieve full entry content by ID(s). Updates `last_accessed` (resets decay clock).

**When to use**: After `kb_search` or `kb_preflight` identifies interesting entries.

**Parameters**:

| Parameter | Type | Description |
|---|---|---|
| `entry_id` | `str \| list[str]` | Single ID or list of IDs (max 20) |

**Returns**: Full entry content including all fields, confidence, staleness warning, tags, hints,
and graph neighbors (two-phase retrieval: fetch entry then fetch neighbors).

---

### `kb_store`

**Purpose**: Create a new entry or update an existing one.

**Parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `short_title` | `str` | required | Brief identifier |
| `long_title` | `str` | required | Descriptive title |
| `knowledge_details` | `str` | required | Full content |
| `entry_type` | `str` | required | One of the four EntryType values |
| `project_ref` | `str \| None` | `None` | Project scope |
| `source_context` | `str \| None` | `None` | Provenance (URL, file, session description) |
| `confidence_level` | `float` | `0.9` | Initial confidence 0.0–1.0 |
| `tags` | `list[str]` | `[]` | Categorization tags |
| `hints` | `dict` | `{}` | Graph hints (supersedes, person, tool, related_entities) |
| `ttl` | `str \| None` | `None` | Expiry duration: "7d", "24h", "2w" |
| `sensitivity` | `str \| None` | `None` | internal / restricted / public |
| `update_entry_id` | `str \| None` | `None` | Update this entry (partial update; omitted fields unchanged) |
| `deactivate_entry_id` | `str \| None` | `None` | Soft-delete this entry alongside the create/update |
| `change_reason` | `str \| None` | `None` | Reason for update (stored in audit log) |

**Notable behavior**:
- Secret scanning runs on `knowledge_details` before storing. Entries with detected secrets are
  rejected unless `KB_SKIP_SAFETY=TRUE`.
- When `update_entry_id` is set, only the provided fields are updated. Fields not passed are
  preserved. `entry_type` defaults to `None` to prevent accidental type overwrite.
- `deactivate_entry_id` and the create/update happen in the same call — atomically in intent
  (non-atomic on Postgres until M4 is resolved).
- Graph enrichment (entity extraction) runs after storing if an extraction LLM is available.

**Returns**: Formatted confirmation with the new/updated entry ID, type, and title.

---

### `kb_store_batch`

**Purpose**: Create multiple entries in a single call. More efficient than multiple `kb_store`
calls because graph enrichment uses a single LLM call for all entries.

**Parameters**:

| Parameter | Type | Description |
|---|---|---|
| `entries` | `list[dict]` | List of entry dicts (max 10). Each dict accepts all `kb_store` parameters. |

**Returns**: Per-entry results showing success or error. Partial failures are reported inline —
successful entries are committed even if some fail.

---

### `kb_ask`

**Purpose**: Explore related knowledge via graph traversal. Returns full entry details. Use
when you need to discover connections, trace decision history, or find everything about a topic.

**Parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `question` | `str` | required | Natural-language question |
| `strategy` | `str` | `"auto"` | One of: auto, decision_trace, timeline, related, connection |
| `scope` | `str \| None` | `None` | e.g., `"project:personal-kb"`, `"tag:architecture"` |
| `target` | `str \| None` | `None` | Target entry ID for connection strategy |
| `include_graph_context` | `bool` | `True` | Include graph neighbor context |
| `limit` | `int` | `10` | Max results |

**Strategies**:

| Strategy | Description |
|---|---|
| `auto` | Hybrid search + optional ReAct agent loop (fast-path if top score > 0.030) |
| `decision_trace` | Follows supersedes chain for a specific decision topic |
| `timeline` | Orders results by creation date — shows how knowledge evolved |
| `related` | BFS graph traversal from a seed concept or entry |
| `connection` | Finds path between two entries in the knowledge graph |

**Returns**: Full entry details with graph context, turns used (for agentic strategies),
and reasoning trace.

---

### `kb_summarize`

**Purpose**: Get a synthesized natural-language answer with `[kb-XXXXX]` citations.

**When to use**: When you need to answer a user question directly from the KB, not when
you need raw entry data.

**Parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `question` | `str` | required | Question to answer |
| `scope` | `str \| None` | `None` | Scope filter (project:X, tag:Y) |
| `limit` | `int` | `10` | Max entries to retrieve for synthesis |

**Notable behavior**: Uses the agentic retrieval loop (when `KB_AGENTIC_SYNTHESIS=TRUE`) plus
a coverage check that fills gaps with additional searches. Synthesis uses a stronger Sonnet 4.6
model (not Haiku) when available, for better prose quality. Disable with `KB_AGENTIC_SYNTHESIS=FALSE`.

**Returns**: Prose answer with inline KB entry citations (`[kb-XXXXX]`).

---

### `kb_preflight`

**Purpose**: Get a project context primer at session start. Pushes relevant context before the
agent starts thinking — addresses unknown unknowns that can't be found via search.

**Parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `question` | `str` | required | Current task or user message (used for scoping) |
| `scope` | `str \| None` | `None` | Explicit project scope override (default: CWD-inferred) |

**Returns**: Compact table of: expiring-soon entries (7-day grace window, max 3), recent
decisions + lessons (last 14 days, max 5), active conventions (confidence ≥ 0.7, max 3).
Format: entry ID + type + short title + age/expiry badge. No full content — use `kb_get` on
entries of interest.

---

### `kb_ingest`

**Purpose**: Intelligent extraction from local files. The LLM reads source content and creates
multiple properly structured KB entries. Deduplicates against existing entries.

**Parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | required | File path, directory, or glob pattern (e.g., `docs/**/*.md`) |
| `project_ref` | `str \| None` | `None` | Assign all extracted entries to this project |
| `dry_run` | `bool` | `False` | Preview what would be extracted without storing |
| `recursive` | `bool` | `True` | Recurse into directories |

**Notable behavior**:
- Extension allowlist: 40+ types including `.md`, `.py`, `.js`, `.ts`, `.yaml`, `.json`,
  `.pdf`, `.html`, and special names (Dockerfile, Makefile, README, CHANGELOG, LICENSE).
- Files over 10MB (configurable via `KB_INGEST_MAX_FILE_SIZE`) are rejected.
- Symlinks are rejected.
- Chunks processed sequentially; secret scanning runs per-chunk before LLM extraction.
- Dedup threshold configurable via `KB_INGEST_DEDUP_THRESHOLD` (default 0.06 RRF score).
- No base-directory restriction (open issue M1 — see ROADMAP.md).

**Returns**: Summary: total files, ingested, skipped (duplicate/flagged), errors, entries created.

---

### `kb_ingest_url`

**Purpose**: Fetch a URL, extract article content from HTML, and ingest it. Handles boilerplate
removal automatically via `trafilatura`. Use when the source is a webpage, documentation site,
or blog post.

**Parameters**:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | required | URL to fetch |
| `content` | `str \| None` | `None` | Pre-fetched content (skips HTTP fetch). Use for authenticated pages. |
| `project_ref` | `str \| None` | `None` | Project scope for extracted entries |
| `dry_run` | `bool` | `False` | Preview without storing |

**Returns**: Same summary format as `kb_ingest`.

---

### `kb_feedback`

**Purpose**: Report KB quality issues. Always available (not manager-gated). Negative feedback
only — for positive signals, the search telemetry (automatic) handles that.

**Parameters**:

| Parameter | Type | Description |
|---|---|---|
| `feedback_type` | `str` | `missing` / `unhelpful` / `friction` |
| `tool_name` | `str` | Which tool triggered the feedback |
| `query_or_params` | `str` | What was searched or called |
| `detail` | `str` | Agent-provided explanation of the issue |

**Feedback types**:
- `missing`: KB lacked information that should be there
- `unhelpful`: Results existed but didn't answer the question
- `friction`: Tool was awkward, slow, or produced confusing output

**Returns**: Confirmation string.

---

### `kb_explore`

**Purpose**: Open the interactive graph explorer web UI.

**Parameters**: None.

**Returns**: URL to the explorer (`http://localhost:8765` by default, configurable via
`KB_EXPLORE_PORT`). The server auto-starts if not already running (`KB_AUTO_EXPLORE=TRUE`).

The explorer provides: force-directed graph visualization, natural-language chat with citations,
write-back tools (update entries, ingest URLs, upload files), and project filtering.

---

### `kb_list_projects`

**Purpose**: List all known project_ref values with entry counts.

**Parameters**: None.

**Returns**: Table of project_ref → entry count.

---

### `kb_list_contributors`

**Purpose**: List contributors with entry counts.

**Parameters**: None.

**Returns**: Table of contributor name → entry count.

---

### `kb_list_teams`

**Purpose**: List teams with entry counts.

**Parameters**: None.

**Returns**: Table of team name → entry count.

---

### `kb_maintain` *(KB_MANAGER=TRUE required)*

**Purpose**: Database maintenance and telemetry. 14 actions available.

**Parameters**:

| Parameter | Type | Description |
|---|---|---|
| `action` | `str` | See table below |
| `entry_id` | `str \| None` | Entry ID for entry-specific actions |
| `days_inactive` | `int \| None` | Inactivity threshold for purge |
| `force` | `bool` | Skip confirmation for destructive actions |
| `confirm` | `bool` | Confirm destructive actions |
| `feedback_type` | `str \| None` | Filter for list_feedback / summarize_feedback |
| `since` | `str \| None` | Time filter for feedback (e.g., "7d", "2w") |

**Actions**:

| Action | Description |
|---|---|
| `stats` | Entry counts, type breakdown, staleness distribution |
| `deactivate` | Soft-delete entry_id |
| `reactivate` | Restore soft-deleted entry_id |
| `rebuild_embeddings` | Re-embed all entries (after model change) |
| `rebuild_graph` | Re-run LLM enrichment on all entries |
| `purge_inactive` | Hard-delete inactive entries older than days_inactive |
| `vacuum` | Run VACUUM on the database |
| `entry_versions` | Show version history for entry_id |
| `list_feedback` | List recent agent feedback (filterable) |
| `summarize_feedback` | LLM theme-cluster of feedback (falls back to raw list) |
| `search_stats` | Telemetry: total queries, zero-result rate, avg top score, top missed queries |
| `list_contributors` | Contributors with counts (alias for kb_list_contributors) |
| `list_audit` | Audit event log |

---

### `kb_bulk_update` *(KB_MANAGER=TRUE required)*

**Purpose**: Metadata-only bulk updates across many entries. Does not update `knowledge_details`.

**Parameters**:

| Parameter | Type | Description |
|---|---|---|
| `filters` | `dict` | Selection filters: project_ref, entry_type, tags, contributor, team |
| `updates` | `dict` | Fields to update: project_ref, entry_type, confidence_level, tags (add/remove), team |
| `dry_run` | `bool` | Preview diff without applying |

**Returns**: Diff table showing what would be / was changed. `dry_run=True` is safe to call
at any time.

---

## Rate Limiting

There is no rate limiting at the MCP protocol layer. Throughput is bounded by:
- Ollama embedding latency (50–200ms per entry)
- LLM inference latency (200ms–2s per call for Haiku)
- SQLite WAL write throughput (suitable for single-user, sequential writes)
- Postgres connection pool (default 1–5 connections, configurable)

For bulk ingestion, use `kb_ingest` (batched pipeline) rather than many individual `kb_store`
calls to minimize LLM overhead.

## Example Code

### Store a decision

```python
# Via MCP tool call
kb_store(
    short_title="JWT expiry: 15min access, 7d refresh",
    long_title="Chose 15-minute access token expiry with 7-day sliding refresh tokens",
    knowledge_details="15-minute access tokens limit the blast radius of token theft. "
        "7-day sliding refresh tokens maintain UX without re-auth prompts. "
        "Refresh rotation on every use (stateful server-side list). "
        "Balances security against UX based on OWASP guidance.",
    entry_type="decision",
    project_ref="auth-service",
    tags=["auth", "jwt", "security"],
    hints={"supersedes": "kb-00031"},
    deactivate_entry_id="kb-00031",
    change_reason="Revised after security review — shortened from 1h to 15min",
)
```

### Search and retrieve

```python
# Find relevant entries
results = kb_search(
    query="JWT token expiry",
    project_ref="auth-service",
    entry_type="decision",
)
# Output shows: kb-XXXXX [decision] short_title | long_title | score 0.82

# Retrieve full content
entry = kb_get("kb-00042")
# Output shows: full entry with hints, tags, graph neighbors
```

### Batch store multiple entries

```python
kb_store_batch(entries=[
    {
        "short_title": "Postgres pool: 1-5 connections",
        "long_title": "Default asyncpg connection pool sized for single-user",
        "knowledge_details": "...",
        "entry_type": "factual_reference",
        "project_ref": "personal-kb",
        "tags": ["postgres", "config"],
    },
    {
        "short_title": "uv run --frozen in hooks",
        "long_title": "Always use --frozen flag in pre-commit hooks that call uv",
        "knowledge_details": "Omitting --frozen causes uv to rebuild the package mid-hook ...",
        "entry_type": "pattern_convention",
        "project_ref": "personal-kb",
        "tags": ["uv", "pre-commit", "hooks"],
    },
])
```

### Ask with decision trace

```python
# Follow a decision's history
kb_ask(
    question="history of authentication approach",
    strategy="decision_trace",
    scope="project:auth-service",
)
```

### Ingest a documentation directory

```python
kb_ingest(
    path="docs/",
    project_ref="personal-kb",
    recursive=True,
    dry_run=True,  # preview first
)
# Then run without dry_run to apply
```

## Pointers

- `src/personal_kb/tools/` — one file per tool, all registration functions
- `src/personal_kb/server.py` — `_INSTRUCTIONS` string (in-MCP tool usage guide)
- `src/personal_kb/server.py` — `_ROLE_PREFIXES` dict (prefix behavior per role)
- `src/personal_kb/tools/formatters.py` — compact output format specification
- `src/personal_kb/models/search.py` — `SearchQuery`, `SearchResult` models
- `src/personal_kb/graph/agent.py` — strategy implementations, fast-path threshold
- `src/personal_kb/graph/planner.py` — `QueryPlanner`, strategy selection
- `CLAUDE.md` — KB_INSTANCE_ROLE behavior, KB_MANAGER gating details
