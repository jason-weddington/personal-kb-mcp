# personal-kb — Architecture

## High-Level Design

personal-kb is a FastMCP server that provides AI agents with structured, queryable long-term
memory. It stores knowledge as typed entries (decisions, patterns, lessons, facts), retrieves
them via hybrid search (BM25 + vector + RRF), traverses relationships via a knowledge graph,
and synthesizes answers using an agentic ReAct loop.

The system is designed for **graceful degradation**: every capability beyond basic CRUD has a
fallback. Vector search falls back to FTS-only when Ollama is unavailable. Graph enrichment is
skipped when no LLM is configured. The agentic query loop falls back to a single hybrid search.
This means the server starts and operates correctly with nothing installed except Python and uv.

Two storage backends are supported: **SQLite** (default, zero-config) and **PostgreSQL**
(for multi-user/team deployments with connection pooling and row-level security).

## Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                       MCP Client                             │
│           (Claude Code, other MCP-compatible agents)         │
└───────────────────────────┬──────────────────────────────────┘
                            │ stdio transport
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                   FastMCP Server (server.py)                  │
│                                                              │
│  Tools: kb_search, kb_store, kb_ask, kb_summarize,           │
│         kb_get, kb_ingest, kb_ingest_url, kb_preflight,      │
│         kb_feedback, kb_explore, kb_maintain, kb_bulk_update │
│         kb_store_batch, kb_list_*                            │
└──────┬──────────┬──────────┬───────────────┬─────────────────┘
       │          │          │               │
       ▼          ▼          ▼               ▼
┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐
│  Search  │ │ Graph  │ │ Ingest │ │  Store /     │
│  Module  │ │ Module │ │ Pipeline│ │  Knowledge   │
│ search/  │ │ graph/ │ │ ingest/ │ │  Store       │
└────┬─────┘ └───┬────┘ └────┬───┘ └──────┬───────┘
     │           │           │            │
     ▼           ▼           ▼            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Database Backend (db/)                     │
│   SQLiteBackend (aiosqlite + sqlite-vec)                     │
│   PostgresBackend (asyncpg + pgvector)                       │
└──────────────────────────────────────────────────────────────┘
       │                         │
       ▼                         ▼
┌─────────────┐          ┌───────────────┐
│   Ollama    │          │   LLM Layer   │
│ (embeddings)│          │   llm/        │
│ qwen3-      │          │ Anthropic /   │
│ embedding   │          │ Bedrock /     │
│ :0.6b       │          │ Ollama        │
└─────────────┘          └───────────────┘
```

## Data Flow

### Store flow (kb_store)

1. Agent calls `kb_store` with content, type, tags, hints
2. `KnowledgeStore` generates `kb-XXXXX` ID via atomic `next_sequence_value()`
3. Entry persisted to DB with `version=1`
4. If Ollama available: `EmbeddingClient.embed()` + `store_embedding()` async
5. `GraphBuilder.build_for_entry()` creates deterministic edges (tags → entry, project → entry,
   hint-specified supersedes/related links)
6. If LLM available: `GraphEnricher.enrich_entry()` extracts entities (person, tool, concept,
   technology) and relationship edges — fuzzy-deduped against existing graph nodes (threshold 0.85)

### Search flow (kb_search)

1. Agent calls `kb_search` with query string and optional filters
2. `hybrid_search()` runs FTS5 BM25 and vector KNN in parallel
3. Results merged via Reciprocal Rank Fusion (RRF_K=60): `score = 1/(k + rank_fts) + 1/(k + rank_vec)`
4. Entries below `min_score_ratio * max_score` filtered out (default 0.5)
5. `compute_effective_confidence()` applied to each result (exponential time decay)
6. If results are sparse, graph hints surfaced to help agent discover related entries
7. Search event recorded in `search_events` table for telemetry

### Agentic query flow (kb_ask with auto strategy)

1. Agent calls `kb_ask(question)` with strategy="auto"
2. Fast-path check: run one hybrid search; if top score > `_FAST_PATH_THRESHOLD` (0.030), skip agent
3. Otherwise, `agentic_query()` starts a ReAct loop (max `KB_AGENTIC_MAX_CALLS` iterations, default 4)
4. Agent has 6 internal tools: `hybrid_search`, `graph_neighbors`, `list_graph_nodes`,
   `decision_chain`, `scope_entries`, `done`
5. Exit rule: 3+ matches → done, 1-2 partial → refine, 0 → retry up to 2 approaches
6. Final result: list of (entry_id, short_title) pairs + turns_used + reasoning

### Ingest flow (kb_ingest)

1. Agent calls `kb_ingest(path)` with optional project_ref
2. `FileIngester` validates extension/size allowlists, skips symlinks
3. Large files chunked by `chunk_content()` at H1/H2 markdown boundaries (16,000 chars default,
   600 char overlap)
4. Each chunk: `detect_secrets_in_content()` — chunks with secrets skipped before LLM sees them
5. `extract_entries()` calls extraction LLM to produce 2-6 `ExtractedEntry` dicts per chunk
6. `DedupAgent.check_duplicate()` hybrid-searches + LLM confirms — skips near-duplicates
7. Surviving entries go through standard store flow (step 2-6 above)

## Key Components

### `search/hybrid.py` — RRF Fusion

Combines FTS5 and vector results using the standard RRF formula from the 2009 Cormack et al.
paper (`k=60`). Falls back to FTS-only when Ollama is unavailable. Records every query in the
`search_events` table for zero-cost telemetry. Applies a relative score threshold to cut noise
from the long tail of results.

### `search/embeddings.py` — EmbeddingClient

Wraps Ollama's embedding API (`qwen3-embedding:0.6b`, 1024 dimensions). Graceful degradation:
returns `None` on any Ollama failure rather than raising. Stores embeddings in the DB via the
backend's `vector_store()` method (sqlite-vec for SQLite, pgvector for Postgres).

### `graph/agent.py` — ReAct Loop

The heart of `kb_ask`. Implements a tool-calling loop where the LLM selects from 6 internal
tools per turn. Uses `json_parser.parse_json_object()` for robust JSON extraction from LLM
output. Fast-path threshold (0.030) skips the LLM entirely when search already found strong
results — this path handles ~62% of queries (8/13 golden queries in the eval corpus).

### `graph/enricher.py` — GraphEnricher

LLM-based entity extraction. Produces 2-6 entity nodes per entry and typed relationship edges.
Fuzzy-deduplication prevents graph fragmentation (prefix-based O(1-10) comparisons, 0.85
similarity threshold). Entity types: `person`, `tool`, `concept`, `technology`. Relationship
types: `uses`, `depends_on`, `implements`, `solves`, `replaces`, `configures`, `learned_from`,
`caused_by`.

### `graph/builder.py` — GraphBuilder

Deterministic graph edges derived from entry metadata: tags, project_ref, and explicit hints
dict. Runs synchronously at store time. No LLM required. Edges marked `{"source": "manual"}`.
LLM-derived edges marked `{"source": "llm"}`.

### `confidence/decay.py` — Exponential Decay

Effective confidence = `base * 2^(-age_days / half_life)` where half-life varies by entry_type:

| EntryType | Half-life |
|---|---|
| FACTUAL_REFERENCE | 90 days |
| DECISION | 365 days (1 year) |
| PATTERN_CONVENTION | 730 days (2 years) |
| LESSON_LEARNED | 1825 days (5 years) |

Anchor point is `max(created_at, last_accessed)` — access resets the decay clock. Entries
below 0.5 effective confidence trigger a staleness warning in search results.

### `ingest/safety.py` — Safety Pipeline

Pre-LLM gate: `detect_secrets_in_content()` using the `detect-secrets` library, per-chunk.
`redact_pii()` using `scrubadub`. Chunks with detected secrets are skipped entirely — they
never reach the LLM extractor. `KB_SKIP_SAFETY=TRUE` bypasses for trusted content.

### `db/backend.py` — Database Protocol

Abstract protocol with async methods. Both `SQLiteBackend` and `PostgresBackend` implement it.
The same application code works against either backend. Backend selected at startup based on
`KB_DATABASE_URL` env var (set → Postgres, unset → SQLite).

### `llm/provider.py` — LLMProvider Protocol

Three implementations: `AnthropicLLMClient` (default, Claude Haiku-4-5), `BedrockLLMClient`
(AWS SDK), `OllamaLLMClient` (local). Each returns `None` from `generate()` when unavailable.
Synthesis (kb_summarize, explorer chat) uses a stronger Sonnet 4.6 model when available.

## Database Schema

### `knowledge_entries` — Primary store

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | kb-XXXXX format |
| `short_title` | TEXT | Brief identifier for compact display |
| `long_title` | TEXT | Descriptive title |
| `knowledge_details` | TEXT | Full entry content |
| `entry_type` | TEXT | factual_reference / decision / pattern_convention / lesson_learned |
| `project_ref` | TEXT | Project scope filter |
| `source_context` | TEXT | Provenance (URL, file path, etc.) |
| `contributor` | TEXT | Creator identity (from KB_CONTRIBUTOR) |
| `team` | TEXT | Team attribution (from KB_TEAM) |
| `updated_by` | TEXT | Last updater identity |
| `sensitivity` | TEXT | internal / restricted / public (classification only, no enforcement) |
| `confidence_level` | REAL | Base confidence 0.0–1.0, default 0.9 |
| `tags` | TEXT | JSON array of strings |
| `hints` | TEXT | JSON dict for graph hints |
| `created_at` | TEXT | ISO datetime |
| `updated_at` | TEXT | ISO datetime |
| `last_accessed` | TEXT | Reset by kb_get (decay anchor) |
| `expires_at` | TEXT | TTL expiry; expired entries excluded from search |
| `superseded_by` | TEXT | Entry ID that replaced this one |
| `is_active` | INTEGER | Soft-delete flag |
| `has_embedding` | INTEGER | Embedding presence flag |
| `version` | INTEGER | Incremented on every update |

### `graph_edges` — Knowledge graph

| Column | Type | Notes |
|---|---|---|
| `source_id` | TEXT | Node ID (entry or concept name) |
| `target_id` | TEXT | Node ID |
| `edge_type` | TEXT | supersedes, uses, depends_on, implements, etc. |
| `properties` | TEXT | JSON `{"source": "manual"|"llm"}` |

### `search_events` — Telemetry

| Column | Type | Notes |
|---|---|---|
| `query_text` | TEXT | Raw query |
| `result_count` | INTEGER | Number of results returned |
| `top_score` | REAL | Best RRF score |
| `match_source` | TEXT | hybrid / fts / vector |
| `created_at` | TEXT | Timestamp |

### `agent_feedback` — Structured feedback

| Column | Type | Notes |
|---|---|---|
| `feedback_type` | TEXT | missing / unhelpful / friction |
| `tool_name` | TEXT | Which tool triggered the feedback |
| `query_or_params` | TEXT | What was searched/called |
| `detail` | TEXT | Agent-provided explanation |
| `created_at` | TEXT | Timestamp |

### FTS5 virtual table

SQLite uses a content-sync FTS5 table on `knowledge_entries` with automatic update triggers.
Postgres uses a `tsvector` column with a GIN index.

### Vector storage

SQLite uses sqlite-vec with float32 vectors. Postgres uses pgvector with cosine distance index.
Default embedding dimension: 1024 (from `qwen3-embedding:0.6b`).

## Security Considerations

**What is enforced:**
- Per-chunk secret scanning (detect-secrets library) before any LLM sees ingested content.
  Chunks with API keys, passwords, or tokens are silently skipped.
- Symlink rejection in kb_ingest (prevents directory traversal via symlinks).
- URL content size limit in kb_ingest_url (prevents memory exhaustion).
- Sensitivity field validation (enum: internal/restricted/public).
- `KB_SKIP_SAFETY=TRUE` explicitly bypasses safety scanning for trusted content only.

**Known gaps (see ROADMAP.md — Audit Fixes):**
- **H1 (multi-user access isolation)**: Attribution exists but no read/write isolation. Any
  contributor can modify any entry. Postgres row-level security is the mitigation path when
  needed.
- **H2 (prompt injection via ingest)**: Ingested file content is interpolated into LLM prompts.
  Malicious files can steer LLM extraction. No easy architectural fix — system prompt hardening
  and output validation are partial mitigations.
- **M1 (no directory restriction)**: kb_ingest accepts any path matching the extension allowlist.
  No base-directory constraint for shared deployments.
- **M4 (Postgres transaction atomicity)**: Multi-step operations (create + delete cascade) are
  not atomic on Postgres. `asyncpg` auto-commits per `execute()` on separate pool connections.

## Performance Considerations

**Hybrid search latency**: FTS5 and vector searches run sequentially (not parallel). Typical
latency: 5–20ms on SQLite for a few thousand entries. PostgreSQL with pgvector adds network
round-trip but scales to millions of entries.

**Embedding latency**: Ollama local inference adds 50–200ms per embedding depending on hardware.
The 10-second timeout (`KB_OLLAMA_TIMEOUT`) prevents blocking on unhealthy Ollama. When Ollama
is unavailable, search silently degrades to FTS-only with no user-visible error.

**Agentic query loop**: Each ReAct iteration makes one LLM call. With `KB_AGENTIC_MAX_CALLS=4`
and ~1s per Haiku call, worst-case is ~4 seconds. The fast-path threshold (0.030) skips the
loop for ~62% of queries. Disable with `KB_AGENTIC_QUERY=FALSE` for latency-sensitive contexts.

**Graph enrichment**: LLM entity extraction adds ~1–2s per stored entry when an extraction LLM
is configured. This runs synchronously at store time. For bulk ingestion, the enricher batches
multiple entries into one LLM call.

**SQLite WAL mode**: Enabled by default. Allows concurrent readers while a writer is active.
Suitable for single-machine multi-session usage.

**Connection pooling (Postgres)**: Pool size configurable via `KB_PG_POOL_MIN` / `KB_PG_POOL_MAX`
(defaults: 1–5). Sized conservatively for single-user deployments; increase for team usage.

## Pointers

- `src/personal_kb/search/hybrid.py` — RRF implementation, telemetry recording
- `src/personal_kb/search/embeddings.py` — Ollama client, graceful degradation
- `src/personal_kb/graph/agent.py` — ReAct loop, fast-path threshold
- `src/personal_kb/graph/enricher.py` — entity extraction, fuzzy dedup
- `src/personal_kb/graph/builder.py` — deterministic graph edges
- `src/personal_kb/confidence/decay.py` — exponential decay formula and half-lives
- `src/personal_kb/ingest/safety.py` — secret scanning, PII redaction
- `src/personal_kb/db/backend.py` — Database protocol abstraction
- `src/personal_kb/db/schema.py` — full table definitions
- `src/personal_kb/server.py` — lifespan, LLM provider wiring, tool registration
- `ROADMAP.md` — open security findings (H1, H2, M1, M4) and roadmap priorities
