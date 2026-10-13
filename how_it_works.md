# How It Works

This document explains the core knowledge base: the entry model, storage, the write path, supersession and the near-duplicate guard, freshness, search, embeddings, the knowledge graph, query and synthesis, LLM providers and the call ledger, ingestion, mental maps as an entry type, preflight, the kb-service process, the web UI, the MCP surfaces and configuration. Everything that runs without an agent asking for it (the hook's seven events, failure cues, turn digests, surprise capture, the write policy, the gotcha slice, the soft gate, failure context, the maps push, the anticipatory listener, nightly map maintenance and the metrics) is documented in [docs/intelligent-kb.md](docs/intelligent-kb.md). Every statement here names the module that implements it; when the two disagree, the code is right and this file is the bug.

Contents: [1 Overview](#1-overview-and-package-map) · [2 Code layout](#2-code-layout) · [3 Entry model](#3-entry-model) · [4 Storage](#4-storage) · [5 Write path](#5-write-path) · [6 Supersession](#6-supersession) · [7 Near-duplicate guard](#7-near-duplicate-guard) · [8 Freshness](#8-freshness-expiry-decay-and-staleness) · [9 Search](#9-search) · [10 Embeddings](#10-embeddings-and-the-retry-queue) · [11 Reading](#11-reading) · [12 Graph](#12-knowledge-graph) · [13 kb_ask](#13-query-kb_ask) · [14 Agent loop](#14-agentic-retrieval-loop) · [15 kb_summarize](#15-answer-synthesis-kb_summarize) · [16 LLM providers and ledger](#16-llm-providers-caching-and-the-call-ledger) · [17 Ingestion](#17-ingestion) · [18 Mental maps](#18-mental-maps) · [19 Preflight](#19-project-preflight-kb_preflight) · [20 kb-service](#20-kb-service) · [21 Routes](#21-rest-route-inventory) · [22 Query streaming](#22-query-streaming-and-the-classifier) · [23 Chat](#23-multi-turn-chat) · [24 Web UI](#24-web-ui) · [25 MCP surfaces](#25-mcp-surfaces) · [26 Maintenance tools](#26-maintenance-tools) · [27 Configuration](#27-configuration-reference) · [28 Degradation](#28-graceful-degradation) · [29 Testing](#29-testing-eval-baselines-and-drift-guards) · [30 Known limitations](#30-known-limitations)

## 1. Overview and package map

Personal KB is a graph-RAG knowledge base for AI agents. `release.sh` builds exactly four wheels from this repo, which is a uv workspace (`[tool.uv.workspace] members = ["packages/*"]`):

| Wheel | Source | Role |
|---|---|---|
| `kb-core` | `packages/kb-core/src/kb_core/` | The engine: storage backends, hybrid search, embeddings, the knowledge graph, agentic retrieval, LLM providers, ingestion. Reads no environment variables. |
| `personal-kb-web-service` | `packages/kb-service/src/kb_service/` | The `kb-service` FastAPI process: REST API under `/api`, MCP over HTTP at `/mcp`, the packaged web SPA, auth, and the automation layer. |
| `personal-kb` | `src/personal_kb/` | The deprecated stdio MCP server, a thin client that forwards each tool call to kb-service over HTTP. It depends on `personal-kb-web-service` as a base dependency, so the `kb-service` console script is always on `PATH` for local mode. |
| `personal-kb-hook` | `packages/personal-kb-hook/src/personal_kb_hook/` | A stdlib-only Claude Code hook (see [intelligent-kb.md §3](docs/intelligent-kb.md#3-the-hook-one-stdlib-process-six-events)). |

One request path serves every client:

```text
agent or browser
  ├─ stdio MCP: personal-kb ── HttpBackend ──────┐
  ├─ MCP over HTTP: /mcp ── InProcessBackend ────┤
  └─ REST /api/* and the web SPA ────────────────┤
                                                 ▼
                    kb-service route handler (kb_service/routes/)
                       │                        └── service DB: users, keys, chat, telemetry, capture tables
                       ▼
                 KnowledgeBase (kb-core) ── data DB: entries, versions, graph, vectors
```

The **data DB** holds the knowledge itself and is owned by kb-core; the **service DB** holds kb-service's own tables (section 20). One kb-service deployment is one KB; a personal KB and a team KB are two deployments of identical code.

## 2. Code layout

The engine modules under `src/personal_kb/` are re-export shims onto `kb_core` (for example `src/personal_kb/db/backend.py` is `from kb_core.db.backend import Cursor, Database, Row`), with one exception and one gap:

- The modules in `db/`, `graph/`, `confidence/`, `ingest/`, `llm/`, `search/`, `store/` and `models/` each re-export the `kb_core` module of the same name; the package `__init__.py` files re-export from those shims.
- The exception is `db/connection.py`, a standalone legacy module with no `kb_core` twin: `create_connection` picks SQLite or Postgres from the client's own environment (`KB_DATABASE_URL`, `KB_DB_PATH`, the `KB_PG_*` settings), and its Postgres path seeds and checks `deployment_config`. Only `scripts/migrate_sqlite_to_pg.py`, `scripts/test_dry_run.py` and tests use it.
- The gap is `kb_core/search/embedder_protocol.py`, which has no shim.
- `tools/coverage.py` maps to `kb_core/coverage.py`, `tools/formatters.py` to `kb_core/formatting.py`, `tools/ttl.py` to `kb_core/ttl.py` and `preflight.py` to `kb_core/preflight.py`.

The non-shim client modules are the MCP tool entry points in `src/personal_kb/tools/` (`kb_store.py`, `kb_store_batch.py`, `kb_search.py`, `kb_get.py`, `kb_ask.py`, `kb_summarize.py`, `kb_preflight.py`, `kb_explore.py`, `kb_ingest.py`, `kb_ingest_url.py`, `kb_feedback.py`, `kb_list.py`, `kb_maintain.py`, `kb_bulk_update.py`, `kb_map_eligibility.py`, the advisory renderer `map_lint.py` and the `_lifespan.py` wiring), plus `src/personal_kb/backend/` (`HttpBackend` and the `Backend` protocol), `server.py`, `config.py`, `daemon.py` and `version_skew.py`.

**Unreachable local-mode code.** The shipped stdio lifespan yields only `{backend: HttpBackend, version_skew_note, deprecation_note}` and `HttpBackend.is_remote` is hard-coded `True`, so every branch guarded by "not remote" is dead in production and exercised only by tests: the `kb_maintain.py` actions `stats`, `rebuild_embeddings`, `rebuild_graph`, `purge_inactive`, `vacuum`, `entry_versions`, `list_feedback`, `summarize_feedback`, `search_stats`, `list_contributors` and `list_audit`; the `kb_ask.py` strategies `decision_trace`, `timeline`, `related` and `connection` and `_strategy_auto_with_planner`; directory ingest in `kb_ingest.py`; and `kb_from_lifespan` in `_lifespan.py`. Over HTTP those paths return an error string (section 25).

Each MCP tool has a twin under `packages/kb-service/src/kb_service/mcp_server/tools/` served at `/mcp`; the two must change together, and the parity rule is in [section 25a](#25a-mcp-over-http).

## 3. Entry model

Every piece of knowledge is a `KnowledgeEntry` (`kb_core/models/entry.py`). `EntryType` has five members: `factual_reference`, `decision`, `pattern_convention`, `lesson_learned` and `mental_map` (an orientation node, section 18).

| Field | Meaning |
|---|---|
| `id` | `kb-XXXXX`, permanent, never recycled |
| `project_ref` | Scope string, free text (case-sensitive) |
| `short_title`, `long_title`, `knowledge_details` | The three text fields; the only text that is embedded |
| `entry_type` | One of the five `EntryType` members; `factual_reference` when omitted on create |
| `source_context` | Free-text provenance (for example `Ingested from <source>`) |
| `contributor`, `team`, `updated_by` | Attribution, injected by the server |
| `sensitivity` | `internal`, `restricted`, `public` or None. A label only: a non-public value renders as `[RESTRICTED]` or `[INTERNAL]`, and nothing enforces it |
| `confidence_level` | 0.0 to 1.0, default 0.9; the base for decay (section 8) |
| `tags` | List of strings |
| `hints` | Free-form dict that drives graph building and several guards |
| `created_at`, `updated_at`, `last_accessed`, `expires_at` | Lifecycle timestamps |
| `superseded_by` | Derived from `supersedes` edges, never written by hand (section 6) |
| `is_active` | False after a soft delete |
| `has_embedding` | True once a vector is stored |
| `version` | Incremented by every update |

`KnowledgeEntry.compose_embedding_text` is the single definition of embedded text: `short_title`, `long_title` and `knowledge_details` joined by single spaces. Tags, hints and `source_context` are not embedded.

**Tags** are stored as one space-joined TEXT column, not a JSON array, and filtered with `(' ' || tags || ' ') LIKE '% tag %'`; `_parse_tags` in `kb_core/db/queries.py` splits on whitespace, so a tag containing a space comes back as two tags. The `'[]'` DDL default on the column is vestigial, because `insert_entry` always writes the joined text.

**Known `hints` keys:**

| Key | Read by | Effect |
|---|---|---|
| `supersedes` | `GraphBuilder`, supersession | kb id or list of kb ids this entry replaces; the only channel that writes a `supersedes` edge |
| `distinct_from` | near-duplicate guard, `GraphBuilder` | kb ids declared genuinely different; mirrored as `distinct_from` edges |
| `related_entities` | `GraphBuilder` | List of kb-id strings or `{id or target, edge_type or type}` dicts; items typed `supersedes` or `distinct_from` are ignored with a warning |
| `person`, `tool` | `GraphBuilder` | `mentions_person` / `uses_tool` edges to lower-cased nodes |
| `resolution` | prevention channels | A corrected belief (see [intelligent-kb.md §4](docs/intelligent-kb.md#4-the-lesson-record-hintsresolution-and-the-trust-model)) |
| `operated_via` | anticipatory listener, on `mental_map` entries | See [intelligent-kb.md §17](docs/intelligent-kb.md#17-the-anticipatory-listener-paused) |
| `surprise_capture` | prevention admission | Stamped by autonomous capture (see [intelligent-kb.md §9](docs/intelligent-kb.md#9-distill-ii-distiller-critic-match-and-expiry)) |
| `write_policy` | audit | Stamped on entries written from a queued candidate (see [intelligent-kb.md §10](docs/intelligent-kb.md#10-distill-iii-queued-stores-shape-5-w1-w9)) |

**Ids.** `next_entry_id` in `kb_core/db/queries.py` returns `f"kb-{val:05d}"` from the single-row `entry_id_seq` table: SQLite does a SELECT then UPDATE inside the caller's transaction, which is safe under its single connection, and Postgres does one atomic `UPDATE ... RETURNING`. Deactivation is a soft delete, so the row and its id remain and ids are never recycled. It does not keep the graph intact: `KnowledgeBase.deactivate` deletes the entry's outgoing edges (unless it is a `mental_map`) and keeps only incoming ones (section 6). The five-digit shape is hard-coded in `kb-\d{5}` regexes in `kb_core/supersession.py`, `kb_core/graph/builder.py`, `kb_service/routes/near_duplicate_guard.py` and the `kb_store` tool, so a six-digit id (`kb-100000`) would fail validation (section 30).

**Attribution.** `kb_service/attribution.resolve_attribution(user)` returns `contributor` = the authenticated user's email and `team` = the `app_config` row `team` (set by an admin through `PUT /api/settings`, which replaces it wholesale; blank clears it). Updates stamp `updated_by` with the user's email, and version rows carry the contributor. kb-service builds its `KnowledgeBase` with a bare `Attribution()`, so only per-request values matter. `KB_CONTRIBUTOR` only gates registration of the contributor and team listing tools, and `KB_TEAM` is not read by the service. Attribution is visibility, not access control: any interactive caller can update any entry.

## 4. Storage

All engine SQL is written once against the `Database` protocol in `kb_core/db/backend.py` with SQLite-flavoured syntax. Its members are `execute`, `executemany`, `executescript`, `commit`, `close`, `fts_search`, `vector_store`, `vector_search`, `vector_delete`, `delete_llm_edges`, `delete_deterministic_edges`, `map_eligibility_counts`, `vacuum`, `next_sequence_value`, `transaction` and `apply_schema`.

**SQLiteBackend** (`kb_core/db/sqlite_backend.py`) wraps one aiosqlite connection opened by `_open_sqlite` in `kb_core/knowledge_base.py`: it creates the parent directory, sets `journal_mode=WAL` and `foreign_keys=ON`, and loads sqlite-vec best-effort (without it the vec0 schema is skipped and vector search returns `[]`). `transaction()` uses ContextVars: `commit()` is suppressed inside the block, the block commits on exit and rolls back on exception, and nested blocks use SAVEPOINTs. Full-text search uses `knowledge_fts`, an external-content FTS5 table (`porter unicode61`) over `short_title`, `long_title`, `knowledge_details` and `tags`, kept in sync by the triggers `knowledge_fts_ai`, `knowledge_fts_ad` and `knowledge_fts_au`, so any UPDATE of an entry re-indexes it. Vectors live in `knowledge_vec`, a vec0 table (`entry_id TEXT PRIMARY KEY, embedding FLOAT[dim] distance_metric=cosine`, default dim 1024); `vector_store` deletes then inserts because vec0 has no upsert. Because kb-service shares the one connection across requests, a background task's commit could commit a half-written request transaction, which is why the embedding worker is off on SQLite (section 10).

**PostgresBackend** (`kb_core/db/postgres_backend.py`) wraps an asyncpg pool (`PostgresBackend.create` defaults to 2..10 connections; kb-service passes `KB_PG_POOL_MIN`/`KB_PG_POOL_MAX`, default 1..5). The `?` placeholders are translated to `$N` at execute time (string literals skipped). `transaction()` pins one pooled connection in a ContextVar and nests with savepoints; `commit()` is a no-op. `apply_schema` runs under `pg_advisory_lock(42)` so concurrent starts converge: it creates the `vector` extension and every table, the trigger function `knowledge_entries_search_trigger` that builds `search_vector` with weights A (`short_title`), B (`long_title`), C (`knowledge_details`) and D (`tags`), the GIN index `idx_entries_fts`, then idempotent column migrations via `information_schema` and a `COLLATE "C"` pin on `embedding_retry_queue.next_attempt_at`. `knowledge_vec` is `vector(dim)` with no ANN index, so vector search is an exact scan ordered by the `<=>` cosine operator.

**Schema and migrations.** There is no migration framework: `kb_core/db/schema.py` (SQLite) and `PostgresBackend._apply_schema_locked` create everything with `CREATE ... IF NOT EXISTS` on every open and add later columns with introspection-guarded `ALTER TABLE ADD COLUMN`. `schema_version` holds `SCHEMA_VERSION = 1`, which nothing consults. Data-DB tables:

| Table | Holds |
|---|---|
| `knowledge_entries` | Entries (base columns plus migrated `last_accessed`, `expires_at`, `contributor`, `team`, `updated_by`, `sensitivity`; Postgres adds `search_vector`) |
| `entry_versions` | One row per create or update (section 5) |
| `knowledge_fts` (SQLite) / `search_vector` column (Postgres) | Full-text index |
| `knowledge_vec` | Embeddings |
| `entry_id_seq` | The id counter |
| `graph_nodes`, `graph_edges` | The knowledge graph (section 12) |
| `ingested_files` | One row per ingested source (section 17) |
| `search_events` | One row per `hybrid_search` call that runs a search, best-effort (sections 9 and 11) |
| `agent_feedback` | `kb_feedback` rows (section 11) |
| `audit_events` | Mutation audit trail (section 5) |
| `embedding_retry_queue` | Entries waiting to be embedded (section 10) |
| `map_eligibility_override`, `map_cluster_ledger` | Nightly map maintenance state (see [intelligent-kb.md §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract)) |
| `deployment_config` | Key/value table created by the schema; only the legacy `src/personal_kb/db/connection.py` seeds and checks it |
| `schema_version` | Unused version marker |

`knowledge_entries` is indexed on `project_ref`, `entry_type`, `is_active`, `contributor` and `team`; there is no index on `superseded_by`, `tags` or `expires_at`. The vector table is sized from the embedding dimension at creation, and nothing reconciles the two, so changing the embedding model or dimension requires recreating `knowledge_vec` and re-embedding.

**IAM authentication.** `kb_core/db/iam_auth.py` (`parse_dsn`, `make_token_factory`, `make_ssl_context`) is wired by `_open_postgres` when `PostgresConfig.iam_auth` is true, so library callers can use RDS/Aurora IAM tokens. kb-service's `main._open_kb` never passes `iam_auth` or `region`, so `KB_PG_IAM_AUTH` and `KB_PG_REGION` affect only `scripts/migrate_sqlite_to_pg.py`, which opens its target through the legacy `src/personal_kb/db/connection.py`.

**Migration scripts.** `scripts/migrate_sqlite_to_pg.py` copies a SQLite KB into Postgres in FRESH mode (ids as-is) or MERGE mode (source ids offset), with `--dry-run`, `--contributor`/`--team` stamping and `--skip-embeddings`. It copies entries, versions, graph tables, `ingested_files` and `audit_events` and re-embeds through Ollama because vectors cannot move between sqlite-vec and pgvector. Its column list omits `expires_at`, so TTL entries migrate as permanent. `scripts/migrate_to_postgres.py` is an older variant.

## 5. Write path

Writes enter through `kb_store` (create or update one entry), `kb_store_batch` (1 to 10 creates), deactivate and reactivate. Every write goes through a kb-service route in `kb_service/routes/kb_write_routes.py` and then `KnowledgeBase` in `kb_core/knowledge_base.py`.

**Write policy.** Each write is classified by its **surface** (`interactive`, `headless` or `autonomous`), resolved from the API key and the `X-KB-Mode` header by `kb_service/write_policy.py`. Interactive writes run the path below unchanged; a create from a non-interactive surface is queued as a candidate instead of stored, and an update, deactivate or reactivate from one is refused with 403. The policy and the queue are documented in [intelligent-kb.md §7](docs/intelligent-kb.md#7-capture-c-the-write-policy-unattended-agents-propose-never-write), and the **machine principal** (the one user named by the `app_config` key `machine_principal_email`, whose writes are treated as autonomous) in [§4](docs/intelligent-kb.md#4-the-lesson-record-hintsresolution-and-the-trust-model).

**Create: tool-layer rules.** The `kb_store` tool (`src/personal_kb/tools/kb_store.py` and its `/mcp` twin) requires `short_title`, `long_title` and `knowledge_details`. `supersedes` is a required parameter: a list of kb ids or the literal `"none"`. The tool rejects `[]` as ambiguous and rejects `"none"` together with a non-empty `hints.supersedes`, while the route treats an absent field, `[]` and `"none"` alike as no supersession. `superseded_by` is rejected on create, `distinct_from` must be a list of kb ids, `sensitivity` must be one of the three values, `knowledge_details` is secret-scanned, a `mental_map` with no outbound pointer is rejected (section 18b), and `ttl` is pre-parsed.

**Create: route order.** `POST /api/kb/store` without `update_entry_id` checks, in order:

1. Required fields (422).
2. `ttl` parse (422).
3. Secret scan of `knowledge_details` (422).
4. Write-policy queue for a non-interactive surface.
5. `hints.resolution` validation and stamping (see [intelligent-kb.md §4](docs/intelligent-kb.md#4-the-lesson-record-hintsresolution-and-the-trust-model)).
6. Orphan `mental_map` check (422).
7. Machine-principal map lint (see section 18c).
8. Superseded map pointer (409 `pointer_superseded`).
9. `hints.supersedes` shape, then `check_supersedes` (422 `supersedes rejected: ...`).
10. `distinct_from` validation (422).
11. Near-duplicate guard (409, section 7).

Then `KnowledgeBase.store` runs, synchronously inside the request: `KnowledgeStore.create_entry` in one transaction (next id, the row, version 1 and an `entry_created` audit row); `_embed_one` (a failure enqueues an `embedding_retry_queue` row and never fails the store); `GraphBuilder.build_for_entry` (best-effort); and `GraphEnricher.enrich_entry` when an extraction LLM exists (best-effort, failures logged with the `enrichment-failure` marker). The effective supersedes set is the sorted union of `body.supersedes` and `hints.supersedes` and is written back into stored `hints.supersedes`; `distinct_from` is stored as `hints.distinct_from`. The tool prints `Created kb-XXXXX (vN)`, the compact entry, `Entry will be embedded when Ollama is available` when `has_embedding` is false, a `Supersedes:` line, and any map-lint advisories.

**Update.** `POST /api/kb/store` with `update_entry_id` applies only the fields supplied. The route checks the write policy (403 for non-interactive surfaces) first, then requires a non-blank `change_reason` (422), rejects `distinct_from` in the field or in hints (422), parses `ttl`, secret-scans only when `knowledge_details` is supplied, returns 404 for a missing entry, stamps any resolution against the stored hints, runs the orphan, lint and superseded-pointer checks against the effective post-update body, and validates only newly added supersedes targets (section 6). `KnowledgeStore.update_entry` refuses an inactive entry, shallow-merges hints (`dict.update`, so a supplied `resolution` replaces the whole stored object and omitted keys survive), replaces tags wholesale when supplied, lets `entry_type` change, stamps `updated_by`, appends a version and writes an `entry_updated` audit row; it never writes `superseded_by`. Deterministic graph edges are rebuilt on every update.

Three update behaviours surprise callers:

- Re-embedding and LLM enrichment run only when `knowledge_details` is supplied, so a title-only update leaves a stale vector even though titles are part of the embedding text.
- The MCP `kb_store` tool always sends `confidence_level` (default 0.9), so every tool-originated update, a tags-only one included, resets the stored confidence to 0.9 unless the caller passes the current value. REST callers that omit the field keep the stored value.
- A TTL can be set or replaced over HTTP and MCP but not cleared: the engine's `clear_expiry` is not exposed by `StoreRequest` or the tool.

**store_batch.** The tool accepts 1 to 10 entries, each with the create fields and `supersedes`; any shape, hints-conflict, `distinct_from`, sensitivity or secret problem rejects the batch, and entries with an invalid `ttl` are split out client-side into a `Failed entries (retry these)` list (their indices remapped to the caller's positions) and never sent. `POST /api/kb/store_batch` validates every entry up front, in order, with failures prefixed `entry {i}: `: resolution, ttl, secret scan, map pointer, machine-principal lint, superseded map pointer (422 here, 409 on a single store), supersedes, `distinct_from`, then the near-duplicate guard (409 with `entry_index`). Entries in one batch are not compared with each other. A non-interactive surface queues every entry. `KnowledgeBase.store_batch` then creates each entry in its own transaction (a per-entry runtime failure is logged and skipped, so `created` can be shorter than `requested`), builds each entry's graph, makes one `embed_batch` call (any id not embedded is enqueued for retry; there is no per-entry fallback when `embed_batch` returns None) and one `enrich_batch` call, then clears the enricher's vocabulary cache. The response is `StoreBatchResponse(requested, created, superseded_ids)`.

**Deactivate, reactivate and hard delete.** `POST /api/kb/entries/{id}/deactivate` (and `kb_store` with `deactivate_entry_id`, which requires `supersedes="none"`) checks the write policy (403), refuses a `mental_map` (422, because a soft delete strips a map's outbound edges; maps are deleted through `DELETE /api/kb/maps/{map_id}`), requires `change_reason` before checking existence (so an unknown id without a reason is 422, not 404), and validates an optional `superseded_by`. `KnowledgeBase.deactivate` then runs in one transaction (section 6). The `knowledge_vec` row is kept; search drops inactive ids. `POST /api/kb/entries/{id}/reactivate` is admin-only and rebuilds graph edges and re-enriches best-effort. `delete_entry_cascade` (vector, versions, edges, node, row) is reachable only through the dead local `purge_inactive` action.

**Versions and audit.** `entry_versions` gets version 1 (`Initial creation`) on create and `existing.version + 1` on update, holding only `knowledge_details`, `confidence_level`, `change_reason` and `contributor` (not titles, tags or hints). `audit_events(event_type, entry_id, contributor, detail, created_at)` is written fire-and-forget for `entry_created`, `entry_updated` (also bulk updates, detail `Bulk metadata update`), `entry_deactivated`, `entry_reactivated`, `superseded_by_changed`, `supersedes_hint_appended`, `near_duplicate_checked`, `embedding_recovered`, `embedding_exhausted`, and map events (`map_deleted`, `map_eligibility_override_set`, `map_eligibility_override_cleared`, `map_cluster_*`). Both tables are write-only over HTTP: no route or working MCP action reads them, so history needs direct database access.

**Secret scanning.** `kb_core/ingest/safety.detect_secrets_in_content` runs detect-secrets line by line with exactly three plugins (`KeywordDetector`, `PrivateKeyDetector`, `BasicAuthDetector`) over `knowledge_details` only (not titles, tags, hints or `source_context`). It runs in the tool (`Error: Potential secrets detected (types). ...`) and again in the route (422 `Secret scan detected sensitive content: types`). `KB_SKIP_SAFETY` is read by each process from its own environment: `TRUE` in the service environment skips the route scan and the `/mcp` tool scan, while the deprecated stdio `kb_store` and `kb_store_batch` scan in the client process (`personal_kb.config.is_safety_skip`), so skipping theirs needs the variable in the stdio server's environment too. If detect-secrets is not installed the check passes (fail-open); kb-service depends on `kb-core[safety]`, so it ships. The stronger `redact_secrets` serves turn digests, not `kb_store`.

## 6. Supersession

Supersession is structural: "B replaces A" is a `supersedes` graph edge from B to A, and `A.superseded_by` is **derived** from those edges, never written by hand. The code lives in `kb_core/supersession.py`.

### The invariant

Every `knowledge_entries` row's `superseded_by` equals the id of its newest qualifying superseder, or NULL if it has none. A qualifying superseder of T is a row S with `is_active = 1` and `entry_type != 'mental_map'`, joined by a `graph_edges` row `(S, T, 'supersedes')` whose `properties` JSON does not carry `source == 'llm'` (parsed in Python with `json.loads`, so no dialect-specific JSON SQL is needed). "Newest" means max `created_at`, which is immutable. `updated_at` is never used, because it moves on every edit and would flip the pointer. `created_at` is parsed with `datetime.fromisoformat`, naive values are treated as UTC, and ties break by id descending. The invariant holds whether T is active or inactive.

`recompute_superseded_by(db, target_ids, *, trigger)` applies the invariant to a set of targets inside the caller's transaction (it opens none of its own). An unchanged target is not written, counted or logged. A changed one gets an UPDATE of `superseded_by` alone, with no version bump and no `updated_at` change, a `superseded_by_changed` audit row whose detail is `{old, new, candidates, trigger}`, and an INFO line `supersession target=... old=... new=... candidates=... trigger=...`. `queries.update_entry` never writes `superseded_by`, so an update from a stale entry snapshot cannot clobber the derived value.

### The writers

`GraphBuilder.build_for_entry` captures the entry's old `supersedes` targets before clearing its edges. It rebuilds the edges, and its last statement recomputes the old targets, the new targets and the entry itself, so dropping a target from `hints.supersedes` clears that target's pointer in the same transaction. No edge is derived from `superseded_by`. The `related_entities` step skips any item typed `supersedes` with a `supersession: ignoring related_entities supersedes edge` warning, so `hints.supersedes` is the single validated channel.

`KnowledgeStore.deactivate_entry` (which takes an optional `change_reason` that becomes the audit detail) and `reactivate_entry` recompute the entry's outgoing targets and the entry itself. `KnowledgeBase.deactivate(entry_id, *, contributor, change_reason, superseded_by)` runs one transaction. When `superseded_by` is given, it appends `entry_id` to the superseder's `hints.supersedes` (a hints-only UPDATE plus a `supersedes_hint_appended` audit row) and inserts the edge. It then deactivates the entry, removes its outgoing non-map edges and recomputes. `recompute_supersession` runs after a bulk `entry_type` change (section 26). Hard delete and direct SQL are not maintained; the reconcile heals them.

### Startup reconcile

`reconcile_supersession(db)` is idempotent and runs in one transaction. First it backfills a missing `supersedes` edge for every active non-map entry whose `hints.supersedes` names an existing kb id that has no edge yet. Then it recomputes every supersedes-edge target and every row whose `superseded_by` is set. kb-service's lifespan calls `kb.reconcile_supersession()` right after opening the KB, inside a try/except that never blocks startup; admins can run the same work on demand with `POST /api/kb/admin/reconcile-supersession` or `kb_maintain` action `reconcile_supersession` (both return `edges_added`, `set_count`, `cleared_count` and the changed rows). It logs `supersession-reconcile edges_added=N set=N cleared=N` and one WARNING `supersession-reconcile drift target=... old=... new=...` per changed row; a failure logs `supersession-reconcile failed: ...`. After the first startup on a new release has done the backfill, any `supersession-reconcile drift` line is a defect signal: some writer left the invariant broken.

### The HTTP contract

`StoreRequest` and `StoreBatchEntry` carry `supersedes: list[str] | "none" | None`. At the route an absent field, `[]` and `"none"` all mean no supersession; the MCP tools are stricter (the parameter is required, `[]` is rejected as ambiguous, and `"none"` conflicts with a non-empty `hints.supersedes`). On create, the effective set is the union of `body.supersedes` and the string items of `hints.supersedes`; a non-string hint item is a 422, never silently filtered. The set is validated with `kb.check_supersedes` before `kb.store`, and the problems are rendered as `supersedes rejected: ...`. A target must be a well-formed kb id that exists and is active. It must not be the writer itself or a mental_map, and it must not already supersede the writer (a cycle). A mental_map writer cannot supersede anything. Cross-project targets are allowed. `StoreResponse.superseded_ids` lists every validated target on create.

On update, `change_reason` is required, and a missing or whitespace-only value is a 422. Supersedes are a monotonic union with the targets the entry already records, and only the newly added targets are validated, so previously recorded targets (possibly inactive by now) are grandfathered. `hints={"supersedes": []}` retracts nothing; the only API path to retract is deactivating the superseder. `superseded_ids` lists only the newly added targets. `/store_batch` applies the create rule per entry in its up-front validation loop (`entry {i}: supersedes rejected: ...`).

`POST /api/kb/entries/{id}/deactivate` takes an optional `{change_reason, superseded_by}` body, checked in this order: auth (401), the write policy (403), the mental_map block (422), a missing change_reason (422), then `superseded_by` must exist and pass `check_supersedes` (422). Only after those does `kb.deactivate` run. Every decision logs one `supersession-route op=... outcome=... mode=... writer=... targets=... problems=...` line. A validated target still unpointed after the write logs `supersession-route build_failed`, because `_build_graph` is best-effort and the next reconcile heals it.

### Superseded map pointers

`POST /api/kb/store` (create and update) and `POST /api/kb/store_batch` reject a `mental_map` write whose pointers name a superseded entry via `check_superseded_map_pointers` in `kb_service/routes/map_write_guards.py`: a superseded pointer is a 409 `pointer_superseded` with the `(pointer, superseded_by)` pairs (422 `entry i: ...` in a batch), and on update only newly added pointers are checked, so existing ones are grandfathered. The nightly loop's map operations apply the same rule (see [intelligent-kb.md §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract)).

### Supersession-aware reads

An entry is superseded exactly when `entry.superseded_by is not None`; reads never re-check that the superseder is active, because the write-side invariant only points at active, non-map superseders. In the fused path of `hybrid_search`, superseded rows are skipped unless `SearchQuery.include_superseded` is set, and the result window grows by one for each hidden row, so only superseded rows are backfilled and other filters still use up their slot. The filter-only path pushes `AND superseded_by IS NULL` into SQL, so LIMIT is never spent on hidden rows; that path is not instrumented. The flag travels as `SearchRequest.include_superseded` (default False) and `HttpBackend.search` always sends it. Ingest dedup opts out with `include_superseded=True` so content the KB already has still dedupes. `kb_preflight` adds `superseded_by IS NULL` to the expiring, recent, conventions and graph-related queries (not the maps query). On the client, `kb_search` appends `[SUPERSEDED by kb-X]` to superseded results and skips superseded graph hints; `kb_get` prefixes a superseded entry with `SUPERSEDED by <id> — <title>`, costing one extra `get_entries` call only when the superseder was not already in the request. `kb_ask` graph expansion, the agent's `graph_neighbors` and `scope_entries` tools, and the exhaust fallback skip superseded entries; `decision_chain` and an agent's explicit final pick are exempt (the pick is logged).

### Audit events and log markers

Audit events: `superseded_by_changed` (every derived-pointer change) and `supersedes_hint_appended` (deactivate with `superseded_by`). Greppable log markers, all `key=value`:

- `supersession` — one line per changed pointer, from `recompute_superseded_by`.
- `supersession-route` — one line per write-route decision; `supersession-route build_failed` when a validated target is still unpointed after the write.
- `supersession-read op=<search|ask_expand|agent_tool|agent_exhaust|agent_final_pick|kb_get>` — every hidden row; `supersession-read invariant_breach` (a superseder that cannot be found) is a defect signal.
- `supersession-client` — client-side tool decisions.
- `supersession-reconcile` — the reconcile summary; `supersession-reconcile drift` per changed row and `supersession-reconcile failed` on error.

## 7. Near-duplicate guard

The service refuses to create an entry that is nearly identical to an existing one unless the request says what the relationship is, because a silent duplicate splits the knowledge and both copies then rot independently. The guard lives in `kb_service/routes/near_duplicate_guard.py` (route glue, 409 payload, telemetry) and `kb_core/near_duplicates.py` (the lookup), exposed as `KnowledgeBase.find_near_duplicates`.

It runs only on create, for both `POST /api/kb/store` and `POST /api/kb/store_batch`, after the supersedes validation. An update never runs it, and an update that carries `distinct_from` (in the field or in `hints`) is a 422. Entries within one batch are not compared with each other. Creates from a non-interactive surface never reach this route guard, because they are queued first; they meet the same floor when the store pipeline writes them (see [intelligent-kb.md §10](docs/intelligent-kb.md#10-distill-iii-queued-stores-shape-5-w1-w9)).

The search is scoped to the new entry's own `project_ref`. The new text is embedded with `KnowledgeEntry.compose_embedding_text`, then `Database.vector_search(limit=NEAR_DUPLICATE_NEIGHBOURS, project_ref=...)` with `NEAR_DUPLICATE_NEIGHBOURS = 20` returns active rows of that project with cosine distance. Eligible rows are non-`mental_map` with `superseded_by IS NULL`; similarity is `round(1 - distance, 4)`. Candidates are eligible rows at or above the floor, sorted by `(-similarity, id)`, truncated to 5. On SQLite the vector search over-fetches the top 200 global neighbours and then filters to the project, so a project whose rows are crowded out of that window can be missed; Postgres is exact.

Exemptions: `mental_map` entries (`exempt_mental_map`), entries with no or blank `project_ref` (`skipped_no_project`), and two fail-open cases: `embedder_unavailable` (no embedder, or the embed call returned nothing) and `search_failed` (the search raised). Expired-but-active entries still count as candidates.

The floor is `KB_NEAR_DUPLICATE_FLOOR`, default 0.88, read by `get_near_duplicate_floor()` on every request. A value above 1.0 disables the guard. The floor is read before the exemptions, so a malformed value raises `ValueError` and every create returns 500, maps included, until it is fixed. 0.88 was chosen from the live distribution: over 30 days, 339 non-map project-scoped creates compared with their nearest older same-project neighbour would have 409'd at 0.92 (2), 0.90 (4), 0.88 (7, 2.1%), 0.85 (15) and 0.80 (47); the median similarity of declared supersedes edges is about 0.70, so the guard catches near-duplicates and does not replace declared supersession.

A conflict is a 409 whose `detail` is `{"error": "near_duplicate", "message", "project_ref", "floor", "candidates": [{id, short_title, entry_type, similarity, updated_at}]}` (plus `entry_index` for a batch, whose message starts `entry {i}: `). Only candidates not already covered by the request are listed. Both `HttpBackend` and `InProcessBackend` `json.dumps` a dict detail, so the agent sees `Error: {"error": "near_duplicate", ...}` with the three escapes spelled out in `message`. Every listed id must be covered by one of them: `update_entry_id=<id>` (same fact: update that entry, with `change_reason`), `supersedes=[<id>]` (the new entry replaces it), or `distinct_from=[<id>]` (genuinely different facts). Each `distinct_from` id must be a well-formed kb id of an existing, active, non-map entry (cross-project is allowed) and must not also be in `supersedes`; otherwise 422.

`distinct_from` is stored on the new entry as `hints.distinct_from`, and `GraphBuilder` mirrors it as a `distinct_from` edge (the target must exist and not be the entry itself). The hint is the source of truth; the edge mirrors it. Suppression is request-scoped: a candidate named in this request's `supersedes` or `distinct_from` is not reported. `kb_ask` and the graph page traverse the edge like any other type.

Every guarded create emits one log line starting `near-duplicate-guard op=` (INFO; WARNING for `embedder_unavailable` and `search_failed`) with `outcome` (`clear|resolved|conflict|exempt_mental_map|skipped_no_project|embedder_unavailable|search_failed`), floor, top similarity, raw and eligible hit counts, `embed_ms`, `search_ms`, candidates, `resolved_by`, `distinct_from_unused`, contributor, title and `text_sha` (first 12 hex digits of the sha256 of the embedding text). It also writes one `audit_events` row with `event_type='near_duplicate_checked'` and the same fields as JSON `detail`. A 409 row has `entry_id` NULL; every other row is written after the store succeeds and carries the created id (with a `near-duplicate-guard-stored` log line). The audit write is best-effort.

These rows are the retune instrument. To re-derive the 409 rate at a candidate floor X, count rows where `detail.top_similarity >= X`, grouped by `detail.outcome`:

```sql
SELECT detail::json->>'outcome' AS outcome, count(*)
FROM audit_events
WHERE event_type = 'near_duplicate_checked'
  AND (detail::json->>'top_similarity')::float >= :X
GROUP BY 1;
```

Missed duplicates are `clear` rows whose `entry_id` later becomes the target of a `supersedes` edge; the escape an agent chose after a 409 is found by joining `conflict` rows to later `resolved` rows with the same `text_sha`.

## 8. Freshness: expiry, decay and staleness

**TTL expiry.** An optional `ttl` sets `expires_at`. `kb_core/ttl.py` `parse_ttl` accepts `^(\d+)([hdw])$` (zero is rejected) and `compute_expires_at` returns now plus the delta in UTC; the service computes it at request time (422 on a bad value) and stores an ISO string. `hybrid_search` drops entries with `now >= expires_at`, in both the fused and filter-only paths, unless `include_expired` is set. `format_entry_meta` renders `[EXPIRES Nd]`, `[EXPIRES Nh]` or `[EXPIRED]`. `kb_get` and direct SQL still return expired entries, and there is no reaper job: an expired entry is hidden from search, never deactivated. Autonomous lessons carry their own TTL (see [intelligent-kb.md §9](docs/intelligent-kb.md#9-distill-ii-distiller-critic-match-and-expiry)).

**Confidence decay.** `kb_core/confidence/decay.py` computes effective confidence at read time, never stored: `effective = base * 2^(-age_days / half_life)`, rounded to 4 places.

| Entry type | Half-life (days) |
|---|---|
| `factual_reference` | 90 |
| `decision` | 365 |
| `pattern_convention` | 730 |
| `lesson_learned` | 1825 |
| any other type (`HALF_LIVES.get(type, 365.0)`) | 365 |
| `mental_map` | exempt (section 18a) |

The age anchor is the later of `updated_at` (or `created_at`) and `last_accessed`, so any update, a bulk update included, resets the clock. `last_accessed` is written only by `POST /api/kb/get` (`touch_accessed` on the found ids), unless the request sets `touch: false`, which the web UI uses for neighbour-title lookups, never by search, ask or summarize. Below `STALENESS_THRESHOLD = 0.5` an entry renders `[STALE]`; `hybrid_search` drops results below `STALE_FILTER_THRESHOLD = 0.3` unless `include_stale` is set.

## 9. Search

`kb_core/search/hybrid.py` `hybrid_search(db, embedder, SearchQuery)` fuses a keyword leg and a semantic leg by rank, because neither alone is reliable: FTS misses paraphrases and vectors miss exact identifiers.

1. **Filter-only path.** An empty or `*` query with any of `project_ref`, `entry_type`, `tags`, `contributor` or `team` runs `_filter_only_search`: newest `created_at` first, superseded rows hidden in SQL, and the SQL LIMIT applied before expiry and staleness filtering, so fewer than `limit` rows can come back. Results carry score 0.0 and `match_source` `filter`. No query and no filters returns `[]`.
2. **Over-fetch.** Otherwise each leg fetches `limit * 3` ids with identical SQL-level filters; the vector leg runs only when an embedder exists.
3. **Reciprocal Rank Fusion.** Each leg adds `1 / (RRF_K + rank + 1)` per id, with `RRF_K = 60` and 0-based rank. Rank 1 in one leg scores 1/61 = 0.0164; rank 1 in both scores 2/61 = 0.0328.
4. **Relative cutoff.** `min_score_ratio` (default 0.5, not exposed over REST or MCP) drops every id below `top * ratio` and reports the count as `filtered_count`. When the top hit is rank 1 in both legs, a hit present in only one leg survives only at that leg's rank 1.
5. **Result walk.** The sorted ids are examined until `limit` results are collected: missing or inactive ids are skipped; superseded ids are skipped unless `include_superseded` (the window grows by one per hidden row, the only backfill); metadata filters are re-checked in Python (tags must all be present as exact, case-sensitive tokens); expired entries are dropped unless `include_expired`; effective confidence is computed and entries below 0.3 are dropped unless `include_stale`. Filtered, expired and stale candidates are not backfilled, so fewer than `limit` results can return.
6. **Telemetry.** Every call that runs a search writes one `search_events` row; a call with no query and no filters returns before recording anything. The insert goes through `_record_search_event`, which catches any failure and only logs a warning, so telemetry never fails a search and a failed insert leaves no row.

`SearchResult` carries `entry`, `score`, `effective_confidence`, `staleness_warning`, `match_source`, `vector_similarity` (`max(0, 1 - distance)`), `fts_matched` and `fts_rank`. `match_source` is `fts`, `hybrid` or `filter`; it is never `vector`, despite the comment in `kb_core/models/search.py`. `SearchQuery.limit` defaults to 10 (range 1 to 50).

**FTS leg.** `kb_core/search/fts.py` forwards to `db.fts_search` and swallows exceptions. On SQLite, `_escape_fts_query` splits on whitespace, strips double quotes and quotes every token, so the FTS5 query is an implicit AND of all tokens with porter stemming and no stopword removal: `how do I deploy production` matches nothing where `deploy production` matches. Natural-language questions therefore often have an empty FTS leg and rely on the vector leg. Ranking is unweighted `bm25(knowledge_fts)`. On Postgres, `plainto_tsquery('english')` also ANDs lexemes but drops English stopwords, and ranking is `-ts_rank_cd` over the weighted A/B/C/D `search_vector`. Both legs filter `is_active = 1` and equality filters in SQL; the tag filter is an AND of `LIKE` clauses, which is case-insensitive for ASCII on SQLite and case-sensitive on Postgres, and `%` or `_` in a tag act as wildcards. Only the fused path re-checks tags exactly. Only rank order feeds RRF, never the raw scores.

**Vector leg.** `kb_core/search/vector.py` embeds the query and calls `search_similar` with the same filters; it returns `[]` when the embedder returns None or the search raises. SQLite with filters over-fetches `max(limit * 10, 100)` neighbours from vec0 and then joins `knowledge_entries` on `is_active = 1` and the filters; without filters it is a plain LIMIT with no join, so inactive ids can occupy rank slots until the result walk drops them. Postgres is an exact scan.

## 10. Embeddings and the retry queue

`kb_core/search/embeddings.py` `EmbeddingClient` talks to Ollama. `is_available()` probes `GET /api/tags`; success is cached for the life of the client, failure is not cached, and the cache resets when an embed call fails, so every embed while Ollama is down pays a fresh probe bounded by the timeout. `embed(text)` posts `{model, input, keep_alive}` to `/api/embed` and returns the first vector or None on any error; `embed_batch` posts a list and returns None on a count mismatch.

| `EmbeddingConfig` field | Default | Env var (kb-service) |
|---|---|---|
| `ollama_url` | `http://localhost:11434` | `KB_OLLAMA_URL` |
| `model` | `qwen3-embedding:0.6b` | `KB_EMBEDDING_MODEL` |
| `dim` | 1024 | `KB_EMBEDDING_DIM` |
| `timeout` | 10.0 s | `KB_OLLAMA_TIMEOUT` |
| `keep_alive` | `30m` | none (see below) |

`embedding=None` on `KbConfig` is a first-class FTS-only mode; there is no silent provider swap. A mismatch between `EmbeddingConfig.dim` and the storage dimension is not reconciled: the insert fails, is logged, and the entry is queued for retry. `KB_OLLAMA_KEEP_ALIVE` is inert on kb-service: `kb_service/config.build_embedding_config` does not pass `keep_alive`, so the engine's 30-minute default always applies, and the NOTE in that function claiming kb-core predates the field is stale.

**Retry queue.** An entry whose embed fails would be invisible to vector search forever, so `kb_core/embedding_retry.py` queues it and retries with backoff. `embedding_retry_queue` has `entry_id` (primary key), `attempts`, `last_error`, `next_attempt_at`, `status` (`pending` or `exhausted`), `created_at` and `updated_at`. `KnowledgeBase._embed_one` and `_enqueue_retry` enqueue whenever an embed returns None, raises, or a batch leaves an id unembedded: a fresh row is due in 60 s, re-enqueueing a pending row only updates `last_error`, and re-enqueueing an exhausted row revives it.

`EmbeddingRetryWorker`, started by `KnowledgeBase.start_embedding_worker`:

- `backfill()` queues every active `has_embedding = 0` entry (due immediately, reviving exhausted rows) at start and every `backfill_interval_seconds` (3600).
- The drain loop runs every `poll_interval_seconds` (60), or every `down_backoff_seconds` (300) while Ollama is unreachable.
- `_claim_due` compare-and-swaps `next_attempt_at` forward by `lease_seconds` (600) for up to `batch_size` (16) rows, so several processes can share one DB.
- Claimed entries are embedded with `embed_batch` on the worker's own client with `request_timeout` 180 s (a cold model load takes about 73 s).
- Success stores the vector, sets `has_embedding`, writes an `embedding_recovered` audit row and deletes the queue row. Failure applies `BACKOFF_SCHEDULE_SECONDS` (60, 300, 900, 3600, 10800) and flips the row to `exhausted` at attempt `MAX_ATTEMPTS = 6`, with an `embedding_exhausted` audit row and an ERROR `embedding-retry-exhausted` line. Inactive or missing entries are dropped.
- A tripwire WARNING fires when active vectorless entries have no queue row.

The worker is Postgres-only: `start_embedding_worker` returns immediately for a `SQLiteBackend`, and `main._open_kb` forces `EmbeddingRetryConfig(enabled=False)` on SQLite. Rows are still enqueued on SQLite and nothing drains them, and no HTTP path rebuilds embeddings. `GET /api/kb/embedding-queue` (admin) returns `pending`, `exhausted`, `oldest_pending_age_seconds`, `next_due_at`, `vectorless_unqueued` (should be 0), `worker_enabled` and `worker_running`. `KB_EMBED_WORKER_ENABLED`, `KB_EMBED_WORKER_BATCH_SIZE`, `KB_EMBED_WORKER_POLL_SECONDS` and `KB_EMBED_WORKER_TIMEOUT` configure it; the down backoff, backfill interval and lease have no env var.

## 11. Reading

**kb_search.** Parameters: `query` (optional), `project_ref`, `entry_type`, `tags` (all must match), `limit` 1 to 50 (tool default 5; `SearchRequest` default 10), and `include_stale`, `include_expired`, `include_superseded` (all default false). The schema also accepts `contributor` and `team`, but any non-None value returns `Error: contributor/team filters are not supported in HTTP mode.`, because `SearchRequest` deliberately omits them. `POST /api/kb/search` builds a `SearchQuery` (with `min_score_ratio` 0.5), passes `contributor=user.email` to `kb.search` as telemetry only, and returns the results with full entries and `filtered_count`. Rendering (`kb_core/formatting.py`):

```text
2 result(s) (3 low-relevance filtered)
[kb-00082] lesson_learned | Short title (90%)
  Long title (when different)
  #tag1 #tag2 | project | @contributor/team  [RESTRICTED]  [EXPIRES 5d]  [STALE]
  [SUPERSEDED by kb-00100]   (only when include_superseded)
```

When fewer than 3 results come back (`_SPARSE_THRESHOLD`), `collect_graph_hints` appends up to 3 lines under `Related entries via graph:` of the form `See also: [kb-id] title (via <edge_type> from <result id>)` or `(via <node id>)`: one-hop entry neighbours, or two hops through a non-entry node such as a tag, skipping superseded entries, using `GET /api/kb/graph/neighbors` with limit 10. The `Vector search unavailable (Ollama offline)` note in the tool is dead code over the service, because `vector_search_available()` always returns true.

**kb_get.** `POST /api/kb/get` takes 1 to 20 ids (the clients batch by 20) and returns one slot per id in request order. Missing or inactive ids are `found=false` (rendered `[kb-id] not found`); expired and superseded entries are returned. `touch_accessed` sets `last_accessed` on every found id in one UPDATE (skipped when the body sets `touch` to false). For a `mental_map` the route also computes pointer-rot (section 18d). The tool prefixes a superseded entry with `SUPERSEDED by kb-X — <title>`, costing one extra call only when the superseder was not requested, and logs `supersession-read invariant_breach` if the superseder cannot be found. Output is `format_entry_full`: header, meta line, `Updated by @x (vN)` when `updated_by` differs from the contributor, then `knowledge_details`.

**Listing.** `GET /api/kb/projects`, `/api/kb/contributors` and `/api/kb/teams` return `{items: [{name, entry_count}]}` over active entries, NULLs excluded, ordered by count. The `kb_list_projects` tool is always registered; `kb_list_contributors` and `kb_list_teams` only when `KB_CONTRIBUTOR` is set in the serving process.

**Search telemetry and feedback.** `hybrid_search` appends a `search_events` row (`query_text`, `result_count`, `top_score` = the best RRF score before filtering and NULL for filter-only searches, `match_source`, `contributor`, `created_at`) through a helper that never raises; there is no retention. `POST /api/kb/feedback` inserts into `agent_feedback` (`feedback_type` constrained to `missing`, `unhelpful` or `friction` by a pydantic Literal and a CHECK, plus `tool_name`, `query_or_params`, `detail`, `contributor`, `team`); the `kb_feedback` tool is always registered and is not blocked by the write policy, so unattended agents can report friction. Nothing over HTTP reads `agent_feedback`, `search_events`, `audit_events` or `entry_versions`, so analysis needs direct database access.

## 12. Knowledge graph

The graph connects entries to each other and to tags, projects, people, tools and concepts, so retrieval can walk from a hit to its neighbours. Tables: `graph_nodes(node_id PK, node_type, properties, created_at)` and `graph_edges(id, source, target, edge_type, properties, created_at, UNIQUE(source, target, edge_type))`, with foreign keys to `graph_nodes` and indexes on `node_type`, `source`, `target` and `edge_type`.

**Deterministic edges.** `GraphBuilder.build_for_entry(entry)` (`kb_core/graph/builder.py`) runs in one transaction with no LLM:

1. Capture the entry's old `supersedes` targets.
2. `delete_deterministic_edges(entry.id)`: delete every outgoing edge whose `properties` lack `source = 'llm'`, so LLM edges survive a metadata-only rebuild. This also deletes `extracted_from` provenance edges, which nothing re-adds.
3. Upsert the entry node with `{short_title, entry_type}`.
4. Each tag gives node `tag:<tag>` and edge `has_tag`.
5. `project_ref` gives `project:<ref>` and `in_project`.
6. `hints.supersedes` gives a `supersedes` edge only for ids that fullmatch `kb-\d{5}` (others are warned and skipped).
7. `hints.distinct_from` gives a `distinct_from` edge only for well-formed, existing, non-self ids.
8. Every distinct `kb-\d{5}` in `knowledge_details` (not titles), excluding self, gives a `references` edge.
9. `hints.related_entities` gives a dict-specified edge type (default `related_to`) or `related_to` for a bare string; the target is not validated and becomes an entry-typed node even if it is not a kb id.
10. `hints.person` gives `person:<lowercased>` and `mentions_person`; `hints.tool` gives `tool:<lowercased>` and `uses_tool`.
11. Last statement: `recompute_superseded_by` over the old targets, the new targets and the entry (section 6).

`_ensure_node` upserts with `ON CONFLICT(node_id) DO UPDATE SET properties = excluded.properties`, so a placeholder call that passes no properties resets an existing node's properties to `{}`; the graph page fills entry labels from `knowledge_entries`, so this is mostly cosmetic. `_add_edge` is `INSERT ... ON CONFLICT DO NOTHING`.

**When edges are built.** `KnowledgeBase.store` builds edges and enriches; `update` rebuilds edges on every update but embeds and enriches only when `knowledge_details` changed; `store_batch` builds each entry's edges and makes one `enrich_batch` call; reactivate rebuilds and re-enriches best-effort; ingestion builds edges per extracted entry and adds `note` nodes with `extracted_from` edges. Writes from a queued candidate set enrich off, so autonomous lessons get deterministic edges but no LLM entity edges (see [intelligent-kb.md §10](docs/intelligent-kb.md#10-distill-iii-queued-stores-shape-5-w1-w9)). A builder failure is caught by `KnowledgeBase._build_graph`, logged, and never fails the write.

**LLM enrichment.** `GraphEnricher.enrich_entry` (`kb_core/graph/enricher.py`) returns without touching the graph when `llm.is_available()` is false. It sends the titles, type, tags, project and the full `knowledge_details` with purpose `enricher` and asks for 2 to 6 specific entities as `{entity, entity_type, relationship}`. Items outside `_VALID_ENTITY_TYPES = {person, tool, concept, technology}` are dropped and at most `_MAX_RELATIONSHIPS = 8` are kept. The relationship label becomes `edge_type` verbatim, unvalidated. In one transaction it deletes the entry's LLM edges and inserts the new ones with `{"source": "llm"}`, so an unparseable reply still clears the old edges. Target nodes are deduplicated: candidates are vocabulary names sharing the first 3 characters, and the best `difflib.SequenceMatcher` ratio at or above `_DEDUP_SIMILARITY_THRESHOLD = 0.85` wins regardless of entity type (so `concept:async-io` can merge with `technology:asyncio`). The vocabulary is `get_graph_vocabulary`'s top 200 non-entry nodes by connection count, tag and project nodes included, loaded lazily and cached on the enricher; only batch paths call `clear_vocab_cache()`, so in the long-running service single stores never refresh it after the first load. Failures are logged at ERROR with `ENRICHMENT_FAILURE_MARKER = "enrichment-failure"`. LLM edges are ignored by the supersession invariant.

**Batch enrichment.** `enrich_batch` sends one block per entry with `knowledge_details` truncated to `_MAX_BATCH_CONTENT = 500` characters, purpose `enricher_batch`, and asks for a JSON object keyed by entry id. Ids not in the batch and non-list values are ignored. An unparseable reply falls back to `enrich_entry` per entry; a None reply returns 0 and clears nothing. On success every entry in the batch has its LLM edges replaced, so an entry the model omitted ends with none. `enrich_all` has no production caller.

**Traversal primitives** (`kb_core/graph/queries.py`, exposed as `KnowledgeBase.graph`):

| Function | Behaviour |
|---|---|
| `get_neighbors(node_id, edge_types, direction, limit=50)` | `(neighbor, edge_type, outgoing|incoming)`; the limit is shared across directions |
| `bfs_entries(start, max_depth=2, edge_types, limit=20)` | Queue BFS returning `(entry_id, depth, path)` for kb-id nodes at depth above 0 |
| `find_path(source, target, max_depth=4)` | `(source, edge_type, target)` hops; `[]` when source equals target, None when unreachable |
| `entries_for_scope(scope, ...)` | Scope grammar: `kb-XXXXX`, `project:X`, `tag:X`, `person:X`, `tool:X`, a bare entry type, else a generic node id; `order_by` whitelist `created_at`, `updated_at`, `confidence_level`, `short_title` (anything else becomes `created_at`) |
| `supersedes_chain(entry_id)` | The supersedes chain, oldest first |
| `get_graph_vocabulary(max_nodes=200)` | `{node_type: [names]}` for non-entry nodes by connection count |

None of these are MCP tools. They are served as REST to any authenticated caller: `GET /api/kb/graph/full` (section 24a), `/api/kb/graph/neighbors` (`limit` 1 to 200, default 50; `direction` `outgoing`, `incoming` or `both`; repeatable `edge_types`), `/api/kb/graph/bfs` (`max_depth` 1 to 5, default 2; `limit` 1 to 100, default 20), `/api/kb/graph/path` (`max_depth` 1 to 10, default 4; `found=false` when unreachable, `found=true` with no hops when source equals target), `/api/kb/graph/supersedes-chain`, `/api/kb/graph/scope-entries` and `/api/kb/graph/vocabulary` (`max_nodes` 1 to 1000, default 200). The agent loop uses them as internal tools (section 14).

## 13. Query (kb_ask)

`kb_ask` returns full entries relevant to a question. Only strategy `auto` is served: both MCP twins answer any other strategy with `Error: strategy '<s>' requires a local KB — only 'auto' is supported in HTTP mode.`, and `AskRequest` has no strategy field. The path is `POST /api/kb/ask {question, scope, include_graph_context, limit 1..50 default 20}` to `KnowledgeBase.ask` to `kb_core/query.retrieve_entries`, with `agentic` and `max_tool_calls` taken from server configuration (a request cannot set them).

- **Default path** (`KB_AGENTIC_QUERY` on and a query LLM present): `agentic_query` (section 14). It takes no scope and no `include_graph_context`, so both arguments are ignored; only `limit` truncates the result. When the agent returns no entries, `retrieve_entries` returns `([], 0)` with no fallback search.
- **Non-agentic path** (`KB_AGENTIC_QUERY` off, or no query LLM): the `QueryPlanner` (when an LLM exists) rewrites only the search text; its planned strategy, scope and target are ignored. Then `_auto_search_entries` runs `hybrid_search` (`include_stale` false, honouring `project:`, `tag:` and entry-type scopes) and expands one hop: `get_neighbors(limit 10)` per hit, adding active, non-superseded kb-id neighbours with context `linked from X via T` or `links to X via T` until `limit`.

Output is a header `[Agent: N tool calls]` (N counts agent LLM turns, 0 for the fast path and the non-agentic path) followed by each entry as `format_entry_full` with a `↳ context` line; no entries gives `[Agent: N tool calls] No results found.` The client allows 300 s for the read (`HttpBackend._LONG_TIMEOUT`).

The `decision_trace`, `timeline`, `related` and `connection` strategies survive only as unreachable code in `src/personal_kb/tools/kb_ask.py`; their graph primitives are reachable as agent tools and as `/api/kb/graph/*` routes (section 12).

## 14. Agentic retrieval loop

`agentic_query` in `kb_core/graph/agent.py` finds entries; it does not answer. Instead of classifying a question into a strategy up front, it exposes graph operations as tools and lets the model compose them. Six internal tools (not MCP tools):

| Tool | Purpose and caps |
|---|---|
| `hybrid_search` | Search with optional filters; `limit` capped at 10, `include_stale` false |
| `graph_neighbors` | Edges from any node; `limit` capped at 20; skips superseded entry neighbours |
| `list_graph_nodes` | Graph vocabulary by type; 20 names per type |
| `decision_chain` | Supersedes chain, labelled original, vN, current |
| `scope_entries` | Entries for a project, tag or entry type; `limit` capped at 20; skips superseded |
| `done` | Final answer: `entry_ids` and `reasoning` |

**Fast path.** First the loop runs `hybrid_search(limit 5, include_stale false)` on the raw question. If the top RRF score is at least `_FAST_PATH_THRESHOLD = 0.03`, it returns those (up to 5) entries with context `search match (score: x)`, zero LLM calls and no graph expansion. With `RRF_K = 60` one leg alone maxes at 0.0164, so the fast path needs the entry in both the FTS and vector legs; with no embedder it never fires and every question enters the loop.

**The loop.** The model receives the question, the weak fast-path results as compact lines (never `knowledge_details`, which keeps the context small across turns) and an instruction to find the best entries, via `llm.generate_chat(messages, purpose="agent", system=...)`, and must answer one JSON object `{tool, args}`. The loop is `range(max_tool_calls)` LLM iterations, and every iteration consumes budget: a tool call, a `done`, a parse failure (`ERROR: Could not parse your response as JSON...`, event `parse_error`) and a rejected repeat of an identical call (`ERROR: You already made this exact tool call...`, event `tool_dedup`). With the default of 4, an agent that finds results makes 3 tool calls and then `done`. The system prompt hard-codes `NEVER make more than 4 tool calls` regardless of `KB_AGENTIC_MAX_CALLS`.

`done` keeps entry ids that exist and are active; an explicit pick of a superseded entry is kept and logged. `done` with no valid ids returns an empty result. On exhaustion, or when the LLM returns None (provider failure breaks the loop), the exhaust path collects every active, non-superseded kb id mentioned anywhere in the conversation (the seeded fast-path list included) as `agent fallback` entries, and only if there are none returns the fast-path results, with reasoning `Exhausted tool call budget`. `_dispatch_tool` has no try/except, so a hallucinated argument (an invalid `entry_type`, a non-numeric `limit`) raises out of `agentic_query` and fails the whole request (500 on `/api/kb/ask`, an `error` event on the stream).

**Toggles.** `KB_AGENTIC_QUERY=FALSE` selects the non-agentic path (section 13). Flags parse only the case-insensitive string `true` as on (`1` and `yes` are off). A non-integer `KB_AGENTIC_MAX_CALLS` fails startup. With the agent off, `agent_turns` is always 0, so the coverage check in section 15 never runs.

**Events** passed to the optional callback: `fast_path`, `agent_started`, `thinking`, `tool_call`, `tool_result`, `tool_dedup`, `parse_error` and `agent_done` (also emitted on exhaustion).

**Eval numbers.** `tests/eval/corpus.json` has 58 entries and `queries.json` 36 queries. The search baseline (`baseline.json`) scores 28 queries at mean MRR 0.9137, NDCG 0.9302 and recall 1.0. The agent baseline (`agent_baseline.json`) scores 13 queries at 1.0 on all three, with `turns_used` 0 for 8 queries (the fast path) and 1 for q04, q05, q06, q09 and q10.

See: `kb_core/graph/agent.py`, `kb_core/query.py`.

## 15. Answer synthesis (kb_summarize)

`kb_summarize` returns a prose answer with `[kb-XXXXX]` citations instead of raw entries. The path is `POST /api/kb/summarize {question, scope, limit 1..50 default 20}` to `KnowledgeBase.summarize` to `kb_core/query.synthesize_answer`, which takes keyword-only `agentic`, `agentic_synthesis` and `max_tool_calls` from server config. The `src/personal_kb` copies of these functions are thin wrappers.

1. **Retrieval.** `retrieve_entries` (section 13) returns `(entries_with_context, agent_turns)`. No entries returns `No entries found matching your question.`
2. **Coverage check.** It runs between retrieval and synthesis when three conditions hold: `KB_AGENTIC_SYNTHESIS` is on, a query LLM exists, and `agent_turns > 0`. `assess_coverage` in `kb_core/coverage.py` makes one call (purpose `coverage`) with each entry's id, short title, tags and the first 200 characters of `knowledge_details`, biased hard toward "no gaps", and expects `{has_gaps, suggested_query, reason}`. A None reply, a malformed reply or any exception returns `CoverageResult(has_gaps=False)`, so it never blocks synthesis. On a gap with a suggested query, `_auto_search_entries` runs with the request's scope and limit and `include_graph_context` true, and `_merge_entries` adds the new entries after the originals, deduplicated by id. The fast path and the non-agentic path never run it.
3. **Synthesis.** The provider is `synthesis_llm` if present, else `query_llm`. `_synthesize` formats each entry as `[kb-id] short_title #tags`, a `[SUPERSEDED by X]` line when applicable, `Context: ...` and the full `knowledge_details` with no truncation, and calls `generate` with purpose `query_synthesis` and `max_tokens` 4096. The system prompt says to answer only from the entries, cite ids, note conflicts and be concise. Retrieval returns at most `limit` entries and the coverage re-search can append up to `limit` more, so the prompt carries up to twice `limit` entries in full (100 at the maximum `limit` of 50). Coverage sees 200-character previews because it only judges relevance; synthesis sees everything because it must be accurate.
4. **Fallbacks.** A None synthesis reply returns `(LLM synthesis failed — showing raw results)` plus `format_entry_full` of every entry; no LLM at all returns `(LLM unavailable — showing raw results)` plus the same.

The synthesis role is a separate client whose model is the constant `claude-sonnet-4-6` (Bedrock `us.anthropic.claude-sonnet-4-6`) on the `KB_QUERY_PROVIDER` provider; there is no env override (section 16). The client allows 300 s for the read; the route docstrings still say 120 s. Events: `synthesis_started {entry_count}` and `synthesis_done`.

## 16. LLM providers, caching and the call ledger

**Providers.** `LLMProvider` (`kb_core/llm/provider.py`) is a runtime-checkable protocol: `is_available()`, `generate(prompt, *, purpose, system=None)`, `generate_chat(messages, *, purpose, system=None)` and `close()`, where None means unavailable. `purpose` is a required keyword with no default, so mypy rejects a new call site that skips it. `ProviderConfig` has three roles, each a `ProviderRoleConfig` bundling the active provider and credentials for all three providers:

- **extraction**: graph enrichment, ingest summarize and extract, the ingest dedup agent;
- **query**: the agent loop, the planner, coverage, the query-stream classifier;
- **synthesis**: `query_synthesis`, `chat` (first choice), `listener_vote`.

`kb_service/config.build_provider_config` sets the extraction provider from `KB_EXTRACTION_PROVIDER` and both the query and synthesis providers from `KB_QUERY_PROVIDER` (there is no separate synthesis variable). Extraction and query use `KB_ANTHROPIC_MODEL` (`claude-haiku-4-5`) or `KB_BEDROCK_MODEL` (`us.anthropic.claude-haiku-4-5-20251001-v1:0`); synthesis swaps in the constants `claude-sonnet-4-6` or `us.anthropic.claude-sonnet-4-6`; Ollama uses `KB_OLLAMA_MODEL` (`qwen3:4b`) for every role. `KnowledgeBase.create` builds one client per role through `_build_llm` (lazy imports); a role whose provider is unknown is None.

- **Anthropic** (`kb_core/llm/anthropic.py`): lazy `AsyncAnthropic` (key from `ANTHROPIC_API_KEY` or SDK discovery), `is_available` true when the SDK imports (no key check), `max_tokens` 4096, per-request timeout `KB_ANTHROPIC_TIMEOUT` (30 s); any exception returns None. `AnthropicProviderConfig` optionally carries `effort` (one of `low`, `medium`, `high`, `xhigh`, `max`; sent as the request's `output_config.effort`) and `thinking` (`disabled` or `adaptive`; sent as `thinking = {"type": mode}`). Either field being `None` (the default) sends nothing, so every request is byte-identical to the pre-effort behaviour; no production path sets either field today — they exist so evals can compare model arms at a chosen effort or with thinking disabled, and invalid values are rejected by the config's `__post_init__`.
- **Bedrock** (`kb_core/llm/bedrock.py`): the Converse API, `max_tokens` 4096, credentials by priority `KB_AWS_PROFILE`, then a profile named `personal_kb_bedrock` if it exists, then `AWS_BEARER_TOKEN_BEDROCK`, then environment credentials; up to 3 retries with 1, 2 and 4 s backoff; an empty or reasoning-only reply counts as failure. It neither caches nor records.
- **Ollama** (`kb_core/llm/ollama.py`): `/api/generate` and `/api/chat`, availability via `GET /api/tags`, timeout `KB_OLLAMA_LLM_TIMEOUT` (120 s).

`json_parser.parse_json_object` and `parse_json_array` strip markdown fences and take the first valid JSON value.

**Purpose to client and model** (code defaults):

| Purpose | Client | Default model | Called by |
|---|---|---|---|
| `agent` | query | `claude-haiku-4-5` | agent loop, each turn |
| `planner` | query | `claude-haiku-4-5` | `QueryPlanner` (agent off) |
| `coverage` | query | `claude-haiku-4-5` | coverage check after an agent-resolved summarize |
| `classifier` | query | `claude-haiku-4-5` | query-stream classifier |
| `enricher` | extraction | `claude-haiku-4-5` | single-entry enrichment |
| `enricher_batch` | extraction | `claude-haiku-4-5` | batch enrichment (store_batch, ingest) |
| `ingest_summarize` | extraction | `claude-haiku-4-5` | ingestion |
| `ingest_extract` | extraction | `claude-haiku-4-5` | ingestion, per chunk |
| `dedup_agent` | extraction | `claude-haiku-4-5` | ingest dedup agent |
| `query_synthesis` | synthesis | `claude-sonnet-4-6` | `kb_summarize` |
| `chat` | synthesis, else query | `claude-sonnet-4-6` | web chat |
| `listener_vote` | synthesis | `claude-sonnet-4-6` | listener, 3 per decision |
| `surprise_detector_shape2`, `surprise_detector_shape3` | detector (Anthropic only) | `claude-sonnet-5-5` | surprise capture |
| `surprise_distiller`, `store_distiller` | distiller (Anthropic only) | `claude-sonnet-5-5` | surprise capture, queued stores |
| `surprise_critic`, `store_critic` | critic (Anthropic only) | `claude-sonnet-5-5` | surprise capture, queued stores |
| `feedback_summary` | — | — | dead stdio local path; never recorded |
| `seed_resolutions` | script's own client | — | `seed_resolutions.py`; never recorded |

The six surprise purposes use dedicated Anthropic clients that ignore `KB_QUERY_PROVIDER` and `KB_ANTHROPIC_MODEL`; their models, call counts and cost are in [intelligent-kb.md §21](docs/intelligent-kb.md#21-what-the-automation-costs).

**Prompt caching.** `AnthropicLLMClient` never sends a top-level `cache_control`, because the API treats that as automatic caching and would write the whole prompt, one-shot user message included, at the write premium. With a system prompt it sends the system as a content-block array with an explicit `{type: ephemeral}` breakpoint: 5-minute TTL by default, 1 hour for purposes in the constructor argument `system_cache_1h_purposes` (kb-service passes `{surprise_distiller}` only). `generate_chat` also puts a rolling breakpoint on the last content block of the last message, so an agent loop reuses the prefix built by the previous turn. A prompt below the model's minimum cacheable size caches nothing.

**LLMProvider.generate_chat()** takes `messages: list[Message]` (`role`, `content`), `purpose` and an optional `system`. Anthropic adds the breakpoints above, Bedrock maps messages to Converse messages, and Ollama uses `/api/chat`. The agent loop and chat use it.

### LLM call ledger (llm_calls)

Every Claude API call made by a kb-service process is recorded, by purpose, in the service-DB table `llm_calls`, so spend and cache behaviour can be attributed to a caller. `kb_core.llm.provider.LLMPurpose` is a Literal of the 20 purposes in the table above; every test double declares `purpose` without a default so the tests fail closed too. Bedrock and Ollama accept `purpose` and ignore it.

The recorder is injected at construction: `AnthropicLLMClient(config, recorder=...)`, and `KnowledgeBase.create`, `create_sqlite` and `create_postgres` take `llm_recorder`, which `_build_llm` attaches to the Anthropic clients it builds from config (never to pre-built overrides). kb-core stays storage- and env-free: the recorder is a synchronous `Callable[[LLMCallRecord], None]`, and `kb_service.llm_calls.record_llm_call` schedules the insert as a task on the running loop so the LLM call is never slowed. A recorder or insert failure only logs a warning (`llm-call recorder failed purpose=...`, `llm_calls insert failed purpose=... model=... response_id=...`) and never fails the call. kb-service passes the recorder to `_open_kb`'s factory and to the surprise detector, distiller and critic clients; an AST test asserts every `AnthropicLLMClient(...)` in kb_service passes `recorder=`. The lifespan drains pending inserts (5 s, then cancels) after the KB closes and before the service pool closes.

Each row holds: `ts` (UTC, seconds, after the final attempt); `purpose`; `model`; `outcome` (`ok` when the response had a text block, `no_text` when it had none, `error` when the call raised, `asyncio.CancelledError` included, with `error_type` the exception class name); `response_id` and `stop_reason`; the usage fields `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_creation_1h_input_tokens` (from `usage.cache_creation.ephemeral_1h_input_tokens`) and `cache_read_input_tokens`, each NULL when the SDK did not report an int; `diagnostics_state`; `previous_message_id`; `cache_miss_reason`, `cache_missed_input_tokens` and `diagnostics_raw` (the response's `diagnostics` object as compact JSON, at most 2000 chars); `system_sha256` (the first 16 hex chars of the SHA-256 of the system blocks, `cache_control` included, NULL without a system prompt); and `latency_ms` (SDK retries included). With `diagnostics_state` `sent`, a NULL `cache_miss_reason` means the API reported no divergence, `pending` means it had not yet computed a reason, and `unparsed` means the shape did not match the documented one (read `diagnostics_raw`; the client logs `llm-diagnostics unparsed shape` once per model per process).

`diagnostics_state` is one of four values. `sent`: the request carried `diagnostics: {previous_message_id: ...}` through the SDK's `extra_body` (the pinned SDK has no parameter for it; the result is read from the response's `model_extra`). `off_config`: `KB_LLM_DIAGNOSTICS` turned it off. `off_disabled`: an earlier 400 disabled it for this client. `fallback`: this call's first attempt got a 400 naming diagnostics and was retried once without it; these rows have a NULL `previous_message_id`.

`previous_message_id` is the last response id for the same `(purpose, model)` in this process, held in a module-level dict in `kb_core/llm/anthropic.py` and shared by every client of that model. It is updated only by a `sent` call that returned a string id, and there is no lock, so the three concurrent `listener_vote` calls all send the same previous id and the last to finish wins.

On a 400 whose message contains `diagnostics`, the client sets `_diagnostics_disabled`, logs `llm-diagnostics disabled model=... reason=400` once, and retries once without the field; the rejected attempt writes no row of its own. `KB_LLM_DIAGNOSTICS` set to `0`, `false`, `off` or `no` stops sending diagnostics while still recording every call.

Only recorder-equipped kb-service clients send diagnostics or write rows. Bedrock and Ollama roles, the stdio server, `packages/kb-service/scripts/seed_resolutions.py` and `scripts/surprise_eval` write none, and `feedback_summary` is reachable only through the dead stdio local path, so neither it nor `seed_resolutions` produces rows in production. `prune_llm_calls` deletes rows older than 90 days (`LLM_CALLS_RETENTION_DAYS = 90`) and runs only inside `surprise_worker.drain_once`, which returns before pruning when `KB_SURPRISE_CAPTURE` is off, so with capture off the table is unbounded.

Reading the ledger: `messages_changed` is the steady state for single-shot purposes whose prompt varies. `system_changed` is expected for `chat` (the system prompt embeds the session's entries) and for `ingest_summarize`/`ingest_extract` (the code versus prose supplement). `listener_vote` calls go through `generate` with no system prompt, and `generate` sets a cache breakpoint only on a system block, so listener votes are never cached and always show zero cache tokens. Prompts below the model's cacheable minimum always show zero cache tokens; the short Haiku-role system prompts usually fall in that range. The per-purpose hit ratio is `SELECT purpose, model, SUM(cache_read_input_tokens)*1.0 / NULLIF(SUM(COALESCE(input_tokens,0)+COALESCE(cache_read_input_tokens,0)+COALESCE(cache_creation_input_tokens,0)),0) FROM llm_calls WHERE outcome='ok' GROUP BY purpose, model`, and the integrity audit `SELECT COUNT(*) FROM llm_calls WHERE outcome='ok' AND input_tokens IS NULL` should be 0. When diagnostics is unavailable, counting distinct `system_sha256` per purpose checks that the cached prefix is stable.

`GET /api/kb/observe/llm-calls` serves the table to admins; the observe API is documented in [intelligent-kb.md §20](docs/intelligent-kb.md#20-measure-telling-whether-the-loop-works).

## 17. Ingestion

Ingestion turns a file, a URL or a text blob into several self-contained entries plus a note node, replacing earlier entries from the same source.

**Endpoints** (`kb_service/routes/ingest_routes.py`, any authenticated user): `POST /api/kb/ingest/text {content, source_name, project_ref, dry_run}`, `POST /api/kb/ingest/url {url, content?, project_ref, dry_run}` (a supplied `content` skips the fetch and HTML extraction, for SSO or JavaScript pages) and `POST /api/kb/ingest/file` (multipart `file`, `project_ref`, `dry_run`). A non-dry run first requires an interactive surface (403 otherwise). Then, unless `KB_SKIP_SAFETY=TRUE`, a missing detect-secrets or scrubadub distribution is a 503, because kb-core would otherwise degrade silently. `/file` streams the upload in 1 MiB chunks, returns 413 above `KB_INGEST_MAX_FILE_SIZE` (10485760; exactly the limit is accepted) and 422 for an empty filename, and uses the file's basename as the source key. A missing extraction LLM or embedder raises `RuntimeError` in kb-core, which becomes 503. Skips and fetch failures are HTTP 200 with `action` `skipped` or `error`. The response mirrors kb-core's `FileResult`: `path`, `action` (`ingested`, `skipped`, `flagged`, `error`, `unchanged` or `dry_run`), `reason`, `entry_count`, `entry_ids`, `summary`, `chunks_processed`, `chunks_skipped` and `chunks_flagged`. The web UI's Home page Add to KB panel is the browser-side caller of these three endpoints: its drop zone and Browse picker upload files, its URL field posts a URL, and it applies an optional project to both.

**Pipeline.** `FileIngester` (`kb_core/ingest/ingester.py`) for a file:

1. Deny-list on the filename (keys, certificates, env and credential files, archives, binaries, media, database files).
2. Extension and name allowlist (about 60 extensions such as `.md`, `.py`, `.ts`, `.yaml`, `.json`, `.html`, `.pdf`, and names such as `Dockerfile`, `Makefile`, `README`).
3. Symlink rejection, then the size limit.
4. Read (PDF through pymupdf, otherwise UTF-8 with replacement).
5. SHA-256 content hash; an unchanged, active `ingested_files` row returns `unchanged`.
6. scrubadub PII redaction over the whole content (types recorded in `ingested_files.redactions`).
7. `summarize_file` (purpose `ingest_summarize`); an unavailable LLM returns `action=error`.
8. `chunk_content`, then per chunk: secret scan (a hit drops the whole chunk), the dedup agent, and `extract_entries` (purpose `ingest_extract`) with titles already extracted from earlier chunks.
9. One transaction of `KnowledgeStore.create_entry` (`source_context` `Ingested from <source>`) plus `GraphBuilder.build_for_entry` per entry. This bypasses `kb.store`, so ingested entries skip the near-duplicate guard, the supersession checks and the write-policy queue.
10. One `embed_batch` and one `enrich_batch` (failures only log; on Postgres the retry worker's backfill later queues vectorless entries).
11. Only after the new entries exist, deactivate the previous ingestion of the same source (`change_reason` `re-ingest: replaced by newer ingestion of the same source`).
12. Upsert the note node (`note:<path or url>`) with `extracted_from` edges, and upsert the `ingested_files` row (`relative_path` UNIQUE, `content_hash`, `note_node_id`, `entry_ids`, `summary`, `file_size`, `file_extension`, `project_ref`, `redactions`, `contributor`, timestamps, `is_active`).

`ingest_text` applies the allowlist to `source_name` and the size limit, with no deny-list; URL ingestion applies the hash, PII redaction and size steps with the URL as the source key. A dry run summarizes and extracts only and returns `action=dry_run` with the entry count. Progress events: `ingest_summarizing`, `ingest_start {total_chunks}`, `ingest_chunk_start`, `ingest_chunk_done`, `ingest_done` and `ingest_error`.

**Chunker.** `chunk_content(content, chunk_size=16000, overlap=600)` returns one chunk when the content fits; otherwise it splits at H1/H2 headings, merges adjacent sections up to the size, sub-splits oversized sections at blank lines then newlines, and prepends the last `overlap` characters of the previous chunk, snapped forward to a newline. Each chunk carries its first heading, passed to the extractor as `Section: ...`. Extraction truncates each chunk to 100_000 characters.

**Safety.** The deny-list applies only to files. A secret hit drops the whole chunk and counts it in `chunks_flagged`, which the stdio formatter labels `redacted (secrets)` although nothing is redacted. `FileIngester` ignores `IngestConfig.skip_safety`, so `KB_SKIP_SAFETY` only bypasses the route's library check; with the libraries installed, chunk skipping and PII redaction always run. `redact_secrets` is not used here, and a file-level `action=flagged` is never produced.

**Dedup agent.** `DedupAgent.check_chunk` (`kb_core/ingest/dedup_agent.py`, enabled by `KB_AGENTIC_INGEST`) searches with the chunk heading and first 500 characters (`include_superseded=True`) and asks the extraction LLM for `extract`, `skip` or `partial` only when the top score reaches `KB_INGEST_DEDUP_THRESHOLD`. The maximum hybrid score is 2/61 = 0.0328, below the default 0.06, so the LLM check is never reached and every chunk is extracted. Identical files are still caught by the content hash.

**Extraction prompts** (`kb_core/ingest/extractor.py`). The summary is 2 to 3 sentences about what is unique to the source, with a code supplement for about 35 source extensions and a prose supplement for documents. Extraction asks for 1 to 10 entries per chunk, each with `short_title`, `long_title`, self-contained `knowledge_details`, an `entry_type` among the four value types and 2 to 5 lower-cased tags, and `[]` when nothing goes beyond public knowledge; `mental_map` is never produced.

**URL fetch.** `ingest_url` uses httpx with `follow_redirects=True` and a 30 s timeout, then `trafilatura.extract(output_format='txt', include_tables=True, favor_precision=True)`; empty output returns an error explaining the page may need JavaScript. There is no scheme, host or private-address check on the URL.

**Thin-client kb_ingest.** The stdio `kb_ingest` uploads single files through `/api/kb/ingest/file`; globs expand against the MCP process's working directory, directories are rejected with `Error: Directory ingest is not supported in HTTP mode — ingest individual files.`, and `recursive` is ignored. `/mcp` never registers `kb_ingest`, because it would read files on the service host; `kb_ingest_url` with `content=` covers that case.

## 18. Mental maps

The `mental_map` entry type is the directory tier of the KB: an orientation node whose body is pointers and structure, never retrievable values, so an agent learns where to look without the map going stale. The settled design lives in `docs/mental-map-prespec.md` §7. This section covers the type itself: storage, the decay exemption, the pointer requirement, the lint, pointer-rot and the maps index. The hook's directory push, map eligibility, the machine principal and nightly maintenance are in [intelligent-kb.md §16](docs/intelligent-kb.md#16-orientation-push-maps-directory-and-tool-inventory) and [§18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract).

### 18a. The mental_map entry type

`EntryType.MENTAL_MAP = "mental_map"` is the fifth member of `EntryType` in `kb_core/models/entry.py`, sharing all storage, versioning, FTS, embedding and graph plumbing with the four value-bearing types. `kb_core/confidence/decay.py:compute_effective_confidence` checks for `MENTAL_MAP` first and returns `base_confidence` without consulting any half-life table:

```python
if entry_type == EntryType.MENTAL_MAP:
    return base_confidence
```

The rationale is double-edged. A map holds no retrievable value of its own, so nothing on it goes stale on a clock. And the access-aware self-heal (`touch_accessed` resets `last_accessed`, the decay anchor, on every `kb_get`) would make a constantly-surfaced map look fresh while its pointers rotted, and a correct but cold map trip stale. Both directions are backwards for an orientation node, so the clock is bypassed entirely and freshness for a map is pointer validity, computed on read (section 18d). `HALF_LIVES.get(entry_type, 365.0)` means a future enum member can never `KeyError` the decay path.

The graph builder turns every distinct `kb-\d{5}` in a map body (other than its own id) into a `references` edge from the map to the detail entry. That `references` edge set is what pointer-rot, the maps-index `pointers`, the listener and the nightly loop's input all read. A map can never supersede anything and is never a qualifying superseder (section 6). `POST /api/kb/entries/{id}/deactivate` refuses a map with 422, because a soft delete would strip its outbound edges; a bad map is hard-deleted with `DELETE /api/kb/maps/{map_id}`, which only the machine principal may call (see [intelligent-kb.md §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract)). Headless and autonomous surfaces cannot create a map.

### 18b. Required outbound pointer

A map is defined by what it points to, so a map with zero outbound pointers is an orphan note and is rejected. An outbound pointer exists when `knowledge_details` contains a `kb-XXXXX` reference (`re.compile(r"kb-\d{5}")`, the builder's own pattern) or `hints.related_entities` contains a dict with a non-empty `id`/`target` or a bare non-empty string. Tag, project, person and tool hints do not count, and neither does `hints.supersedes` (a map cannot supersede).

Enforcement points:

- The client tool (`src/personal_kb/tools/kb_store.py` `_mental_map_has_pointer`) returns `ORPHAN_MAP_ERROR` on create before any backend call: *"A mental_map entry requires at least one outbound pointer (a kb-XXXXX reference in knowledge_details, or a related_entities hint). A map with zero pointers is an orphan note, not a map."*
- The server (`kb_service/routes/map_write_guards.py` `_check_orphan_mental_map`) returns 422 on create and on update, evaluated on the effective post-update body and hints (stored values merged with the request, so a tags-only update of a good map passes), per `store_batch` entry (`entry {i}: ...`), and in the nightly loop's `create_map` operation. Its message is shorter than `ORPHAN_MAP_ERROR`.
- The kb-core library path itself is unguarded; only direct `KnowledgeBase` calls skip the check.

Superseded pointers are rejected separately (section 6). A map whose only pointers are in `related_entities` passes validation but produces `related_to` edges, not `references` edges, so it is invisible to the maps-index pointer list, pointer-rot, the listener and the nightly loop.

### 18c. Fact-free map lint

The §7.3 invariant, "no retrievable value in assertion position", is checked by a deterministic regex lint in `kb_core/map_lint.py`. `lint_map_body(text)` is a pure function that never raises and returns `list[MapLintFinding(code, message)]` with `MapLintCode` values evaluated in this order:

1. `kb-\d{5}` references are stripped first, so pointer digits never read as numerals.
2. `url`: `https?://\S+`.
3. `path`: `~/`-rooted, or `/seg/seg`; the pattern is unanchored, so `a/b/c` matches through `/b/c`.
4. `env_var`: `\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b`.
5. `dotted_identifier`: `\b[A-Za-z_]\w*\.[A-Za-z_]\w*`.
6. `quoted_literal`: a double-quote or backtick pair (single quotes are ignored so apostrophes pass).
7. `config_numeral`: any decimal, any integer of 4 or more digits, or any numeral directly after `=` or `:`, except a numeral followed by a plural noun (`\s*[A-Za-z]+s\b`), which is a count of parts (`three stages`, `12 nodes`) and allowed.
8. `over_budget`: a body longer than `map_body_budget(n)` = `MAP_BODY_BASE_CHARS = 900` + `MAP_BODY_PER_POINTER_CHARS` (175) × n, where n is the number of distinct `kb-[0-9]{4,5}` ids in the raw body (`map_pointer_ids`), gets the note "cut orientation prose, not pointer glosses".

The discriminator is value versus pointer: would a reader act on this string directly (a retrievable value, forbidden) or follow it to a source (a pointer, fine)? The budget is an advisory proxy, not the definition.

`src/personal_kb/tools/map_lint.py` and its kb-service twin only render findings as `Map lint (advisory): ...`. `kb_store` runs the lint on every map create and on updates that supply `knowledge_details`, and prepends the advisories above the `Created/Updated` block (below backend warnings) via `_prepend_map_advisories`; `kb_store_batch` attaches them per entry. The lint is advisory for every caller except the configured machine principal, for whom any finding is a 422 on `/store`, `/store_batch` and the nightly map operations; `POST /api/kb/map-lint` is a dry run of the same check (see [intelligent-kb.md §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract)).

### 18d. Pointer-rot on read

Because maps do not decay, the freshness signal moves to retrieval. `POST /api/kb/get` (`kb_service/routes/kb_read_routes.get_entries`) computes it server-side for each found `mental_map`: `kb.graph.neighbors(id, direction="outgoing")` with no edge-type filter, keeping neighbour ids that match `kb-\d{5}` (tag, project, person and tool nodes drop out before any lookup), deduplicated, then `kb.get` per target. A missing target is skipped. A target with `superseded_by` set yields `PointerRotTarget(target, superseded_by)`, which wins even if the target is also inactive, because naming the replacement is actionable; an inactive target yields `PointerRotTarget(target, None)`. The list is sorted by target id. A deactivated map itself is reported not found. The tools render it with `_render_pointer_rot`, two-space indented under the full entry:

```text
  Pointer-rot:
    [kb-00310] superseded by [kb-00342]
    [kb-00214] deactivated
```

Non-map output is unchanged. There is no freshness inheritance: a map's confidence never depends on its leaves, which would manufacture a permanent-staleness trap.

### 18e. Maps index

`kb_core/preflight.py:_maps_sql` selects a project's maps:

```python
"SELECT id, short_title, long_title, knowledge_details "
"FROM knowledge_entries "
"WHERE is_active = 1 AND project_ref = ? "
"AND entry_type = 'mental_map' "
# + optional team clause
"ORDER BY created_at DESC"
```

`build_project_context` renders the result as the `Maps:` section that leads preflight output (section 19), one line per map `  - [<id>] <short_title> — <long_title>`, omitted when the project has none. `KnowledgeBase.maps_for_project` uses the same query and extracts each map's pointers from `knowledge_details`.

`GET /api/kb/maps-index` (`kb_service/routes/maps_routes.py`, any authenticated user) selects the distinct `project_ref` values that have an active map, calls `maps_for_project` per project, drops projects with none, and returns `MapsIndexResponse{projects: [{project_ref, maps: [{id, short_title, long_title, pointers}]}]}`, projects sorted ascending and maps `created_at DESC`. `pointers` lists the kb ids in the map body, deduplicated, excluding the map's own id. Maps are neither expiry- nor superseded-filtered, the index is computed per request from the live DB (there is no sidecar file), and a DB failure is a 500. The hook consumes it as described in [intelligent-kb.md §16](docs/intelligent-kb.md#16-orientation-push-maps-directory-and-tool-inventory).

## 19. Project preflight (kb_preflight)

`kb_preflight` is a no-LLM table of contents for one project, called at session start. The path is `kb_preflight(project_ref, since?)` to `GET /api/kb/preflight?project_ref=&since=` (422 on a malformed or zero `since`, which uses the TTL grammar `Nh`, `Nd`, `Nw`) to `KnowledgeBase.preflight` to `kb_core/preflight.build_project_context`, pure SQL.

Output starts with the header `<project_ref> (N entries)` and the line `Use kb_get to read full details for any entry below.`, then up to five sections:

1. **Maps** — every active `mental_map` of the project, newest first, uncapped (section 18e). Not expiry- or superseded-filtered.
2. **Expiring** — `expires_at` between 7 days ago and 30 days ahead, ascending, at most 5, with badges `[EXPIRED today]`, `[EXPIRED Nd ago]`, `[EXPIRES Nd]` or `[EXPIRES Nh]`.
3. **Recent decisions & lessons** — `decision` and `lesson_learned` entries not yet expired, newest first, at most 5; `since` narrows to a window and appends it to the heading.
4. **Conventions** — `pattern_convention` entries not yet expired, at most 5, ignoring `since`.
5. **Related (via graph)** — decisions and lessons from other projects reached through `has_tag` edges on tags used by at least 2 of this project's entries, at most `_MAX_RELATED = 5`, each with `(via #tag)`.

Expiring, recent, conventions and related exclude superseded entries. When every section is empty the result is `No entries found for project '<ref>'.` Team scoping (`(team = ? OR team IS NULL)` on all five queries, maps included) applies only when the `KnowledgeBase` attribution carries a team, which kb-service never sets, so it is off in practice. The thin client appends the version-skew and deprecation notes (section 25) when present.

**Design note.** The MCP server does not infer the project from its working directory, because an MCP subprocess's working directory is unreliable. Agents pass `project_ref` explicitly; the hook, which does receive a reliable `cwd`, resolves projects with `.kb_project` (see [intelligent-kb.md §3](docs/intelligent-kb.md#3-the-hook-one-stdlib-process-six-events)).

## 20. kb-service

`kb_service/main.py` builds the FastAPI app (`title="Personal KB Web Service"`). The lifespan runs, in order:

1. `check_database_config()`: setting `KB_DB_PATH` together with a non-blank `KB_DATABASE_URL` or `KB_SERVICE_DATABASE_URL` is a startup `RuntimeError` naming only the variables, because the Postgres URL would otherwise silently win.
2. `init_db()`: the service-DB schema.
3. `_open_kb()`: a non-empty `KB_DATABASE_URL` opens `create_postgres` (pool from `KB_PG_POOL_MIN`/`KB_PG_POOL_MAX`, embedding, provider, ingest, agentic and retry configs from `kb_service/config.py`, `llm_recorder=record_llm_call`); otherwise `create_sqlite` at `KB_DB_PATH` (default `~/.local/share/personal_kb/knowledge.db`) with the embedding worker forced off. The `KnowledgeBase` singleton is stored on `app.state.kb`.
4. The supersession reconcile (section 6), best-effort.
5. An `asyncio.Lock` for surprise drains, then `kb.start_embedding_worker()` (section 10).
6. The surprise capture worker, when `KB_SURPRISE_CAPTURE` is `shadow` or `on` and `KB_DATABASE_URL` is set (see [intelligent-kb.md §8](docs/intelligent-kb.md#8-distill-i-surprise-detection)).
7. `write_policy.log_startup`.
8. A fresh FastMCP app for `/mcp` (section 25a), whose own lifespan wraps the `yield`.

Shutdown, in a `finally`: clear `app.state.mcp_http_app`, stop the surprise worker, stop the embedding worker, `kb.close()`, `drain_pending_llm_calls()` (5 s), `close_db()`.

**Service DB.** `kb_service/database.get_db()` opens an asyncpg pool on `KB_SERVICE_DATABASE_URL`; when it is unset and `KB_AUTH_MODE=none`, a SQLite `service.db` beside the data DB; otherwise it raises, so a hosted deploy that lost its URL fails loudly instead of starting userless. `init_db()` creates 17 tables idempotently: `users`, `api_keys` (with a `surface` column), `invites`, `password_resets`, `app_config`, `chats`, `chat_messages` and `llm_calls`, documented here, plus `whisper_telemetry`, `listener_decisions`, `failure_events`, `gate_decisions`, `turn_events`, `surprise_candidates`, `surprise_detections`, `surprise_distillations` and `surprise_dry_runs`, documented in [intelligent-kb.md §23](docs/intelligent-kb.md#23-data-the-loop-keeps-tables-and-retention). Timestamps are ISO strings and JSON columns are text. `kb_service/db_sqlite.SqlitePool` adapts aiosqlite to the asyncpg surface: one connection under an `asyncio.Lock`, WAL and a 5000 ms busy timeout, `$N` placeholders rewritten to `?N`, `FOR UPDATE` stripped, dict rows and asyncpg-style status strings. On SQLite, `ADD COLUMN IF NOT EXISTS` is emulated through `PRAGMA table_info`, constraint statements are skipped, and `_SQLITE_CHECK_REBUILDS` rebuilds an old table in one transaction when its stored DDL lacks a newer CHECK (logged `service_db check_rebuild`).

**Authentication** (`kb_service/auth.py`). `KB_AUTH_MODE` is read on every call: `jwt` (default) or `none`; anything else is a `ValueError`. `resolve_principal(token)` is the single path for REST and `/mcp` and returns an `AuthPrincipal(user, api_key_id, auth_method)`:

- `none` mode returns a synthetic admin (`local@localhost`, `auth_method` `none`) for every caller without touching the DB, ignoring any token. The thin client sends the sentinel key `local-no-auth`. Every caller that can reach the port is admin, and `kb-service serve --host` has no loopback guard.
- A missing token is 401 `Not authenticated`.
- The token is tried as a JWT first (HS256, secret `JWT_SECRET`, falling back to the literal `dev-secret-change-me` with no startup refusal, 72 h expiry, `sub` = user id), then as an API key.
- API keys are `kb_` + `secrets.token_urlsafe(32)`, stored only as SHA-256 hex in `api_keys.key_hash`, and shown once at creation; listings show `hash_prefix`, the first 8 hex characters.

FastAPI dependencies: `get_current_principal` (stores the principal on `request.state.kb_principal`), `get_current_user`, `require_admin` (403 `Admin only`) and `get_current_user_sse`, which accepts only a JWT in the `?token=` query parameter because `EventSource` cannot set headers. Passwords are bcrypt.

**Users and admin.** Registration is invite-only: `POST /api/auth/register {email, password, invite_token}` validates the invite and creates the user in one transaction (400 invalid token, 410 used, 409 email taken). Admins issue invites (`POST /api/admin/invites`, a link to `/register?token=...` on `public_base_url`; invites never expire), reset links valid for 12 hours (`POST /api/admin/users/{id}/password-reset`, consumed by the public `POST /api/auth/password-reset`), promote and delete users. `POST /api/admin/shutdown` exists only in `none` mode and only from loopback; the local daemon handshake uses it (section 25).

**CLI** (`kb_service/cli.py`, console script `kb-service`, talking directly to the service DB):

| Subcommand | Effect |
|---|---|
| `create-admin --email --password` | The first admin (both arguments required; registration is invite-only) |
| `create-user --email --password` | A non-admin user, for example a machine account |
| `make-admin --email` | Promote |
| `serve [--host 127.0.0.1] [--port 8000]` | Run the service |
| `metrics repeat-rate` | See [intelligent-kb.md §20](docs/intelligent-kb.md#20-measure-telling-whether-the-loop-works) |
| `list-keys`, `set-key-surface` | API-key surfaces; see [intelligent-kb.md §7](docs/intelligent-kb.md#7-capture-c-the-write-policy-unattended-agents-propose-never-write) |
| `set-machine-principal --email` | See [intelligent-kb.md §4](docs/intelligent-kb.md#4-the-lesson-record-hintsresolution-and-the-trust-model) |

**Settings.** `GET /api/settings` (any user) returns `{team, client_install_spec}`; `PUT /api/settings` (admin) replaces the team. `client_install_spec` is `KB_SERVICE_CLIENT_INSTALL_SPEC`, defaulting to `personal-kb @ git+https://github.com/jason-weddington/personal-kb-mcp`.

**Unauthenticated endpoints.** `GET /api/health` returns `{status: "ok", version, install_id}`, where `version` is the installed `personal-kb-web-service` version and `install_id` is the serving process's `sys.prefix` (the thin client's daemon handshake). `GET /api/kb/runtime` returns `{auth: "none" | "jwt"}` so the SPA can skip login.

**CORS** allows `http://localhost:5173`, `https://localhost` and `https://$HOSTNAME`. A `RequestValidationError` handler logs `/api/kb/turn` 422s (location and type only) and returns FastAPI's default body.

**Logging.** `configure_logging` sets the `kb_service` and `kb_core` loggers to `KB_LOG_LEVEL` (INFO when unset; an unknown value falls back to INFO with one WARNING) and leaves the root logger at WARNING.

## 21. REST route inventory

Auth values: **none** (unauthenticated); **user** (any authenticated user, JWT or API key); **admin** (`require_admin`); **machine principal** (an authenticated user that `is_machine_principal` accepts, else 403); **JWT ?token=** (`get_current_user_sse`, API keys rejected). Every `/api/kb/*` write route is also classified by the write policy ([intelligent-kb.md §7](docs/intelligent-kb.md#7-capture-c-the-write-policy-unattended-agents-propose-never-write)).

| Path | Method | Module (`kb_service/routes/`) | Auth | Doc |
|---|---|---|---|---|
| `/api/health` | GET | `main.py` | none | §20 |
| `/mcp` | GET, POST, DELETE | `main.py`, `mcp_server/endpoint.py` | bearer token | §25a |
| `/api/auth/register` | POST | `auth_routes.py` | none | §20 |
| `/api/auth/login` | POST | `auth_routes.py` | none | §20 |
| `/api/auth/logout` | POST | `auth_routes.py` | user | §20 |
| `/api/auth/me` | GET | `auth_routes.py` | user | §20 |
| `/api/auth/password` | POST | `auth_routes.py` | user | §20 |
| `/api/auth/password-reset` | POST | `auth_routes.py` | none | §20 |
| `/api/auth/api-keys` | POST, GET | `auth_routes.py` | user | §20 |
| `/api/auth/api-keys/{key_id}` | DELETE | `auth_routes.py` | user (own keys) | §20 |
| `/api/admin/invites` | POST, GET | `admin_routes.py` | admin | §20 |
| `/api/admin/invites/{token}` | DELETE | `admin_routes.py` | admin | §20 |
| `/api/admin/users` | GET | `admin_routes.py` | admin | §20 |
| `/api/admin/users/{user_id}/promote` | POST | `admin_routes.py` | admin | §20 |
| `/api/admin/users/{user_id}/password-reset` | POST | `admin_routes.py` | admin | §20 |
| `/api/admin/users/{user_id}` | DELETE | `admin_routes.py` | admin | §20 |
| `/api/admin/shutdown` | POST | `admin_routes.py` | none mode, loopback only | §20, §25 |
| `/api/settings` | GET | `settings_routes.py` | user | §20 |
| `/api/settings` | PUT | `settings_routes.py` | admin | §20 |
| `/api/kb/runtime` | GET | `kb_routes.py` | none | §20 |
| `/api/kb/search` | POST | `kb_routes.py` | user | §9, §11 |
| `/api/kb/get` | POST | `kb_read_routes.py` | user | §11, §18d |
| `/api/kb/graph/full` | GET | `kb_read_routes.py` | user | §24a |
| `/api/kb/graph/neighbors` | GET | `kb_read_routes.py` | user | §12 |
| `/api/kb/graph/bfs` | GET | `kb_read_routes.py` | user | §12 |
| `/api/kb/graph/path` | GET | `kb_read_routes.py` | user | §12 |
| `/api/kb/graph/supersedes-chain` | GET | `kb_read_routes.py` | user | §12 |
| `/api/kb/graph/scope-entries` | GET | `kb_read_routes.py` | user | §12 |
| `/api/kb/graph/vocabulary` | GET | `kb_read_routes.py` | user | §12 |
| `/api/kb/preflight` | GET | `kb_read_routes.py` | user | §19 |
| `/api/kb/projects` | GET | `kb_read_routes.py` | user | §11 |
| `/api/kb/contributors` | GET | `kb_read_routes.py` | user | §11 |
| `/api/kb/teams` | GET | `kb_read_routes.py` | user | §11 |
| `/api/kb/browse/recent` | GET | `browse_routes.py` | user | §24b |
| `/api/kb/browse/top` | GET | `browse_routes.py` | user | §24b |
| `/api/kb/browse/stale` | GET | `browse_routes.py` | user | §24b |
| `/api/kb/browse/projects` | GET | `browse_routes.py` | user | §24b |
| `/api/kb/browse/project/{project_ref:path}` | GET | `browse_routes.py` | user | §24b |
| `/api/kb/store` | POST | `kb_write_routes.py` | user | §5 |
| `/api/kb/store_batch` | POST | `kb_write_routes.py` | user | §5 |
| `/api/kb/entries/{entry_id}/deactivate` | POST | `kb_write_routes.py` | user | §5, §6 |
| `/api/kb/entries/{entry_id}/reactivate` | POST | `kb_write_routes.py` | admin | §5 |
| `/api/kb/bulk_update` | POST | `kb_write_routes.py` | admin | §26 |
| `/api/kb/admin/reconcile-supersession` | POST | `kb_write_routes.py` | admin | §6 |
| `/api/kb/feedback` | POST | `kb_write_routes.py` | user | §11 |
| `/api/kb/ask` | POST | `query_routes.py` | user | §13 |
| `/api/kb/summarize` | POST | `query_routes.py` | user | §15 |
| `/api/kb/query/stream` | POST | `query_routes.py` | JWT ?token= | §22 |
| `/api/kb/ingest/text` | POST | `ingest_routes.py` | user | §17 |
| `/api/kb/ingest/url` | POST | `ingest_routes.py` | user | §17 |
| `/api/kb/ingest/file` | POST | `ingest_routes.py` | user | §17 |
| `/api/chat/stream` | POST | `chat_routes.py` | JWT ?token= (404 in none mode) | §23 |
| `/api/chat/create` | POST | `chat_routes.py` | user (404 in none mode) | §23 |
| `/api/chat/history` | GET | `chat_routes.py` | user (404 in none mode) | §23 |
| `/api/chat/{chat_id}/messages` | GET | `chat_routes.py` | user (404 in none mode) | §23 |
| `/api/chat/{chat_id}` | DELETE | `chat_routes.py` | user (404 in none mode) | §23 |
| `/api/kb/maps-index` | GET | `maps_routes.py` | user | §18e |
| `/api/kb/embedding-queue` | GET | `embedding_queue_routes.py` | admin | §10 |
| `/api/kb/map-eligibility` | GET | `map_eligibility_routes.py` | admin | [intelligent §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract) |
| `/api/kb/map-eligibility/override` | POST | `map_eligibility_routes.py` | admin | [intelligent §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract) |
| `/api/kb/map-eligibility/override/clear` | POST | `map_eligibility_routes.py` | admin | [intelligent §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract) |
| `/api/kb/map-lint` | POST | `map_lint_routes.py` | user | §18c, [intelligent §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract) |
| `/api/kb/map-loop-input` | GET | `map_loop_routes.py` | user | [intelligent §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract) |
| `/api/kb/map-worklist` | GET | `map_loop_routes.py` | user | [intelligent §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract) |
| `/api/kb/map-op` | POST | `map_op_routes.py` | machine principal | [intelligent §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract) |
| `/api/kb/maps/{map_id}` | DELETE | `map_delete_routes.py` | machine principal | [intelligent §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract) |
| `/api/kb/cluster-ledger/match` | POST | `cluster_ledger_routes.py` | user | [intelligent §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract) |
| `/api/kb/cluster-ledger/decline` | POST | `cluster_ledger_routes.py` | user | [intelligent §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract) |
| `/api/kb/pointer-candidates` | POST | `nudge_routes.py` | user | [intelligent §18](docs/intelligent-kb.md#18-map-maintenance-somnus-the-server-side-contract) |
| `/api/kb/listener` | POST | `listener_routes.py` | user | [intelligent §17](docs/intelligent-kb.md#17-the-anticipatory-listener-paused) |
| `/api/kb/telemetry/whispers` | POST | `telemetry_routes.py` | user | [intelligent §20](docs/intelligent-kb.md#20-measure-telling-whether-the-loop-works) |
| `/api/kb/event` | POST | `event_routes.py` | user | [intelligent §5](docs/intelligent-kb.md#5-capture-a-the-failure-cue-index) |
| `/api/kb/event/heartbeat` | GET | `event_routes.py` | user | [intelligent §5](docs/intelligent-kb.md#5-capture-a-the-failure-cue-index) |
| `/api/kb/turn` | POST | `turn_routes.py` | user | [intelligent §6](docs/intelligent-kb.md#6-capture-b-turn-digests) |
| `/api/kb/turn/heartbeat` | GET | `turn_routes.py` | user | [intelligent §6](docs/intelligent-kb.md#6-capture-b-turn-digests) |
| `/api/kb/surprise/drain` | POST | `surprise_routes.py` | user | [intelligent §8](docs/intelligent-kb.md#8-distill-i-surprise-detection) |
| `/api/kb/surprise/candidates` | GET | `surprise_routes.py` | user | [intelligent §11](docs/intelligent-kb.md#11-shadow-mode-and-the-audit-endpoint) |
| `/api/kb/prevention` | GET | `prevention_routes.py` | user | [intelligent §12](docs/intelligent-kb.md#12-deliver-i-get-apikbprevention-and-the-hooks-prevention-cache) |
| `/api/kb/prevention/decisions` | POST | `prevention_routes.py` | user | [intelligent §14](docs/intelligent-kb.md#14-prevent-the-pretooluse-soft-gate) |
| `/api/kb/prevention/stats` | GET | `prevention_routes.py` | user | [intelligent §20](docs/intelligent-kb.md#20-measure-telling-whether-the-loop-works) |
| `/api/kb/metrics/repeat-rate` | GET | `metrics_routes.py` | user | [intelligent §20](docs/intelligent-kb.md#20-measure-telling-whether-the-loop-works) |
| `/api/kb/observe/turn-events` | GET | `observe_routes.py` | admin | [intelligent §20](docs/intelligent-kb.md#20-measure-telling-whether-the-loop-works) |
| `/api/kb/observe/failure-events` | GET | `observe_routes.py` | admin | [intelligent §20](docs/intelligent-kb.md#20-measure-telling-whether-the-loop-works) |
| `/api/kb/observe/surprise-detections` | GET | `observe_routes.py` | admin | [intelligent §20](docs/intelligent-kb.md#20-measure-telling-whether-the-loop-works) |
| `/api/kb/observe/llm-calls` | GET | `observe_routes.py` | admin | §16, [intelligent §20](docs/intelligent-kb.md#20-measure-telling-whether-the-loop-works) |
| `/api/kb/observe/listener-decisions` | GET | `observe_routes.py` | admin | [intelligent §20](docs/intelligent-kb.md#20-measure-telling-whether-the-loop-works) |
| `/openapi.json` | GET | FastAPI default (`main.py`) | none | §21 |
| `/docs` | GET | FastAPI default (`main.py`) | none | §21 |
| `/docs/oauth2-redirect` | GET | FastAPI default (`main.py`) | none | §21 |
| `/redoc` | GET | FastAPI default (`main.py`) | none | §21 |
| `/{full_path:path}` | GET | `main.py` (`mount_frontend`) | none | §24 |

`map_write_guards.py` and `near_duplicate_guard.py` in the same directory are helpers, not routers. The `FastAPI(...)` constructor in `main.py` sets no `docs_url`, `redoc_url` or `openapi_url`, so FastAPI's default `/openapi.json`, `/docs`, `/docs/oauth2-redirect` and `/redoc` are registered ahead of the SPA catch-all and served without authentication; they expose the API schema, not KB data. When a built SPA is found, `mount_frontend` also mounts `/assets` as static files and registers the `/{full_path:path}` catch-all, which serves any other GET and returns 404 under `api/` and `.well-known/` (section 24).

## 22. Query streaming and the classifier

`POST /api/kb/query/stream` (`kb_service/routes/query_routes.py`) powers the web UI's ask box: the Graph page's query mode (§24a). It authenticates only with a JWT in `?token=` (API keys are rejected) and takes `{question}`. The work runs as an asyncio task feeding an `asyncio.Queue` that the SSE generator drains. Events, in order:

1. `classified {mode}`.
2. Every engine event under its own name (`fast_path`, `agent_started`, `thinking`, `tool_call`, `tool_result`, `tool_dedup`, `parse_error`, `agent_done`, `synthesis_started`, `synthesis_done`), each followed by a `status` event when `kb_service/sse.event_to_status` has text for it. Status text exists only for `agent_started`, `tool_call` (per tool), `thinking`, `synthesis_started`, `fast_path` and the `ingest_*` events.
3. `synthesis_result {answer, question, entry_ids}` in summarize mode, where `entry_ids` comes from the `fast_path` and `agent_done` events, so entries added by the coverage re-search are missing; or `entries {entries: [{id, short_title, entry_type, tags, context}], turns_used}` in explore mode.
4. `error {message: "ExcType: detail"}` on failure.
5. Always last, `stream_end {}`.

Explore mode calls `kb.ask(question)` and summarize mode `kb.summarize(question)`, both with the default limit 20 and server-side agentic flags.

**Classifier.** `kb_service/classifier.classify_query` makes one call on `kb.query_llm` (purpose `classifier`, Haiku by default). Its prompt defines `explore` as visually navigating the graph and `summarize` as any request for information, says "when in doubt choose summarize", and expects one word. The reply is lower-cased and trimmed: an exact `explore` or `summarize` wins, else a reply containing `summarize` is summarize, else explore; an exception or None is also explore. With no query LLM the mode is explore without a call.

**Event callback.** The optional `event_callback` threads through `KnowledgeBase.ask` and `summarize` into `kb_core/query.py` and `kb_core/graph/agent.py`; callers that do not stream pass None.

## 23. Multi-turn chat

The web Chat page holds a conversation grounded in KB entries, with follow-ups and optional single-step tool use. `kb_service/chat.py` holds the logic and `kb_service/chat_history.py` the persistence (tables `chats` and `chat_messages`); the routes are in `kb_service/routes/chat_routes.py`.

**Routes.** `POST /api/chat/stream` (JWT `?token=` only; body `{session_id?, message, mode?, seed_question?, seed_answer?, seed_entry_ids?}`), `POST /api/chat/create` (idempotent for the owner, 409 for an id owned by another user), `GET /api/chat/history`, `GET /api/chat/{chat_id}/messages` (404 unless owned) and `DELETE /api/chat/{chat_id}` (also evicts the in-memory session). A router-level guard answers 404 for every route when `KB_AUTH_MODE=none`, because history lives in the auth DB, and the SPA hides the page in that mode.

**Sessions.** `ChatSession` objects are cached in a module dict keyed `(user_id, session_id)` with no eviction or TTL. An uncached session id is rebuilt from the database with `ChatSession.from_saved`, which re-extracts entry ids from assistant messages. A new session persists a chat row (title from the first 80 characters) and any seed messages; each turn saves the user and assistant messages.

**A reply.** The chat LLM is `kb.synthesis_llm`, else `kb.query_llm`; with neither the stream emits `error` `LLM not available`. Each `reply()`:

1. Appends the user message and trims history: a no-op at 100_000 characters or fewer (`_MAX_CONVERSATION_CHARS`) or 3 or fewer messages; otherwise it keeps the first two messages (the grounding seed) and drops from the front of the rest.
2. Runs `kb.search(limit 5, include_stale false)` on the message and adds new ids to `session.entry_ids`.
3. Builds the system prompt from the chat prompt, the write-tools prompt (always included) and the full text of every known active entry, emits `chat_thinking`, and calls `generate_chat(purpose="chat")`.
4. If the reply contains a JSON tool block (fenced or bare, the first object with a `tool` key), dispatches exactly one tool, emits `chat_tool_result {tool, success, entry_ids}`, appends `Tool result: ...` and makes one more `generate_chat` call. A reply therefore costs at most two LLM calls; there is no loop.

Tools: `get_entry` (read-only; adds the entry to context); `update_entry` (requires `change_reason`, `sensitivity` among the three values, a secret scan unless `KB_SKIP_SAFETY`, an optional `ttl`, and stamps `updated_by` with the user); `ingest_url` (needs an extraction LLM and an embedder). A None LLM reply yields a canned apology.

**Events.** `chat_session {session_id}` first, then `chat_thinking`, optional `chat_tool_result`, `chat_done {new_entries}`, `chat_response {answer, session_id}`, `error` on failure, and `stream_end`. The Chat page sends `{session_id, message, mode: "chat"}`. The Graph page chat panel (section 24a) sends `seed_question`, `seed_answer`, `seed_entry_ids` and `mode` (`summarize` or `explore`) with its first message and the returned `session_id` afterwards, so a Graph conversation is saved to chat history at its first follow-up as one `chats` row whose first two messages are the seed question and answer.

## 24. Web UI

The SPA lives in `packages/kb-service/frontend/` (Vite, React 19, MUI 7, react-router 7, react-markdown, and react-force-graph-2d bundled by Vite). react-force-graph-2d and the Graph page's search Autocomplete load in the Graph page's lazily loaded chunk (`registry.tsx` imports `pages/Graph` with `React.lazy`), so other pages do not download them. It is served by the same kb-service process.

Pages:

- **Home** (`/`): a Browse panel on the left and the KB Stats card on the right (stacked above the list on narrow screens). Entries counts active, non-superseded entries with or without a project (the unfiltered total of `/api/kb/browse/recent`); a No project row appears when some have no project, so the project cards plus that row add up to Entries. Projects, Contributors and Maps come from `/api/kb/projects`, `/api/kb/contributors` and `/api/kb/maps-index`. The Browse panel has four tabs: New (newest or recently updated entries), Top (most connected or recently read), Stale (decayed or expired) and Projects (a filterable grid of project cards that opens one project's entries, mental maps first). Its state lives in the URL: `view` (absent means New), `type` (an entry-type filter shared by the entry views) and `project` (the open project in the Projects view). Clicking a row, or Enter on a focused row, opens the entry drawer, and ArrowDown and ArrowUp move between rows. Data comes from the browse endpoints in section 24b. Home has no Ask panel; `/ask` now redirects to `/chat`. The right column also holds an Add to KB panel under the KB Stats card: drop files or URLs (or use the Browse picker or the URL field), give an optional project, and each file or URL becomes a job in a queue that runs one at a time in submission order, showing queued, an indeterminate progress bar while ingesting, or the result (entry ids open the drawer; skipped, unchanged and flagged show as non-failures); after a job ingests, the stats card and the browse list refresh without a reload.
- **Search** (`/search`): query, project, entry type, tags and limit, posted to `/api/kb/search`.
- **Chat** (`/chat`): conversation list from `/api/chat/history`, new and delete, optimistic bubbles and a thinking indicator. Hidden and redirected to `/` in `none` mode.
- **Graph** (`/graph`): section 24a.
- **Settings** (`/settings`): password change, API-key create and revoke with a one-time dialog showing `claude mcp add --transport http personal-kb <origin>/mcp --header "Authorization: Bearer <key>"`, the equivalent `mcpServers` JSON and a deprecated stdio snippet (with `KB_INSTANCE_ROLE=team` when a team is set), and the admin team editor. The account, API-key and team cards render only in `jwt` mode.
- **Entry detail** (`/entries/:id`) and the `EntryDrawer` component, both showing pointer-rot and superseded banners and a Connections section of one-hop neighbours grouped by edge type and direction (group key `${edge_type}:${dir}`, the group header showing the edge type with underscores as spaces, an arrow and the member count): every group is an accordion of one summary row per member, defaulting expanded for up to 12 members and collapsed for bigger ones (an entry row shows the short title, long title, type-coloured dot and project; a non-entry row the raw node id with a type chip), every row clickable (see the Graph page below). The entry's tags render once as clickable chips that open `tag:<tag>` and the project as a clickable project chip, with no separate has_tag or in_project groups; the drawer keeps its shell and back stack and its body (markdown, superseded and pointer-rot banners) stays expanded with no Details control. On the Graph page a chip walks the selection to that node in the graph (resetting the type chips and Scope first if the node is filtered out); elsewhere an entry chip opens that entry in the drawer and a non-entry chip (tag, project, tool and so on) opens the Graph focused on it.
- **Admin** (`/admin/invites`, `/admin/users`), behind `AdminRoute`.
- `/login`, `/register` (invite token from `?token=`) and `/reset-password`.

`frontend/src/api.ts` keeps the JWT in localStorage under `kb-token`, converts keys between camelCase and snake_case for the auth, admin and settings calls (KB routes are read as raw snake_case), and on any 401 clears the token and redirects to `/login` unless already there, which prevents the refresh loop a redirect from the login page itself would cause. The stream is read with `fetch` and a `ReadableStream` in `streaming.ts`, whose optional `AbortSignal` lets a caller cancel the fetch.

**Static serving.** `main.resolve_static_dir` picks `KB_SERVICE_STATIC_DIR`, else the checkout's `frontend/dist`, else the UI packaged in the wheel at `kb_service/static`. `mount_frontend` serves `/assets` and an SPA catch-all that 404s `api/*` and `.well-known/*` and never serves a path outside the dist directory. The packaged copy is committed: `scripts/build_static_ui.sh` (run by `release.sh`) builds it, copies it into `kb_service/static` and writes `.source-hash`, and `scripts/check_static_ui_fresh.sh` fails when the recorded hash differs from the current frontend source.

**kb_explore** only returns the Graph page URL: `KB explorer is hosted at <url>/graph — open it in a browser.` The stdio tool uses `PERSONAL_KB_URL` with any trailing slash stripped; the `/mcp` twin derives the URL from the request through `public_base_url`, which honours `KB_SERVICE_PUBLIC_URL`. No explorer process is started per MCP session.

### 24a. Graph page data

`GET /api/kb/graph/full` calls `extract_graph_data(db)` in `packages/kb-service/src/kb_service/graph_export.py`, which runs four queries: inactive entry ids; `graph_nodes` with a connection count; all `graph_edges`; and active entry metadata (`id`, `short_title`, `long_title`, `entry_type`, `tags`, `confidence_level`, `contributor`, `project_ref`). Inactive entry nodes, non-entry nodes with no edges, and edges touching a dropped node are dropped. Each node is `{id, label, type, val, properties}`, where `label` is the entry's short title (or the part of the id after the first `:` for other nodes), `val = max(connections, 1)`, and entry properties merge the metadata above. Each edge is `{source, target, type, properties}`. `stats` is `{node_count, edge_count}`.

The Graph page draws a force-directed canvas with one colour per node type:

| Type | Colour |
|---|---|
| entry | `#e0e0e0` |
| tag | `#00bcd4` |
| project | `#ff9800` |
| person | `#ffc107` |
| tool | `#4caf50` |
| concept | `#9c27b0` |
| technology | `#2196f3` |
| note | `#78909c` |
| unknown types (always shown) | `#666` |

The Graph page is the tuned standalone explorer ported into the app shell (`components/GraphExplorer.tsx`, `graph/explorerModel.ts`, `graph/drawNode.ts`): it fills the viewport under the 64 px app bar with the sidebar still present, on the dark `#0a0a0f` canvas in both themes. Each node is a circle of radius `sqrt(val) * 2.5` (plus 6 when highlighted, 3 for a query result, plus a 600 ms pulse of up to 6), and highlighted, visited and result nodes glow (shadow blur 25, 20 and 30). Labels sit on a dark pill below the node and fade in with zoom from scale 3.0 to 5.0, are truncated to 20 characters below scale 2.0, and are always shown for the hovered node, the highlighted node and the focused node's neighbourhood. Clicking a node focuses it: only the node and its one-hop neighbours (and the links touching it) stay visible, the camera moves over 800 ms to zoom 4 (an entry is shifted 55 units so it sits left of the 440 px inspector when the canvas is at least 960 px wide), and at 850 ms every node opens the non-modal 440 px inspector, which spans the full column height with a fixed header and one scroll container for the connections and the entry Details section (no nested scroll inside a group): the header shows the node type, label and id (an entry adds its type chip, long title, a compact `project · confidence · updated` line, its tag chips and its project chip; a non-entry node gets a summary line `N entries · <top two entry types with counts> · <rest> other` over its entry members), and its connections grouped by edge type and direction (group key `${edge_type}:${dir}`, each group header showing the edge type with underscores as spaces, an arrow (→ out, ← in) and the member count), every group an accordion of one summary row per member (an entry row shows the short title, long title, type-coloured dot and project; a non-entry row the raw node id with a type chip) that defaults expanded for up to 12 members and collapsed for bigger ones, every row built from the loaded graph payload with no fetch and clickable to walk to that node (the same select path as a canvas click, so ArrowLeft returns). The entry inspector's Details section starts expanded and fetches the entry on open (once per shown id, still collapsible). Groups with more than 12 members in the inspector gain a title filter box (a case-insensitive substring match over each member's label and long title) and facet chips computed client-side from the loaded graph (the group members' entry types, project refs and co-occurring non-entry neighbours of the members, excluding the inspected node), grouped into labelled rows (a Type row, a Project row, then one row per other neighbour node type present) each showing its top 6 by member count with ties by label ascending and a `+N more` button that reveals that row only; every facet starts on (filled with its legend colour) and clicking one switches it off — a grey outlined chip — hiding every member matching any off facet, with each chip's count computed over the whole group and unchanged as facets switch off; clicking a facet never walks to that node; the group header reads e.g. `has tag ← 69 of 118` while a filter is active, with a Clear control that turns every facet back on and empties the title text; and while a group filter is active the hidden members of that group and their links drop out of the visible canvas focus neighbourhood. Escape and a background click reset the focus and fit the connected nodes with `zoomToFit(400, 40)`, then clamp the zoom to 6 if the fit zoomed in further; while the inspector is open, the first Escape only closes it. Back and Forward IconButtons (aria-labels `Back` and `Forward`, a `ChevronLeft`/`ChevronRight` icon) sit at the start of the search row, and ArrowLeft/ArrowRight walk the same browser-style history: every focus (a canvas click, an inspector row, a search pick, `?focus`, a citation fly) appends a record holding the node, the camera (zoom, cx, cy) and the inspector view state as they were just before you left it, truncating everything forward of the cursor, so Back and Forward restore the camera plus the inspector view snapshot (expanded groups, active filters and the scroll position) together, and the history keeps at most 500 records by dropping the oldest; Escape and a background click unfocus and fit but keep the history and the cursor, so Back while nothing is focused re-focuses the record at the cursor (the node you just left), the one in-page thing that clears it is a filter change that stops rendering the focused node (closing the query or chat panel keeps it), and a page reload clears it; every node focused this session carries a trailing check mark on its inspector member rows. Links are not hit-tested (no hover tooltip, and a click on a link counts as a background click), because force-graph repaints its hit-test canvas every 800 ms and painting every link into it made a visible hitch on a graph with tens of thousands of edges. While the layout is still settling (a focus right after the page loads, or after a filter change), the camera follows the focused node so it stays in view, until any pointer or wheel input on the canvas. Every connection row in the inspector and the entry drawer is clickable: on the Graph page it walks the selection to that node (pushing history so ArrowLeft returns), first resetting the type chips and scope when the node is filtered out; on other pages an entry chip opens it in the drawer and any other chip opens the Graph at `/graph?focus=<node id>`, which focuses that node once the graph has loaded and then drops the parameter from the URL. The search box ranks nodes client-side by label, id, long title, tags, entry type and project, putting an exact label match first, then higher connection count (`val`), and shows 12 results; picking one (or Enter) flies to that node. Type chips switch node types on and off; a chip that is off reads grey (outlined with grey text and border), never a coloured outline. Scope limits the canvas to one project's entries plus their one-hop non-entry neighbours of an active type; unknown types are always shown. A stats line reads `<rendered> of <total> nodes · <rendered> of <total> edges`. `?focus=<node id>` (URL-encoded) flies to that node once the graph has loaded, then removes the parameter; an unknown id logs `[graph] openNode: node not rendered` at debug level and leaves the default view. Adding `?perf=1` to the URL shows an FPS, settle-time and load-time readout and logs one `[graph-perf]` console line per graph with node and link counts, `settle_ms`, `load_ms` and `fps`.

**Query mode.** A query starts from the ask bar, or from Enter on free text in the search box when no suggestion is highlighted, and streams `POST /api/kb/query/stream`. Closing or re-asking aborts the client's fetch only; the server's ask or summarize task is not cancelled and runs to completion. Events become node states in `graph/queryModel.ts`: `graph_neighbors` `node_id`, `decision_chain` `entry_id` and `tool_result` ids become visited; `fast_path`, `agent_done`, `entries` and `synthesis_result` ids, plus any kb ids cited in a summarize answer, become results; a node is never downgraded and never queued twice. Reveals run 250 ms apart, each with a pulse, particles and a fit over every marked node (500 ms, 60 px), then a fit over the results (600 ms, 40 px), both capped at zoom 6. Unmarked non-neighbour nodes and their labels are dimmed while a query is open. The panel shows the mode chip, the status, an error, the answer with its sources, or the explore list; results that are hidden or not in the graph are labelled and open the entry drawer directly. Escape or Close cancels the stream, clears the marks and refits. Query mode works in `none` auth mode, where the stream endpoint authenticates a synthetic user. Every query logs one `[graph-query]` console line with its outcome, mode, path, mark and result counts (rendered, hidden, not in graph) and its malformed or unknown events, and a `[graph-query]` warning appears when an event is unhandled or malformed, or when no marked node is in the graph.

**Chat panel.** The chat panel exists only in `jwt` auth mode and is hidden in `none` mode, like the Chat page. A 420 px panel opens to the right of the explorer and the explorer narrows; below the sm breakpoint the panel covers the explorer. A summarize answer opens the panel with the question and answer as its first messages, and the query results panel is hidden while it is open. An explore result opens it beside the query results panel, with `Found N entries: [kb-...]. Ask me anything about these results.` and a Summarize these button that sends `Summarize these entries for me.`. Enter sends and Shift+Enter adds a new line; Escape blurs the input, and a second Escape ends the conversation (an Escape while the citation snackbar shows only dismisses the snackbar). `chat_done` `new_entries` that are on the graph are revealed as results through the query reveal queue, and the others get a caption in the chat. `[kb-NNNNN]` citations, including comma lists inside one pair of brackets, are buttons. Each flies to its node, whose inspector then opens at 850 ms over the panel until closed, or shows a snackbar with an Open entry action when the node is not rendered. Citations in the query results panel's answer fly in both auth modes. While the panel is open, the entry camera shift is 2.5 units when the canvas width plus 420 px is at least 960 px. When the canvas width changes during a query, the camera refits the results (600 ms, padding 40) in place of the resize refit. The close button and Escape end the conversation and the query and reset the view, and a new query replaces it. The `[graph-chat]` console lines (seed, citation not rendered, new_entries, stream ended without response, session id mismatch, re-seeding, unlinked citations) are what to read when the panel does not open or a reply goes missing.

### 24b. Home browse data

The Browse panel reads five GET endpoints in `packages/kb-service/src/kb_service/routes/browse_routes.py`, all user-authenticated, which call the read-only queries in `packages/kb-core/src/kb_core/browse.py` with `kb.db`. `/api/kb/browse/recent` takes `sort` (`created` or `updated`), `/api/kb/browse/top` takes `signal` (`degree` or `read`, echoed in the response), `/api/kb/browse/stale` takes `kind` (`decayed` or `expired`), and `/api/kb/browse/project/{project_ref:path}` takes `sort` (`newest`, `updated` or `confidence`) and `q`; these four also take `entry_type`, `limit` (1 to 100, default 25) and `cursor`. `/api/kb/browse/projects` takes `q`, `limit` (default 24) and `cursor`. The `:path` converter lets a `project_ref` containing `/` (sent as `%2F`) reach the handler. An unknown `sort`, `signal`, `kind` or `entry_type`, or a `limit` out of range, is a FastAPI 422; a malformed cursor raises `BrowseParamError` (422) and a project with no visible entries raises `ProjectNotFound` (404 `Unknown project: <ref>`). Any other exception is re-raised as a 500, so a bad stored row never masquerades as a client error.

Every query reads one portable SELECT over `knowledge_entries` with `is_active = 1 AND superseded_by IS NULL`, so inactive and superseded entries are hidden everywhere. Expired and decayed entries are marked (`expired`, `effective_confidence`), not hidden, in New, Top and project lists, and project cards count them. All ordering happens in Python over parsed timestamps, with naive timestamps read as UTC, and ties always end with id ascending. New `created` orders by `created_at` descending; New `updated` keeps only entries whose `updated_at` is later than `created_at` and orders by `updated_at` descending. Top `degree` orders by graph degree descending, then `created_at` descending, and drops entries with no edges; Top `read` keeps entries with a `last_accessed` and orders by it descending. Stale `decayed` keeps entries with effective confidence below `STALE_FILTER_THRESHOLD` and orders by effective confidence ascending, then by days since the decay anchor descending; Stale `expired` keeps entries with `now >= expires_at` and orders by `expires_at` ascending. Projects order by last activity (the latest `updated_at`) descending, then `project_ref`, with `counts_by_type` in entry-type declaration order and `q` a case-insensitive substring of the ref. A project view puts mental maps first, then sorts by `created_at`, `updated_at` or effective confidence, all descending, and its `q` matches the id and both titles case-insensitively.

Effective confidence is computed as in section 8, anchored on the later of `updated_at` and `last_accessed`, with `half_life_for` giving the half-life (none for `mental_map`). Graph degree is section 24a's connection count over `graph_edges`: each edge counts once for its source and once for its target, and a self-loop counts once. Recently read is `last_accessed`, which only `POST /api/kb/get` writes, so opening an entry in the drawer counts as a read and resets its decay clock, while the drawer's neighbour-title lookups pass `touch: false`. Top uses graph degree because the KB has no per-entry retrieval counter: `search_events` stores no entry ids, `audit_events` records writes only, `whisper_telemetry` and `gate_decisions` cover only maps and resolutions, and `turn_events.items` holds raw per-turn tool JSON in the service DB, which is not a per-entry counter.

The cursor is a decimal offset string, so rows can be skipped or repeated when entries change between pages; the UI drops repeated ids when it appends a page. A row with an unknown entry type or an unparseable timestamp is skipped, counted in `skipped` (shown as `N unreadable` under the list) and logged once per call as `browse-skip fn=<function> skipped=<n> unknown_type=<n> bad_timestamp=<n> ids=<first 10>` at WARNING. Each handled request logs one `browse-route` line: on success at INFO with `view`, `mode`, `entry_type`, `project_ref`, `q_len`, `limit`, `cursor`, `outcome=ok`, `scanned`, `matched`, `returned`, `skipped`, `duration_ms` and `user_id`; on a 422 or 404 at INFO with `outcome=bad_param` or `outcome=not_found` and `detail`; and on any other exception at WARNING with `outcome=error` and the exception type.

## 25. MCP surfaces

**stdio thin client.** `src/personal_kb/server.py` `create_server()` registers 18 tools at most: `store`, `store_batch`, `search`, `get`, `ask`, `summarize`, `ingest`, `ingest_url`, `feedback`, `preflight`, `explore`, `map_eligibility`, `map_eligibility_override` and `list_projects` always; `maintain` and `bulk_update` when `KB_MANAGER=TRUE`; `list_contributors` and `list_teams` when `KB_CONTRIBUTOR` is set. `KB_INSTANCE_ROLE` sets the name prefix (`personal` gives `personal_kb_`, `team` gives `team_kb_`, otherwise `kb_`) and prepends a role paragraph to the instructions. The lifespan opens only an `HttpBackend`; a remote URL without `PERSONAL_KB_API_KEY` fails startup. Logs go to stderr and to `~/.local/share/personal_kb/log.txt` (truncated each start) at `KB_LOG_LEVEL`, default WARNING here.

**HttpBackend** (`src/personal_kb/backend/http.py`) is the client's only data path. Timeouts: connect 10 s, read 60 s, write 60 s, pool 10 s, with read 300 s for `ask_auto`, `summarize`, `ingest_file` and `ingest_url`. Every request carries `Authorization: Bearer <key>` and `User-Agent: personal-kb/<version>`; when `HEADLESS_BUILD_ENGINE` is set it adds `X-KB-Engine: <engine>` and `X-KB-Harness: claude-code`. It never sends `X-KB-Mode`, so the surface comes from the key. Each operation maps to one route in section 21 (`get_entries` batches ids by 20; `ingest_file` uploads multipart with the file's basename). Connection and timeout errors become `BackendHttpError(0, "cannot reach KB service at <url>: ...")`. `_map_error` renders 401 as `KB service authentication failed (401). Check PERSONAL_KB_API_KEY.`, passes through a 403 detail starting `write policy: `, renders any other 403 as `admin privileges required (403): ...`, returns the detail for 404 and 409, and renders anything else as `KB service returned N: detail`.

**Configuration.** `PERSONAL_KB_URL` defaults to `http://127.0.0.1:8765` (local mode). `PERSONAL_KB_API_KEY` defaults to the sentinel `local-no-auth` only for loopback hosts (`127.0.0.1`, `localhost`, `::1`).

**Local daemon.** For a loopback URL the lifespan calls `daemon.ensure_daemon(url)` first, so a fresh install works with no configuration:

1. `GET /api/health` (2 s). Healthy with `install_id` equal to this client's `sys.prefix`: reuse it.
2. Healthy but from another install: replace it at most once per process, via `POST /api/admin/shutdown`, else SIGTERM of the pidfile pid when `/proc/<pid>/cmdline` contains `kb-service`, then wait up to 10 s for the port.
3. Otherwise take an `O_EXCL` pidfile lock (`kb-daemon.pid` in `~/.local/share/personal_kb/`, overridable with `PERSONAL_KB_DAEMON_STATE_DIR`). The winner spawns `kb-service serve --port <port>` detached with `KB_AUTH_MODE=none` added to the inherited environment, logging to `kb-daemon.log`; losers poll `/api/health` every 0.5 s for up to 30 s. A stale pidfile is reclaimed once.

The daemon is never stopped when an MCP session ends. A failure raises `RuntimeError` naming the daemon log.

**Version skew and deprecation.** For a non-loopback URL, `version_skew.check_version_skew` compares `/api/health.version` with the client's version and, when the client is newer (dev or local suffixes count as newer), logs a WARNING; `stdio_deprecation_note` returns the `claude mcp add --transport http ...` migration text (server name `team-kb` when `KB_INSTANCE_ROLE=team`). Both notes are appended to `kb_preflight` output; loopback installs get neither.

**HTTP-mode limitations** (both MCP surfaces):

| Tool | Limitation |
|---|---|
| `kb_ask` | Only `strategy="auto"` |
| `kb_search` | `contributor` or `team` returns an error |
| `kb_maintain` | Only `deactivate`, `reactivate` and `reconcile_supersession` (section 26) |
| `kb_ingest` | stdio only; single files, directories rejected |
| `kb_map_eligibility`, `kb_map_eligibility_override` | Admin-only at the route |
| `kb_bulk_update` | Admin-only; `dry_run` defaults to true |

### 25a. MCP over HTTP

kb-service serves the same MCP tool set over MCP streamable HTTP at `/mcp`. The tool code lives in `packages/kb-service/src/kb_service/mcp_server/`, never imports `personal_kb`, and makes no HTTP call to itself: each tool calls the kb-service route handler in-process.

**Endpoint.** `main.py` appends `Route("/mcp", endpoint=McpEndpoint(), methods=["GET", "POST", "DELETE"])` after the API routers and before the SPA catch-all. `McpEndpoint` (`mcp_server/endpoint.py`) is an ASGI class: it parses `Authorization: Bearer <token>` (any other scheme, or an empty token, counts as missing), resolves it with `resolve_principal`, and answers a failure with the same JSON `{"detail": ...}` as REST plus `WWW-Authenticate: Bearer` on 401. On success it writes `kb_principal` and `kb_app` into the ASGI scope state and delegates to `app.state.mcp_http_app`; if that is not set (no lifespan) it answers 503 `MCP endpoint not started`. The inner app is FastMCP's `http_app(path="/mcp", stateless_http=True, json_response=True)`: no MCP session ids, a fresh transport per request, JSON responses instead of SSE, and 405 for GET and DELETE.

**Per-lifespan session manager.** The kb-service lifespan builds a new `create_mcp_server()` and `http_app` on every entry, logs the served tool names, and enters the app's own lifespan around the `yield`, because mcp 1.26's `StreamableHTTPSessionManager.run()` can only run once per instance. The first statement of the lifespan's `finally` resets `app.state.mcp_http_app` to None. A single exception raised through the yield comes back wrapped in an `ExceptionGroup` from the session manager's task group; the lifespan unwraps it so callers see the original exception.

**Tool set.** `mcp_server/server.py` copies the stdio server's instructions, tool prefixes and registration gates (`KB_INSTANCE_ROLE`, `KB_MANAGER`, `KB_CONTRIBUTOR`), all read from the kb-service environment, as is `KB_SKIP_SAFETY`. At most 17 tools are served: `{prefix}ingest` is never registered, and its instructions bullet is removed, because it reads files on the machine running the tool, which over `/mcp` is the service host. Each tool keeps only the stdio tool's HTTP-mode behaviour (section 25). The middleware list is `[McpCallLogMiddleware, SurfaceToolFilterMiddleware]`: the first writes the `mcp-call` log line below, the second shows a headless or autonomous caller only the allow-list `NON_INTERACTIVE_TOOL_BASES` (the read tools, `store` and `store_batch`, which queue candidates, and `feedback`), hides every other tool from `tools/list`, refuses it on `tools/call`, and logs `mcp-surface tool= outcome=hidden surface=` (see [intelligent-kb.md §7](docs/intelligent-kb.md#7-capture-c-the-write-policy-unattended-agents-propose-never-write)).

**Client setup.** `claude mcp add --transport http personal-kb https://<host>/mcp --header "Authorization: Bearer <key>"`. The server name must be `personal-kb` or `team-kb`, because personal-kb-hook's `PostToolUse` matcher keys on it. A reverse proxy in front of kb-service needs a read timeout of at least 300 s, because `kb_ask`, `kb_summarize` and `kb_ingest_url` run LLM calls inside one request. The client's raw headers, including `X-KB-Mode`, `X-KB-Harness` and `X-KB-Engine`, reach the write policy.

**AuthPrincipal.** `kb_service.auth.AuthPrincipal` is a frozen dataclass of `user`, `api_key_id` (the `api_keys.id` row, or None) and `auth_method` (`api_key`, `jwt` or `none`). `resolve_principal(token)` is the one resolution path for both REST and `/mcp` (section 20). The REST dependency `get_current_principal` stores the result on `request.state.kb_principal`, and `get_current_user` depends on it.

**InProcessBackend.** `mcp_server/context.backend_for_request()` reads the principal and app from the current request (via `fastmcp.server.dependencies.get_http_request`) and fails closed with a `RuntimeError` (and an ERROR `mcp-principal-missing` line) when either is absent. `mcp_server/backend.InProcessBackend` mirrors `HttpBackend` method for method: it builds the same request body dict, validates it into the route's request model, awaits the route handler with every parameter passed by keyword, then runs `jsonable_encoder` on the result and parses it with the same statements `HttpBackend` uses. Each call gets a fresh shim `Request` whose scope carries the app, the principal and the client's raw headers, so a handler sees `request.app.state` and `request.state.kb_principal` as on REST. FastAPI dependencies do not run on this path, so the admin gate is replicated: methods in `ADMIN_ONLY_METHODS` (`reactivate`, `bulk_update`, `reconcile_supersession`, `map_eligibility`, `set_map_eligibility_override`, `clear_map_eligibility_override`) raise `BackendHttpError(403, "Admin only")` for a non-admin before the handler runs. Handler `HTTPException`s become `BackendHttpError(status, detail)`, request-model validation errors become 422, and anything else is logged and becomes 500 `Internal Server Error`; the tools render these with `mcp_server/errors.map_error`, byte-identical to the stdio client's `_map_error`.

**Drift guards.** `packages/kb-service/tests/test_mcp_backend.py` spies on each of the 20 handlers and asserts the backend passes exactly the handler's parameters; walks the FastAPI dependency tree of each handler and fails if any handler gains a dependency other than `get_current_user`, `get_current_principal`, `_bearer` and (for the admin six) `require_admin`, because such a dependency would be bypassed on `/mcp`; and checks that the set of admin-gated handlers equals `ADMIN_ONLY_METHODS`. `test_mcp_import_purity.py` fails on any `personal_kb` import under `src/kb_service`.

**Parity rule.** During the stdio overlap every tool exists twice, in `src/personal_kb/tools/` and in `packages/kb-service/src/kb_service/mcp_server/tools/`; change both together. `tests/test_mcp_http_parity.py` (root suite) compares the stdio server and `/mcp` server tool by tool (description, parameters, output schema, annotations, title, tags, meta) for the default and the team/manager/contributor environments, and checks the instructions differ only by the `kb_ingest` bullet. It then runs the same 25-call sequence through both channels against fresh SQLite KBs (stdio tools over `HttpBackend` on an ASGI transport, and `/mcp` tool calls) and compares the outputs with timestamps masked, plus a table of client-side rejections that must not reach the backend at all. `tests/integration/test_mcp_http_smoke.py` starts a real `kb-service serve` and drives it with the official MCP client.

Writes through `/mcp` go through the same route handlers as REST, so the `supersession-route`, `resolution-route` and `near-duplicate-guard` log lines and audit rows fire for MCP writes too.

**Observability.** All lines are %-style `key=value` records under the `kb_service` logger (INFO by default); arguments, tokens and header values other than the User-Agent are never logged.

- `mcp-endpoint started path=/mcp prefix= tools= names=` (INFO, once per lifespan) lists the served tools; `mcp-endpoint not_started method=` (WARNING) accompanies a 503.
- `mcp-auth denied status= reason= key_fp= ua=` (INFO) for every rejected request: reason is `missing`, `invalid_key`, `user_not_found` or `other`; `key_fp` is the first 8 hex digits of the token's sha256, or `none`.
- `mcp-call tool= outcome= worst_status= duration_ms= user_id= key_id= auth= ua=` (INFO), exactly one per tool call from `McpCallLogMiddleware`: outcome is `exception` when the tool raised, `tool_error` when any backend call returned 400 or above, else `ok`; `worst_status` is the highest backend status, or `none`.
- `mcp-backend op= status=403 reason=admin_gate user_id= key_id=` (WARNING) for an admin-gate denial, and `mcp-backend op= status=500 user_id= key_id=` (ERROR, with traceback) for an unexpected handler exception.
- `mcp-principal-missing has_principal= has_app=` (ERROR) when a tool call reaches `backend_for_request` without an authenticated principal.

See: `packages/kb-service/src/kb_service/mcp_server/`, `kb_service/auth.py`, `kb_service/main.py`, `src/personal_kb/server.py`.

## 26. Maintenance tools

`KB_MANAGER=TRUE` in the serving process registers `kb_maintain` and `kb_bulk_update`.

**kb_maintain** declares 14 actions, but only three run over HTTP: `deactivate` (needs `entry_id` and `change_reason`, optional `superseded_by` matching `kb-\d{5}`), `reactivate` (admin) and `reconcile_supersession` (admin; renders `Supersession reconcile: edges_added= set= cleared=` plus one `drift <target>: old -> new` line per change). Every other action (`stats`, `rebuild_embeddings`, `rebuild_graph`, `purge_inactive`, `vacuum`, `entry_versions`, `list_feedback`, `summarize_feedback`, `search_stats`, `list_contributors`, `list_audit`) returns `Error: action X is not supported in HTTP mode — run it on the KB service host.`, and an unknown action returns the sorted action list.

**kb_bulk_update** forwards to `POST /api/kb/bulk_update` (admin; `dry_run` defaults to true). Allowed filter keys: `contributor`, `team`, `project_ref` (null matches unset), `entry_type`, `tags`, `entry_ids`. Allowed update keys: `project_ref`, `entry_type`, `confidence_level`, `tags_add`, `tags_remove`, `team`. Empty or unknown keys are 422 (the tool also rejects empty filters or updates client-side). `KnowledgeStore.bulk_update` updates matching active rows, bumps `version` and `updated_at` (which resets decay), stamps `updated_by`, and writes a `Bulk metadata update` version row and audit event. It does not rebuild graph edges, so `has_tag` and `in_project` edges stay stale until each entry is next updated; only an `entry_type` change triggers `recompute_supersession`. A non-dry run from a non-interactive surface is a write-policy 403.

## 27. Configuration reference

This table lists every core environment variable. Read-by values: **service** (kb-service, through `kb_service/config.py`, `main.py`, `auth.py` or `database.py`; kb-core itself reads no environment, which an AST test enforces), **client** (the stdio `personal-kb` process), **script** (an operator script only). Automation variables (`KB_SURPRISE_*`, `KB_SOFT_GATE_*`, `KB_WRITE_POLICY_DEFAULT_SURFACE`, `KB_LISTENER_*`, `KB_DELIVER_OBSERVED_ONCE` and the hook's variables) are in [intelligent-kb.md §22](docs/intelligent-kb.md#22-production-state-and-switches).

| Variable | Read by | Default | Effect |
|---|---|---|---|
| `KB_DATABASE_URL` | service | unset | Non-empty selects Postgres for the data DB; else SQLite |
| `KB_DB_PATH` | service | `~/.local/share/personal_kb/knowledge.db` | SQLite data DB; also locates `service.db` in none mode; error if set with a Postgres URL |
| `KB_SERVICE_DATABASE_URL` | service | unset | Postgres service DB; required unless `KB_AUTH_MODE=none` |
| `KB_AUTH_MODE` | service | `jwt` | `jwt` or `none` (synthetic admin for every caller) |
| `JWT_SECRET` | service | `dev-secret-change-me` | JWT signing key; the default is not refused |
| `KB_SERVICE_PUBLIC_URL` | service | request scheme and host | Base for invite and reset links and the `/mcp` `kb_explore` URL |
| `KB_SERVICE_CLIENT_INSTALL_SPEC` | service | `personal-kb @ git+https://github.com/jason-weddington/personal-kb-mcp` | Shown by `GET /api/settings` |
| `KB_SERVICE_STATIC_DIR` | service | unset | First choice for the SPA directory |
| `KB_EMBEDDING_MODEL` | service | `qwen3-embedding:0.6b` | Ollama embedding model |
| `KB_EMBEDDING_DIM` | service | `1024` | Vector size; must match the stored table |
| `KB_OLLAMA_URL` | service | `http://localhost:11434` | Ollama for embeddings and the Ollama LLM role |
| `KB_OLLAMA_TIMEOUT` | service | `10.0` | Embedding request timeout (s) |
| `KB_OLLAMA_KEEP_ALIVE` | service | `30m` | **Inert**: read but never passed to the engine, which always uses 30m |
| `KB_OLLAMA_MODEL` | service | `qwen3:4b` | Ollama LLM for every role when a provider is `ollama` |
| `KB_OLLAMA_LLM_TIMEOUT` | service | `120.0` | Ollama LLM timeout (s) |
| `KB_EMBED_WORKER_ENABLED` | service | `TRUE` | Embedding retry worker (Postgres only) |
| `KB_EMBED_WORKER_BATCH_SIZE` | service | `16` | Rows claimed per pass |
| `KB_EMBED_WORKER_POLL_SECONDS` | service | `60.0` | Drain interval |
| `KB_EMBED_WORKER_TIMEOUT` | service | `180.0` | Worker embed request timeout (s) |
| `KB_PG_POOL_MIN`, `KB_PG_POOL_MAX` | service | `1`, `5` | Data-DB asyncpg pool |
| `KB_PG_IAM_AUTH`, `KB_PG_REGION` | script | unset, `us-east-1` | **Not honoured by kb-service**; only `scripts/migrate_sqlite_to_pg.py` via the legacy connection module |
| `KB_EXTRACTION_PROVIDER` | service | `anthropic` | Provider for the extraction role (`anthropic`, `bedrock`, `ollama`) |
| `KB_QUERY_PROVIDER` | service | `anthropic` | Provider for the query and synthesis roles |
| `ANTHROPIC_API_KEY` | service | unset | Anthropic credentials |
| `KB_ANTHROPIC_MODEL` | service | `claude-haiku-4-5` | Extraction and query model; synthesis uses `claude-sonnet-4-6` regardless |
| `KB_ANTHROPIC_TIMEOUT` | service | `30.0` | Anthropic request timeout (s) |
| `KB_BEDROCK_MODEL` | service | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Extraction and query model on Bedrock |
| `KB_BEDROCK_REGION` | service | `us-east-1` | Bedrock region |
| `KB_BEDROCK_TIMEOUT` | service | `60.0` | Bedrock timeout (s) |
| `KB_AWS_PROFILE` | service | unset | Bedrock profile; else a profile named `personal_kb_bedrock` |
| `AWS_BEARER_TOKEN_BEDROCK`, `AWS_ACCESS_KEY_ID` | service | unset | Further Bedrock credential sources |
| `KB_AGENTIC_QUERY` | service | `TRUE` | Agent loop for `kb_ask`/`kb_summarize` retrieval |
| `KB_AGENTIC_SYNTHESIS` | service | `TRUE` | Coverage check before synthesis |
| `KB_AGENTIC_MAX_CALLS` | service | `4` | Agent LLM turns; non-integer fails startup |
| `KB_INGEST_MAX_FILE_SIZE` | service | `10485760` | Ingest size limit (bytes) |
| `KB_INGEST_CHUNK_SIZE`, `KB_INGEST_CHUNK_OVERLAP` | service | `16000`, `600` | Chunker (characters) |
| `KB_AGENTIC_INGEST` | service | `TRUE` | Wires the ingest dedup agent |
| `KB_INGEST_DEDUP_THRESHOLD` | service | `0.06` | Dedup agent score floor (above the reachable maximum, section 17) |
| `KB_SKIP_SAFETY` | service, client | unset | `TRUE` skips the service's store, chat, surprise-capture and ingest-route secret checks; in the stdio client it skips the `kb_store`/`kb_store_batch` scan |
| `KB_NEAR_DUPLICATE_FLOOR` | service | `0.88` | Near-duplicate cosine floor; above 1.0 disables |
| `KB_LLM_DIAGNOSTICS` | service | on | `0`, `false`, `off`, `no` stop sending cache diagnostics |
| `KB_MANAGER` | service, client | unset | `TRUE` registers `kb_maintain` and `kb_bulk_update` |
| `KB_CONTRIBUTOR` | service, client | unset | Any value registers the contributor and team list tools; attributes nothing |
| `KB_TEAM` | legacy client config only | unset | **Unused by the service**; team comes from `PUT /api/settings` |
| `KB_INSTANCE_ROLE` | service, client | unset | `personal` or `team` tool-name prefix and role paragraph |
| `KB_LOG_LEVEL` | service, client | INFO (service), WARNING (client) | Log level |
| `PERSONAL_KB_URL` | client | `http://127.0.0.1:8765` | kb-service base URL; loopback triggers the local daemon |
| `PERSONAL_KB_API_KEY` | client | `local-no-auth` on loopback only | Bearer key |
| `PERSONAL_KB_DAEMON_STATE_DIR` | client | `~/.local/share/personal_kb` | Daemon pidfile and log directory |
| `HEADLESS_BUILD_ENGINE` | client | unset | Sent as `X-KB-Engine` with `X-KB-Harness: claude-code` |
| `HOSTNAME` | service | `localhost` | Adds `https://$HOSTNAME` to the CORS origins |

## 28. Graceful degradation

Every optional dependency fails toward a reduced but working KB:

- **Ollama down.** Search is FTS-only; every embed while it is down pays one `/api/tags` probe; failed entries are queued for retry, which self-heals only on Postgres (section 10). The near-duplicate guard fails open (`embedder_unavailable`).
- **No extraction LLM.** No enrichment; ingestion returns 503.
- **No query LLM.** No agent, planner, coverage or classifier call: `kb_ask` runs the non-agentic auto search and the query stream falls back to explore.
- **No synthesis LLM.** Synthesis uses the query LLM, and with neither returns raw entries.
- **Coverage failure.** Never blocks synthesis (section 15).
- **Search failure in the near-duplicate guard.** Fails open (`search_failed`).
- **Enrichment failure.** Logged with the `enrichment-failure` marker; the write succeeds.
- **Graph builder failure.** Logged; the next supersession reconcile heals the pointers.
- **detect-secrets missing.** The store scan passes (fail-open), the ingest routes return 503, and turn digests are not stored (see [intelligent-kb.md §6](docs/intelligent-kb.md#6-capture-b-turn-digests)).

## 29. Testing, eval baselines and drift guards

Commands: `uv run pytest -m "not eval"` for the main suite (the default for every iteration loop) and `(cd packages/personal-kb-hook && uv run --project ../.. pytest)` for the hook suite. The root `pyproject.toml` declares the uv workspace, so `uv sync` installs the hook package editable into the main `.venv` for the cross-package tests.

**Eval baselines.** The search baseline (`tests/eval/baseline.json`) is deterministic, using a `ControlledEmbedder`, and is regenerated by `uv run pytest tests/eval/test_baseline.py -s` for any ranking change. The agent baseline (`tests/eval/agent_baseline.json`) hits a live Anthropic API, carries the `eval` marker, and rewrites its JSON when run; it is run deliberately, never in a build loop. Current numbers are in section 14. The agent baseline test reads three optional, eval-only environment variables that no shipped package ever reads: `KB_EVAL_ANTHROPIC_MODEL` (a model id), `KB_EVAL_ANTHROPIC_EFFORT` (`low`, `medium`, `high`, `xhigh` or `max`, sent as `output_config.effort`) and `KB_EVAL_ANTHROPIC_THINKING` (`disabled` or `adaptive`, sent as `thinking.type`), passing them onto the `AnthropicProviderConfig` it builds; unset variables leave the config exactly as it always was. The surprise-detector eval harness has equivalent per-arm knobs (`--model MODEL[@EFFORT][+nothink]`), documented in [its README](scripts/surprise_eval/README.md).

**Drift guards** and what each pins:

| Guard | Pins |
|---|---|
| `tests/test_mcp_http_parity.py` | stdio and `/mcp` tool tables, instructions and 25-call output parity |
| `packages/kb-service/tests/test_mcp_backend.py` | `InProcessBackend` parameters, handler dependencies, `ADMIN_ONLY_METHODS` |
| `packages/kb-service/tests/test_mcp_import_purity.py` | No `personal_kb` import in kb-service |
| `packages/kb-service/tests/test_write_policy.py::test_write_route_classification` | Every write route is classified by the write policy |
| `tests/test_cue_drift_guard.py` | The hook's `cues_lite` copy matches `kb_core.cues` |
| `tests/test_gate_decision_drift_guard.py` | The hook's failure-context gate-log rows validate as kb_service `GateDecisionRow` |
| `tests/test_local_defaults_drift_guard.py` | Hook local-mode URL and key equal `personal_kb.config` |
| `tests/test_turn_digest_drift_guard.py` | The hook's turn-digest caps and body match kb_service's contract |
| `packages/kb-service/tests/test_contracts_service.py`, `packages/personal-kb-hook/tests/test_contracts_hook.py` | The `contracts/` wire fixtures (see [intelligent-kb.md §19](docs/intelligent-kb.md#19-harness-portability-talos-and-the-wire-contracts)) |
| `packages/kb-service/tests/test_llm_calls.py` | AST check that every `AnthropicLLMClient(...)` in kb_service passes `recorder=` |
| `packages/kb-core/tests/test_import_purity.py` | AST check that kb_core reads no `os.environ` and stays light to import |
| `scripts/check_static_ui_fresh.sh` | The packaged SPA matches the frontend source |
| `tests/test_docs_coverage.py` | Every route the kb-service app registers has a row, per method, in the section 21 table (mounts must appear in that section), and every environment variable the four shipped packages read appears in this file or in `docs/intelligent-kb.md` |

**Keeping the docs current.** A change that adds a feature, route, env var, table or behaviour updates this file (core) or `docs/intelligent-kb.md` (automation) in the same commit. `tests/test_docs_coverage.py` enforces the part a test can check: it fails when the app gains a route or method with no row in section 21, or when code in `src/personal_kb`, `kb_core`, `kb_service` or `personal_kb_hook` reads an environment variable that neither doc names. It finds reads as string literals passed to `os.environ.get`, `os.getenv`, `os.environ.setdefault` or `os.environ[...]`, plus any whole `KB_*` or `PERSONAL_KB_*` string literal, which catches reads through helpers and named constants; a name built at runtime (an f-string) is not seen. Tables, behaviour and prose stay a review responsibility.

## 30. Known limitations

As of 2026-10-11. Each item is a defect or gap verified in code; remove it from this list when it is fixed.

| Limitation | Module | Consequence |
|---|---|---|
| Ingest dedup agent is inert | `kb_core/ingest/dedup_agent.py` | The default threshold 0.06 exceeds the maximum RRF score 0.0328, so overlapping files are always re-extracted |
| Tool updates reset confidence | `src/personal_kb/tools/kb_store.py` and twin | Every MCP update writes `confidence_level` 0.9 unless the caller passes the current value |
| Title-only updates keep the old vector | `kb_core/knowledge_base.py` (`update`) | Search ranks the entry on stale titles |
| TTL cannot be cleared over HTTP | `kb_service/models_kb.py` (`StoreRequest`) | An expiry can only be replaced, not removed |
| IAM auth not wired in kb-service | `kb_service/main.py` (`_open_kb`) | `KB_PG_IAM_AUTH` has no effect on the service |
| `KB_OLLAMA_KEEP_ALIVE` inert | `kb_service/config.py` (`build_embedding_config`) | The knob does nothing; the code comment explaining why is stale |
| SQLite embedding retries never drain | `kb_core/embedding_retry.py`, `kb_service/main.py` | Entries that failed to embed on SQLite stay vector-less |
| Migration drops TTLs | `scripts/migrate_sqlite_to_pg.py` | Expiring entries become permanent after migration |
| Agent tool dispatch is unguarded | `kb_core/graph/agent.py` (`_dispatch_tool`) | A hallucinated tool argument fails the whole ask or summarize request |
| Enricher vocabulary never refreshes for single stores | `kb_core/graph/enricher.py` | Long-running services dedupe new entities against a stale vocabulary |
| Five-digit id regexes | `kb_core/supersession.py`, `kb_core/graph/builder.py`, `kb_service/routes/near_duplicate_guard.py`, `kb_store` tool | Ids from `kb-100000` on would fail validation |
| Default `JWT_SECRET` accepted | `kb_service/auth.py` | A deploy that forgets the secret signs tokens with a public string |
| `none` mode makes every caller admin | `kb_service/auth.py`, `kb_service/cli.py` (`serve`) | Binding a no-auth service to a non-loopback host exposes admin to the network |
| URL ingest has no private-address guard | `kb_core/ingest/ingester.py` | The service host fetches any URL an interactive user supplies |
| Feedback, telemetry, audit and versions unreadable over HTTP | `kb_service/routes/`, `kb_maintain` | Analysis needs direct database access |
| Bulk update leaves graph edges stale | `kb_service/routes/kb_write_routes.py` (`bulk_update`) | `has_tag` and `in_project` edges lag until each entry's next update |
| Dead "Vector search unavailable" note | `kb_search` tools | Never shown over the service |
| Chunk secret skips labelled "redacted" | `src/personal_kb/tools/kb_ingest.py`, `kb_ingest_url` twin | Dropped chunks are reported as redacted |
| File-level `flagged` action never produced | `kb_core/ingest/ingester.py` | The enum member is unreachable |
| `match_source` comment lists `vector` | `kb_core/models/search.py` | Misleads readers; the value is never produced |
