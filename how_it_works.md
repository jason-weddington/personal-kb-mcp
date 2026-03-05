# How It Works

Personal KB is a Model Context Protocol (MCP) server that gives AI assistants a persistent, searchable memory. Built on FastMCP with an async stdio transport, it stores knowledge entries in SQLite and retrieves them through a combination of full-text search, vector similarity, and a knowledge graph. The system has three core layers: a storage engine that keeps entries, versions, embeddings, and graph data in a single SQLite file; a retrieval layer that ranks results through hybrid search and confidence decay; and a graph layer that connects entries to each other and to named entities. Every component is designed to degrade gracefully — if Ollama is down or no LLM is configured, the server still stores entries and serves full-text search.

## Storage Engine

Everything lives in one SQLite database file, typically at `~/.local/share/personal_kb/knowledge.db`. The connection is initialized with WAL mode for better concurrent read performance and foreign keys enabled for referential integrity. The row factory is set to `aiosqlite.Row` so queries return dict-like objects. All database operations are async via aiosqlite.

The `knowledge_entries` table is the center of the schema. Each entry has a text primary key following the `kb-XXXXX` format (zero-padded to five digits), along with fields for titles, content, entry type, project reference, tags, hints, confidence level, timestamps, and flags for active status and embedding presence. The ID sequence is managed by a separate `entry_id_seq` table that holds a single integer, atomically read-and-incremented on every insert. The formatting happens in `db/queries.py:next_entry_id`, which reads the current value, bumps it by one, and returns the zero-padded string.

Versioning is built into every write operation. When an entry is created, the `KnowledgeStore` inserts both the entry row and an initial version record in `entry_versions` with version number 1 and a change reason of "Initial creation." On every update, the version number increments, a new version record captures the updated knowledge details, confidence level, and change reason, and the entry's `updated_at` timestamp resets. This means the full history of every entry is preserved and can be queried through the versions table. The version table has a unique constraint on `(entry_id, version_number)` to prevent duplicates.

Tags are stored as space-separated text in the entries table rather than as JSON arrays. This was an intentional design choice: it allows the FTS5 content-sync triggers to index tags directly alongside the other text fields without any JSON parsing at index time. When tags come out of the database, `db/queries.py:_parse_tags` splits them on whitespace. When they go in, `db/queries.py:insert_entry` joins the list with spaces. The trade-off is that individual tags cannot contain spaces, but that is a reasonable constraint for short categorization labels.

See: `db/schema.py`, `db/connection.py`, `db/queries.py`, `store/knowledge_store.py`, `models/entry.py`

### Database Abstraction and PostgreSQL

All source code operates against a `Database` protocol defined in `db/backend.py`, not against a concrete database library. The protocol specifies 12 methods: `execute`, `executemany`, `executescript`, `commit`, `close`, `fts_search`, `vector_store`, `vector_search`, `vector_delete`, `delete_llm_edges`, `vacuum`, and `apply_schema`. Two implementations exist: `SQLiteBackend` (wrapping aiosqlite + sqlite-vec + FTS5) and `PostgresBackend` (wrapping asyncpg + pgvector + tsvector/GIN).

`create_connection()` in `db/connection.py` dispatches between the two: if `KB_DATABASE_URL` is set to a `postgresql://` URL, it creates a `PostgresBackend`; otherwise it creates a `SQLiteBackend` at the configured file path. Explicit `:memory:` always uses SQLite (for tests).

`PostgresBackend` translates `?` placeholders to `$1, $2, ...` via a pre-compiled regex at execute time. Each `execute()` call acquires a connection from the asyncpg pool, runs the query, and releases the connection. `commit()` is a no-op since asyncpg auto-commits. FTS uses tsvector with ts_rank_cd instead of FTS5/BM25, and vector search uses pgvector's `<=>` cosine distance operator instead of sqlite-vec.

### Aurora IAM Authentication

For AWS deployments, the server supports RDS/Aurora IAM authentication via `KB_PG_IAM_AUTH=TRUE`. All IAM-specific logic lives in `db/iam_auth.py`, keeping AWS/boto3 knowledge out of `PostgresBackend`. The module has three components:

**DSN parsing** (`parse_dsn`): Extracts host, port, and username from the PostgreSQL connection URL via `urllib.parse.urlparse()`. The host and username are required for token generation; port defaults to 5432.

**Token factory** (`make_token_factory`): Returns a zero-arg closure that calls `boto3.client('rds').generate_db_auth_token()`. The boto3 RDS client is created once when the factory is built (captured in the closure); each call generates a fresh SigV4-signed token valid for ~15 minutes. The `import boto3` is lazy — it only runs when IAM auth is enabled, so plain Postgres users never import boto3.

**SSL context** (`make_ssl_context`): Returns `ssl.create_default_context()` with certificate verification enabled (`verify_mode=CERT_REQUIRED`, `check_hostname=True`). IAM auth requires TLS. Users can override the CA bundle via the `SSL_CERT_FILE` environment variable.

The connection flow in `_create_postgres()`: when `is_pg_iam_auth()` returns True, it parses the DSN, builds a token factory, creates an SSL context, and passes both as `password` and `ssl` kwargs to `PostgresBackend.create()`. asyncpg's `create_pool()` accepts `password` as a callable, invoking it for each new connection — so token refresh is automatic. AWS credentials come from the standard boto3 chain (environment variables, `~/.aws/credentials` profiles, instance roles).

`PostgresBackend.create()` accepts optional `password` (callable) and `ssl` (SSLContext) kwargs, forwarding them to `asyncpg.create_pool()` only when not None. This keeps the backend generic — it has no awareness of IAM, boto3, or AWS.

See: `db/iam_auth.py`, `db/connection.py`, `db/postgres_backend.py`

## Full-Text Search

The FTS5 virtual table `knowledge_fts` indexes four columns: `short_title`, `long_title`, `knowledge_details`, and `tags`. It uses external content mode (`content='knowledge_entries'`), meaning the FTS index does not store its own copy of the text — it reads from the main table via rowid. The tokenizer is `porter unicode61`, which applies Porter stemming for English morphology and Unicode 6.1 normalization for broad character support.

Three triggers keep the FTS index in lockstep with the content table. The `knowledge_fts_ai` trigger fires after every INSERT and adds the new row's text to the index. The `knowledge_fts_au` trigger fires after every UPDATE and performs a delete-then-insert: it first removes the old text (using the special FTS5 `'delete'` command with the old row values), then inserts the new text. The `knowledge_fts_ad` trigger fires after DELETE and removes the old text. Because the triggers are defined with `AFTER` timing, they see the final state of each operation.

Search queries go through `search/fts.py:fts_search`, which joins the FTS results back to `knowledge_entries` via rowid to access the full entry. The function uses BM25 scoring via `bm25(knowledge_fts)`, where FTS5 returns negative scores with more-negative values indicating stronger matches. Results are ordered by score (ascending, since more negative is better) and capped at a caller-specified limit.

Before hitting FTS5, the raw query string is escaped by `_escape_fts_query`, which splits the input on whitespace and wraps each token in double quotes. This prevents FTS5 syntax errors from special characters like colons, hyphens, or parentheses that FTS5 would otherwise interpret as query operators. The escaped tokens are joined with spaces, which FTS5 treats as implicit AND.

Filtering happens in the SQL WHERE clause after the FTS MATCH. Project reference and entry type filter with exact equality. Tag filtering uses the pattern `(' ' || e.tags || ' ') LIKE '% tag %'` — padding the stored tags with spaces on both sides and searching for the tag surrounded by spaces ensures exact word boundary matching rather than substring matching.

See: `db/schema.py` (triggers), `search/fts.py`

## Vector Embeddings

When Ollama is available, every entry also gets a dense vector embedding for semantic similarity search. The embedding client in `search/embeddings.py` talks to Ollama's `/api/embed` endpoint. The default model is `qwen3-embedding:0.6b`, which produces 1024-dimensional vectors — both configurable via environment variables.

The text fed to the embedding model is the concatenation of `short_title`, `long_title`, and `knowledge_details`, joined with spaces. This is defined as the `embedding_text` property on the `KnowledgeEntry` model, ensuring every component that needs the text for embedding uses the same concatenation.

Vectors are stored in a `knowledge_vec` virtual table powered by the sqlite-vec extension. The table schema is `vec0(entry_id TEXT PRIMARY KEY, embedding FLOAT[1024])`. Loading the sqlite-vec extension requires special handling with aiosqlite: a zero-argument closure captures the underlying synchronous `db._conn` object, enables extension loading, calls `sqlite_vec.load(db._conn)`, then disables extension loading. This closure is executed via `await db._execute(closure)` to run on aiosqlite's background thread.

Embedding vectors are serialized to binary using `struct.pack` with a format string like `"1024f"`, producing a compact 4-byte-per-float representation that sqlite-vec expects. KNN search uses the `WHERE embedding MATCH ?` syntax with the serialized query vector, and sqlite-vec returns results ordered by Euclidean distance (lower is better).

The embedding client uses a success-only caching pattern for availability checks. `is_available()` pings Ollama's `/api/tags` endpoint. On success, it caches `_available = True` and skips the ping on future calls. On failure, it sets `_available = None` (not False), meaning the next call will retry the ping rather than permanently giving up. This design lets the system recover automatically if Ollama starts up after the server does.

See: `search/embeddings.py`, `search/vector.py`, `db/schema.py`

## Hybrid Search and Reciprocal Rank Fusion

When both FTS and vector search produce results, they are combined using Reciprocal Rank Fusion (RRF) [1]. The implementation lives in `search/hybrid.py:hybrid_search`.

The process starts by over-fetching: both FTS and vector search are asked for `limit * 3` results (three times the caller's requested limit). Over-fetching improves re-ranking quality because entries that appear in both lists — which are the strongest signals of relevance — are more likely to be captured even if they rank modestly in one list.

Each entry's RRF score is computed as: `score(d) = sum of 1/(K + rank + 1)` across all lists in which it appears, where K is a constant set to 60 (the value from the original RRF paper [1]) and rank is the entry's zero-based position in each list. An entry that appears in both FTS and vector results gets the sum of its two reciprocal rank scores, naturally boosting entries confirmed by both retrieval methods. Entries from only one list still get a score, just lower.

After RRF scoring, entries are sorted by combined score in descending order (higher is better, the inverse of the raw FTS and distance scores). The top entries up to the caller's limit are then looked up to get their full data, confidence decay is applied, and stale entries (effective confidence below 0.3) are filtered out unless the caller explicitly requests them via `include_stale`.

Each result carries a `match_source` field: `"hybrid"` if both FTS and vector contributed results, or `"fts"` if vector search returned nothing (because Ollama was unavailable or no embeddings exist).

See: `search/hybrid.py`

## Sparse Graph Hints

When `kb_search` returns fewer than 3 results, the system augments the response with graph-connected entries that the user might find relevant. This idea draws on Think-on-Graph's iterative graph reasoning [2] and GraphRAG's community-level summarization [3], adapted to a much smaller scale: instead of multi-hop reasoning chains or hierarchical community summaries, we do a simple 1-2 hop BFS and format the results as hints. The `collect_graph_hints` function in `tools/kb_search.py` walks the knowledge graph outward from each search result to find related entries that didn't match the query directly.

For each result entry, the function calls `get_neighbors` with a limit of 10. Non-entry neighbors — tags, concepts, tools, and other intermediate nodes — get a second hop: the function calls `get_neighbors` again on the intermediate node to find entries connected through it. For example, if a search result has a `has_tag` edge to `tag:security`, and another entry also has a `has_tag` edge to `tag:security`, that second entry becomes a hint candidate. Direct entry-to-entry neighbors (from `supersedes`, `references`, etc.) are also collected but only need the single hop.

A `seen_ids` set tracks all result entry IDs and any entries already collected as hints, preventing duplicates. Deactivated entries are filtered out — only entries with `is_active = True` are included. The function returns up to 3 formatted hint strings like `"See also: [kb-00042] Title (via tag:security)"`, where the `via` label identifies the intermediate node or edge type that connects the hint to the original result.

The sparse threshold and max hints are constants (`_SPARSE_THRESHOLD = 3`, `_MAX_HINTS = 3`). When results are plentiful, hints are skipped entirely — the function is never called.

See: `tools/kb_search.py`, `tools/formatters.py`

## Confidence Decay

Every entry's confidence degrades over time using exponential decay. The formula is: `effective = base_confidence * 2^(-age_days / half_life)`. This uses base-2 exponential decay, meaning after exactly one half-life, an entry retains half its original confidence.

Half-lives vary by entry type, reflecting how quickly different kinds of knowledge go stale:

- **factual_reference**: 90 days (3 months) — facts like version numbers and API details change frequently
- **decision**: 365 days (1 year) — decisions persist but the context that justified them shifts
- **pattern_convention**: 730 days (2 years) — coding standards and conventions are durable
- **lesson_learned**: 1,825 days (5 years) — hard-won debugging insights and experiential knowledge stick

The decay clock anchors on whichever is more recent: `updated_at` or `last_accessed`. This access-aware approach is inspired by MemoryBank's dynamic memory management [4] and the recency-relevance-importance scoring in Generative Agents [5] — the insight being that retrieval frequency is a valid signal for knowledge value, not just recency of creation. Editing an entry resets its decay via `updated_at`, because if you've verified and updated a piece of knowledge it should be treated as fresh. Retrieving an entry via `kb_get` resets its decay via `last_accessed`, because actively-used knowledge shouldn't rot just because it hasn't been edited. The `db/queries.py:touch_accessed` function batch-updates `last_accessed` to the current time for all entries returned by a `kb_get` call. Crucially, search alone does not reset the decay clock — only explicit retrieval via `kb_get` does. This means an entry that keeps appearing in search results but is never opened will still decay, while one that a user regularly retrieves stays fresh.

Two thresholds govern how decay affects results. At 50% effective confidence, a staleness warning is attached to the result, suggesting the user verify the information is still current. At 30%, the entry is filtered out of search results entirely (unless the caller passes `include_stale=True`). For a factual reference with the default 0.9 confidence, the warning appears around 90 days (one half-life, when 0.9 drops to ~0.45) and the hard cutoff around 160 days (when 0.9 drops to ~0.27).

See: `confidence/decay.py`, `db/queries.py` (`touch_accessed`)

## The Knowledge Graph

The knowledge graph is stored in two tables: `graph_nodes` and `graph_edges`. Nodes have a text primary key (`node_id`), a type, JSON properties, and a creation timestamp. Edges have a source, target, edge type, JSON properties, and a unique constraint on `(source, target, edge_type)` to prevent duplicate edges.

Node IDs follow a human-readable convention that encodes the type: `tag:python`, `project:personal-kb`, `person:jason`, `tool:aiosqlite`, `concept:async-io`, `technology:fastapi`, `note:path/to/file.md`. Entry nodes use their entry ID directly (e.g., `kb-00042`). This naming convention makes the graph browsable and query results interpretable without joining back to other tables.

Graph construction happens in two tiers: deterministic edges derived directly from entry data, and LLM-enriched edges extracted by a language model.

### Deterministic Edges

The `GraphBuilder` in `graph/builder.py` runs on every store and update operation. It follows a delete-and-rebuild model: first it clears all outgoing edges from the entry's node, then re-derives them from the entry's current data. This approach is simpler and more robust than incremental diffing — it guarantees the graph always reflects the entry's current state.

The builder creates edges in this order:

1. **Entry node** — upserted with properties including `short_title` and `entry_type`
2. **Tag edges** — for each tag, creates a `tag:X` node and a `has_tag` edge
3. **Project edge** — if `project_ref` is set, creates a `project:X` node and an `in_project` edge
4. **Supersedes edges** — from the `supersedes` key in hints, creates edges to older entries
5. **Superseded-by edges** — if the entry has a `superseded_by` field, creates the reverse edge
6. **Text references** — scans `knowledge_details` for `kb-XXXXX` patterns and creates `references` edges to those entries, deduplicating matches
7. **Related entities** — from `hints.related_entities`, creates edges with caller-specified types (defaulting to `related_to`)
8. **Person hints** — from `hints.person`, creates `person:X` nodes and `mentions_person` edges
9. **Tool hints** — from `hints.tool`, creates `tool:X` nodes and `uses_tool` edges

### LLM Enrichment

The `GraphEnricher` in `graph/enricher.py` adds a second layer of edges by asking an LLM to extract entities and relationships from each entry's content. The LLM works with a closed set of four entity types: `person`, `tool`, `concept`, and `technology`. The system prompt instructs it to extract 2-6 entities per entry, returning a JSON array where each object specifies an entity name, entity type, and relationship type. Relationship types are open-ended — the LLM can use whatever describes the connection best (`uses`, `depends_on`, `implements`, `solves`, `replaces`, etc.).

The enricher's response parsing is defensive: it strips markdown code fences, finds the JSON array via regex, validates each item's structure and entity type, and caps results at 8 relationships. All LLM-derived edges are marked with `{"source": "llm"}` in their properties, which enables selective clearing. When re-enriching an entry, the enricher deletes only edges where `json_extract(properties, '$.source') = 'llm'`, preserving all deterministic edges. The enricher also ensures the entry node exists (via `ON CONFLICT DO NOTHING`) before adding edges, avoiding foreign key violations if the deterministic builder hasn't run yet.

Before creating a new entity node, the enricher checks for near-duplicates in the existing graph — an entity resolution step inspired by GraphRAG [3] and LightRAG [6], where entity deduplication is identified as critical for small graphs where fragmentation degrades connectivity. It loads the current graph vocabulary — all non-entry node IDs grouped by type — via `get_graph_vocabulary()`, then compares each LLM-extracted entity name against every existing name using `difflib.SequenceMatcher`. If a match scores at or above 0.85, the enricher reuses the existing node instead of creating a new one. This matching is cross-type: if the LLM extracts `concept:asyncio` but `technology:asyncio` already exists in the graph, the enricher will merge to the existing node. The vocabulary cache is loaded once per `enrich_entry` or `enrich_batch` call, and new entities are registered in the cache immediately so later edges in the same batch can resolve against them.

Enrichment never breaks storage. The entire enrichment call is wrapped in a try/except in `kb_store.py:_enrich_graph`, so failures are logged and swallowed.

See: `graph/builder.py`, `graph/enricher.py`, `db/schema.py`

## Query Strategies (kb_ask)

The `kb_ask` tool supports five query strategies, each suited to different kinds of questions.

**auto** is the default strategy. It runs hybrid search (FTS + vector) to find matching entries, then expands results by walking one hop through the graph from each search hit. For each hit, it calls `get_neighbors` with a limit of 10 and adds any neighboring entry nodes that aren't already in the result set. This provides context around search results — if you find a decision entry, you might also see the entries it supersedes or the tools it references.

**decision_trace** searches for decision-type entries using FTS, then walks the `supersedes` chain in both directions for each hit. The chain walk in `graph/queries.py:supersedes_chain` follows `supersedes` edges backward (what this entry supersedes) and forward (what supersedes this entry), building a chronologically ordered list from oldest to newest. The formatted output labels each entry as "original decision", "supersedes kb-XXXXX", or "current" to show the decision's evolution.

**timeline** takes a scope (like `project:personal-kb` or `tag:sqlite`) and returns all matching entries sorted by `created_at`. It uses `graph/queries.py:entries_for_scope`, which interprets scope strings as project filters, tag lookups, person/tool graph traversals, or entry type filters depending on the prefix. This strategy is useful for understanding the history of a topic or project.

**related** performs a breadth-first search from a starting node with a maximum depth of 2. The BFS in `graph/queries.py:bfs_entries` uses a standard queue, tracking visited nodes to avoid cycles and collecting entry nodes (those matching `kb-XXXXX` format) encountered along the way. It returns each entry's depth and full path from the start node. This strategy answers questions like "what else relates to aiosqlite?"

**connection** finds the shortest path between two nodes using BFS with a maximum depth of 4. The `graph/queries.py:find_path` function returns a list of `(source, edge_type, target)` triples forming the path, or None if no path exists. This strategy answers questions like "how are these two concepts connected?"

### Agentic Query Planning

When a query LLM is available, the `auto` strategy delegates to a ReAct agent loop in `graph/agent.py` that can plan, execute, evaluate, and retry — replacing the single-shot query planner that had to pick the right strategy blindly.

The agent is designed around a key insight: instead of classifying a question into one of five strategies upfront, dissolve the strategy taxonomy into tools and let the agent compose them. The agent has access to six internal tools (not exposed as MCP tools):

| Tool | Purpose |
|---|---|
| `hybrid_search` | Full-text + vector search with optional filters |
| `graph_neighbors` | Get edges from any node in the graph |
| `list_graph_nodes` | Browse the knowledge graph vocabulary by type |
| `decision_chain` | Follow supersedes chains for decision entries |
| `scope_entries` | List entries for a project, tag, or entry type |
| `done` | Return final answer with entry IDs and reasoning |

**Fast-path optimization.** Before entering the agent loop, the system runs a hybrid search with the raw question. If the top result's RRF score exceeds a threshold (`_FAST_PATH_THRESHOLD = 0.030`), the results are returned immediately with zero LLM calls. In practice, this handles the majority of straightforward keyword queries — on the eval corpus, 8 of 13 golden queries resolve via fast-path. The fast-path results are also seeded into the agent's initial context so it doesn't waste a turn repeating the same search.

**The agent loop.** Each turn, the agent receives the full conversation history (flattened into a single string for the `generate()` call — there is no chat API) and a system prompt describing the available tools, evaluation rules, and the hard cap. The agent responds with a single JSON object — either a tool call or a final answer. Tool calls are dispatched to the corresponding async function, and the formatted result is appended to the conversation as a user message. Final answers trigger entry ID lookups and return an `AgentResult` with the entries, turn count, and reasoning.

**Hard cap.** The loop is bounded at 4 tool calls (configurable via `KB_AGENTIC_MAX_CALLS`), enforced by the orchestrator. The system prompt also restates the cap ("NEVER make more than 4 tool calls") as belt-and-suspenders. On exhaustion without a `done` call, the agent extracts all `kb-XXXXX` IDs mentioned in the conversation as a best-effort fallback, then falls back to fast-path results if nothing was found.

**Error handling.** If the LLM returns None (provider failure), the loop breaks immediately and falls back to fast-path results. If the LLM returns unparseable JSON, an error message is injected into the conversation and the loop continues, consuming one tool call from the budget. This gives the LLM a chance to self-correct.

**Compact output.** The agent only sees `format_entry_compact()` output — titles, types, scores, and metadata, but never `knowledge_details`. This keeps the context window lean across multiple turns. Full entry details are fetched only when formatting the final result for the caller.

**Toggle.** Set `KB_AGENTIC_QUERY=FALSE` to bypass the agent and fall back to the single-shot `QueryPlanner` from `graph/planner.py`, which translates natural language into a structured `QueryPlan` (strategy, scope, target, search query) via a single LLM call. The planner receives graph statistics and vocabulary as context for entity resolution.

**Eval results.** On the eval corpus (32 entries, 13 scorable golden queries), the agent achieves perfect MRR, recall@5, and NDCG@5 (all 1.000), up from 0.853/1.000/0.889 with hybrid search alone. The three queries where hybrid search ranked the right entry 2nd-4th (q05 REST auth, q06 CORS, q10 encoding bug) are all resolved in a single agent turn.

See: `graph/agent.py`, `tools/kb_ask.py`, `graph/planner.py`

## Answer Synthesis (kb_summarize)

The `kb_summarize` tool provides a higher-level interface than `kb_ask` by adding LLM synthesis on top of retrieval. The core logic lives in `summarize_question()`, an extracted function that takes the database, embedder, query LLM, question, optional scope, and limit — keeping it testable without FastMCP context.

### Retrieval

The first step calls `retrieve_entries()` from `tools/kb_ask.py`, which returns a tuple of `(entries_with_context, agent_turns_used)`. Each entry is paired with a context string describing how it was found (e.g., `"search match (score: 0.0312)"` or `"linked from kb-00042 via supersedes"`). The `agent_turns_used` count tracks how many LLM tool calls the agentic query loop made — 0 means the fast-path resolved the query without touching the LLM.

### Coverage Assessment

When the retrieval involved the agent (not fast-path), the system runs a coverage check before synthesis. The check fires only when all four conditions are met: `KB_AGENTIC_SYNTHESIS` is `TRUE` (the default), a query LLM is available, the LLM implements the `LLMProvider` protocol, and `agent_turns > 0`. Fast-path results and single-shot planner results skip coverage entirely — if retrieval was confident enough to skip the agent, coverage checking would add latency for no benefit.

The `assess_coverage()` function in `tools/coverage.py` makes a single LLM call. The prompt includes the question and a compact summary of each retrieved entry — entry ID, short title, tags, and the first 200 characters of `knowledge_details` (not the full content, to keep the prompt lean). The LLM is biased toward "no gaps" — it only flags a gap when an obvious, specific concept is missing, not when the answer could theoretically be more complete. The response is a JSON object with three fields: `has_gaps` (boolean), `suggested_query` (a short search query to fill the gap, or null), and `reason` (brief explanation).

If coverage finds gaps and provides a suggested query, the system runs a second retrieval via `_auto_search_entries()` — hybrid search plus graph expansion using the LLM's suggested query. The extra entries are merged into the original set by `_merge_entries()`, which deduplicates by entry ID while preserving the original ordering.

Coverage assessment is designed to never block synthesis. The outer wrapper catches all exceptions and returns `CoverageResult(has_gaps=False)` on any failure — LLM returning None, unparseable JSON, or unexpected errors. The JSON parsing follows the same defensive pattern as the graph enricher: strip markdown fences, find a JSON object via regex, validate fields, and fall back on parse failure.

### Synthesis

The `_synthesize()` function receives the question and the full structured entries — not compact summaries but complete `KnowledgeEntry` objects with their `knowledge_details`. Each entry is formatted as a block with its ID, short title, tags, optional context string, and full knowledge details. The system prompt instructs the LLM to answer only from the provided entries, cite entry IDs in `[kb-XXXXX]` format, note conflicting information with citations to both sources, and be concise. The LLM produces a natural-language answer grounded in the retrieved entries.

The distinction between coverage and synthesis prompts is intentional: coverage sees 200-character previews (enough to judge relevance without burning tokens), while synthesis sees everything (necessary for accurate, detailed answers).

### Fallback

If the query LLM is unavailable, synthesis is skipped and the tool returns raw entries formatted with `format_entry_full()`, prefixed with "(LLM unavailable — showing raw results)". If synthesis is attempted but the LLM returns None, the same fallback fires with "(LLM synthesis failed — showing raw results)". This three-layer design — primary synthesis, failed-synthesis fallback, no-LLM fallback — ensures the tool always returns something useful.

See: `tools/kb_summarize.py`, `tools/coverage.py`, `tools/kb_ask.py`

## The Entry Pipeline (kb_store)

When `kb_store` is called to create or update an entry, the full pipeline runs in sequence with each step isolated from failures in subsequent steps.

**Step 1: Store the entry.** For a new entry, `KnowledgeStore.create_entry` allocates the next `kb-XXXXX` ID, inserts the entry row, and creates the initial version record. For an update, it bumps the version number, creates a new version record, and resets `updated_at` and `has_embedding` (since the content changed and needs re-embedding). Both paths commit to the database, so the entry is durably stored before anything else runs.

**Step 2: Generate and store the embedding.** The embedding client sends the entry's `embedding_text` to Ollama, gets back a float vector, serializes it with `struct.pack`, and upserts it into the `knowledge_vec` table (delete then insert, since vec0 does not support `ON CONFLICT`). On success, the entry's `has_embedding` flag is set to true. If Ollama is unreachable or embedding fails, this step is skipped and the entry remains searchable via FTS only.

**Step 3: Build deterministic graph edges.** The graph builder clears all outgoing edges from the entry node, then re-derives them from tags, project ref, hints, and text references. This step runs regardless of whether embedding succeeded.

**Step 4: Enrich graph via LLM.** If an extraction LLM is configured, the enricher sends the entry to the LLM, parses out entities and relationships, and adds them as graph edges. This step runs regardless of whether the previous steps succeeded.

Each step is wrapped in its own try/except block. A failure in embedding does not prevent graph building, and a failure in graph enrichment does not affect the stored entry or its embedding. The entry is always returned to the caller with its current state.

See: `tools/kb_store.py`, `store/knowledge_store.py`

## Token Efficiency

MCP tool responses consume tokens in the calling LLM's context window, so the server uses a two-phase retrieval pattern to minimize waste. Search results from `kb_search` include only compact metadata — entry ID, type, titles, tags, project, and confidence — but omit the `knowledge_details` field, which is often the bulk of the content. When the caller needs full details for specific entries, it calls `kb_get` with one or more entry IDs (up to 20) to retrieve the complete content. This means a search over hundreds of entries sends back a manageable summary, and the caller only pays the token cost for entries it actually wants to read.

The formatting layer lives in `tools/formatters.py` and provides shared functions used across all tools. `format_entry_compact` produces a 2-3 line summary (header with ID, type, short title, confidence percentage; long title if different from short; tag and project metadata). `format_entry_full` adds the `knowledge_details` body and optionally a context line (like "via tag:python" in `kb_ask` results). `format_result_list` assembles a list of formatted entries with a count header, optional notes, and optional graph hint lines. `format_graph_hint` produces the one-liner hint format used by sparse graph hints.

`kb_get` also resets the confidence decay clock for retrieved entries. After formatting results, it calls `touch_accessed` to batch-update `last_accessed` on all successfully retrieved entries. This ties access-aware decay directly to the tool that indicates genuine user interest — reading the full content of an entry signals it is still useful.

### Batch Storage (kb_store_batch)

The `kb_store_batch` tool accepts up to 10 entries in a single call. Each entry goes through the standard pipeline — create, embed, build graph — individually. But graph enrichment is batched: instead of making one LLM call per entry, the enricher's `enrich_batch` method sends all entries in a single prompt and parses a JSON object keyed by entry ID from the response. This reduces LLM round-trips from N to 1. If the batch response fails to parse, the enricher falls back to per-entry enrichment so storage never fails due to a parsing issue.

The core logic is extracted into `batch_store_entries()`, a standalone async function that takes a list of entry dicts and the server lifespan context. This extraction keeps the business logic testable without requiring a FastMCP context.

See: `tools/formatters.py`, `tools/kb_get.py`, `tools/kb_store_batch.py`

## File Ingestion (kb_ingest)

The `kb_ingest` tool reads files from disk and converts them into knowledge entries through a multi-step pipeline orchestrated by `ingest/ingester.py:FileIngester`.

**Step 1: Deny-list check.** The first thing checked, before anything else, is whether the filename matches a deny pattern. The deny list in `ingest/safety.py` covers private keys (`.pem`, `.key`, `id_rsa`), environment files (`.env`), credentials files (`credentials.json`, `token.json`), binary formats, images, audio, video, and database files. This runs before the extension allowlist because it is a security boundary — even if a file has an allowed extension, it should be blocked if its name matches a sensitive pattern.

**Step 2: Extension allowlist.** The file's extension is checked against a set of supported text formats. The allowlist includes documentation formats (`.md`, `.txt`, `.rst`, `.org`), programming languages (`.py`, `.js`, `.ts`, `.go`, `.rs`, and many more), configuration formats (`.yaml`, `.toml`, `.json`, `.xml`), and shell scripts. Files with no extension are checked against a set of known names like `Dockerfile`, `Makefile`, `README`, and `LICENSE`.

**Step 3: File size limit.** The file must be under 5MB by default (configurable via `KB_INGEST_MAX_FILE_SIZE`). This prevents memory issues and excessive LLM token usage.

**Step 4: UTF-8 content read.** The file is read as UTF-8 with `errors="replace"` to handle non-UTF-8 bytes gracefully rather than crashing.

**Step 5: SHA-256 hash dedup.** The content's SHA-256 hash is compared against the `ingested_files` table. If a previous ingestion of the same file produced the same hash and the record is active, the file is skipped as unchanged. This makes re-running ingestion on a directory cheap — only modified files are reprocessed.

**Step 6: Safety pipeline.** The content passes through detect-secrets (using KeywordDetector, PrivateKeyDetector, and BasicAuthDetector — entropy detectors were dropped due to instability in detect-secrets 1.5) and scrubadub (which redacts PII like names, emails, and phone numbers). Both libraries are optional dependencies — if not installed, their checks are silently skipped. Files with detected secrets are flagged and not ingested. PII-redacted content continues through the pipeline with the redactions recorded.

**Step 7: LLM summarization.** The file's content (truncated at 100,000 characters) is sent to the query LLM with a system prompt requesting a 2-3 sentence summary. The prompt is supplemented based on file type: code files (`.py`, `.js`, etc.) get guidance to focus on high-level purpose rather than implementation details, while prose files (`.md`, `.txt`, etc.) get guidance to focus on key insights and conclusions. The summary becomes part of the note node's properties in the graph.

**Step 8: Content chunking.** The `chunk_content()` function in `ingest/chunker.py` splits large content into manageable pieces for extraction. If the content fits within the chunk size (default 16,000 characters, configurable via `KB_INGEST_CHUNK_SIZE`), it returns a single chunk. Otherwise, it splits at H1/H2 heading boundaries using the regex `^#{1,2}\s+`, greedily merges adjacent sections up to the chunk size limit, and falls back to paragraph breaks (`\n\n`) then newline breaks (`\n`) when a section still exceeds the limit. Each chunk is a `Chunk` dataclass carrying the text, a sequential index, the start character position in the original document, and the first H1/H2 heading found in the chunk (extracted and stored for context propagation to the extraction prompt). Adjacent chunks share an overlap region (default 600 characters, configurable via `KB_INGEST_CHUNK_OVERLAP`) snapped to a newline boundary to avoid splitting mid-sentence.

**Step 9: KB-aware dedup per chunk.** When agentic ingestion is enabled (`KB_AGENTIC_INGEST=TRUE`, the default), the `DedupAgent` in `ingest/dedup_agent.py` checks each chunk against the existing KB before extraction. The agent constructs a search query from the chunk's heading and first ~500 characters, runs `hybrid_search` with a limit of 5, and checks the top result's score against a threshold (default 0.06, configurable via `KB_INGEST_DEDUP_THRESHOLD`). If no results match or the top score falls below the threshold, the chunk proceeds directly to extraction without an LLM call. If results exceed the threshold, the agent sends the chunk text (first 2,000 characters) and the matching KB entry summaries to the LLM, which returns one of three verdicts: `"skip"` (knowledge is >80% covered — skip extraction entirely), `"partial"` (some overlap but new knowledge exists — extract with awareness of existing entries), or `"extract"` (mostly new knowledge — extract normally). For "partial" verdicts, the LLM also returns `existing_titles` — a list of KB entry titles that overlap — which are injected into the extraction context alongside previously extracted titles from earlier chunks. Graceful degradation is built in: any failure in search, LLM, or JSON parsing defaults to `"extract"`, so the system never loses data due to a dedup error.

**Step 10: LLM entry extraction.** Each chunk's text (truncated at 100,000 characters) is sent to the LLM with a system prompt asking for structured knowledge entries in JSON format. The LLM returns an array of objects, each with a short title, long title, knowledge details, entry type, and tags. The parser strips markdown fences, extracts the JSON array via regex, validates each object's fields and entry type, and caps at 10 entries per chunk. Two parameters prevent duplicate extraction across chunks: `previously_extracted` is a running list of short titles from entries already extracted from earlier chunks of the same file, and `chunk_heading` is the first H1/H2 heading from the current chunk. Both are included in the extraction prompt so the LLM skips concepts already covered. When the dedup agent returns a "partial" verdict, the `existing_titles` from the KB are appended to `previously_extracted`, creating a combined context that prevents re-extracting concepts covered by both earlier chunks and existing KB entries.

**Step 11: Entry storage.** Each extracted entry goes through the full `kb_store` pipeline: create the entry, generate and store the embedding, build deterministic graph edges, and enrich via LLM. Each entry runs independently, so a failure on one does not block the others. The entry's `source_context` is set to `"Ingested from {relative_path}"` for traceability.

**Step 12: Note node and edges.** A note node with ID `note:{relative_path}` is created in the graph, carrying the file path and summary in its properties. An `extracted_from` edge is added from each extracted entry to the note node, linking entries back to their source file.

**Step 13: Record in ingested_files.** The file's path, hash, note node ID, entry IDs, summary, size, extension, project reference, redactions, and timestamps are recorded in the `ingested_files` table.

**Re-ingestion** is handled automatically. If a file was previously ingested but its hash has changed, the old entries are deactivated (soft-deleted), old graph edges are removed, and the file goes through the full pipeline again. The `ingested_files` record is updated in place rather than recreated.

### URL Ingestion

The `kb_ingest` tool also accepts pre-fetched content with a source URL, enabling ingestion of web pages, wiki articles, and other text fetched by the calling agent. The tool accepts two additional parameters: `content` (the pre-fetched text) and `source_url` (the URL for attribution and dedup). When `content` is provided, `source_url` is required and `path` is ignored.

The content path calls `ingest_content()` on `FileIngester`, which runs a pipeline parallel to `ingest_file()` but skips all filesystem-specific steps: no deny-list check, no extension allowlist, and no file size check (the content is already in memory). Safety runs through `run_content_safety()` in `ingest/safety.py`, which performs secret detection and PII redaction without the deny-list check — a function extracted from `run_safety_pipeline()` so that file-mode ingestion still calls the full pipeline while content-mode calls only the content-relevant subset.

The rest of the pipeline is identical: SHA-256 hash for dedup (keyed on `source_url` in the `ingested_files` table's `relative_path` column), LLM summarization, chunking, dedup agent, extraction with running context, entry storage, and graph construction. The note node uses `note:{source_url}` as its ID, with `{"url": source_url, "summary": summary}` in properties (compared to `{"path": rel_path, "summary": summary}` for file-sourced notes). The file extension is derived from the URL path via `urllib.parse.urlparse` and `Path.suffix`, defaulting to an empty string when the URL has no path extension. File size is computed as `len(content.encode())` rather than a filesystem stat call.

See: `ingest/safety.py`, `ingest/extractor.py`, `ingest/chunker.py`, `ingest/dedup_agent.py`, `ingest/ingester.py`, `tools/kb_ingest.py`

## Dual LLM Architecture

The server uses two independent LLM slots: one for extraction (graph enrichment during storage) and one for queries (planning in `kb_ask` and synthesis in `kb_summarize`). Each slot can be independently configured to use Anthropic, Bedrock, or Ollama as its backend, controlled by the `KB_EXTRACTION_PROVIDER` and `KB_QUERY_PROVIDER` environment variables (values: `anthropic`, `bedrock`, or `ollama`). Both default to `anthropic`.

This separation exists because the two use cases have different performance characteristics. Extraction runs on every store operation and produces structured JSON — it benefits from a fast, inexpensive model. Query planning and synthesis are interactive and user-facing — they benefit from a more capable model. Running both on Anthropic (Claude Haiku) works well for most setups, but you could run extraction on a local Ollama model to reduce API costs while keeping Anthropic for queries, or use Bedrock for both in an AWS environment.

All three backends implement the `LLMProvider` protocol defined in `llm/provider.py`: `is_available()` checks if the backend is reachable, `generate()` sends a prompt with an optional system message and returns the response text or None, and `close()` releases resources. The protocol is decorated with `@runtime_checkable` so it can be used with `isinstance()` checks at runtime.

The **Anthropic client** (`llm/anthropic.py`) uses lazy SDK import — the `anthropic` package is only imported when `_get_client()` is first called. If the package is not installed, the client returns None from `_get_client()` and all `generate()` calls return None. This means the server can run without the anthropic package installed if only Ollama is used. The client also uses success-only caching: `_available` is set to True after a successful `generate()` call, but reset to None (not False) on failure, allowing retries. Availability checking is lightweight — it just verifies the SDK is importable and `ANTHROPIC_API_KEY` is set, deferring the actual API call to the first `generate()`.

The **Bedrock client** (`llm/bedrock.py`) uses the `aws-sdk-bedrock-runtime` package, a natively async SDK generated from the Smithy service model. It supports two authentication modes. The primary path uses bearer token auth via the `AWS_BEARER_TOKEN_BEDROCK` environment variable — this required monkey-patching the SDK because the Bedrock service model declares `httpBearerAuth` support but the codegen doesn't wire it up. The `_configure_bearer_auth` function injects a custom `BearerAuthScheme` (built on the existing smithy_http `APIKeyAuthScheme` plumbing) into the SDK's Config object and patches the auth scheme resolver to prefer bearer auth when a token is present. The fallback path uses traditional SigV4 signing via `AWS_ACCESS_KEY_ID` with the `EnvironmentCredentialsResolver` from smithy_aws_core. If neither credential is set, the client disables itself. Like the other clients, it uses lazy SDK import and success-only caching. The SDK is an optional dependency (`[project.optional-dependencies] aws`).

The **Ollama client** (`llm/ollama.py`) talks to Ollama's `/api/generate` endpoint over httpx. It uses the same success-only caching pattern as the embedding client: ping `/api/tags` to check availability, cache success, retry on failure. The model and timeout are configured separately from the embedding model — `KB_OLLAMA_MODEL` (default `qwen3:4b`) and `KB_OLLAMA_LLM_TIMEOUT` (default 120 seconds).

The factory function `_create_llm()` in `server.py` selects the right client class based on the provider string. During server lifespan setup, two LLM clients are created (one per slot), and the extraction client is wrapped in a `GraphEnricher` instance while the query client is passed directly to the lifespan context for use by `kb_ask` and `kb_summarize`.

See: `llm/provider.py`, `llm/anthropic.py`, `llm/bedrock.py`, `llm/ollama.py`, `server.py`

## Graceful Degradation

The system is designed to always store entries and serve full-text search at minimum, even when every optional dependency is unavailable.

When **Ollama is unreachable**, the embedding client's `is_available()` returns False, `embed()` returns None, and `store_embedding()` is never called. The `has_embedding` flag stays False on the entry. In hybrid search, `vector_search()` returns an empty list, and RRF operates on FTS results alone. The `match_source` in results will be `"fts"` instead of `"hybrid"`. If Ollama comes back later, the success-only caching means the next embedding attempt will retry the ping and start working. The `kb_maintain` tool can backfill missing embeddings in bulk.

When **the extraction LLM is unavailable** (no configured provider reachable for the extraction slot), the `graph_enricher` is set to None in the lifespan context. The `_enrich_graph` helper in `kb_store.py` checks for None and returns immediately. Entries still get deterministic graph edges from the builder — tags, project references, text references, and hint-based edges all work without an LLM. The graph just lacks the entity-level edges (concept, technology, tool, person relationships) that enrichment would add.

When **the query LLM is unavailable**, `kb_ask`'s auto strategy skips the agent loop and runs hybrid search directly with the user's raw question. The check in `_strategy_auto_with_planner` tests `query_llm is not None` before entering the agentic path. Without a query LLM, `kb_summarize` falls back to showing raw search results prefixed with "(LLM unavailable — showing raw results)" — still useful, just not synthesized into prose.

When **the anthropic package is not installed**, the `AnthropicLLMClient._get_client()` method catches `ImportError` and returns None. All subsequent `generate()` calls return None. The server starts normally and the provider slot acts as if the LLM is permanently unavailable, falling through to the same degradation paths described above.

When **the aws-sdk-bedrock-runtime package is not installed**, the `BedrockLLMClient._get_client()` method catches `ImportError` and returns None, following the same pattern as the Anthropic client. When the package is installed but neither `AWS_BEARER_TOKEN_BEDROCK` nor `AWS_ACCESS_KEY_ID` is set, `is_available()` returns False and logs a warning.

When **detect-secrets is not installed**, `detect_secrets_in_content()` catches `ImportError` and returns None (as opposed to an empty list, which would mean "scanned and found nothing"). The ingestion pipeline treats None as "scan not performed" and continues without flagging.

When **scrubadub is not installed**, `redact_pii()` catches `ImportError` and returns the original content unchanged with an empty redactions list. Content passes through without PII redaction.

The overall design principle is that each component checks its own dependencies at call time, returns a neutral result (None, empty list, or unchanged input) when those dependencies are missing, and the calling code handles neutral results by skipping the dependent step. No component throws an exception for a missing optional dependency, and no step's failure prevents subsequent steps from running.

## Graph Explorer Visualization

The graph explorer renders the entire knowledge graph as an interactive force-directed visualization in the browser, powered by [force-graph](https://github.com/vasturiano/force-graph) (a d3-force wrapper for Canvas2D). It runs as a FastAPI web server with LLM-powered query capabilities, SSE streaming, and multi-turn chat. A static `file://` fallback exists if the port is already in use.

### Graph Data Extraction

`explorer/graph_data.py:extract_graph_data()` runs three SQL queries to build the visualization payload: one for all rows in `graph_nodes` (id, type, label, properties), one for all rows in `graph_edges` (source, target, edge_type, properties), and one for entry metadata from `knowledge_entries` (id, short_title, entry_type, project_ref, tags, created_at, confidence). The function returns a dict with `nodes`, `edges`, and `stats` (total counts for entries, nodes, edges, and a breakdown of entries by type). Node types are preserved from the graph — `entry`, `tag`, `project`, `person`, `tool`, `concept`, `technology`, and `note`.

### Renderer and Template

`explorer/renderer.py:render_explorer_html()` takes the graph data dict and produces a single self-contained HTML string. The graph data is injected as a JSON literal inside a `<script>` tag — no external API calls needed for the base visualization. The template uses force-graph loaded from CDN.

Each node type has a fixed color: entries are gray (`#e0e0e0`), tags are cyan (`#00bcd4`), projects are orange (`#ff9800`), people are amber (`#ffc107`), tools are green (`#4caf50`), concepts are purple (`#9c27b0`), technologies are blue (`#2196f3`), and notes are blue-gray (`#78909c`). Nodes are drawn on canvas with `nodeCanvasObject` — entry nodes display their short title as a label, while entity nodes show their human-readable label. Node radius scales with connection count. Edge colors are derived from the source node's type color at reduced opacity.

The search bar (top-left) filters nodes by label with autocomplete. Selecting a node from the dropdown flies the camera to it and highlights it with a colored ring. A legend (bottom-left) shows node type colors with counts.

### Web Server (Query-Driven Mode)

The web infrastructure lives in `web/`.

**App factory** (`web/app.py`): Two factory functions create the FastAPI application. `create_app_with_deps(db, embedder, query_llm)` is used when the explorer is launched from the `kb_explore` MCP tool — it receives the database, embedder, and LLM client directly from the MCP lifespan context, sharing connections instead of creating new ones. `create_app()` is used by the standalone `personal-kb-web` CLI entry point — it creates its own database connection, embedder, and LLM client in a lifespan handler, mirroring the MCP server's startup sequence. Both store dependencies on `app.state`.

**Routes** (`web/routes.py`): Five endpoints. `GET /` serves the explorer HTML (calls `extract_graph_data()` → `render_explorer_html()`). `GET /api/graph` returns the raw graph data as JSON (for live refresh). `POST /api/query/stream` is the SSE endpoint that streams query events. `GET /api/entry/{entry_id}` returns full entry details (including `knowledge_details`) for on-demand loading in the info panel. `POST /api/chat/stream` handles multi-turn follow-up conversations via SSE (see Chat section below).

**SSE streaming**: The `/api/query/stream` endpoint receives a JSON body with a `question` field. It first classifies the query (explore vs. summarize), then runs the appropriate retrieval function as an `asyncio.create_task`. An `asyncio.Queue` bridges the async event callback and the SSE generator — the `event_callback` pushes events to the queue, and the generator drains the queue and yields formatted SSE lines. Each agent event is translated to a human-readable status message via `event_to_status()` and sent as a `status` SSE event. When the task completes, the generator yields the final results (entry list or synthesized answer) and a `stream_end` sentinel.

**Query classification** (`web/classifier.py`): A single Haiku LLM call routes each query to either `explore` (browse and discover — "what connects to Python?", "show me debugging entries") or `summarize` (direct answer needed — "why did we choose FastAPI?", "explain the pipeline"). The classifier defaults to `explore` on any failure — LLM unavailable, parse error, or garbage response. The `summarize` keyword is extracted even from verbose LLM responses ("The classification is: summarize") via substring matching.

**Event callback pipeline**: The `event_callback` parameter threads through `retrieve_entries()` in `kb_ask.py` and `summarize_question()` in `kb_summarize.py`, down to `agentic_query()` in `graph/agent.py`. The agent emits 8 event types: `fast_path` (strong matches found, skipping agent), `agent_started` (entering ReAct loop), `thinking` (before each LLM call), `tool_call` (before dispatching a tool), `tool_result` (after a tool returns), `agent_done` (final answer), `parse_error` (LLM response couldn't be parsed), and on exhaustion (max tool calls reached). `summarize_question()` adds two more: `synthesis_started` (before calling the synthesis LLM) and `synthesis_done` (after). All existing callers pass `None` for the callback — zero behavior change to MCP tools.

### Frontend Query Features

The frontend detects whether it's served by the web server (`location.protocol !== 'file:'`). When served, the search bar placeholder changes to "Search nodes or ask a question..." and pressing Enter with free-form text (containing spaces or no autocomplete matches) triggers a query instead of a node search.

**Traversal animation**: As the agent explores the graph, nodes transition through visual states via a staggered animation queue (250ms delay between nodes). Visited nodes (touched by tool calls) glow orange (`#ffaa00`) with a 20px shadow blur. Final result nodes glow green (`#00ff88`) with a 30px shadow blur and +3px radius. Labels are always shown for visited and result nodes regardless of zoom level. Traversal particles emit along edges from visited nodes. The camera progressively widens via `zoomToFit` on the accumulated set of visited nodes (rather than jumping node-to-node). Deactivated entries are filtered out of the visualization entirely — `extract_graph_data()` collects inactive entry IDs and excludes their nodes and any edges touching them.

**Info panel**: Clicking an entry node opens a panel (right side) showing metadata with bold white labels (Type, Tags, Project, Confidence, By). Confidence is displayed as a percentage (e.g., "95%"). An expandable "Full Entry..." accordion fetches the full `knowledge_details` on demand from `/api/entry/{id}` and renders it as markdown. The accordion caches fetched content to avoid re-fetching.

**Chat panel**: For `summarize` queries, the response opens a multi-turn chat panel instead of a static response panel. The chat panel appears at the search bar's position — the search bar fades out (`opacity 0.25s`, `translateY(4px)`), then the chat panel slides in with a `scaleY(0.05 → 1)` animation over 0.3s (transform-origin: top left). The panel contains the original question (right-aligned user bubble) and synthesized answer (left-aligned assistant bubble), with a text input and Send button at the bottom. Pressing Enter or clicking Send posts to `/api/chat/stream`, shows a typing indicator with animated dots, and streams the response into a new assistant bubble. The close button is a prominent 28px grey circle with a bold white X. Closing the chat reverses the animation (scaleY collapse → search bar fades back in after 300ms).

**SSE event handling**: The frontend reads the SSE stream with `fetch()` + `ReadableStream`, parsing `event:` and `data:` lines. JSON parse errors and event handler errors are caught in separate try/catch blocks to prevent one from swallowing the other. Status messages appear in a fixed status line below the search bar, updating in real time as the agent works ("Searching knowledge base...", "Exploring neighbors of tag:python...", "Synthesizing answer from 5 entries...").

### kb_explore Tool Integration

The `kb_explore` MCP tool (`tools/kb_explore.py`) tries the web server first. It calls `create_app_with_deps()` with the MCP lifespan's database, embedder, and LLM client, starts uvicorn as an `asyncio.create_task`, and opens the browser. A module-level `_web_server_task` variable prevents starting duplicate servers across multiple `kb_explore` calls in the same session. If the port is already in use (`OSError`), it falls back to the static `file://` mode — writing a temp HTML file and opening it with `webbrowser.open()`.

The standalone `personal-kb-web` CLI (`web/cli.py`) runs the same web app via `uvicorn.run()` on port 8765, opening the browser after a 1-second delay. This is useful for browsing the KB outside of an MCP session.

### Multi-Turn Chat

When a `summarize` query completes, the frontend opens a chat panel seeded with the original question and synthesized answer. Follow-up messages are sent to `/api/chat/stream`, which maintains conversation state server-side.

**ChatSession** (`web/chat.py`): Holds the conversation as a `list[Message]` (where `Message = dict[str, str]` with `role` and `content` keys). The `seed()` method initializes the conversation with the original Q+A pair and the entry IDs from the summarize result. On each `reply()`, the session: (1) appends the user message, (2) trims history if over budget, (3) runs `hybrid_search` with `limit=5` to find entries relevant to the follow-up, (4) builds a system prompt that includes `_CHAT_SYSTEM_PROMPT` plus the full `knowledge_details` of all known entries, (5) calls `llm.generate_chat(messages, system=system)`, and (6) appends the assistant response. New entry IDs discovered during retrieval are accumulated in `self.entry_ids` and reported back via the `chat_done` event so the frontend can highlight them on the graph.

**Token budget**: `_MAX_CONVERSATION_CHARS = 100_000` (~25K tokens). When the total character count exceeds this, `_trim_history()` preserves the first 2 messages (the seed Q+A that grounds the conversation) and the most recent messages, dropping middle turns until the budget is met. No summarization pass — just a sliding window.

**Session store**: Sessions are stored in-memory in a module-level dict keyed by UUID. `get_or_create_session()` creates a new session if no valid `session_id` is provided. Sessions are ephemeral — they don't survive server restarts.

**SSE protocol**: The `/api/chat/stream` endpoint accepts `{message, session_id?, seed_question?, seed_answer?, seed_entry_ids?}`. It emits: `chat_session` (with the session ID), `chat_thinking` (before LLM call), `chat_done` (with newly discovered entry IDs), `chat_response` (with the answer text and session ID), and `stream_end`. Errors yield an `error` event with exception details.

**LLMProvider.generate_chat()**: Added to the `LLMProvider` protocol alongside the existing `generate()` method. Takes `messages: list[Message]` and optional `system` prompt, returns `str | None`. Each backend implements it natively: Anthropic passes messages directly to `messages.create()`, Bedrock maps to `BRMessage` objects for the Converse API, and Ollama uses `/api/chat` (not `/api/generate`). The ReAct agent loop in `graph/agent.py` also uses `generate_chat()` — the old `_build_prompt()` function that flattened messages into a single string was deleted.

See: `explorer/graph_data.py`, `explorer/renderer.py`, `web/app.py`, `web/routes.py`, `web/chat.py`, `web/classifier.py`, `web/events.py`, `web/cli.py`, `tools/kb_explore.py`

## References

[1] G. V. Cormack, C. L. A. Clarke, and S. Büttcher. "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods." *SIGIR 2009*. https://dl.acm.org/doi/10.1145/1571941.1572114

[2] J. Sun et al. "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph." *arXiv:2307.07697*, 2023. https://arxiv.org/abs/2307.07697

[3] D. Edge et al. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." Microsoft Research, 2024. https://arxiv.org/abs/2404.16130

[4] W. Zhong et al. "MemoryBank: Enhancing Large Language Models with Long-Term Memory." *AAAI 2024*. https://arxiv.org/abs/2305.10250

[5] J. S. Park et al. "Generative Agents: Interactive Simulacra of Human Behavior." *UIST 2023*. https://arxiv.org/abs/2304.03442

[6] Z. Guo et al. "LightRAG: Simple and Fast Retrieval-Augmented Generation." *arXiv:2410.05779*, 2024. https://arxiv.org/abs/2410.05779

The research survey that informed the entity deduplication, access-aware decay, and sparse graph hints features is documented in the KB itself (entries kb-00111 and kb-00112), produced by a multi-agent paper review of ~12 GraphRAG papers from the HuggingFace graphrag-papers collection.
