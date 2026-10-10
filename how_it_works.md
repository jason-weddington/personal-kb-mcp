# How It Works

Personal KB is a graph-RAG knowledge base for AI assistants, split across three
distributions: the reusable **`kb-core`** engine (the graph-RAG brain — hybrid
search, the knowledge graph, agentic retrieval, pluggable storage and LLM
backends), the **`personal-kb-web-service`** hosted backend (a FastAPI + React
SPA that wraps the engine behind an authed HTTP API and serves the graph
explorer, chat, listener gate, and maps index), and the **`personal-kb`** MCP
server in this repo, which is now a **thin client** of that service — it
forwards each MCP tool call to the appropriate `/api/kb/*` route. A second thin
client — **`personal-kb-hook`** under `packages/personal-kb-hook/` — runs as a
Claude Code `SessionStart` / `UserPromptSubmit` / `Stop` hook and surfaces
mental-map orientation pointers into the model's context.

This document covers what lives in **this repo** today: the MCP tool entry
points under `src/personal_kb/tools/`, the project-preflight + mental-map
orientation pull (`src/personal_kb/preflight.py`, `src/personal_kb/tools/kb_preflight.py`,
`src/personal_kb/tools/map_lint.py`), the standalone hook package, and the
anticipatory-listener whisper loop that the hook spawns. Engine internals —
storage, FTS5, vector embeddings, hybrid search and Reciprocal Rank Fusion,
the knowledge graph (deterministic edges + LLM enrichment), confidence decay,
the store / ingest pipelines, the dual-LLM architecture, and graceful
degradation — are documented in the kb-core internals doc:
[`packages/kb-core/docs/how_it_works.md`](packages/kb-core/docs/how_it_works.md).

**The shim arrangement.** As part of the kb-core extraction wave, every engine
module that used to live under `src/personal_kb/<module>` is now a thin
**re-export shim** that imports from `kb_core`. For example,
`src/personal_kb/db/backend.py` is a 9-line shim doing
`from kb_core.db.backend import Cursor, Database, Row`, and
`src/personal_kb/tools/coverage.py` is a shim doing
`from kb_core.coverage import ...`. The path mappings are:

- `db/*`, `graph/*`, `confidence/*`, `ingest/*`, `llm/*`, `search/*`, `store/*`,
  `models/*` — kb_core path mirrors the old path 1:1 (e.g.
  `packages/kb-core/src/kb_core/graph/agent.py`,
  `packages/kb-core/src/kb_core/db/queries.py`,
  `packages/kb-core/src/kb_core/confidence/decay.py`).
- Lifted top-level helpers, **with renames**: `tools/coverage.py` →
  `packages/kb-core/src/kb_core/coverage.py`, and the formatters were lifted
  AND renamed to `packages/kb-core/src/kb_core/formatting.py`. `tools/ttl.py`
  and `preflight.py` were also lifted (`kb_core/ttl.py`, `kb_core/preflight.py`).
- The MCP tool entry points stay in this repo: `src/personal_kb/tools/kb_ask.py`,
  `kb_summarize.py`, `kb_preflight.py`, `kb_search.py`, `kb_get.py`,
  `kb_store.py`, `kb_store_batch.py`, `kb_explore.py`, `kb_ingest.py`,
  `kb_ingest_url.py`, `kb_maintain.py`, `kb_list.py`, `kb_bulk_update.py`,
  `kb_feedback.py`, plus `map_lint.py` and the `_lifespan.py` wiring. These
  have no `kb_core` counterpart — they exist precisely to be the MCP-facing
  surface.
- The standalone hook package `packages/personal-kb-hook/` is untouched by the
  extraction (it has its own zero-dep distribution and is described below).

The sections that follow document those server-side / hook-side concerns. For
how the engine itself works underneath, follow the kb-core link above.

## Query Strategies (kb_ask)

The `kb_ask` tool supports five query strategies, each suited to different kinds of questions.

**auto** is the default strategy. It runs hybrid search (FTS + vector) to find matching entries, then expands results by walking one hop through the graph from each search hit. For each hit, it calls `get_neighbors` with a limit of 10 and adds any neighboring entry nodes that aren't already in the result set. This provides context around search results — if you find a decision entry, you might also see the entries it supersedes or the tools it references.

**decision_trace** searches for decision-type entries using FTS, then walks the `supersedes` chain in both directions for each hit. The chain walk in `kb_core/graph/queries.py:supersedes_chain` follows `supersedes` edges backward (what this entry supersedes) and forward (what supersedes this entry), building a chronologically ordered list from oldest to newest. The formatted output labels each entry as "original decision", "supersedes kb-XXXXX", or "current" to show the decision's evolution.

**timeline** takes a scope (like `project:personal-kb` or `tag:sqlite`) and returns all matching entries sorted by `created_at`. It uses `kb_core/graph/queries.py:entries_for_scope`, which interprets scope strings as project filters, tag lookups, person/tool graph traversals, or entry type filters depending on the prefix. This strategy is useful for understanding the history of a topic or project.

**related** performs a breadth-first search from a starting node with a maximum depth of 2. The BFS in `kb_core/graph/queries.py:bfs_entries` uses a standard queue, tracking visited nodes to avoid cycles and collecting entry nodes (those matching `kb-XXXXX` format) encountered along the way. It returns each entry's depth and full path from the start node. This strategy answers questions like "what else relates to aiosqlite?"

**connection** finds the shortest path between two nodes using BFS with a maximum depth of 4. The `kb_core/graph/queries.py:find_path` function returns a list of `(source, edge_type, target)` triples forming the path, or None if no path exists. This strategy answers questions like "how are these two concepts connected?"

### Agentic Query Planning

When a query LLM is available, the `auto` strategy delegates to a ReAct agent loop in `kb_core/graph/agent.py` that can plan, execute, evaluate, and retry — replacing the single-shot query planner that had to pick the right strategy blindly.

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

**The agent loop.** Each turn, the agent receives the full conversation history as a structured message list via `llm.generate_chat(messages, system=...)`, along with a system prompt describing the available tools, evaluation rules, and the hard cap. The agent responds with a single JSON object — either a tool call or a final answer. Tool calls are dispatched to the corresponding async function, and the formatted result is appended to the conversation as a user message. Repeated identical tool calls are detected and rejected with an error message, preventing infinite loops when the agent gets stuck. Final answers trigger entry ID lookups and return an `AgentResult` with the entries, turn count, and reasoning.

**Hard cap.** The loop is bounded at 4 tool calls (configurable via `KB_AGENTIC_MAX_CALLS`), enforced by the orchestrator. The system prompt also restates the cap ("NEVER make more than 4 tool calls") as belt-and-suspenders. On exhaustion without a `done` call, the agent extracts all `kb-XXXXX` IDs mentioned in the conversation as a best-effort fallback, then falls back to fast-path results if nothing was found.

**Error handling.** If the LLM returns None (provider failure), the loop breaks immediately and falls back to fast-path results. If the LLM returns unparseable JSON, an error message is injected into the conversation and the loop continues, consuming one tool call from the budget. This gives the LLM a chance to self-correct.

**Compact output.** The agent only sees `format_entry_compact()` output (from `kb_core.formatting`) — titles, types, scores, and metadata, but never `knowledge_details`. This keeps the context window lean across multiple turns. Full entry details are fetched only when formatting the final result for the caller.

**Toggle.** Set `KB_AGENTIC_QUERY=FALSE` to bypass the agent and fall back to the single-shot `QueryPlanner` from `kb_core/graph/planner.py`, which translates natural language into a structured `QueryPlan` (strategy, scope, target, search query) via a single LLM call. The planner receives graph statistics and vocabulary as context for entity resolution.

**Eval results.** On the eval corpus (32 entries, 13 scorable golden queries), the agent achieves perfect MRR, recall@5, and NDCG@5 (all 1.000), up from 0.853/1.000/0.889 with hybrid search alone. The three queries where hybrid search ranked the right entry 2nd-4th (q05 REST auth, q06 CORS, q10 encoding bug) are all resolved in a single agent turn.

See: `kb_core/graph/agent.py`, `src/personal_kb/tools/kb_ask.py`, `kb_core/graph/planner.py`

## Answer Synthesis (kb_summarize)

The `kb_summarize` tool provides a higher-level interface than `kb_ask` by adding LLM synthesis on top of retrieval. The core logic lives in `summarize_question()`, an extracted function that takes the database, embedder, query LLM, optional synthesis LLM, question, optional scope, and limit — keeping it testable without FastMCP context. When a synthesis LLM is available (typically Sonnet via `model_override`), it is preferred for generating the final prose answer; otherwise the query LLM is used. The dual-LLM split itself is documented in the kb-core internals doc.

### Retrieval

The first step calls `retrieve_entries()` from `src/personal_kb/tools/kb_ask.py`, which returns a tuple of `(entries_with_context, agent_turns_used)`. Each entry is paired with a context string describing how it was found (e.g., `"search match (score: 0.0312)"` or `"linked from kb-00042 via supersedes"`). The `agent_turns_used` count tracks how many LLM tool calls the agentic query loop made — 0 means the fast-path resolved the query without touching the LLM.

### Coverage Assessment

When the retrieval involved the agent (not fast-path), the system runs a coverage check before synthesis. The check fires only when all four conditions are met: `KB_AGENTIC_SYNTHESIS` is `TRUE` (the default), a query LLM is available, the LLM implements the `LLMProvider` protocol, and `agent_turns > 0`. Fast-path results and single-shot planner results skip coverage entirely — if retrieval was confident enough to skip the agent, coverage checking would add latency for no benefit.

The `assess_coverage()` function in `kb_core/coverage.py` makes a single LLM call. The prompt includes the question and a compact summary of each retrieved entry — entry ID, short title, tags, and the first 200 characters of `knowledge_details` (not the full content, to keep the prompt lean). The LLM is biased toward "no gaps" — it only flags a gap when an obvious, specific concept is missing, not when the answer could theoretically be more complete. The response is a JSON object with three fields: `has_gaps` (boolean), `suggested_query` (a short search query to fill the gap, or null), and `reason` (brief explanation).

If coverage finds gaps and provides a suggested query, the system runs a second retrieval via `_auto_search_entries()` — hybrid search plus graph expansion using the LLM's suggested query. The extra entries are merged into the original set by `_merge_entries()`, which deduplicates by entry ID while preserving the original ordering.

Coverage assessment is designed to never block synthesis. The outer wrapper catches all exceptions and returns `CoverageResult(has_gaps=False)` on any failure — LLM returning None, unparseable JSON, or unexpected errors. The JSON parsing follows the same defensive pattern as the graph enricher: strip markdown fences, find a JSON object via regex, validate fields, and fall back on parse failure.

### Synthesis

The `_synthesize()` function receives the question and the full structured entries — not compact summaries but complete `KnowledgeEntry` objects with their `knowledge_details`. Each entry is formatted as a block with its ID, short title, tags, optional context string, and full knowledge details. The system prompt instructs the LLM to answer only from the provided entries, cite entry IDs in `[kb-XXXXX]` format, note conflicting information with citations to both sources, and be concise. The LLM produces a natural-language answer grounded in the retrieved entries.

The distinction between coverage and synthesis prompts is intentional: coverage sees 200-character previews (enough to judge relevance without burning tokens), while synthesis sees everything (necessary for accurate, detailed answers).

### Fallback

If the query LLM is unavailable, synthesis is skipped and the tool returns raw entries formatted with `format_entry_full()` (from `kb_core.formatting`), prefixed with "(LLM unavailable — showing raw results)". If synthesis is attempted but the LLM returns None, the same fallback fires with "(LLM synthesis failed — showing raw results)". This three-layer design — primary synthesis, failed-synthesis fallback, no-LLM fallback — ensures the tool always returns something useful.

See: `src/personal_kb/tools/kb_summarize.py`, `kb_core/coverage.py`, `src/personal_kb/tools/kb_ask.py`

## Graph Explorer Visualization

The graph explorer renders the entire knowledge graph as an interactive force-directed visualization in the browser, powered by [force-graph](https://github.com/vasturiano/force-graph) (a d3-force wrapper for Canvas2D). It now lives in the **personal-kb-web-service** repo — the React + MUI SPA under `frontend/` consumes JSON from FastAPI routes under `src/kb_service/`, the `personal-kb` MCP server merely opens the hosted URL. A static `file://` fallback is no longer needed; the explorer is always served by the hosted backend.

### Graph Data Extraction

`kb_service.graph_export.extract_graph_data()` (in the web service repo) runs SQL queries against the kb-core data DB to build the visualization payload: rows from `graph_nodes` (id, type, label, properties), rows from `graph_edges` (source, target, edge_type, properties), and entry metadata from `knowledge_entries` (id, short_title, entry_type, project_ref, tags, created_at, confidence). The function returns a dict with `nodes`, `edges`, and `stats` (total counts for entries, nodes, edges, and a breakdown of entries by type). Node types are preserved from the graph — `entry`, `tag`, `project`, `person`, `tool`, `concept`, `technology`, and `note`. Inactive entry nodes are excluded; non-entry nodes with no edges (orphans) are excluded; edges touching excluded nodes are excluded.

### Renderer and Template

The renderer is now a React component tree in `personal-kb-web-service/frontend/src/`. The graph data is fetched from the backend at page load and rendered into a `<canvas>` element bound to force-graph. The template uses force-graph loaded from CDN.

Each node type has a fixed color: entries are gray (`#e0e0e0`), tags are cyan (`#00bcd4`), projects are orange (`#ff9800`), people are amber (`#ffc107`), tools are green (`#4caf50`), concepts are purple (`#9c27b0`), technologies are blue (`#2196f3`), and notes are blue-gray (`#78909c`). Nodes are drawn on canvas with `nodeCanvasObject` — entry nodes display their short title as a label, while entity nodes show their human-readable label. Node radius scales with connection count. Edge colors are derived from the source node's type color at reduced opacity.

The search bar (top-left) filters nodes by label with autocomplete. Selecting a node from the dropdown flies the camera to it and highlights it with a colored ring. A legend (bottom-left) shows node type colors with counts.

### Web Server (Query-Driven Mode)

The web infrastructure lives in `packages/kb-service/src/kb_service/`.

**App factory** (`kb_service/main.py`): The FastAPI application is created in a lifespan handler that opens a single kb-core `KnowledgeBase` from `KB_DATABASE_URL` (the data DB), plus an asyncpg pool for the service/auth DB at `KB_SERVICE_DATABASE_URL`. The two DBs are kept distinct: kb-core owns the data schema, the service owns the auth tables (`users`, `api_keys`, `invites`, `password_resets`). The `KnowledgeBase` is stored on `app.state.kb` and shared by every request.

**Routes** (`kb_service/routes/`): The service mounts route modules for auth (`auth_routes.py`), admin (`admin_routes.py`), KB reads/writes (`kb_read_routes.py`, `kb_write_routes.py`, `kb_routes.py`), maps index (`maps_routes.py`), ingest (`ingest_routes.py`), query (`query_routes.py`), chat (`chat_routes.py`), the anticipatory listener (`listener_routes.py`, described below), and settings (`settings_routes.py`). The standalone `personal-kb-web` CLI from the old design is gone — the hosted service replaces it.

**SSE streaming**: The query stream endpoint receives a JSON body with a `question` field. It first classifies the query (explore vs. summarize) via `kb_service/classifier.py`, then runs the appropriate retrieval function as an `asyncio.create_task`. An `asyncio.Queue` bridges the async event callback (in `kb_service/sse.py`) and the SSE generator — the `event_callback` pushes events to the queue, and the generator drains the queue and yields formatted SSE lines. Each agent event is translated to a human-readable status message and sent as a `status` SSE event. When the task completes, the generator yields the final results (entry list or synthesized answer) and a `stream_end` sentinel.

**Query classification** (`kb_service/classifier.py`): A single Haiku LLM call routes each query to either `explore` (browse and discover — "what connects to Python?", "show me debugging entries") or `summarize` (direct answer needed — "why did we choose FastAPI?", "explain the pipeline"). The classifier defaults to `explore` on any failure — LLM unavailable, parse error, or garbage response. The `summarize` keyword is extracted even from verbose LLM responses ("The classification is: summarize") via substring matching.

**Event callback pipeline**: The `event_callback` parameter threads through `retrieve_entries()` in `src/personal_kb/tools/kb_ask.py` and `summarize_question()` in `src/personal_kb/tools/kb_summarize.py`, down to `agentic_query()` in `kb_core/graph/agent.py`. The agent emits 8 event types: `fast_path` (strong matches found, skipping agent), `agent_started` (entering ReAct loop), `thinking` (before each LLM call), `tool_call` (before dispatching a tool), `tool_result` (after a tool returns), `agent_done` (final answer), `parse_error` (LLM response couldn't be parsed), and on exhaustion (max tool calls reached). `summarize_question()` adds two more: `synthesis_started` (before calling the synthesis LLM) and `synthesis_done` (after). Callers that don't need streaming pass `None` for the callback.

### Frontend Query Features

The SPA detects whether the search bar input is a free-form question and triggers a query instead of a node search. Pressing Enter on free-form text (containing spaces or no autocomplete matches) opens the query stream.

**Traversal animation**: As the agent explores the graph, nodes transition through visual states via a staggered animation queue (250ms delay between nodes). Visited nodes (touched by tool calls) glow orange (`#ffaa00`) with a 20px shadow blur. Final result nodes glow green (`#00ff88`) with a 30px shadow blur and +3px radius. Labels are always shown for visited and result nodes regardless of zoom level. Traversal particles emit along edges from visited nodes. The camera progressively widens via `zoomToFit` on the accumulated set of visited nodes (rather than jumping node-to-node). Deactivated entries are filtered out of the visualization entirely — `extract_graph_data()` collects inactive entry IDs and excludes their nodes and any edges touching them.

**Info panel**: Clicking an entry node opens a panel (right side) showing metadata with bold white labels (Type, Tags, Project, Confidence, By). Confidence is displayed as a percentage (e.g., "95%"). An expandable "Full Entry..." accordion fetches the full `knowledge_details` on demand from the entry endpoint and renders it as markdown. The accordion caches fetched content to avoid re-fetching.

**Chat panel**: For `summarize` queries, the response opens a multi-turn chat panel instead of a static response panel. The chat panel appears at the search bar's position — the search bar fades out (`opacity 0.25s`, `translateY(4px)`), then the chat panel slides in with a `scaleY(0.05 → 1)` animation over 0.3s (transform-origin: top left). The panel contains the original question (right-aligned user bubble) and synthesized answer (left-aligned assistant bubble), with a text input and Send button at the bottom. Pressing Enter or clicking Send posts to the chat stream endpoint, shows a typing indicator with animated dots, and streams the response into a new assistant bubble. Closing the chat reverses the animation (scaleY collapse → search bar fades back in after 300ms).

**SSE event handling**: The frontend reads the SSE stream with `fetch()` + `ReadableStream`, parsing `event:` and `data:` lines. JSON parse errors and event handler errors are caught in separate try/catch blocks to prevent one from swallowing the other. Status messages appear in a fixed status line below the search bar, updating in real time as the agent works ("Searching knowledge base...", "Exploring neighbors of tag:python...", "Synthesizing answer from 5 entries...").

### Auto-Start and kb_explore Tool Integration

In the thin-client split, the MCP server no longer hosts the explorer locally — the hosted web service serves it permanently. The `kb_explore` MCP tool (`src/personal_kb/tools/kb_explore.py`) is now a tiny wrapper that returns the hosted explorer URL for the caller to open in a browser. The legacy `KB_AUTO_EXPLORE`, `KB_EXPLORE_PORT`, uvicorn-as-asyncio-task, port-stealing, and static `file://` fallback machinery is gone; the equivalent runs once in the web service's own lifespan, not per MCP session.

### Multi-Turn Chat

When a `summarize` query completes, the frontend opens a chat panel seeded with the original question and synthesized answer. Follow-up messages are sent to the chat stream endpoint, which maintains conversation state server-side.

**ChatSession** (`packages/kb-service/src/kb_service/chat.py`): Holds the conversation as a `list[Message]` (where `Message = dict[str, str]` with `role` and `content` keys). The `seed()` method initializes the conversation with the original Q+A pair and the entry IDs from the summarize result. On each `reply()`, the session: (1) appends the user message, (2) trims history if over budget, (3) runs `hybrid_search` (via the shared `KnowledgeBase`) with `limit=5` to find entries relevant to the follow-up, (4) builds a system prompt that includes the chat system prompt plus the full `knowledge_details` of all known entries, (5) calls `llm.generate_chat(messages, system=system)`, and (6) appends the assistant response. New entry IDs discovered during retrieval are accumulated in `self.entry_ids` and reported back via the `chat_done` event so the frontend can highlight them on the graph.

**Write tools**: When write deps are available (the per-request `Attribution` carries the user's contributor/team), the chat session gains a mini-ReAct loop with three tools: `get_entry` (fetch full entry details), `update_entry` (modify an existing entry), and `ingest_url` (fetch and ingest a URL). The LLM emits tool calls as `{"tool": "...", "args": {...}}` JSON blocks, which `_parse_tool_call()` extracts from the response. `_dispatch_tool()` executes the tool, injects the result as a user message, and re-queries the LLM for a follow-up response. `get_entry` is always available (read-only); `update_entry` and `ingest_url` require write deps.

**Token budget**: `_MAX_CONVERSATION_CHARS = 100_000` (~25K tokens). When the total character count exceeds this, the trimmer preserves the first 2 messages (the seed Q+A that grounds the conversation) and the most recent messages, dropping middle turns until the budget is met. No summarization pass — just a sliding window.

**Session store**: Sessions are stored in-memory in a module-level dict keyed by `(user_id, session_id)`. Recently-used `ChatSession` objects are kept in memory; there is no eviction or TTL (home-lab MVP). The chat-history routes (`kb_service/chat_history.py`) persist message logs into the service DB so a user's chats survive restarts.

**SSE protocol**: The chat stream endpoint accepts `{message, session_id?, seed_question?, seed_answer?, seed_entry_ids?}`. It emits: `chat_session` (with the session ID), `chat_thinking` (before LLM call), `chat_done` (with newly discovered entry IDs), `chat_response` (with the answer text and session ID), and `stream_end`. Errors yield an `error` event with exception details.

**LLMProvider.generate_chat()**: Part of the `LLMProvider` protocol (in `kb_core/llm/provider.py`) alongside the existing `generate()` method. Takes `messages: list[Message]` and optional `system` prompt, returns `str | None`. Each backend implements it natively: Anthropic passes messages directly to `messages.create()`, Bedrock maps to `BRMessage` objects for the Converse API, and Ollama uses `/api/chat` (not `/api/generate`). The ReAct agent loop in `kb_core/graph/agent.py` also uses `generate_chat()`.

See: `packages/kb-service/src/kb_service/graph_export.py`, `packages/kb-service/src/kb_service/chat.py`, `packages/kb-service/src/kb_service/classifier.py`, `packages/kb-service/src/kb_service/sse.py`, `packages/kb-service/src/kb_service/routes/`, `src/personal_kb/tools/kb_explore.py`

## Project Preflight (kb_preflight)

The `kb_preflight` tool is a lightweight project context primer that agents call at session start to get up to speed on a project. It takes a `project_ref` and returns a compact table-of-contents of relevant entries — no LLM calls, pure SQL against indexed columns.

The output has five sections. The Maps section lists all of a project's mental maps (uncapped — they're a small curated set); the other four cap at 5 entries each:

1. **Maps** — `mental_map` entries for this project, rendered as `- [kb-XXXXX] short_title — long_title`. The `Maps:` section leads the output: it is the orientation directory an agent reads first to decide which maps to pull. Implemented by `_maps_sql()` in `kb_core/preflight.py`, which filters on `entry_type = 'mental_map'` and orders by `created_at DESC` with no limit (maps are a small curated set, unlike the recent/conventions sections which cap at 5). See the Mental Maps section below for how this in-process pull pairs with the on-disk push index.
2. **Expiring entries** — entries with `expires_at` in a window from 7 days ago (grace period for recently expired) to 30 days ahead. Sorted by expiry date ascending, so the most urgent appear first. Each line includes an expiry badge: `[EXPIRED 2d ago]`, `[EXPIRES 5d]`, or `[EXPIRES 12h]`.
3. **Recent decisions & lessons** — entries with `entry_type` of `decision` or `lesson_learned`, sorted by `created_at` descending. An optional `since` parameter (same TTL format as `kb_store` — `7d`, `2w`, `24h`) narrows to a time window; omitting it shows all.
4. **Active conventions** — `pattern_convention` entries, always shown regardless of `since`.
5. **Related (via graph)** — decisions and lessons from *other* projects that share tags with the current project. Found via 2-hop graph traversal: project entries → shared tag nodes (threshold: 2+ entries using that tag) → entries from other projects. Each line shows the connecting tag: `(via #api)`.

Output format is compact — entry ID, type, and short title only. Agents use `kb_get` to read full details for entries that look relevant.

**Team filtering.** When `KB_TEAM` is set, all four queries automatically scope to `(team = ? OR team IS NULL)`, so team members see their own entries plus global (unscoped) entries, but not other teams' knowledge.

**Design note.** An earlier approach tried CWD-based project detection — the MCP server subprocess inherits the client's working directory, so we attempted fuzzy matching against known `project_ref` values to inject context automatically at startup. This was removed because CWD is unreliable for subprocess spawning. The current design is explicit: agents call `kb_preflight(project_ref="my-project")` when they need context.

See: `kb_core/preflight.py`, `src/personal_kb/tools/kb_preflight.py`

## Supersession

Supersession is structural: "B replaces A" is a `supersedes` graph edge from B to A, and `A.superseded_by` is **derived** from those edges, never written by hand. The code lives in `kb_core/supersession.py`.

### The invariant

Every `knowledge_entries` row's `superseded_by` equals the id of its newest qualifying superseder, or NULL if it has none. A qualifying superseder of T is a row S with `is_active = 1` and `entry_type != 'mental_map'`, joined by a `graph_edges` row `(S, T, 'supersedes')` whose `properties` JSON does not carry `source == 'llm'` (parsed in Python with `json.loads`, so no dialect-specific JSON SQL is needed). "Newest" means max `created_at`, which is immutable. `updated_at` is never used, because it moves on every edit and would flip the pointer. `created_at` is parsed with `datetime.fromisoformat`, naive values are treated as UTC, and ties break by id descending. The invariant holds whether T is active or inactive.

`recompute_superseded_by(db, target_ids, *, trigger)` applies the invariant to a set of targets inside the caller's transaction (it opens none of its own). An unchanged target is not written, counted or logged. A changed one gets an UPDATE of `superseded_by` alone, with no version bump and no `updated_at` change. It also gets a `superseded_by_changed` audit row whose detail is `{old, new, candidates, trigger}`, plus an INFO line `supersession target=... old=... new=... candidates=... trigger=...`. `queries.update_entry` no longer writes `superseded_by` at all, so an update from a stale entry snapshot cannot clobber the derived value.

### The writers

`GraphBuilder.build_for_entry` captures the entry's old `supersedes` targets before clearing its edges. It rebuilds the edges, and its last statement recomputes the old targets, the new targets and the entry itself, so dropping a target from `hints.supersedes` clears that target's pointer in the same transaction. Builder step 5, which wrote a reversed edge from `entry.superseded_by`, is removed. Step 7 (`related_entities`) skips any item typed `supersedes` with a `supersession: ignoring related_entities supersedes edge` warning, so `hints.supersedes` is the single validated channel.

`KnowledgeStore.deactivate_entry` (which gains an optional `change_reason` that becomes the audit detail) and `reactivate_entry` recompute the entry's outgoing targets and the entry itself. `KnowledgeBase.deactivate(entry_id, *, contributor, change_reason, superseded_by)` runs one transaction. When `superseded_by` is given, it appends `entry_id` to the superseder's `hints.supersedes` (a hints-only UPDATE plus a `supersedes_hint_appended` audit row) and inserts the edge. It then deactivates the entry, removes its outgoing non-map edges and recomputes. Hard delete and direct SQL are not maintained; the startup reconcile heals them.

### Startup reconcile

`reconcile_supersession(db)` is idempotent and runs in one transaction. First it backfills a missing `supersedes` edge for every active non-map entry whose `hints.supersedes` names an existing kb id that has no edge yet. Then it recomputes every supersedes-edge target and every row whose `superseded_by` is set. kb-service's lifespan calls `kb.reconcile_supersession()` right after opening the KB, inside a try/except that never blocks startup. It logs `supersession-reconcile edges_added=N set=N cleared=N` and one WARNING `supersession-reconcile drift target=... old=... new=...` per changed row; a failure logs `supersession-reconcile failed: ...`. After the first post-deploy startup has done the backfill, any `supersession-reconcile drift` line is a defect signal: some writer left the invariant broken.

### The HTTP contract

`StoreRequest` and `StoreBatchEntry` carry `supersedes: list[str] | "none" | None`. An absent field (an older client), `[]` and `"none"` all mean no supersession. On create, the effective set is the union of `body.supersedes` and the string items of `hints.supersedes`; a non-string hint item is a 422, never silently filtered. The set is validated with `kb.check_supersedes` before `kb.store`, and the problems are rendered as `supersedes rejected: ...`. A target must be a well-formed kb id that exists and is active. It must not be the writer itself or a mental_map, and it must not already supersede the writer (a cycle). A mental_map writer cannot supersede anything. Cross-project targets are allowed. `StoreResponse.superseded_ids` lists every validated target on create.

On update, `change_reason` is required, and a missing or whitespace-only value is a 422 checked before everything else. Supersedes are a monotonic union with the targets the entry already records, and only the newly added targets are validated, so previously recorded targets (possibly inactive by now) are grandfathered. `hints={"supersedes": []}` retracts nothing; the only API path to retract is deactivating the superseder. `superseded_ids` lists only the newly added targets. `/store_batch` applies the create rule per entry in its up-front validation loop (`entry {i}: supersedes rejected: ...`).

`POST /api/kb/entries/{id}/deactivate` takes an optional `{change_reason, superseded_by}` body, checked in this order: auth (401), the mental_map block (422), a missing change_reason (422), then `superseded_by` must exist and pass `check_supersedes` (422). Only after those does `kb.deactivate` run. Because of that order, an unknown id with no change_reason is a 422, not a 404. Every decision logs one `supersession-route op=... outcome=... mode=... writer=... targets=... problems=...` line. A validated target still unpointed after the write logs `supersession-route build_failed`, because `_build_graph` is best-effort and the next startup reconcile heals it.

### Maps and superseded pointers

`POST /api/kb/map-op` rejects a `create_map` whose body points at a superseded entry, and an `add_pointer` whose added entry is superseded. Both return 409 `pointer_superseded`, with a second trail line carrying the `(pointer, superseded_by)` pairs. `strike_gap` and `propose_gap` are unchanged, so pointers already in a map are grandfathered. `GET /api/kb/map-loop-input` adds `AND superseded_by IS NULL` to its own candidate CTE (the shared `MAPPABLE_ENTRY_WHERE_SQL` and the map-eligibility counts are unchanged), so somnus is never offered a superseded entry and the 409 is unreachable in normal operation.

`POST /api/kb/store` (create and update) and `POST /api/kb/store_batch` apply the same rule to `mental_map` writes via `check_superseded_map_pointers` in `map_write_guards.py`: a superseded pointer is a 409 `pointer_superseded` (422 `entry i: ...` in a batch), and on update only newly added pointers are checked, so existing ones are grandfathered.

### Supersession-aware reads

An entry is superseded exactly when `entry.superseded_by is not None`; reads never re-check that the superseder is active, because the write-side invariant only points at active, non-map superseders. In the fused path of `hybrid_search`, superseded rows are skipped unless `SearchQuery.include_superseded` is set, and the result window grows by one for each hidden row, so only superseded rows are backfilled and other filters still use up their slot as before. The filter-only path pushes `AND superseded_by IS NULL` into SQL, so LIMIT is never spent on hidden rows; that path is not instrumented. The flag travels as `SearchRequest.include_superseded` (default False, so older clients get hidden) and `HttpBackend.search` always sends it. Dedup ingestion opts out with `include_superseded=True` so content the KB already has still dedupes. `kb_preflight` adds `superseded_by IS NULL` to the expiring, recent, conventions and graph-related queries (not the tag or maps queries). On the client, `kb_search` appends `[SUPERSEDED by kb-X]` to superseded results and skips superseded graph hints; `kb_get` prefixes a superseded entry with `SUPERSEDED by <id> — <title>`, costing one extra `get_entries` call only when the superseder was not already in the request. `kb_ask` graph expansion, the agent's `graph_neighbors` and `scope_entries` tools, and the exhaust fallback skip superseded entries; `decision_chain` and an agent's explicit final pick are exempt (the pick is logged). Every skip is logged as `supersession-read op=<search|ask_expand|agent_tool|agent_exhaust|agent_final_pick|kb_get> ...`; any `supersession-read invariant_breach` line is a defect signal.

## Near-duplicate guard

The hosted service refuses to create an entry that is nearly identical to an existing one unless the request says what the relationship is. The guard lives in `kb_service/routes/near_duplicate_guard.py` (route glue, 409 payload, telemetry) and `kb_core/near_duplicates.py` (the lookup), and is exposed on the facade as `KnowledgeBase.find_near_duplicates`.

It runs only on create, for both `POST /api/kb/store` and `POST /api/kb/store_batch`, after the supersedes validation. An update never runs it, and an update that carries `distinct_from` (in the field or in `hints`) is a 422. Entries within one batch are not compared with each other.

The search is scoped to the new entry's own `project_ref`. The new text is embedded with `KnowledgeEntry.compose_embedding_text` (the same definition the stored vector uses), then `Database.vector_search(limit=20, project_ref=...)` returns active rows of that project with cosine distance. Eligible rows are non-`mental_map` with `superseded_by IS NULL`; similarity is `1 - distance`. Candidates are eligible rows at or above the floor, sorted by `(-similarity, id)`, truncated to 5. On SQLite the vector search over-fetches the top 200 global neighbours and then filters to the project, so a project whose rows are crowded out of that window can be missed; Postgres is exact.

Exemptions: `mental_map` entries, entries with no (or blank) `project_ref`, and two fail-open cases: `embedder_unavailable` (no embedder, or the embed call returned nothing) and `search_failed` (the search raised). Expired-but-active entries still count as candidates.

The floor is `KB_NEAR_DUPLICATE_FLOOR`, default 0.88, read on every request. A value above 1.0 disables the guard. A malformed value raises `ValueError` and creates return 500 until it is fixed. 0.88 was chosen from the live distribution: over 30 days, 339 non-map project-scoped creates compared with their nearest older same-project neighbour would have 409'd at 0.92 (2), 0.90 (4), 0.88 (7, 2.1%), 0.85 (15) and 0.80 (47); the median similarity of declared supersedes edges is about 0.70, so the guard catches near-duplicates and does not replace declared supersession.

A conflict is a 409 whose `detail` is `{"error": "near_duplicate", "message", "project_ref", "floor", "candidates": [{id, short_title, entry_type, similarity, updated_at}]}` (plus `entry_index` for a batch, whose message starts `entry {i}: `). Only candidates not already covered by the request are listed. Every listed id must be covered by one of three escapes: `update_entry_id=<id>` (same fact: update that entry, with `change_reason`), `supersedes=[<id>]` (the new entry replaces it; older clients: `hints={"supersedes": [...]}`), or `distinct_from=[<id>]` (genuinely different facts; older clients: `hints={"distinct_from": [...]}`). Each `distinct_from` id must be a well-formed kb id of an existing, active, non-map entry (cross-project is allowed) and must not also be in `supersedes`; otherwise 422.

`distinct_from` is stored on the new entry as `hints.distinct_from`, and `GraphBuilder` mirrors it as a `distinct_from` edge (the target must exist and not be the entry itself; a `distinct_from` item in `related_entities` is ignored with a warning). The hint is the source of truth; the edge mirrors it. Suppression is request-scoped: a candidate named in this request's `supersedes` or `distinct_from` is not reported. A reader should treat an edge in either direction between P and Q as "declared distinct"; this item implements no such reader. `kb_ask` / `kb_explore` traverse the edge like any other type.

Every guarded create emits one log line starting `near-duplicate-guard op=` (INFO; WARNING for `embedder_unavailable` and `search_failed`) with `outcome` (`clear|resolved|conflict|exempt_mental_map|skipped_no_project|embedder_unavailable|search_failed`), floor, top similarity, raw/eligible hit counts, `embed_ms`, `search_ms`, candidates, `resolved_by`, `distinct_from_unused`, contributor, title and `text_sha` (first 12 hex digits of the sha256 of the embedding text). It also writes one `audit_events` row with `event_type='near_duplicate_checked'` and the same fields as JSON `detail`. A 409 row has `entry_id` NULL; every other row is written after the store succeeds and carries the created id (with a `near-duplicate-guard-stored` log line). The audit write is best-effort.

These rows are the retune instrument. To re-derive the 409 rate at a candidate floor X, count rows where `detail.top_similarity >= X`, grouped by `detail.outcome`:

```sql
SELECT detail::json->>'outcome' AS outcome, count(*)
FROM audit_events
WHERE event_type = 'near_duplicate_checked'
  AND (detail::json->>'top_similarity')::float >= :X
GROUP BY 1;
```

Two follow-ups: missed duplicates are `clear` rows whose `entry_id` later becomes the target of a `supersedes` edge; and the escape agents chose after a 409 is found by joining `conflict` rows to later `resolved` rows with the same `text_sha`.

## Resolutions (hints.resolution)

A resolution is a corrected belief stored in an entry's `hints` under `resolution`; future sessions receive it through the prevention channels. The format is owned by `kb_service/prevention.py` (the reader), and `kb_service/resolution_hint.py` is the producer-side validator: `corrected_fact` (required, at most 1000 characters), optional `wrong_belief`, `evidence`, `cue` (`tool`, `target_class`, optional `args_prefix`), `provenance` (`capture`, `grounding`, `event_id`), `observed_sessions` and `scope` (`project` or `global`). `provenance.event_id` is an additive key the reader ignores.

`POST /api/kb/store` (create and update) and `POST /api/kb/store_batch` validate any request whose hints carry `resolution` before every other guard, so a shape error is a 422 (`entry i: ...` in a batch, where one bad entry stores nothing) and never a 409. A resolution on a `mental_map` entry is a 422, because the reader excludes maps. A Bash cue's `target_class` must be a fixed point of `kb_core.cues.target_class`.

Stamping: the machine principal's resolution is always `capture='autonomous'` (a supplied `deliberate` is overridden and logged as `resolution_capture_forced`); any other writer defaults to `deliberate`. Grounding defaults to `asserted`; `observed` requires an `event_id`. An autonomous, non-observed resolution cannot replace a deliberate one (a stored resolution with no provenance counts as deliberate) and is a 422 `deliberate_protected`. An update that omits `resolution` leaves the stored one untouched (shallow hint merge in kb-core).

A superseder must carry the resolution forward to keep it alive: nothing is copied automatically, and a write that supersedes an entry holding a resolution without carrying one logs the WARNING `resolution_dropped_by_supersede`.

Every request entry that carries a resolution emits one INFO line starting `resolution-route` (op, outcome, reason, writer, capture, grounding, cue, scope, prior capture).

`packages/kb-service/scripts/seed_resolutions.py` seeds resolutions from existing procedure entries over HTTP. It is a dry run unless `--apply` is given. `--pinned-file` supplies `{entry_id: resolution}` written verbatim (no LLM; apply it with a non-machine-principal key or the capture is downgraded and reported as `capture_forced`). Other entries pass a regex prefilter, an LLM draft, and an ordered guard: `llm_unparseable`, `declined`, `empty_fact`, `bad_scope`, `not_normalized`, `not_two_word`, `example_mismatch`, `denylisted`, `class_collision`. Writes use a `change_reason` starting `seed_resolutions:` (the after-the-fact query key) and are re-fetched and verified. `--audit` is read-only and re-checks every stored resolution against the live validator and cues normalizer; run it after any cues normalizer change.

## Mental Maps

The `mental_map` entry type is the directory tier of the KB: an orientation node whose body is *pointers and structure*, never retrievable values. The settled design lives in `docs/mental-map-prespec.md` §7; this section documents the shipped code that implements it. The push half — the `personal-kb-hook` CLI — is documented in the next section; what follows is everything in the MCP server itself that makes maps a distinct entry type.

### The `mental_map` entry type and decay exemption

`EntryType.MENTAL_MAP = "mental_map"` is the fifth member of the `EntryType` enum in `kb_core/models/entry.py`, sharing all storage, versioning, FTS, embedding, and graph plumbing with the four value-bearing types. The one place it diverges is `kb_core/confidence/decay.py:compute_effective_confidence`, which checks for `MENTAL_MAP` first and early-returns `base_confidence` without consulting any half-life table:

```python
if entry_type == EntryType.MENTAL_MAP:
    return base_confidence
```

The rationale is double-edged. First, a map holds no retrievable value of its own — there is nothing on it that goes stale on a clock. Second, the system's access-aware self-heal — `kb_core/db/queries.py:touch_accessed` resets `last_accessed` on every `kb_get`, which is the decay anchor — would make a constantly-surfaced map look "fresh" while its pointers rotted, and a correct-but-cold map trip stale. Both directions are backwards for an orientation node, so the entire clock is bypassed. Freshness for a map is *pointer-validity*, computed on retrieval (see "On-GET pointer-rot" below), not a half-life decay.

The lookup of the four remaining types is `HALF_LIVES.get(entry_type, 365.0)`. The `.get()` with a one-year default is the §7.2 same-commit guard rail: a future enum member can never `KeyError` the decay path.

See: `kb_core/models/entry.py`, `kb_core/confidence/decay.py`.

### Required-outbound-pointer validation on store

A map is defined by what it points *to*. A map with zero outbound pointers is, definitionally, an orphan note — not a map. `src/personal_kb/tools/kb_store.py:_mental_map_has_pointer()` enforces this **before** `create_entry` runs, so an orphan never produces a row or a version record:

```python
if entry_type == EntryType.MENTAL_MAP and not _mental_map_has_pointer(
    knowledge_details, hints
):
    return ORPHAN_MAP_ERROR
```

The check is a closed checklist that mirrors `kb_core/graph/builder.py`'s edge-producing logic exactly. An outbound pointer exists iff *any* of these is true:

1. `knowledge_details` contains a `kb-XXXXX` reference (matched with the same `re.compile(r"kb-\d{5}")` the builder uses);
2. the `related_entities` hint contains either a dict with a non-empty `id`/`target`, or a bare non-empty string.

Neither the client copy in `kb_store.py` nor the server-side guard (`kb_service/routes/map_write_guards.py::_mental_map_has_pointer`) counts a `supersedes` hint or `superseded_by`: a mental_map cannot supersede anything (see Supersession), and the builder no longer derives any edge from `superseded_by`.

Tag, project, person, and tool hints do **not** count — those are categorization, not orientation. Mirroring the builder's exact predicate set means a future change to what counts as a "pointer" needs to be made in exactly one place; the validator follows automatically.

The error message returned to the caller is a single constant, `ORPHAN_MAP_ERROR`:

> *"A mental_map entry requires at least one outbound pointer (a kb-XXXXX reference in knowledge_details, or a related_entities hint). A map with zero pointers is an orphan note, not a map."*

This is the only place in the `kb_store` pipeline where mental_map content is *rejected*. The fact-free lint below is advisory only and never blocks.

See: `src/personal_kb/tools/kb_store.py` (`_mental_map_has_pointer`, `ORPHAN_MAP_ERROR`), `kb_core/graph/builder.py`.

### Advisory fact-free lint

The §7.3 invariant — *"no retrievable value in assertion position"* — is enforced as a **deterministic, regex-based, advisory-only** heuristic in `src/personal_kb/tools/map_lint.py`. The module's contract is deliberately narrow:

- Pure function: `lint_map_body(text) -> list[str]`. No I/O, no LLM, no network. Never raises.
- Every returned string starts with the literal prefix `Map lint (advisory): `. The module exposes no `Error:` path. The store always succeeds; the warnings are informational.
- It mirrors the shape of `_check_secrets` in `kb_store.py` — same call-site pattern — but never signals rejection.

The lint runs on every mental_map create and every mental_map update where a fresh `knowledge_details` body was supplied; metadata-only updates are skipped. In `kb_store`, gating happens at the call site (not inside `format_store_result`, which is shared with non-map stores), and warnings are prepended *above* the `Created/Updated` compact block but *below* any backend warning via `_prepend_map_advisories`. `kb_store_batch` runs the same lint per-entry and attaches the warnings to the entry's own block.

The heuristic categories, in order of how `lint_map_body` evaluates them:

1. **`kb-XXXXX` references are stripped first.** Pointers are the desired content; their digits must not later read as a retrievable numeral. The cleaned text is what every subsequent rule sees.
2. **URLs** — `https?://\S+`. A retrievable link belongs in a `factual_reference` the map points to.
3. **File paths** — `~/`-rooted, or an absolute `/a/b…` path with at least two segments. Retrievable values, not orientation pointers.
4. **`ENV_VAR`-style tokens** — all-caps starting with a letter, with at least one underscore (e.g. `KB_DB_PATH`). Retrievable config, not pointers.
5. **Dotted code identifiers** — `module.func` patterns (`\b[A-Za-z_]\w*\.[A-Za-z_]\w*`). Retrievable signatures.
6. **Quoted literals** — `"..."` or backticked `` `...` `` pairs. Single quotes are deliberately excluded to avoid flagging ordinary apostrophes in prose.
7. **Config-like numerals** — decimals (e.g. `0.06`), integers with ≥4 digits (e.g. `8767`, `51820`), or any numeral immediately preceded by `=` or `:`. **Counts-of-parts are exempt**: a numeral followed by a plural noun (`three stages`, `12 nodes`) is a pointer-in-disguise — it tells you how many edges to expect — so the `_COUNT_TAIL_RE` of `\s*[A-Za-z]+s\b` is checked first and matching numerals are allowed regardless of magnitude.
8. **Advisory size budget.** Bodies longer than `map_body_budget(pointer_count)` characters — `MAP_BODY_BASE_CHARS = 900` plus `MAP_BODY_PER_POINTER_CHARS = 175` per distinct `kb-` pointer, counted format-agnostically — get the note "*cut orientation prose, not pointer glosses*." Per §7.3 the budget is an advisory proxy, not the definition of fact-free — the discriminator is value-vs-pointer, not byte count, and a compositional budget keeps per-pointer glosses (craft the nightly loop must never regenerate) from being squeezed out of well-pointed maps.

The discriminator the prespec frames it with: *would a reader act on this number/string directly* (forbidden — a retrievable value) *or follow it to a source* (fine — a pointer)?

See: `src/personal_kb/tools/map_lint.py`, `src/personal_kb/tools/kb_store.py` (`_prepend_map_advisories`, gating), `src/personal_kb/tools/kb_store_batch.py`.

### On-GET pointer-rot

Because maps don't decay on a clock (§7.4), the freshness signal moves to retrieval. `src/personal_kb/tools/kb_get.py:_pointer_rot_note` runs only when the retrieved entry's type is `MENTAL_MAP`; for every other entry type the function returns `None` immediately, so non-map `kb_get` output is **byte-identical** to before.

For a mental_map, the function resolves the entry's **outbound** graph edges via `get_neighbors(db, entry.id, direction="outgoing")` with no `edge_types` filter — it sees every outgoing edge, of which only those whose target matches `kb-\d{5}` are treated as pointers. (Tag/project/person/tool nodes are filtered out by the kb-id regex check before any extra DB lookup is made.) A target is *rotted* if either of these holds:

- `target.superseded_by is not None` — there is a replacement; or
- `target.is_active is False` — the target has been deactivated.

When **both** apply, the superseded form wins, because naming the actionable replacement is more useful than just flagging "gone." Targets are deduplicated (a target reached by multiple edge types surfaces once) and sorted ascending by id so the output is stable. The rendered block is two-space-indented to nest cleanly under the standard full-entry render:

```text
  Pointer-rot:
    [kb-00310] superseded by [kb-00342]
    [kb-00214] deactivated
```

Two implementation details worth calling out:

- The function deliberately bypasses `kb_get`'s top-level *"not is_active → not found"* short-circuit when resolving targets — a deactivated target is precisely the rot signal we want to surface, not hide. It calls `kb_core/db/queries.py:get_entry` directly, which has no `is_active` filter.
- This is the **on-GET** check, not a new always-on badge subsystem (per §7.4). It deliberately reuses the supersedes edges the graph builder already maintains; no new edge type, no new index, no badge in `format_entry_compact`. There is also **no freshness inheritance** — a map's effective confidence does not depend on its leaves' confidence, which would manufacture a permanent-staleness trap.

See: `src/personal_kb/tools/kb_get.py` (`_pointer_rot_note`), `kb_core/graph/queries.py:get_neighbors`, `kb_core/db/queries.py:get_entry`.

### The Maps index pull half

`kb_preflight` is the in-process pull half of the §7.7 surfacing design. The relevant SQL lives in `kb_core/preflight.py:_maps_sql`:

```python
"SELECT id, short_title, long_title "
"FROM knowledge_entries "
"WHERE is_active = 1 AND project_ref = ? "
"AND entry_type = 'mental_map' "
# + optional team clause
"ORDER BY created_at DESC"
```

`build_project_context` runs this query alongside the other preflight queries, and renders results into a **`Maps:`** section that leads the output (before Expiring / Recent / Conventions / Related). Each line follows the format `  - [<id>] <short_title> — <long_title>` — id plus both titles, no type label (redundant inside a Maps section), with U+2014 EM DASH between the two titles. When the project has no maps, the section is omitted entirely; an empty Maps block never renders.

The same predicate — `entry_type = 'mental_map'`, `is_active = 1`, optional team scope, `ORDER BY created_at DESC` (no limit) — is reused by the hosted web service's `GET /api/kb/maps-index` route (`packages/kb-service/src/kb_service/routes/maps_routes.py`), which computes the per-project maps index on demand from the live DB so the on-disk JSONL the hook reads is always fresh by construction. The push half — how the hook actually pulls and caches that index — is described in the CLI Hook section below.

See: `kb_core/preflight.py` (`_maps_sql`, the `Maps:` block in `build_project_context`), `src/personal_kb/tools/kb_preflight.py`, `packages/kb-service/src/kb_service/routes/maps_routes.py`.

## CLI Hook (personal-kb-hook)

The repo ships a console script — `personal-kb-hook` — that is wired into Claude Code's `SessionStart`, `UserPromptSubmit`, and `Stop` hooks. It is the "push" half of the mental_map surfacing design (the in-process `kb_preflight` Maps section, documented above, is the "pull" half) AND the launcher for the anticipatory-listener whisper loop (described in the next section). The hook is intentionally tiny and stdlib-only — argparse, json, urllib, pathlib, sys. It is published as a **separate package** so its install footprint is genuinely zero-third-party-dependency.

### Package layout

The runtime package lives at `packages/personal-kb-hook/src/personal_kb_hook/`. Its modules are:

| Module | Role |
|---|---|
| `cli.py` | Console-script entry point: parses stdin payload, branches on `hook_event_name` |
| `resolver.py` | `.kb_project` walk-up resolver |
| `roster.py` | Multi-KB roster loader (`kbs.json`) |
| `http_index.py` | Maps-index fan-out across the roster (consumes `/api/kb/maps-index`) |
| `index_reader.py` | `MapEntry` / `MapKey` types |
| `render.py` | Factual non-imperative directory and whisper rendering |
| `suppression.py` | Per-session "only on change" scratch |
| `paths.py` | Cache and config path helpers |
| `listener.py` | Listener env gate, transcript extraction, cache helpers, worker spawn |
| `listener_worker.py` | Detached multi-KB fan-out worker for the listener gate |
| `events.py` | Harness-event forwarding: `PostToolUseFailure` to `POST /api/kb/event` |
| `prevention.py` | Prevention channels: SessionStart gotcha slice and PreToolUse soft gate |
| `cues_lite.py` | Vendored lexical cue functions (stdlib-only copy of `kb_core.cues`) |
| `telemetry.py` | Whisper-efficacy telemetry helpers (stdlib-only, silent-on-failure) |
| `tool_inventory.py` | SessionStart inventory of personal script directories |
| `defaults.py` | Local-mode defaults for the hook's service URL and API key |
| `whisper_debug.py` | Ephemeral local whisper-decision debug log |
| `turn_digest.py` | Stop-time turn digest: window, items, body, per-session counter, spawn |
| `turn_sender.py` | Detached sender that POSTs a digest body to `/api/kb/turn` |

The runtime package imports **only the Python standard library**. A test in the runtime package's own suite (`packages/personal-kb-hook/tests/test_hook_cli.py::test_hook_package_is_stdlib_only_and_does_not_import_main_package`) walks every `.py` file in the package, parses the AST, and asserts every `import` / `from ... import` resolves to either a `sys.stdlib_module_names` root or the package's own `personal_kb_hook` namespace.

### Maps index — service-computed, hook-consumed

The hook used to read an on-disk JSONL maps index written by the MCP server. In the thin-client split the maps index is computed by the hosted web service — `GET /api/kb/maps-index` (in `packages/kb-service/src/kb_service/routes/maps_routes.py`) queries the live DB on each request, using the exact predicate of `kb_core/preflight.py:_maps_sql`, and returns the same shape: one entry per project with at least one active map, sorted ascending by `project_ref`. The hook's `http_index.load_index(roster)` fans the request across every KB in the roster concurrently (using stdlib `urllib`) and returns the merged `dict[str, list[tuple[label, MapEntry]]]` keyed by project. Per-session suppression remains the hook's job (`packages/personal-kb-hook/src/personal_kb_hook/suppression.py`).

### `.kb_project` walk-up resolver

`personal_kb_hook.resolver.resolve_project(cwd)` walks from `Path(cwd)` through each parent up to the filesystem root, returning the first non-blank, non-comment line of the first `.kb_project` it finds. The file is **committed to the repo** — portable across machines and users, no per-user TOML, no git-origin lookups. The walk is tolerant: a falsy `cwd`, a missing/unreadable/empty/comment-only `.kb_project`, or any unexpected I/O error returns `None` and never raises.

This is the v1 scope anchor for both the `SessionStart` and `UserPromptSubmit` hooks. `cwd` is reliably present in both hook payloads (unlike the MCP server subprocess, where CWD is unreliable — see the design note in the kb_preflight section). FTS/keyword matching on the prompt is **out of scope for v1**; it is deferred to a v1.1 within-project map refiner that picks *which* of a multi-map project's maps to surface, not *which project*.

### Suppression scratch + compact bypass

Hooks are stateless between turns; without a scratch file, "only on change" is unimplementable. `personal_kb_hook.suppression.should_emit()` and `mark_emitted()` read/write `~/.cache/personal_kb/injected-<session_id>.json` holding `{"last_scope": ..., "surfaced_map_ids": [...]}` (with `surfaced_map_ids` carried as `[label, id]` pairs since the multi-KB extension).

The hook emits only when maps exist for the resolved scope AND at least one of:

* `payload.source == "compact"` (a compaction event re-seeds the context, bypassing the subset check),
* no scratch file yet,
* the resolved scope differs from `last_scope` (the user moved to another repo),
* the resolved map ids are not already a subset of `surfaced_map_ids` (new maps were created since the last surface).

After a successful emit, `mark_emitted()` unions the new ids into `surfaced_map_ids` and sets `last_scope` to the freshly resolved scope. Atomic write: `tempfile.NamedTemporaryFile` → `os.replace`.

### Output

`personal_kb_hook.render.render_directory()` produces:

```text
Maps for <project_ref> — [<id>] <short_title>: <long_title>; [<id>] <short_title>: <long_title>
```

The separator after `<project_ref>` is U+2014 EM DASH (matching `kb_core/preflight.py`'s separator). Entries are joined with `"; "` (semicolon-space). An entry whose `long_title` is empty renders as `[<id>] <short_title>` with no trailing `": "`.

The output is **factual, never imperative**. `render.py` defines a `BANNED_TOKENS` frozenset (`load`, `use`, `read`, `fetch`, `pull`, `open`, `retrieve`, `get`, `review`, `consult`); a unit test lowercases the rendered string and asserts none of those substrings appear. Imperative phrasing trips prompt-injection defenses and gets surfaced to the user instead of read by the model — that's the failure mode we are designing around.

For `--format=claude-json`, `render_claude_json()` wraps the same directory string in `{"hookSpecificOutput": {"hookEventName": <event>, "additionalContext": <directory>}}` — the envelope Claude Code understands.

### Why the hook never talks to the DB directly

The MCP server starts as an `stdio` subprocess. A `SessionStart` hook can fire before that subprocess has connected; a hook that called the MCP would race that startup. SQLite directly is also off the table — it would couple the hook to the database backend (the deployed service is Postgres), and it would need read-only file locking semantics across instances. The HTTP maps-index sidesteps both problems: the hosted service is the single owner of truth, the hook is read-only, and HTTP failure modes are well-defined (timeout → no surface, error JSON → no surface).

### Running tests for both halves

Both halves carry their own test suite:

* **Main repo** (`uv run pytest -m "not eval"`): MCP tool entry-point tests, the drift-guard / round-trip tests that pair with the hook, and the listener / hook integration tests reachable from this repo.
* **Standalone hook** (`(cd packages/personal-kb-hook && uv run --project ../.. pytest)`): cli / resolver / render / suppression / index-reader / roster / listener / listener_worker, plus the "no third-party imports" assertion.

The root `pyproject.toml` declares `[tool.uv.workspace] members = ["packages/*"]` and lists `personal-kb-hook` as a workspace dev dep, so `uv sync` installs the standalone package editable into the main `.venv` (needed by the cross-package tests).

See: `packages/personal-kb-hook/src/personal_kb_hook/{cli,resolver,index_reader,render,suppression,paths,roster,http_index,listener,listener_worker}.py`.

## Anticipatory Listener

The anticipatory listener is the third push surface (alongside `kb_preflight` and the maps directory): when an AI agent finishes a turn whose work clearly *touches* one or more orientation maps' domains, the listener whispers pointers to those maps into the *next* prompt. It is opt-in and retrieve-and-cite. **Multi-KB whisper went LIVE 2026-06-15 (`kb-01839`); design of record is `kb-01828` (v2); listener design is `kb-01725`.** **Plural (up to 2) pointers per KB per request went live with GTD 66ea1fe4** — a map is a SUBJECT AREA, and evidence can legitimately implicate more than one; the gate vote was reframed from single-id unanimity to a majority-of-3 vote over a SET of candidate ids, and a non-primary pointer must additionally clear an evidence bar (see `listener_routes.py::_meets_second_slot_bar`) to avoid riding along with a stronger match. This doc section otherwise predates that change (and the detail-match retrieval fix, GTD bf40d4f1) in places — read the current `listener_routes.py`/`listener_worker.py`/`render.py` docstrings for the as-built detail. The listener has two halves: a server route in `personal-kb-web-service` that returns candidate pointers for one KB, and a hook-side worker in `personal-kb-hook` that fans the request across the multi-KB roster, arbitrates the results, and queues the whisper for the next user prompt.

### Server route — `POST /api/kb/listener`

The route lives at `packages/kb-service/src/kb_service/routes/listener_routes.py`. Its contract:

- **Request body** (`ListenerRequest` in `packages/kb-service/src/kb_service/models.py`): `{text: str, cwd_project: str | None, operating: list[str], source_label: str | null}`.
- **Response body** (`ListenerResponse`): `{pointers: [{id, short_title}, ...], pointer: {id, short_title} | null, reason: str}` — `pointers` (0..2, evidence order) is the current field; `pointer` is a deprecated single-item alias (`pointers[0]` or `null`) kept because the hook's `_post_one_kb` still reads it as a fallback for pre-GTD-66ea1fe4 servers.

The route's behaviour, step by step:

1. **Kill switch.** The env var `KB_LISTENER_ENABLED` is read **per request, server-side**, with a default of `'FALSE'`. The route returns `{pointer: null}` immediately unless the value uppercases to exactly `'TRUE'`. The pilot ran with the gate off by default on the dev server; flipping it on is an explicit operator action.
2. **Candidate retrieval.** `kb.search` is called over `EntryType.MENTAL_MAP` with `limit=5` using the request text as the query. The retrieval is just the kb-core hybrid search — no special listener-only index.
3. **Rule A — same-project drop.** Every candidate whose `entry.project_ref` equals the request's `cwd_project` is dropped. When `cwd_project` is `null`, Rule A drops nothing. The intent: a map of your *current* project is not a useful surprise — preflight already showed it.
4. **Rule B — operated_via drop.** Every candidate whose `entry.hints['operated_via']` (a string) is in the request's `operating` list is dropped. Maps with no `operated_via` hint are never dropped by Rule B. The intent: if an agent is already actively operating a tool, naming its map is noise.
5. **Short-circuit.** If no candidates survive A+B, or `kb.synthesis_llm is None`, return `{pointer: null}`.
6. **LLM gate — majority-of-3.** The route fires **3 concurrent** `kb.synthesis_llm.generate(prompt)` calls. The synthesis LLM is the *synthesis slot* (Sonnet in the pilot configuration) — the model is NOT hard-coded in the route; it's whatever the `KnowledgeBase` was configured with. Each vote is parsed (a `NONE` answer or any non-candidate id becomes `None`). Each voter returns a SET of up to 2 candidate ids; an id is emitted only when it appears in at least 2 of the 3 voters' sets (majority-of-3, retrieve-and-cite: ids must come from the surviving candidate set), and any pointer beyond the first must also clear the second-slot evidence bar (>=2 distinct detail hits, or best detail rank <=3).
7. **Prompt project label.** When the gate prompt mentions the project the agent is working in, the label resolves `source_label → cwd_project → 'unknown'` (in that order, first non-None wins).

The verdict is intentionally strict. The asymmetric cost is the whole point of the listener design (kb-01725): a false positive (surfacing an irrelevant map into the next prompt) is much worse than a false negative (staying silent), so the gate stays silent unless all three votes agree on the same candidate.

### Hook-side multi-KB fan-out

The hook lives in `packages/personal-kb-hook/`. Its listener-side modules are:

- `personal_kb_hook.listener` — env gate (`is_listener_enabled`), transcript extraction (`extract_manifest`), cache helpers, and the detached-worker spawner (`spawn_worker`).
- `personal_kb_hook.listener_worker` — the detached subprocess that does the actual multi-KB fan-out, arbitration, and cache merge.
- `personal_kb_hook.roster` — the multi-KB roster loader (`load_roster()`).
- `personal_kb_hook.cli` — wires the `Stop` event to spawn the worker and the `UserPromptSubmit` event to consume the worker's output.

The end-to-end flow on a Claude Code `Stop` event:

1. **Env gate.** `cli.py` calls `listener.is_listener_enabled()`, which requires **all three** of `PERSONAL_KB_URL`, `PERSONAL_KB_API_KEY`, and `PERSONAL_KB_LISTENER` (lowercased value in `{'1', 'true'}`) to be set; otherwise the listener does nothing. Headless dispatch runs (`HEADLESS_BUILD_ENGINE` set) also return here unless `KB_LISTENER_HEADLESS=TRUE`: they never get another `UserPromptSubmit`, the only whisper delivery path, so the worker's Sonnet votes would be wasted. The roster telemetry flush still runs first.
2. **Manifest extraction.** `listener.extract_manifest(transcript_path)` does a bounded tail-scan (the last 256 KiB) of the Claude Code JSONL transcript and pulls the **last assistant record's** text blocks, head-capped to 4000 chars, plus the **sorted, de-duplicated** list of `'mcp:<server>'` strings derived from every tool_use block name across the tail window via the regex `^mcp__(.+?)__`. The gate text is instead the Stop payload's `last_assistant_message` when it is a non-empty string (the transcript is written asynchronously and may lag the current turn); the transcript text is the fallback and the transcript still supplies `operating`. Both paths share `listener.normalize_text` (4000-char cap, 200-char minimum; shorter means the listener exits cleanly), and the request carries `text_source` (`last_assistant_message` | `transcript`), stored on the `listener_decisions` row.
3. **Worker spawn.** `cli.py` writes the request JSON (`{text, cwd_project, operating, source_label}` — `cwd_project` and `source_label` are both the project-ref resolved from the cwd walk-up, since the hook has no separate "session label" concept) to a `NamedTemporaryFile` and calls `listener.spawn_worker(req_tmp_path, cache_path)`. The spawn uses `subprocess.Popen` with `start_new_session=True`, stdout/stderr redirected to `DEVNULL`, and **no `wait()` or `communicate()`** — the hook returns immediately so the agent's `Stop` event is not blocked. The worker is a fully detached `python -m personal_kb_hook.listener_worker`.
4. **Roster load.** Inside the worker, `roster.load_roster()` reads `<XDG_CONFIG_HOME or ~/.config>/personal_kb/kbs.json` (the underscore directory, pinned by kb-01828) and returns a list of typed `KbEntry(label, url, key)` records. An empty roster (`[]`) is an explicit no-op (zero POSTs, cache untouched, exit 0); a missing/unreadable file falls back to a synthesized single `KbEntry(label='personal', ...)` from the `PERSONAL_KB_URL` + `PERSONAL_KB_API_KEY` env vars when both are present.
5. **Per-KB POST.** For each roster entry, `_post_one_kb` does ONE POST to `{entry.url.rstrip('/')}/api/kb/listener` with `Authorization: Bearer {entry.key}`, `Content-Type: application/json`, a 30-second timeout, **stdlib `urllib` only**, and a single broad `try / except Exception → []`. Any failure for one KB (timeout, HTTPError, URLError, JSONDecodeError, non-dict body, missing `pointers`/`pointer` key, validation failure) yields an empty pointers list for that label and the loop continues. `pointers` (0..2, evidence order) is read first; the singular `pointer` is a fallback for a server that predates GTD 66ea1fe4. The worker NEVER raises into the operating system.
6. **Suppress-only arbitration.** `_arbitrate(candidates, roster_entries, source_label)` is **client-side, suppress-only**. The steps, in order: (a) drop every null pointer; (b) title-dedup by `short_title.strip().casefold()`, keeping one winner per title group; (c) a defensive one-pointer-per-KB cap. The tie-break order between same-title duplicates is: the KB whose `label` equals the request's `source_label` IF that label is in the roster, else the KB labelled `'personal'` if in-roster, else the first roster entry. Arbitration **NEVER** elevates, re-scores, or reorders by relevance — it only suppresses duplicates. Each surviving KB contributes at most ONE pointer.
7. **Cache merge.** The post-arbitration `winners` list (`[(label, pointer), ...]`) is atomically merged into the listener cache at `~/.cache/personal_kb/injected-<session_id>.json` as a per-KB-provenance `pending` list `[{label, id, short_title, long_title}, ...]`. `whispered_map_ids` is preserved in the `[label, id]` shape, with tolerant back-parse of any pre-multi-KB bare-id strings.

On the next `UserPromptSubmit` event, `cli.py` reads the same cache, filters `pending` against `whispered_map_ids` (so a pointer is only whispered ONCE), applies the same tie-break ordering, and groups the survivors by label before rendering — **up to two pointers per KB are whispered on ONE line** (GTD 66ea1fe4), via `render.render_whisper([...], label=..., multi_kb=...)`, in factual non-imperative form (singular/plural header, "Possibly relevant map(s) — "). The rendered lines (one per KB) are appended below the directory output as the next prompt's `additionalContext`. After successful emission, the whispered pairs are merged into `whispered_map_ids` so the next prompt's surface set never repeats.

### Arbitration

The hook's client-side `_arbitrate` (`packages/personal-kb-hook/src/personal_kb_hook/listener_worker.py`) is suppress-only: it drops null pointers, de-duplicates by title with a tie-break, and caps the result at one pointer per KB. `personal-kb-hook` is intentionally not a dependency of the service (it is stdlib-only), so the service never imports it. Listener evaluation harnesses are maintained outside this repo.

See: `packages/kb-service/src/kb_service/routes/listener_routes.py`, `packages/kb-service/src/kb_service/models.py` (`ListenerRequest`, `ListenerResponse`, `ListenerPointer`), `packages/personal-kb-hook/src/personal_kb_hook/listener.py`, `packages/personal-kb-hook/src/personal_kb_hook/listener_worker.py`, `packages/personal-kb-hook/src/personal_kb_hook/cli.py`, `packages/personal-kb-hook/src/personal_kb_hook/roster.py`.

## Failure-cue index

The failure-cue index records every failed tool call an agent makes, so we can measure how often the same mistake recurs in a later session. It is record-only today: nothing is delivered back into the session.

**Normalizer — `kb_core.cues`.** `packages/kb-core/src/kb_core/cues.py` is the single source of truth for turning a raw failure into a stable `cue_key`. It is stdlib-only, does no I/O and reads no environment, and it is not re-exported from `kb_core/__init__.py`. `build_cue(tool, tool_input, error_text, kb_project, cwd)` truncates the error to 4000 chars and composes `extract_target`, `target_class`, `normalize_error`, `resolve_cue_project` and `host_class_from_cwd` into a frozen `FailureCue`.

**What a cue is.** `extract_target` takes the Bash `command`, the file tools' `file_path`, NotebookEdit's `notebook_path`, or the Glob/Grep `pattern`. `target_class` reduces that to a program name for Bash (plus the subcommand for a fixed set of CLIs such as `git push` or `uv run`) or to `ext:<suffix>` for file tools. `normalize_error` strips ANSI codes, keeps the `Exit code N` line as `exit N`, keeps the first two lines that contain an error keyword (or else the last line), replaces uuids, timestamps, `kb-N` ids, `toolu_` ids, hex runs, paths, line numbers and multi-digit numbers with placeholders, lowercases the result and caps it at 200 chars. `resolve_cue_project` prefers the `.kb_project` value and otherwise uses the cwd basename, with any dispatch run-id suffix stripped.

**The key.** `cue_key` is the first 16 hex chars of sha256 over `CUE_NORMALIZER_VERSION`, tool, target_class, normalized_error and project. `host_class` (`darwin` / `linux` / `windows` / `unknown`) is stored as a reporting cut only and is deliberately not part of the key, so the same mistake counts as a repeat across machines. Any rule change in the module must bump `CUE_NORMALIZER_VERSION`; the version is stamped into the key, the table, the API response and the backfill report.

**Hook side.** `personal-kb-hook` handles the `PostToolUseFailure` event in `personal_kb_hook.events.post_failure`. It builds one `post_tool` event whose `event_id` is `cc:{session_id}:{tool_use_id}`, with `mode` set to `headless` when `HEADLESS_BUILD_ENGINE` is set and `project` taken from the `.kb_project` walk-up. The event carries only the scalar tool_input values (strings capped at 2000 chars) and the error capped at 4000 chars. It POSTs once with a 1.5 s timeout and writes zero stdout. Undeliverable events go to the hook drop log `~/.cache/personal_kb/event-drops.jsonl` (`paths.get_event_drop_log_path`), which is reset once it exceeds 256 KB.

**Server ingest — `POST /api/kb/event`.** This lives in `packages/kb-service/src/kb_service/routes/event_routes.py` and always returns HTTP 200 for a schema-valid body. A non-`post_tool` type returns `unsupported-type`, `is_error=false` returns `not-failure`, and a blank tool_name or error returns `missing-fields`. Otherwise the route calls `build_cue` and inserts one row into the service-DB table `failure_events` with `ON CONFLICT (event_id) DO NOTHING`; the response's `reason` is then `recorded` or `duplicate`. Every normalized column comes from the `FailureCue`. An empty normalized error is flagged with anomaly `empty_error`, and a Bash failure with no program class with `empty_bash_target`. The row is still inserted in both cases. A DB error is logged at WARNING and returned as `write-failed`, never as a 5xx. `ts` is normalized to UTC at second precision and falls back to `received_ts`.

**Heartbeat — `GET /api/kb/event/heartbeat`.** The `hours` parameter defaults to 24 and must be between 1 and 720. The endpoint groups the window's rows by (harness, mode, host) with a count, an anomaly count, the latest `received_ts` and `MAX(hook_version)`, and it returns the in-process `_OUTCOMES` counter of route reasons. It counts failures only, so a zero row cannot be told apart from a dead pipeline; check the hook drop log as the cross-check.

**Offline backfill — `packages/kb-service/scripts/failure_cue_baseline.py`.** This is a read-only report over Claude Code transcript `.jsonl` files: no DB and no network. Failures are `tool_result` blocks with `is_error: true`, and the script normalizes them with the same `build_cue`. Results starting with `<tool_use_error>`, `Permission to use `, `The user doesn't want to proceed` or `[Request interrupted by user` are excluded, because the live hook never sees them. Failures are deduplicated by `tool_use_id`.

**Repeat-rate definition.** For each cue, `s0` is the session with the minimum (earliest ts on the cue, session_id). A later session counts as a repeat when its earliest hit on the cue is at least `--min-gap-hours` (default 24) after `s0`'s, and `repeat_rate = repeat_sessions / sessions`. The report gives `aggregate_rate` over all cues and `aggregate_rate_ge2` over cues seen in at least 2 sessions. It also prints per-cue, per-ISO-week, per-host_class and per-mode tables. It warns of under-collapse when more than 90% of cues are singletons, and of over-collapse when one cue holds more than 20% of failures. Any future live-side rate over `failure_events` must use `WHERE is_interrupt = 0` to match.

## Prevention channels

Two channels deliver known corrections back into a session before the mistake repeats: a SessionStart **gotcha slice** (context) and a PreToolUse **soft gate** (a one-time deny). Neither makes any model or LLM call.

**Resolution format.** A resolution is an active, non-superseded, non-`mental_map` KB entry whose `hints` JSON carries a `resolution` dict: a required non-blank `corrected_fact`, plus optional `wrong_belief`, `evidence`, `cue` `{tool, target_class, args_prefix}`, `provenance` `{capture: deliberate|autonomous, grounding: observed|asserted}`, `observed_sessions` and `scope` (`project` or `global`). Malformed entries are skipped and counted, never raised. A resolution is observed-once when `capture` is `autonomous` and `observed_sessions < 2`. The parser lives in `packages/kb-service/src/kb_service/prevention.py`, whose module docstring is the contract of record. Surprise capture's distiller (see `### Distillation and delivery` under `## Surprise capture`) is the producer of autonomous resolutions; supersedes corrections still feed the slice too.

**Admission rule.** Only cues with `tool == 'Bash'` and a two-word `target_class` (such as `git push`) enter the gate index, capped at 200. File-tool classes (`ext:py`), one-word classes (`git`) and cue-less resolutions reach the slice only. An optional `args_prefix` narrows a cue: after the class tokens, flags are dropped and the remaining tokens must start with the prefix, so `git remote add` can be gated while `git remote -v` is not. Observed-once resolutions are excluded from the index until `observed_sessions >= 2`, whatever their grounding, unless `hints.surprise_capture.shape` is 1 or 2 (any other value or none keeps them out); they reach the slice either way.

**`GET /api/kb/prevention`.** Resolves the project with `kb_core.cues.resolve_cue_project(project, cwd)`, loads the project's resolutions (project-scoped first, then global) and up to 20 supersedes corrections, and returns the gate settings, the index (empty unless the gate is enabled), the slice (trusted resolutions first, then corrections, then at most 5 observed-once resolutions, within 20 items) and the rendered `slice_text` (at most 4000 chars). Any failure returns an inert HTTP 200 response. The payload also carries a top-level `surprise_capture` field (`off|shadow|on`, from `KB_SURPRISE_CAPTURE`), reported on every path including the no-project branch and the inert fallback. `diagnostics.index_excluded_observed_once` counts observed-once Bash two-word-class resolutions with observed grounding that were held out of the index (0 when the gate is off), and the `prevention_fetch` log line carries it plus the `index_ids`; an observed-once cue that still reaches the index logs a `tripwire=observed_once_in_index` WARNING.

**`POST /api/kb/prevention/decisions`.** Batch ingest (at most 500 rows) into the service-DB table `gate_decisions`, idempotent on `decision_id`. The `decision` enum has 7 members: `denied`, `would_deny`, `skipped_already_denied`, `skipped_cap`, `retry`, `armed` (one per successful SessionStart or Stop fetch) and `summary` (per-flush hook health counters). Gate decisions use a batch endpoint because PreToolUse forbids network: the hook logs to `gate-log-<session>.jsonl` and flushes at Stop. For the same reason `POST /api/kb/event` still answers `pre_tool` with `unsupported-type`.

**Gate semantics.** Each resolution is denied at most once per session, and a session gets at most 2 denies. Shadow mode (the default) records `would_deny` instead of denying, but consumes the same deny-once and cap budget, so would-deny counts equal what live denies would have been. After a real deny, only the next call to the same tool counts as the retry (`retry_changed_command` 1 or 0). A retry still pending at Stop is recorded as abandoned, with `retry_changed_command` NULL.

**`GET /api/kb/prevention/stats`.** Reports decision counts, armed sessions per host (a host absent here has a dead or unwired gate), pre_tool error totals and invariant violations (sessions over the cap, repeated denies of one resolution). The kill criterion is `retry_changed_share` = changed retries / non-abandoned retries: kill the gate if it is below 0.30. It is measured from real denies only, so it is None in shadow mode. `would_deny_precision` is the share of `would_deny` rows followed, in the same session, by a non-interrupt `failure_events` row with the same tool and target_class at or after the decision's ts.

## Surprise capture

The feature turns per-turn digests sent by the hook into detected surprises and distilled lessons. `KB_SURPRISE_CAPTURE` takes `off|shadow|on`, default `off`; any other value means `off`, and it is read on every call. The current value at processing time decides shadow vs on, and the stored per-row `capture_mode` is informational only.

### Turn digests

`POST /api/kb/turn` ingests one digest per turn (`TurnDigestRequest`). Caps: user_prompt 4000 chars, final_message 4000, assistant text 2000, tool target 500, tool-result excerpt 1500, at most 200 items; the whole body is at most 65536 bytes. Items are a discriminated union on `kind` (`assistant_text`, `tool_call`, `tool_result`). Unknown keys are ignored. The `event_id` must be exactly `<session_id>:<turn_index>` with no harness prefix (a mismatch is a 422), and ingest is idempotent on it: the first write wins, and a re-post with different content answers `duplicate-mismatch` (logged, stored row unchanged). The size check runs before the capture switch, so an oversize body gets 413 even when capture is off.

With `KB_SURPRISE_CAPTURE` unset or `off` the route answers `capture-off` and never touches the DB. The switch is also reported to the hook as the top-level `surprise_capture` field of `GET /api/kb/prevention`, including on the no-project branch and the inert fallback. The ingest path reads neither `KB_LISTENER_ENABLED` nor `PERSONAL_KB_LISTENER`.

Secrets are redacted before storage with `kb_core.ingest.safety.redact_secrets` over user_prompt, item text/target/excerpt and final_message only (a drift-guard test fails when a new string field is added). It uses six detect-secrets plugins plus four regex fallbacks, replaces a whole offending line with `[REDACTED:<type>]`, and replaces a whole field containing a private key with `[REDACTED:Private Key]`. Redacted values are cut back to their caps, which can cut a marker or the field tail (logged as `retruncated`). If detect-secrets is unavailable the digest is not stored (`redaction-unavailable`). Known misses: secrets with no key name or token shape, e.g. `curl -u user:pass` or an opaque value on its own line.

Rows go to the service-DB `turn_events` table. `anomaly` holds a comma-joined subset of `empty_bash_target_class`, `orphan_tool_result` (only when the digest is not truncated) and `empty_turn`. `processed_at` NULL is the single pending marker; `mark_turn_digests_processed` is the only writer, and a single-id call returns 1 only to the caller that flipped the row, so consumers can use it as an atomic claim. `capture_mode` is informational only.

`kb_service.turn_digest` exposes the consumer helpers: `surprise_capture_mode`, `list_pending_turn_digests`, `mark_turn_digests_processed`, `get_session_turn_digests` and `prune_turn_events`. Pruning deletes rows (pending or processed) received more than `TURN_EVENTS_RETENTION_DAYS` = 30 days ago (a constant, no env override). It schedules nothing itself; the drain's call to it is documented under `### Hook digest sender

At every `Stop` the hook calls `turn_digest.stop(payload)`, directly after `prevention.refresh(payload)` and before the listener gate and the headless listener skip. The only gate is the `surprise_capture` mode cached from the top level of `GET /api/kb/prevention` (`shadow` or `on`); `PERSONAL_KB_LISTENER` and `KB_LISTENER_HEADLESS` do not matter. The per-session counter in `turn-state-<session>.json` (`next_turn_index`, `last_uuid`) advances on every Stop whatever the mode, and `event_id` is `<session_id>:<turn_index>`. Other sessions' state files are removed after 31 days, longer than the server's 30-day `turn_events` retention, so a removed counter cannot reuse a retained `event_id`.

The window is read synchronously from the last 8 MiB of the transcript, walking newest to oldest until the first record that is either the previous digest's `last_uuid` or a human prompt. A human prompt is a `user` record that is not sidechain, meta or a compact summary, whose `origin.kind` (when present) is `human`, with no `tool_result` blocks and text not starting with a slash-command echo, `local-command-` or `[Request interrupted by user` prefix. Items are built in file order: non-empty assistant text (2000 chars), `tool_use` calls (target 500 chars, `target_class` from the full target via `cues_lite`) and `tool_result` blocks (a 746-char head, `\n[...]\n` and a 747-char tail when over 1500 chars). The final message comes only from `last_assistant_message`, since the transcript can lag.

The body has 14 keys and is limited to 64 KiB: items are dropped oldest first (`truncated` true), and when even an empty list does not fit the digest is dropped as `too_large`. The body goes to a temp file and a detached `python -m personal_kb_hook.turn_sender` POSTs it once with a 10 s timeout and no retry, then deletes the file. Failures land in `event-drops.jsonl` with op `turn_digest`; the local `turn-digest-log-<session>.jsonl` records one `stop` row per Stop and one `send` row per send, and is never POSTed. See `packages/personal-kb-hook/README.md` for the reasons and the heartbeat reconciliation.

### Detection`.

`GET /api/kb/turn/heartbeat` returns counts by (harness, mode, host) with truncated, redacted, anomaly and empty-project counts, the pending count and oldest pending `received_ts`, `sessions_with_gaps` (window-relative) and the per-process route outcomes. A 422 on `/api/kb/turn` is logged as `turn_event reason=invalid` with only the error location and type, never the input.

### Detection

`kb_service.surprise_worker.drain_once(pool, kb)` turns pending digests into detections. It reads the CURRENT `KB_SURPRISE_CAPTURE` value once per call, through `turn_digest.surprise_capture_mode()`, at processing time; the stored `capture_mode` decides nothing. The drain path ignores `KB_LISTENER_ENABLED` and `PERSONAL_KB_LISTENER`. In mode `off` it returns zeros with no DB access, no model call and no log line.

Every `shadow`/`on` pass, whether from the route or a worker cycle, first calls `prune_turn_events` (30-day retention, `TURN_EVENTS_RETENTION_DAYS`). It then reads every pending digest (`processed_at` NULL, the single pending marker) ordered by (session, turn_index), loads each session's stored digests once, and runs the three detectors on each pending digest. Only after detection does it claim the digest with a single-id `mark_turn_digests_processed` call. A claim that returns anything but 1 logs `surprise_drain tripwire=double_detect` and writes nothing for that digest. The claim sits outside the insert transaction, because the pool-level helper would deadlock on the SQLite pool's non-reentrant lock if called inside `pool.acquire()`. If the insert transaction then fails, the digest stays processed with no rows, an ERROR `surprise_drain lost_after_claim` is logged and the exception propagates; nothing un-claims it. No model call ever runs while a connection is held.

Shape 1 is deterministic (`detector_model` `rule:shape1`). Within each digest a Bash `tool_call` pairs with the `tool_result` that has the same `tool_use_id`; calls with no result are dropped. Each call is classified per command segment with `kb_core.cues.bash_segments` (split on `&&`, `||`, `;`, `|` and newlines; `cd` segments skipped; falling back to the call's `target_class` when no segment classifies), and classes in `SHAPE1_IGNORED_CLASSES` (cat, ls, head, tail, grep, rg, find, which, test, echo, pwd, stat, wc, file) are skipped. Shape 1 pairs on any shared segment class: a failed call is pending under every one of its classes, and a later success with the success in the current digest pairs with the first pending failure sharing a class (walking the success's classes in order), popping that failure from all its classes; the shared class is the pair's class and is stored as the optional `cue_target_class` key of `detector_output` (omitted when it equals the class of the whole wrong belief), which the shape-1 cue uses. The most recent failure wins, and a pair whose two commands are identical after strip is dropped. Its output is wrong_belief = failed command, corrected_fact = successful command, evidence_excerpt = the failure's excerpt, confidence 1.0.

Shapes 2 and 3 make one model call each, sequentially, with `SURPRISE_DETECTOR_SYSTEM`. Shape 2 asks whether the human's message corrects a factual claim from the assistant's previous turn. It is skipped as `no_prev` (turn 0), `turn_gap` (previous turn missing), `no_user_prompt`, or `no_prev_text` (previous turn has no final message and no assistant text). Shape 3 asks whether a later tool result in the same turn contradicts an earlier assistant claim. It is skipped as `no_claim_before_result` unless a non-empty assistant text precedes some tool result. The detector replies with one JSON object whose five keys are `surprise`, `wrong_belief`, `corrected_fact`, `evidence_excerpt` and `confidence`. A verdict becomes a candidate only if confidence is at least the threshold and the evidence excerpt is grounded. Grounded means it occurs, after whitespace collapse and casefold, verbatim in the human message (shape 2) or in a tool-result excerpt (shape 3).

The detector uses its own Anthropic client, cached per model and never closed: `KB_SURPRISE_DETECTOR_MODEL`, default `claude-sonnet-5-5`. It never uses the KB's synthesis or query providers, and `KB_QUERY_PROVIDER` does not affect it. `KB_SURPRISE_MIN_CONFIDENCE` (default 0.7) is read once per pass; a non-numeric value or one outside [0, 1] falls back to 0.7 with a `bad_min_confidence` WARNING. The effective threshold is recorded on each model-call row as `details.min_confidence`. A failed or absent model call still marks the digest processed and is not retried.

`surprise_candidates` columns: id, shape (1-3), session_id, project, turn_event_ids (JSON text; for shape 1 the failure and success digests, shape 2 the previous and current, shape 3 the current), detector_model, detector_output (JSON text with the four verdict fields), status, entry_id and created_at. Statuses: `pending` = detected in mode on, awaiting distillation; `shadow` = detected in mode shadow, terminal, never distilled and never written to the KB, so flipping shadow to on does not replay shadow-era candidates; `rejected` = no KB write for any terminal distill outcome (not only a not-durable judgement; see `surprise_distillations.outcome`); `written` = a new KB entry was created (entry_id set); `merged` = folded into an existing entry, including `same_session` (entry_id set). The drain only writes `pending` or `shadow`.

`surprise_detections` is the decision log. It has one row per shape per claimed digest, negatives included, the way `listener_decisions` and `gate_decisions` work. Its outcomes are `candidate`, `not_applicable`, `no_llm`, `llm_error`, `unparseable`, `invalid_fields`, `no_surprise`, `low_confidence` and `ungrounded`, with a reason, the model, confidence, `candidate_id` (set iff the outcome is `candidate`), a raw response excerpt (up to 2000 chars), prompt/response sizes and latency. `details` carries the shape-1 counters or `min_confidence`, plus the digest flags `truncated`, `out_of_order` and `turn_gap`. Every row records `detector_version` = `SURPRISE_DETECTOR_VERSION`, which is bumped on any change to the prompts, parser, grounding, default threshold or shape-1 rules. Each pass logs one INFO heartbeat line, `surprise_drain mode=... pruned=... digests=... llm_calls=...`.

`POST /api/kb/surprise/drain` runs one pass. It uses the same auth as the other `/api/kb` routes (any authenticated user, not admin-only) and takes no body. It returns `digests_processed`, `candidates` (the rows inserted or distilled in this pass), `entries_written` and `entries_merged`; in mode off that is 200 with zeros and empty lists. The route and the background worker share one in-process `asyncio.Lock` (`app.state.surprise_drain_lock`), and the per-digest claim covers separate processes, so no digest is processed twice.

`SurpriseCaptureWorker` runs `drain_once` every 60 seconds (`SURPRISE_WORKER_POLL_SECONDS`) under that lock. A failed cycle is logged and the loop continues. It starts at lifespan only when the mode is shadow or on AND `KB_DATABASE_URL` points at a Postgres kb-core backend. On a SQLite kb-core backend (local mode, kb-bench) it never starts, and digests are processed only by `POST /api/kb/surprise/drain`. The mode is read at startup, so flipping it from off needs a restart for the worker; the route needs none.

`distill_candidates(pool, kb, candidates, mode)` is the distillation step described in `### Distillation and delivery`. The drain calls it only in mode on, with every `pending` candidate in id order, and it returns the KB entries written and merged. In shadow the pass ends after inserting its rows.

### Distillation and delivery

`surprise_worker.distill_candidates(pool, kb, candidates, mode)` turns pending candidates into autonomous `lesson_learned` entries. It runs only in mode `on`, as decided by the CURRENT `KB_SURPRISE_CAPTURE` value `drain_once` reads at processing time; for any other mode, or an empty list, it returns at once with no DB, KB or model access and no log line. In `shadow` the drain stores candidates with status `shadow` and stops with no distiller call, and those candidates are never replayed when the mode flips to `on`. The distill path ignores `KB_LISTENER_ENABLED` and `PERSONAL_KB_LISTENER`.

The distiller has its own Anthropic client from `get_distiller_llm()`, cached per model and never closed: `KB_SURPRISE_DISTILL_MODEL`, default `claude-sonnet-5-5`, read on every call. It never uses `get_detector_llm`, `KB_SURPRISE_DETECTOR_MODEL` or the KB's query/synthesis providers, and `KB_QUERY_PROVIDER` does not affect it. It is None only when the Anthropic client cannot be imported.

Candidates are processed sequentially in id order through S0-S13. S0 re-reads the candidate's status and skips anything not `pending` with a `tripwire=double_distill` WARNING (no row). Then: an empty project is `no_project`; a wrong belief containing `[REDACTED:` is `redacted` (reason `wrong_belief`), so a redacted secret can never become a fake recurrence key; an evidence excerpt containing `KB soft gate (deny once` is `gate_induced`, so the gate's own denies never manufacture a lesson or a recurrence. A shape-1 candidate gets a cue (S4). S5 looks for an exact match among the project's resolutions (observed-once included): the same wrong belief after whitespace collapse and casefold, and an equal cue (tool, target_class), with the first match in load order winning. An exact match goes straight to the match step with no model call. Otherwise, with no distiller LLM the candidate stays `pending` (counted `no_llm`, no row, no WARNING), so exact merges still happen while the model is unavailable.

The model call (S7) sends `SURPRISE_DISTILLER_SYSTEM` and a prompt built from the shape description, the project and the detector's wrong belief, corrected fact and evidence, followed by `DISTILLER_INSTRUCTIONS` (decide durability; `why` for a not-durable reply; length limits; never include secrets) and `DISTILLER_SCHEMA_LINE` `{"durable": true|false, "why": str, "short_title": str, "long_title": str, "corrected_fact": str, "lesson": str}`. The parser has four rejects: `llm_error` (no reply; reason `exception` or `none`), `unparseable`, `not_durable` (anything but `durable: true`; reason is the `why`, up to 200 chars) and `invalid_fields` (a missing or blank title, fact or lesson). Accepted fields are cut to 80/200/500/2000 chars. A verdict containing `[REDACTED:` is `redacted` (reason `verdict`). Then the titles and details are scanned with `detect_secrets_in_content` unless `skip_safety` is set (`secret_detected`, reason the finding types), and the resolution is validated with `validate_and_stamp_resolution(is_machine=True)` (`invalid_resolution`, reason the error code).

The resolution written per candidate is `{corrected_fact, wrong_belief, evidence, provenance, observed_sessions: 1, scope: "project"}`, with provenance `autonomous/observed` and `event_id` = the candidate's last turn event (the triggering digest; `autonomous/asserted` when there is none). Only shape 1 carries a cue, `{tool: "Bash", target_class}`, and only when that class is a fixed point of `kb_core.cues.target_class`; shapes 2 and 3 never do. Scope is always `project`. A promoted shape-1 cue has no `args_prefix`, so once gated it also denies the first corrected command of that class in a session (a retry passes).

S11 then runs `kb.find_near_duplicates` at `KB_NEAR_DUPLICATE_FLOOR` (0.88; a malformed value falls back to 0.88 with a `bad_near_duplicate_floor` WARNING). A same-project hit is a cosine match on the top candidate. With no embedder or a failed search it fails open and writes a new entry, so local installs without embeddings cannot cosine-merge.

The match step (S12) never writes in these cases, all `covered`: `missing` (the entry is gone), `no_resolution` (a plain entry), `deliberate` (any capture other than autonomous, including a resolution without provenance), `global` (scope global) and `cue_mismatch` (a different cue, reachable only via cosine). If the candidate's session is already behind the entry, from `hints.surprise_capture.sessions` or the session part of `provenance.event_id`, the outcome is `same_session` with no write. Otherwise it is `merged`: one `kb.update` (enrich off, `updated_by` `surprise-capture`, `change_reason` starting `surprise_capture:`) raises `resolution.observed_sessions` by one, leaves every other resolution key (evidence included) unchanged, and appends the session, candidate id and triggering event id to `hints.surprise_capture`. A new entry (S13, `written`) is a `lesson_learned` with contributor `surprise-capture`, tags `surprise-capture` and `shape-<n>`, confidence 0.7, enrich off, and no supersedes or distinct_from.

`hints.surprise_capture` holds `shape` (the first candidate's shape as an int, written with the new entry and never overwritten by a merge), `sessions` and `candidate_ids` (newest 100 each) and `event_ids`: the triggering digest's event id, recorded on write and appended on every merge, newest 20 kept. Recurrence promotion follows from the D7 rule in `## Prevention channels`: an autonomous resolution seen in one session is observed-once and reaches the slice only, labelled unconfirmed; the merge from a second session lifts `observed_sessions` to 2 and its Bash two-word cue enters the soft-gate index. The exception is first sighting: shape 1 (detected deterministically from tool output) and shape 2 (the user's own correction) are trusted at once, so an observed-once resolution whose stored shape is 1 or 2 is gate-eligible immediately (its slice label still reads unconfirmed), while shape 3 (a model judgment) and an unknown shape wait for the second sighting.

There are 14 outcomes: `written`, `merged`, `same_session`, `covered`, `not_durable`, `redacted`, `gate_induced`, `llm_error`, `unparseable`, `invalid_fields`, `invalid_resolution`, `secret_detected`, `no_project` and `kb_error` (any other exception, reason the exception class; the batch continues). The candidate status becomes `written` for written, `merged` for merged and same_session, and `rejected` for every other (no-write) outcome. Each decision is one `surprise_distillations` row (no mode column, since every row is an `on` decision) with the reason, the new and matched entry ids, match kind, similarity, near-duplicate status and floor, cue class, observed_sessions before/after, the parsed `verdict` JSON for audit (NULL for `secret_detected`), the distiller model and `distiller_version` = `SURPRISE_DISTILLER_VERSION`, a redacted raw response excerpt, prompt/response sizes and latency. Each row logs one INFO `surprise_distill decision ...` line (no lesson text) and failure outcomes add a `distill_failed` WARNING; each call logs one `surprise_distill summary ...` heartbeat with per-outcome counters, `promoted`, `no_llm`, `double_distill` and `aborted`. `SURPRISE_DISTILLER_VERSION` is bumped on any change to the prompt, parser, builders, match rules or step order.

Persistence assumes a single writer per service DB (the drain lock). The status UPDATE is conditional on `pending`; losing it logs the double_distill tripwire and inserts no row. A failure inside the persistence transaction logs `persist_failed` and propagates. The KB write and the candidate update are not atomic across the two databases. No model call runs while a connection is held.
