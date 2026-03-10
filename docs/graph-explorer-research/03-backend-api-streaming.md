# Backend API & Streaming Architecture

**Researcher**: backend-designer
**Date**: 2026-03-05

---

## Data Model (from codebase analysis)

### Graph tables (`schema.py`)

- `graph_nodes`: `node_id TEXT PK`, `node_type TEXT`, `properties JSON`, `created_at TEXT`
- `graph_edges`: `source TEXT FK`, `target TEXT FK`, `edge_type TEXT`, `properties JSON`, `UNIQUE(source, target, edge_type)`
- Indexes on source, target, and edge_type

### Node types (open — not a closed enum)

- `entry` — knowledge entries (`kb-XXXXX`)
- `tag` — tag nodes (`tag:python`)
- `project` — project scoping (`project:personal-kb`)
- `person` — person references (`person:jason`)
- `tool` — tool references (`tool:sqlite`)
- `concept`, `technology` — LLM-enriched entities
- `note` — ingested file/URL sources

### Edge types (also open)

- Deterministic: `has_tag`, `in_project`, `supersedes`, `references`, `mentions_person`, `uses_tool`, `related_to`, `extracted_from`
- LLM-enriched: arbitrary types like `depends_on`, `implements`, with `{"source": "llm"}` in properties

---

## Proposed API Endpoints

### REST (Graph Data)

```
GET /api/graph
  Query: ?scope=project:X&depth=2&node_types=entry,tag,concept
  Returns: { nodes: [...], edges: [...] }
  Uses: entries_for_scope() + get_neighbors() BFS

GET /api/graph/nodes
  Query: ?type=tag&limit=200
  Returns: { nodes: [{id, type, properties, connection_count}] }
  Uses: get_graph_vocabulary()

GET /api/graph/nodes/:node_id/neighbors
  Query: ?edge_types=has_tag,references&direction=both&limit=50
  Returns: { neighbors: [{id, edge_type, direction}] }
  Uses: get_neighbors()

GET /api/graph/path/:source/:target
  Query: ?max_depth=4
  Returns: { path: [{source, edge_type, target}] }
  Uses: find_path()

GET /api/entries/:entry_id
  Returns: Full KnowledgeEntry JSON
  Uses: get_entry()

GET /api/entries
  Query: ?scope=tag:python&order_by=created_at&limit=20
  Returns: { entries: [...] }
  Uses: entries_for_scope()

GET /api/search
  Query: ?q=sqlite+performance&project_ref=X&limit=10
  Returns: { results: [{entry, score, match_source}], filtered_count }
  Uses: hybrid_search()
```

### Streaming (Query Events)

```
POST /api/query/stream
  Body: { question: "...", mode: "explore"|"ask" }
  Response: SSE stream of events
```

---

## Agent Loop Integration Points

The ReAct agent loop (`graph/agent.py:252-356`) is the core streaming target.

### Exact flow with event injection points

```
agentic_query()
  │
  ├─ 1. Fast-path: hybrid_search()
  │     → if top_score >= 0.030, return immediately
  │     EVENT: {type: "fast_path", top_score: 0.034, result_count: 5}
  │
  ├─ 2. Build seed message with question + weak results
  │     EVENT: {type: "agent_started", question: "...", seed_results: 3}
  │
  └─ 3. Loop (up to 4 iterations):
       │
       ├─ LLM call: llm.generate(prompt, system=_AGENT_SYSTEM_PROMPT)
       │   EVENT: {type: "llm_thinking", turn: 1}
       │
       ├─ Parse response → _ToolCall or _FinalAnswer
       │
       ├─ If _FinalAnswer:
       │   EVENT: {type: "done", entry_ids: [...], reasoning: "..."}
       │
       ├─ If _ToolCall:
       │   EVENT: {type: "tool_call", tool: "hybrid_search", args: {...}}
       │   │
       │   └─ _dispatch_tool() executes one of 6 tools:
       │       - hybrid_search
       │         EVENT: {type: "search_results", query: "...", count: 5, entries: [...]}
       │       - graph_neighbors
       │         EVENT: {type: "graph_traverse", node: "kb-00001", neighbors: 7}
       │       - list_graph_nodes
       │         EVENT: {type: "vocab_browse", types: {...}}
       │       - decision_chain
       │         EVENT: {type: "chain_found", entry: "kb-00001", chain_length: 3}
       │       - scope_entries
       │         EVENT: {type: "scope_results", scope: "project:X", count: 12}
       │
       └─ If parse failure:
           EVENT: {type: "parse_error", turn: 2}
```

### Proposed implementation

Add an optional `event_callback` parameter to `agentic_query()`:

```python
async def agentic_query(
    db, embedder, llm, question, scope=None,
    event_callback: Callable[[dict], Awaitable[None]] | None = None,
) -> AgentResult:
```

The existing code works unchanged when `callback=None`. The web layer provides a callback that pushes events to SSE:

```python
# Around agent.py line 334:
if event_callback:
    await event_callback({
        "type": "tool_call",
        "tool": parsed.tool,
        "args": parsed.args,
        "turn": turns_used,
    })

result_text = await _dispatch_tool(parsed, db, embedder)

if event_callback:
    await event_callback({
        "type": "tool_result",
        "tool": parsed.tool,
        "result_length": len(result_text),
        "entry_ids": _ENTRY_ID_RE.findall(result_text),
    })
```

~20 lines of changes total in agent.py.

---

## Streaming Event Catalog

| Event Type | When | Key Fields |
|---|---|---|
| `query_started` | Request received | `question`, `strategy`, `mode` |
| `fast_path_check` | After initial search | `top_score`, `threshold`, `passed` |
| `fast_path_result` | If fast-path succeeds | `entry_ids`, `scores` |
| `agent_started` | Agent loop begins | `seed_count`, `max_turns` |
| `llm_thinking` | Before LLM call | `turn`, `prompt_length` |
| `tool_call` | Agent picks a tool | `turn`, `tool`, `args` |
| `tool_result` | Tool returns | `turn`, `tool`, `result_count`, `entry_ids` |
| `graph_traversal` | graph_neighbors returns | `from_node`, `neighbors[]` |
| `parse_error` | LLM output unparseable | `turn`, `raw_snippet` |
| `agent_done` | Agent returns answer | `entry_ids`, `reasoning`, `turns_used` |
| `coverage_check` | Synthesis coverage | `has_gaps`, `suggested_query` |
| `synthesis_started` | LLM synthesis begins | `entry_count` |
| `synthesis_done` | Answer complete | `answer`, `citations` |
| `entries` | Entry data payload | `entries[]` (full objects) |
| `stream_end` | Stream closing | `total_time_ms`, `total_turns` |
| `error` | Any failure | `message`, `recoverable` |

---

## Query Routing

### kb_ask flow

1. Strategy dispatch: auto, decision_trace, timeline, related, connection
2. For auto: agentic loop or single-shot planner → hybrid_search + graph expansion
3. Returns formatted text with full entry details

### kb_summarize flow

1. `retrieve_entries()` — shares same retrieval path as kb_ask
2. Optional `assess_coverage()` — checks if entries answer the question
3. `_synthesize()` — LLM generates natural language answer with citations
4. Falls back to raw entry list if LLM unavailable

### Web UI routing

- **"Explore" mode** → kb_ask semantics (returns entries + graph context, frontend visualizes traversal)
- **"Ask" mode** → kb_summarize semantics (returns synthesized answer, cited entries shown as cards)
- Both share `retrieve_entries()` as common backbone. Divergence is in output.

---

## SQLite Concurrent Access

WAL mode (already enabled in `connection.py:51`):
- Multiple simultaneous readers ✓
- One writer at a time (the MCP server)
- Readers see consistent snapshots
- `.db-wal` file must be accessible to both processes

For a read-only web server:
```python
conn = await aiosqlite.connect(db_path)
await conn.execute("PRAGMA journal_mode=WAL")
await conn.execute("PRAGMA query_only=ON")
```

For Postgres deployments, asyncpg pool handles concurrent access natively.

---

## Integration Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Web Browser    │────▶│   Web Server     │────▶│   SQLite DB │
│  (Svelte + FG)  │◀────│  (aiohttp/FastAPI)│◀────│  (WAL mode) │
│                  │ SSE │                  │     │             │
└─────────────────┘     └────────┬─────────┘     └──────┬──────┘
                                 │                       │
                                 │ imports               │ shared
                                 ▼                       │ file
                        ┌────────────────┐               │
                        │  personal_kb   │───────────────┘
                        │  (as library)  │
                        │  - hybrid.py   │
                        │  - agent.py    │
                        │  - queries.py  │
                        └────────────────┘
                                 │
                        ┌────────┴────────┐
                        │   MCP Server    │
                        │  (stdio, works  │──── same DB file
                        │   unchanged)    │
                        └─────────────────┘
```

**Key insight**: The web server imports `personal_kb` as a library. All query functions (`hybrid_search`, `agentic_query`, `get_neighbors`, etc.) are standalone async functions that take a `Database` object — NOT tied to FastMCP. The web server creates its own `Database` and calls these functions directly.

No code duplication needed.

---

## What would need to change in personal_kb

Minimal changes:

1. **`graph/agent.py`**: Optional `event_callback` param to `agentic_query()` and `_dispatch_tool()`. ~20 lines.
2. **`tools/kb_summarize.py`**: Optional callback in `summarize_question()` for coverage/synthesis events.
3. **`db/connection.py`**: Expose `create_readonly_connection()` or document read-only usage.
4. **No changes to**: schema.py, queries.py, hybrid.py, builder.py, models/, formatters.py.

Everything is already well-factored for this.

---

## Framework Recommendation

**FastAPI with Starlette SSE**:
- Same Python async ecosystem
- `StreamingResponse` for SSE is trivial
- Can import personal_kb modules directly
- uvicorn for serving
- Static file serving for frontend build

```python
@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    async def event_generator():
        yield sse_event("query_started", {"question": request.question})

        queue = asyncio.Queue()
        async def emit(event: dict):
            await queue.put(event)

        # Run agent query in background task
        task = asyncio.create_task(
            agentic_query(db, embedder, llm, request.question, event_callback=emit)
        )

        while not task.done() or not queue.empty():
            event = await queue.get()
            yield sse_event(event["type"], event)

        result = task.result()
        yield sse_event("entries", serialize_entries(result.entries))
        yield sse_event("stream_end", {"total_time_ms": elapsed})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```
