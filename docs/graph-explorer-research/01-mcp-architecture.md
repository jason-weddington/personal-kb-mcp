# MCP Protocol Constraints & Web Server Embedding

**Researcher**: mcp-architect
**Date**: 2026-03-05

---

## Can we embed a web server inside an MCP server?

**Yes, cleanly.** The MCP server uses stdio transport (stdout reserved for JSON-RPC), but a web server on `localhost:PORT` uses TCP — no conflict. Two viable approaches:

### Approach A: Embedded (same process)

Spawn an `aiohttp` web server on a background asyncio task within the same process. It shares the same event loop and DB connection directly. The lifespan context already has everything wired up (`db`, `store`, `embedder`, `query_llm`, etc.).

**Advantages:**
- Direct access to the live agent loop — can hook `event_callback` into `agentic_query()` for real-time streaming
- No DB sharing concerns — same connection
- Single process to manage

**Constraints:**
- stdout is reserved for MCP stdio transport — the web server must bind to a TCP port, never write to stdout
- All logging already goes to stderr (established convention in the project)
- The aiohttp server runs as a background task, not blocking the MCP event loop

### Approach B: Separate process

A standalone `uv run personal-kb-explorer` command that opens the same SQLite DB independently.

**Advantages:**
- Clean process isolation
- Can run without the MCP server

**Constraints:**
- SQLite WAL mode needed for concurrent readers (already enabled in `connection.py:51`)
- Cannot observe live query execution — can only read DB state
- Must load sqlite-vec extension independently
- Must replicate config parsing (DB path, embedding settings)

### Recommendation

**Embedded for the full vision** (live query streaming). **Separate process works for Phase 1** (static graph view only — no live queries, just read the DB and render).

---

## The `launch_explorer` tool pattern

A `kb_explore` MCP tool that opens the user's browser is straightforward:

```python
import webbrowser
webbrowser.open("http://localhost:PORT")
```

Python's `webbrowser` module is stdlib, works on macOS/Linux/Windows. The tool would:
1. Start the web server if not already running (background task)
2. Open the browser
3. Return a message like "Explorer opened at http://localhost:8765"

### Alternative: Self-contained HTML file

For Phase 1, even simpler — the tool generates a self-contained HTML file with graph data inlined as JSON:
1. Query all graph nodes/edges from DB
2. Render into an HTML template with force-graph (CDN)
3. Write to a temp file
4. Return the file path (or open with `webbrowser.open(f"file:///{path}")`)

**Zero server needed.** The agent calls the tool, the HTML file opens in the browser.

---

## MCP primitives — could we use resources/sampling?

**Resources**: MCP resources are for exposing data to the client (agent), not to humans in a browser. A `graph://` resource URI could expose graph data, but the MCP client (Claude Code, Cursor, etc.) wouldn't render it visually.

**Sampling**: MCP sampling lets the server request LLM completions from the client. Not relevant for visualization.

**Conclusion**: MCP primitives don't help here. The web UI is a separate concern that happens to share the same data layer.

---

## Existing MCP servers with web UIs

No widely-known MCP servers embed web UIs directly. However:
- Several MCP servers expose HTTP endpoints alongside stdio (e.g., for webhooks)
- The FastMCP framework supports SSE transport as an alternative to stdio — this could serve both MCP clients and web browsers
- **GraphRAG Workbench** (Microsoft) is the closest prior art — a web app that visualizes a knowledge graph with a chat panel, though it's not an MCP server

---

## SQLite concurrent access details

WAL mode (already enabled) allows:
- Multiple simultaneous readers
- One writer at a time
- Readers see a consistent snapshot (won't see partial writes)
- `.db-wal` file must be accessible to both processes (same filesystem)

For a read-only web server:
```python
conn = await aiosqlite.connect(db_path)
await conn.execute("PRAGMA journal_mode=WAL")
await conn.execute("PRAGMA query_only=ON")  # safety
```

For Postgres deployments, the asyncpg connection pool handles concurrent access natively.

---

## Architecture recommendation

```
Phase 1: Self-contained HTML file (no server)
  kb_explore tool → query graph → render HTML → open browser

Phase 2: Embedded web server (same process)
  MCP server starts aiohttp on background task
  Web server shares DB connection + can hook into agent loop
  kb_explore tool → ensure server running → open browser

Phase 3: Full streaming
  SSE endpoint streams agent loop events
  Frontend animates graph traversal in real-time
```
