# personal-kb — Codebase Guide

## Code Organization

```
src/personal_kb/
├── __init__.py
├── __main__.py              # Entry point: personal-kb CLI
├── config.py                # 28 environment variables, typed getters
├── server.py                # FastMCP server, lifespan, tool registration
├── preflight.py             # CWD-based project context injection
├── confidence/
│   └── decay.py             # Exponential decay, half-lives per entry_type
├── db/
│   ├── backend.py           # Database protocol (abstract interface)
│   ├── sqlite_backend.py    # aiosqlite + sqlite-vec implementation
│   ├── postgres_backend.py  # asyncpg + pgvector implementation
│   ├── connection.py        # Backend factory (SQLite vs Postgres)
│   ├── schema.py            # Table definitions, migration
│   ├── queries.py           # Reusable query functions
│   └── iam_auth.py          # AWS RDS IAM token signing (boto3)
├── search/
│   ├── hybrid.py            # RRF fusion of FTS + vector results
│   ├── fts.py               # BM25 full-text search (FTS5 / tsvector)
│   ├── vector.py            # KNN vector search (sqlite-vec / pgvector)
│   └── embeddings.py        # Ollama embedding client
├── graph/
│   ├── agent.py             # ReAct agent loop (kb_ask agentic strategy)
│   ├── enricher.py          # LLM entity extraction and relationship edges
│   ├── builder.py           # Deterministic edges from tags/project/hints
│   ├── queries.py           # BFS traversal, path-finding, supersedes chain
│   └── planner.py           # Query plan: strategy, scope, search query
├── ingest/
│   ├── ingester.py          # Main pipeline: FileIngester, FileResult
│   ├── chunker.py           # Markdown-aware content chunking
│   ├── extractor.py         # LLM extraction: ExtractedEntry list
│   ├── safety.py            # Secret scanning, PII redaction
│   ├── dedup_agent.py       # KB-aware deduplication check
│   └── html_extract.py      # Trafilatura HTML content extraction
├── llm/
│   ├── provider.py          # LLMProvider protocol (generate, generate_chat)
│   ├── anthropic.py         # Claude Haiku-4-5 + Sonnet 4.6
│   ├── bedrock.py           # AWS Bedrock SDK (async-native)
│   ├── ollama.py            # Ollama chat/completion client
│   └── json_parser.py       # Robust JSON extraction from LLM output
├── models/
│   ├── entry.py             # KnowledgeEntry, EntryType enum
│   ├── search.py            # SearchQuery, SearchResult
│   └── version.py           # Version metadata
├── store/
│   └── knowledge_store.py   # KnowledgeEntry CRUD, ID generation
├── tools/                   # One file per MCP tool
│   ├── kb_search.py
│   ├── kb_ask.py
│   ├── kb_get.py
│   ├── kb_store.py
│   ├── kb_store_batch.py
│   ├── kb_ingest.py
│   ├── kb_ingest_url.py
│   ├── kb_summarize.py
│   ├── kb_explore.py
│   ├── kb_feedback.py
│   ├── kb_maintain.py
│   ├── kb_list.py           # kb_list_projects, kb_list_contributors, kb_list_teams
│   ├── kb_bulk_update.py
│   ├── kb_preflight.py
│   ├── formatters.py        # Compact output formatting for MCP responses
│   ├── coverage.py          # Coverage check for kb_summarize
│   └── ttl.py               # TTL string parsing (7d, 24h, 2w)
├── explorer/
│   ├── graph_data.py        # Force-graph data extraction from DB
│   ├── renderer.py          # HTML page rendering
│   └── static/              # Explorer frontend assets
└── web/
    └── cli.py               # personal-kb-web entry point
```

Tests mirror source layout:
```
tests/
├── confidence/    # decay tests
├── db/            # backend tests
├── eval/          # search quality baselines (see testing.md)
├── explorer/      # graph explorer tests
├── graph/         # agent, enricher, builder, queries, planner tests
├── ingest/        # pipeline stage tests
├── integration/   # end-to-end integration tests
├── llm/           # provider tests
├── search/        # FTS, vector, hybrid tests
├── store/         # CRUD tests
└── tools/         # per-tool MCP handler tests
```

## Style Guide

**Python version**: 3.13+ required. Use `from __future__ import annotations` in all source files
for deferred evaluation of type hints.

**Imports**: `isort` ordering enforced by ruff (`I` rules). Standard lib → third-party → local.
Type-only imports in `if TYPE_CHECKING:` blocks (enforced by `TCH` rules).

**Line length**: 100 characters. `ruff` + `ruff-format` enforce formatting.

**Naming**: PEP 8 (enforced by `N` rules). Module-level constants in `UPPER_SNAKE_CASE`.
Private helpers prefixed with `_`.

**Docstrings**: Google convention (`D` rules). Required on all public functions, classes, and
modules. Not required on test functions (D rules suppressed under `tests/`).

**Type annotations**: All function signatures annotated. `mypy` strict mode enforced via
pre-commit hook (`mypy src/`). `Any` usage discouraged; use `Protocol` or specific union types.

**Return types**: Always `-> ReturnType`. `-> None` for functions with no return value.

**Logging**: All log output goes to `stderr`. `stdout` is reserved for MCP stdio transport.
The server configures logging at startup (`logging.basicConfig(..., stream=sys.stderr)`).
Log level controlled by `KB_LOG_LEVEL` (default `WARNING`).

**Error handling**: Raise specific exceptions from the standard library. Never swallow
exceptions silently. LLM clients return `None` (not raise) for availability failures.

**Async**: All database and LLM operations are async. Use `aiosqlite` for SQLite, `asyncpg`
for PostgreSQL. Do not block the event loop with synchronous I/O.

**Pre-commit hooks**: trailing-whitespace, end-of-file-fixer, check-yaml, check-toml,
gitleaks (secret detection), ruff (lint + format), conventional-commit (main only),
semantic-release (post-commit on main), vulture (dead code, 80% min confidence), mypy,
coverage (pre-push, 80% threshold, excludes `@pytest.mark.eval` tests).

## Common Patterns

### Database operations

Always use the `db` object from the FastMCP lifespan context. Never create a database
connection directly in tool handlers.

```python
# In a tool handler (via lifespan context)
db: Database = ctx.request_context.lifespan_context["db"]
rows = await db.execute("SELECT id FROM knowledge_entries WHERE is_active=1", ())
```

For multi-step operations that must be atomic, use the `transaction()` context manager:

```python
async with db.transaction():
    await db.execute("INSERT INTO knowledge_entries ...", params)
    await db.execute("INSERT INTO graph_edges ...", edge_params)
```

Note: Postgres transaction wrapper is not yet implemented (see ROADMAP.md M4). Multi-step ops
on Postgres are currently non-atomic.

### Entry ID generation

Entry IDs are generated atomically via `db.next_sequence_value()`. This returns the next
integer from a monotonic sequence and is safe under concurrency. The integer is formatted as
`f"kb-{n:05d}"` producing the `kb-XXXXX` format.

Advisory locks ensure atomicity on Postgres multi-instance deployments.

### Graceful LLM degradation

All LLM clients implement `LLMProvider`:

```python
result = await llm.generate(prompt)
if result is None:
    # LLM unavailable — skip enrichment or fall back
    return
```

Never assume an LLM client is available. Always handle `None` returns.

### Tool registration pattern

Each tool lives in its own file with a `register_kb_*` function:

```python
def register_kb_search(mcp: FastMCP, prefix: str = "") -> None:
    tool_name = f"{prefix}kb_search"

    @mcp.tool(name=tool_name)
    async def kb_search(ctx: Context, query: str, ...) -> str:
        db = ctx.request_context.lifespan_context["db"]
        ...
```

The `prefix` is `""` (default), `"personal_"`, or `"team_"` based on `KB_INSTANCE_ROLE`.

### Compact output formatting

Tool responses are formatted by `tools/formatters.py` for token efficiency. Search results
use a compact table: entry ID, type badge, short title, long title, score. Full content is
only returned by `kb_get` (explicitly requested).

### Configuration access

Import typed getters from `config.py` rather than accessing `os.environ` directly:

```python
from personal_kb.config import get_db_path, is_manager_mode, get_embedding_dim
```

All getters have defined defaults. `KB_MANAGER` gating uses `is_manager_mode()`.

### `uv run --frozen` in hooks

Pre-commit hooks that call `uv run` use `--frozen` to prevent uv from rebuilding the package
mid-hook. Rebuilding can cause spurious "files were modified by pre-commit" failures.

## Testing

See `docs/testing.md` for the full testing guide. Quick reference:

```bash
uv run pytest                           # run all tests (not eval)
uv run pytest -m "not eval" --cov       # with coverage report
uv run pytest tests/search/             # specific module
uv run pytest tests/eval/test_baseline.py -s  # regenerate search baseline
```

Test files use `from __future__ import annotations`, test classes with `Test` prefix,
`-> None` on all test methods, `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed).

## Developer Onboarding

**Prerequisites**: Python 3.13+, uv, Ollama (optional but recommended).

**Setup**:
```bash
git clone <repo>
cd personal-kb-mcp
uv sync
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg \
    --hook-type post-commit --hook-type pre-push
```

**Running locally**: `uv run personal-kb` (stdio MCP server, connect via Claude Code config).

**Adding a new tool**:
1. Create `src/personal_kb/tools/kb_<name>.py` with `register_kb_<name>(mcp, prefix)`.
2. Import and call `register_kb_<name>` in `server.py`.
3. Add tests in `tests/tools/test_kb_<name>.py`.
4. Update `docs/api-docs.md` with the tool's parameters and behavior.

**Adding a new LLM provider**:
1. Implement the `LLMProvider` protocol in `src/personal_kb/llm/<provider>.py`.
2. Add the provider name to `_create_llm()` in `server.py`.
3. Add the provider to `KB_EXTRACTION_PROVIDER` / `KB_QUERY_PROVIDER` accepted values.

**Schema migrations**: Schema changes go in `db/schema.py`. The schema runs at startup via
`executescript()`. For additive changes (new columns, new tables), add `IF NOT EXISTS` /
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` guards so existing databases are migrated safely.

## Anti-patterns

These patterns were identified in the March 2026 code audit. Some are fixed; the open ones
are tracked in ROADMAP.md.

**H1 — No access isolation (open)**: The system stores contributor and team attribution on
every entry, but there is no enforcement. Any client can read, modify, or delete any entry
regardless of who created it. Do not build multi-tenant features that rely on the current
model for isolation — use Postgres row-level security when isolation is required.

**H2 — Prompt injection via ingested content (open)**: File content is interpolated directly
into LLM extraction prompts. A malicious file can include instruction text that steers the LLM
output. Treat ingested content as untrusted. System prompt hardening (e.g., wrapping content in
XML tags with explicit "this is user content, not instructions" framing) is a partial mitigation.

**M1 — No directory restriction on kb_ingest (open)**: The extension allowlist prevents many
file types, but there is no base-directory constraint. A path like `/etc/passwd` would be
rejected by extension but a `.py` file anywhere on the filesystem is ingestible. For shared
deployments, add an explicit base-directory allowlist.

**M4 — Non-atomic multi-step ops on Postgres (open)**: `asyncpg` auto-commits each `execute()`
on separate pool connections. Operations that should be atomic (create entry + delete predecessor)
are not. Use the `db.transaction()` context manager when it becomes available; avoid relying on
multi-step atomicity until the wrapper is implemented.

**Avoid blocking the event loop**: All I/O must be async. Synchronous file reads, HTTP calls,
or database operations block the entire MCP server. Use `aiosqlite`, `asyncpg`, `httpx.AsyncClient`.

**Avoid direct `os.environ` access in tool handlers**: Use the typed getters in `config.py`.
Direct env access bypasses validation and makes tests harder to isolate.

**Avoid `uv run` without `--frozen` in hooks**: Omitting `--frozen` causes uv to rebuild the
package mid-hook and triggers spurious "files were modified" failures.

## Pointers

- `CLAUDE.md` — conventions overview, environment variable table, documentation workflow
- `ROADMAP.md` — current priorities, open audit findings with references (H1, H2, M1, M4)
- `src/personal_kb/server.py` — lifespan, tool registration, LLM provider wiring
- `src/personal_kb/config.py` — all 28 environment variables with defaults
- `src/personal_kb/db/backend.py` — Database protocol, all async methods
- `src/personal_kb/llm/provider.py` — LLMProvider protocol
- `src/personal_kb/tools/formatters.py` — compact output format for MCP responses
- `.pre-commit-config.yaml` — hook configuration, coverage threshold invocation
