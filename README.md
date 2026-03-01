# Personal Knowledge MCP Server

A persistent knowledge base for AI coding agents, exposed as an [MCP](https://modelcontextprotocol.io/) server. Agents store technical decisions, debugging insights, patterns, and facts — the server builds a knowledge graph automatically and answers natural language queries with cited, synthesized responses.

No installation needed — just add the MCP config below and your client handles the rest.

## Features

- **Hybrid search** — BM25 full-text search + vector similarity (via Ollama embeddings), fused with Reciprocal Rank Fusion
- **Knowledge graph** — Automatically built from entries (deterministic edges for tags/projects + LLM-extracted entities like tools, concepts, people)
- **Graph-aware queries** — 5 traversal strategies (auto, decision_trace, timeline, related, connection) with LLM query planning
- **Synthesized answers** — `kb_summarize` retrieves relevant entries and uses Claude Haiku to produce cited prose answers
- **File ingestion** — Bulk-import existing notes, code, and docs from disk with LLM-powered extraction
- **Graceful degradation** — Every optional component (Ollama, Anthropic, vector search) fails gracefully; core storage and FTS always work

## Prerequisites

- **Python 3.13+** and **[uv](https://docs.astral.sh/uv/)** (Python package manager)
- **[Ollama](https://ollama.com/)** — optional, for local vector embeddings
- **LLM provider** (pick one, optional but recommended):
  - **Anthropic API key** — simplest setup
  - **AWS Bedrock bearer token** — use Claude through your AWS account
  - **Ollama** — fully local, no API keys needed

### What works without each dependency

| Component | Without Ollama | Without LLM provider |
|---|---|---|
| Store entries | Works | Works |
| Full-text search (FTS5) | Works | Works |
| Vector similarity search | Disabled | Works (needs Ollama) |
| Graph building (deterministic) | Works | Works |
| Graph enrichment (LLM entities) | Disabled (or use Ollama LLM) | Disabled |
| Query planning (`kb_ask` auto) | Disabled (or use Ollama LLM) | Disabled |
| Answer synthesis (`kb_summarize`) | Disabled (or use Ollama LLM) | Disabled |
| File ingestion (`kb_ingest`) | Disabled (or use Ollama LLM) | Disabled |

At minimum, you get a fully functional knowledge store with full-text search and a deterministic knowledge graph. Add Ollama for vector search; add any LLM provider for the smart features.

## Quick Start

### With Anthropic (simplest)

Add this to your MCP client config — Claude Code (`~/.claude/mcp.json`), Claude Desktop (`claude_desktop_config.json`), etc.:

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jason-weddington/personal-kb-mcp.git", "personal-kb"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

That's it. `uvx` installs and runs the server automatically.

### Fully local (Ollama, no API keys)

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jason-weddington/personal-kb-mcp.git", "personal-kb"],
      "env": {
        "KB_EXTRACTION_PROVIDER": "ollama",
        "KB_QUERY_PROVIDER": "ollama"
      }
    }
  }
}
```

Pull the models first:

```bash
ollama pull qwen3-embedding:0.6b   # for vector search
ollama pull qwen3:4b               # for LLM features (graph enrichment, query planning, synthesis)
```

### AWS Bedrock

Bedrock requires a clone install because of a forked dependency (`smithy-json`). Once the upstream fix merges, `uvx` will work here too.

```bash
git clone https://github.com/jason-weddington/personal-kb-mcp.git
cd personal-kb-mcp
uv sync --extra aws
```

Then add the MCP config pointing to your clone:

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/personal-kb-mcp", "personal-kb"],
      "env": {
        "KB_EXTRACTION_PROVIDER": "bedrock",
        "KB_QUERY_PROVIDER": "bedrock",
        "AWS_BEARER_TOKEN_BEDROCK": "your-bearer-token",
        "KB_BEDROCK_REGION": "us-east-1"
      }
    }
  }
}
```

Uses the cross-region inference profile `us.anthropic.claude-haiku-4-5-20251001-v1:0` by default (override with `KB_BEDROCK_MODEL`).

> **Legacy SigV4 auth** also works — set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` instead of `AWS_BEARER_TOKEN_BEDROCK`. Bearer token auth is preferred.

### Ollama setup (if using embeddings or local LLM)

```bash
# Install Ollama: https://ollama.com/download

ollama pull qwen3-embedding:0.6b   # for vector search
ollama pull qwen3:4b               # only if using Ollama as LLM provider
```

## Tools

### `kb_store`

Store or update a knowledge entry. Each entry has a short title, long title, full content, entry type, optional tags, and optional project reference. Updates create version records preserving full history. Graph edges and embeddings are built automatically on store.

### `kb_store_batch`

Store multiple entries in a single call (max 10). More efficient than repeated `kb_store` — uses a single LLM call for graph enrichment across all entries.

### `kb_search`

Hybrid search combining BM25 full-text search with vector similarity (when Ollama is available). Returns compact summaries (no `knowledge_details`). Supports filtering by project, entry type, and tags. Results include confidence scores with staleness decay.

### `kb_get`

Retrieve full details for one or more entries by ID. Use after `kb_search` to read the complete `knowledge_details` of interesting results.

### `kb_ask`

Answer questions by traversing the knowledge graph. Supports 5 strategies:

- **auto** — Hybrid search + expand results via graph neighbors (default). When an LLM is available, a query planner translates natural language into the optimal strategy automatically.
- **decision_trace** — Follow `supersedes` chains to trace how a decision evolved over time.
- **timeline** — Chronological entries for a given scope (project, tag, etc.).
- **related** — BFS from a starting node through graph edges.
- **connection** — Find paths between two nodes in the graph.

### `kb_summarize`

Answer a question with a synthesized natural language response. Retrieves relevant entries via the auto strategy, then uses Claude Haiku to produce a coherent answer with `[kb-XXXXX]` citations. Falls back to raw search results when the LLM is unavailable.

### `kb_ingest`

Ingest files from disk into the knowledge base (only available when `KB_MANAGER=TRUE`). Reads files, runs safety checks, and uses an LLM to summarize and extract structured knowledge entries.

```
kb_ingest(file_path="/path/to/notes", project_ref="my-project", dry_run=True)
```

**Pipeline:** deny-list check → extension filter → size limit → SHA-256 dedup → secret detection → PII redaction → LLM summarize → LLM extract → store entries → build graph

- Supports single files or entire directories (recursive by default)
- Files become `note:` nodes in the graph, with `extracted_from` edges linking entries to sources
- Re-ingestion detects content changes via hash and replaces old entries
- `dry_run=True` previews extraction without storing anything
- Supports `.md`, `.txt`, `.py`, `.js`, `.ts`, `.yaml`, `.json`, `.toml`, and many more text formats
- Skips binaries, images, archives, keys, `.env` files, and other sensitive formats
- Optional safety libraries: `uv sync --extra safety` installs `detect-secrets` and `scrubadub`

### `kb_maintain`

Administrative operations (only available when `KB_MANAGER=TRUE`):

- `stats` — Database overview with counts
- `deactivate` / `reactivate` — Soft-delete and restore entries
- `rebuild_embeddings` — Re-embed entries (all or only missing)
- `rebuild_graph` — Full graph reconstruction from all active entries
- `purge_inactive` — Hard-delete entries inactive for N+ days
- `vacuum` — Optimize database (PRAGMA optimize + VACUUM)
- `entry_versions` — Show version history for an entry

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| **Core** | | |
| `KB_DATABASE_URL` | _(unset)_ | PostgreSQL URL — when set, uses Postgres instead of SQLite |
| `KB_DB_PATH` | `~/.local/share/personal_kb/knowledge.db` | SQLite database file path (ignored when `KB_DATABASE_URL` is set) |
| `KB_LOG_LEVEL` | `WARNING` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `KB_MANAGER` | _(unset)_ | Set to `TRUE` to enable `kb_maintain` and `kb_ingest` tools |
| `KB_INGEST_MAX_FILE_SIZE` | `512000` | Max file size in bytes for ingestion |
| **Anthropic (cloud LLM)** | | |
| `ANTHROPIC_API_KEY` | _(unset)_ | API key — required for Anthropic provider |
| `KB_ANTHROPIC_MODEL` | `claude-haiku-4-5` | Model for graph enrichment, query planning, and synthesis |
| `KB_ANTHROPIC_TIMEOUT` | `30.0` | Request timeout in seconds |
| **Ollama (local LLM)** | | |
| `KB_OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `KB_OLLAMA_MODEL` | `qwen3:4b` | Model for generation tasks |
| `KB_OLLAMA_LLM_TIMEOUT` | `120.0` | Generation timeout in seconds |
| **Ollama embeddings** | | |
| `KB_EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Model for vector embeddings |
| `KB_EMBEDDING_DIM` | `1024` | Embedding vector dimensions |
| `KB_OLLAMA_TIMEOUT` | `10.0` | Embedding timeout in seconds |
| **Bedrock (AWS-managed Claude)** | | |
| `AWS_BEARER_TOKEN_BEDROCK` | _(unset)_ | Bearer token for Bedrock auth (preferred method) |
| `KB_BEDROCK_MODEL` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock model ID (cross-region inference profile) |
| `KB_BEDROCK_REGION` | `us-east-1` | AWS region for Bedrock |
| `KB_BEDROCK_TIMEOUT` | `30.0` | Request timeout in seconds |
| **Provider selection** | | |
| `KB_EXTRACTION_PROVIDER` | `anthropic` | LLM for graph enrichment (`anthropic`, `bedrock`, or `ollama`) |
| `KB_QUERY_PROVIDER` | `anthropic` | LLM for query planning and synthesis (`anthropic`, `bedrock`, or `ollama`) |

> **Legacy SigV4 auth:** Bedrock also supports `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` for traditional IAM credentials. Bearer token auth (`AWS_BEARER_TOKEN_BEDROCK`) is the preferred method.

## Provider Architecture

The server uses two independent LLM slots, each configurable to use Anthropic (direct API), Bedrock (AWS-managed Claude), or Ollama (local):

- **Extraction LLM** (`KB_EXTRACTION_PROVIDER`) — Enriches the knowledge graph by extracting entities and relationships from stored entries.
- **Query LLM** (`KB_QUERY_PROVIDER`) — Plans graph queries from natural language questions and synthesizes answers in `kb_summarize`.

Both default to `anthropic`. You can mix providers (e.g., `bedrock` for extraction, `ollama` for queries). Vector embeddings always use Ollama and are independent of the provider settings.

## Development

```bash
git clone https://github.com/jason-weddington/personal-kb-mcp.git
cd personal-kb-mcp
uv sync

uv run pytest                    # run tests
uv run ruff check src/ tests/    # lint
uv run personal-kb               # run server directly
```

For Bedrock support: `uv sync --extra aws`. For secret/PII detection in `kb_ingest`: `uv sync --extra safety`. For PostgreSQL: `uv sync --extra postgres`.

## So you started with SQLite...

SQLite is the default and it works great — most users will never need to change. But if your KB has grown large, you're running the server on a shared machine, or you just prefer Postgres, switching is straightforward.

### What changes

| | SQLite | PostgreSQL |
|---|---|---|
| **Full-text search** | FTS5 with BM25 | tsvector + GIN with ts_rank_cd |
| **Vector search** | sqlite-vec (vec0) | pgvector |
| **JSON queries** | `json_extract()` | `->>` operator |
| **Concurrency** | WAL mode (single-writer) | Full MVCC |
| **Setup** | Zero — it's a file | Postgres + pgvector extension |

Everything else — entries, graph, versions, ingested files — works identically. The same MCP tools, the same entry format, the same search results.

### Prerequisites

A running PostgreSQL 15+ instance with the [pgvector](https://github.com/pgvector/pgvector) extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

And the `asyncpg` optional dependency:

```bash
# If running from a clone:
uv sync --extra postgres

# If using uvx, add the extra:
uvx --from "personal-kb-mcp[postgres]" personal-kb
```

### Migrate your data

```bash
# Preview what will be migrated (read-only):
python scripts/migrate_to_postgres.py postgresql://user:pass@localhost/my_kb --dry-run

# Run the migration:
python scripts/migrate_to_postgres.py postgresql://user:pass@localhost/my_kb

# If your SQLite DB is in a non-default location:
python scripts/migrate_to_postgres.py postgresql://localhost/my_kb --sqlite /path/to/knowledge.db
```

The script copies all data tables, then rebuilds embeddings via Ollama automatically. If Ollama isn't running, it skips embeddings gracefully — you can rebuild later. Use `--skip-embeddings` to skip intentionally.

### Switch your MCP config

Update your MCP client config to set `KB_DATABASE_URL`:

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "personal-kb-mcp[postgres]", "personal-kb"],
      "env": {
        "KB_DATABASE_URL": "postgresql://user:pass@localhost/my_kb",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

When `KB_DATABASE_URL` is set, the server uses PostgreSQL. When it's not set, it uses SQLite (the `KB_DB_PATH` file). You can switch back and forth — both backends are always available.

### If embeddings were skipped

Embeddings can't be copied between backends (sqlite-vec uses packed binary, pgvector uses native arrays), so the migration script re-embeds via Ollama. If Ollama wasn't running during migration, or you used `--skip-embeddings`, rebuild them manually:

```
kb_maintain rebuild_embeddings (force=True)
```

The KB works immediately without embeddings — you just won't get vector search results until the rebuild finishes. FTS and graph search work from the start.

### Keeping SQLite as a backup

The migration is additive — it doesn't modify your SQLite database. Your original file at `~/.local/share/personal_kb/knowledge.db` stays intact. To fall back, just remove `KB_DATABASE_URL` from your config.
