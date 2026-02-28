# Personal Knowledge MCP Server

A persistent knowledge base for AI coding agents, exposed as an [MCP](https://modelcontextprotocol.io/) server. Agents store technical decisions, debugging insights, patterns, and facts — the server builds a knowledge graph automatically and answers natural language queries with cited, synthesized responses.

```bash
curl -fsSL https://raw.githubusercontent.com/jason-weddington/personal-kb-mcp/main/install.sh | bash
```

## Features

- **Hybrid search** — BM25 full-text search + vector similarity (via Ollama embeddings), fused with Reciprocal Rank Fusion
- **Knowledge graph** — Automatically built from entries (deterministic edges for tags/projects + LLM-extracted entities like tools, concepts, people)
- **Graph-aware queries** — 5 traversal strategies (auto, decision_trace, timeline, related, connection) with LLM query planning
- **Synthesized answers** — `kb_summarize` retrieves relevant entries and uses Claude Haiku to produce cited prose answers
- **File ingestion** — Bulk-import existing notes, code, and docs from disk with LLM-powered extraction
- **Graceful degradation** — Every optional component (Ollama, Anthropic, vector search) fails gracefully; core storage and FTS always work

## Prerequisites

- **Python 3.13+** and **[uv](https://docs.astral.sh/uv/)** (Python package manager)
- **[Ollama](https://ollama.com/)** — for local vector embeddings (optional but recommended)
- **Anthropic API key** or **AWS credentials** — for graph enrichment, query planning, and answer synthesis (optional but recommended; either provider works)

### What works without each dependency

| Component | Without Ollama | Without Anthropic |
|---|---|---|
| Store entries | Works | Works |
| Full-text search (FTS5) | Works | Works |
| Vector similarity search | Disabled | Works (needs Ollama) |
| Graph building (deterministic) | Works | Works |
| Graph enrichment (LLM entities) | Disabled (or use Ollama LLM) | Disabled (or use Ollama LLM) |
| Query planning (`kb_ask` auto) | Disabled (or use Ollama LLM) | Disabled (or use Ollama LLM) |
| Answer synthesis (`kb_summarize`) | Disabled (or use Ollama LLM) | Disabled (or use Ollama LLM) |
| File ingestion (`kb_ingest`) | Disabled (or use Ollama LLM) | Disabled (or use Ollama LLM) |

At minimum, you get a fully functional knowledge store with full-text search and a deterministic knowledge graph. Add Ollama for vector search; add an Anthropic API key (or Ollama LLM models) for the smart features.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/jdkern11/personal_kb.git
cd personal_kb
uv sync

# 2. Install Ollama and pull the embedding model
# See https://ollama.com/download for Ollama installation
ollama pull qwen3-embedding:0.6b

# 3. (Optional) Pull an Ollama LLM for fully-local operation
ollama pull qwen3:4b

# 4. Run the setup script to verify everything works
./setup.sh
```

Or run the setup script directly — it handles steps 2-4 and checks your environment:

```bash
git clone https://github.com/jdkern11/personal_kb.git
cd personal_kb
uv sync
./setup.sh
```

## MCP Client Configuration

Add the server to your MCP client config.

**Claude Code** (`.mcp.json` in your project root or `~/.claude/mcp.json` globally):

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/personal_kb", "personal-kb"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "personal-kb": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/personal_kb", "personal-kb"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Replace `/path/to/personal_kb` with the actual path where you cloned the repo. The `ANTHROPIC_API_KEY` is optional — omit it to run without Anthropic features.

### AWS Bedrock setup

To use Claude via AWS Bedrock instead of the Anthropic API directly:

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/personal_kb", "personal-kb"],
      "env": {
        "KB_EXTRACTION_PROVIDER": "bedrock",
        "KB_QUERY_PROVIDER": "bedrock",
        "AWS_ACCESS_KEY_ID": "AKIA...",
        "AWS_SECRET_ACCESS_KEY": "...",
        "KB_BEDROCK_REGION": "us-east-1"
      }
    }
  }
}
```

This requires the optional AWS dependency: `uv sync --extra aws`. Uses the cross-region inference profile `us.anthropic.claude-haiku-4-5-20251001-v1:0` by default (override with `KB_BEDROCK_MODEL`).

### Fully local setup (no API keys)

To use Ollama for all LLM features instead of Anthropic:

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/personal_kb", "personal-kb"],
      "env": {
        "KB_EXTRACTION_PROVIDER": "ollama",
        "KB_QUERY_PROVIDER": "ollama"
      }
    }
  }
}
```

This requires `ollama pull qwen3:4b` (or whichever model you set in `KB_OLLAMA_MODEL`).

## Tools

### `kb_store`

Store or update a knowledge entry. Each entry has a short title, long title, full content, entry type, optional tags, and optional project reference. Updates create version records preserving full history. Graph edges and embeddings are built automatically on store.

### `kb_search`

Hybrid search combining BM25 full-text search with vector similarity (when Ollama is available). Supports filtering by project, entry type, and tags. Results include confidence scores with staleness decay.

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
| `KB_DB_PATH` | `~/.local/share/personal_kb/knowledge.db` | SQLite database file path |
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
| `KB_LLM_TIMEOUT` | `120.0` | Generation timeout in seconds |
| **Ollama embeddings** | | |
| `KB_EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Model for vector embeddings |
| `KB_EMBEDDING_DIM` | `1024` | Embedding vector dimensions |
| `KB_OLLAMA_TIMEOUT` | `10.0` | Embedding timeout in seconds |
| **Bedrock (AWS-managed Claude)** | | |
| `KB_BEDROCK_MODEL` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock model ID (cross-region inference profile) |
| `KB_BEDROCK_REGION` | `us-east-1` | AWS region for Bedrock |
| `KB_BEDROCK_TIMEOUT` | `30.0` | Request timeout in seconds |
| **Provider selection** | | |
| `KB_EXTRACTION_PROVIDER` | `anthropic` | LLM for graph enrichment (`anthropic`, `bedrock`, or `ollama`) |
| `KB_QUERY_PROVIDER` | `anthropic` | LLM for query planning and synthesis (`anthropic`, `bedrock`, or `ollama`) |

## Provider Architecture

The server uses two independent LLM slots, each configurable to use Anthropic (direct API), Bedrock (AWS-managed Claude), or Ollama (local):

- **Extraction LLM** (`KB_EXTRACTION_PROVIDER`) — Enriches the knowledge graph by extracting entities and relationships from stored entries.
- **Query LLM** (`KB_QUERY_PROVIDER`) — Plans graph queries from natural language questions and synthesizes answers in `kb_summarize`.

Both default to `anthropic`. You can mix providers (e.g., `bedrock` for extraction, `ollama` for queries). Vector embeddings always use Ollama and are independent of the provider settings.

## Development

```bash
uv run pytest                    # run tests
uv run ruff check src/ tests/    # lint
uv run personal-kb               # run server directly
```
