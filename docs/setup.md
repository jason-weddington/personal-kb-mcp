# personal-kb — Setup Guide

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.13 | Required. Use `python3 --version` to check. |
| uv | latest | Package manager. Install: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Ollama | latest | Optional but strongly recommended for vector search. |
| PostgreSQL | 15+ | Optional. SQLite (zero-config) is the default backend. |
| pgvector | latest | Required only if using PostgreSQL for vector search. |

**Operating systems**: macOS (with Xcode Command Line Tools), Linux. Windows not tested.

**Disk space**: ~500MB for Ollama embedding model (`qwen3-embedding:0.6b`). The SQLite database
grows with entries (~1KB per entry + embedding size).

## Installation Steps

### 1. Clone and install

```bash
git clone https://github.com/jason-weddington/personal-kb-mcp.git
cd personal-kb-mcp

# Install all dependencies (including optional dev group)
uv sync

# Install pre-commit hooks (required for contributors)
uv run pre-commit install \
    --hook-type pre-commit \
    --hook-type commit-msg \
    --hook-type post-commit \
    --hook-type pre-push
```

### 2. Install Ollama and pull embedding model

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Pull embedding model (1024-dim, ~200MB)
ollama pull qwen3-embedding:0.6b

# Optional: pull LLM for local query planning / enrichment
ollama pull qwen3:4b
```

**Without Ollama**: The server starts and operates in FTS-only mode. Vector search is disabled,
all other features work normally.

### 3. Configure Claude Code (MCP client)

Add to your Claude Code MCP config (typically `~/.claude/settings.json` or project-level
`.mcp.json`):

```json
{
  "mcpServers": {
    "personal-kb": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/personal-kb-mcp", "personal-kb"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "KB_CONTRIBUTOR": "your-name"
      }
    }
  }
}
```

Replace `/path/to/personal-kb-mcp` with the absolute path to your clone.

### 4. Set optional environment variables

At minimum, set `ANTHROPIC_API_KEY` for LLM-powered features (graph enrichment, agentic query,
synthesis). Without it, the server runs in search-only mode with Ollama fallback.

For team usage, set `KB_CONTRIBUTOR` and `KB_TEAM` for attribution tracking.

### 5. Automated setup (macOS/Linux)

The `setup.sh` script automates steps 1-4:

```bash
./setup.sh
```

It checks Xcode CLT (macOS), installs uv, ensures Python 3.13, downloads Ollama and the
embedding model, prompts for `ANTHROPIC_API_KEY`, and generates the MCP config JSON.

## Configuration

All configuration is via environment variables. Pass them in the MCP server env block or
export them in your shell. No config files required.

### Database & storage

| Variable | Default | Description |
|---|---|---|
| `KB_DB_PATH` | `~/.local/share/personal_kb/knowledge.db` | SQLite file path. Created on first start. |
| `KB_DATABASE_URL` | (unset) | PostgreSQL DSN (e.g., `postgresql://user:pass@host/dbname`). When set, Postgres is used instead of SQLite. |

### Embeddings (Ollama)

| Variable | Default | Description |
|---|---|---|
| `KB_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `KB_EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Embedding model name |
| `KB_EMBEDDING_DIM` | `1024` | Vector dimensions (must match model output) |
| `KB_OLLAMA_TIMEOUT` | `10.0` | Embedding request timeout in seconds |
| `KB_OLLAMA_MODEL` | `qwen3:4b` | Ollama LLM model (for local query planning / enrichment) |
| `KB_OLLAMA_LLM_TIMEOUT` | `120.0` | Ollama LLM request timeout in seconds |

### LLM providers

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | (unset) | Anthropic API key. Required for Anthropic-powered features. |
| `KB_ANTHROPIC_MODEL` | `claude-haiku-4-5` | Anthropic model for extraction and query planning |
| `KB_ANTHROPIC_TIMEOUT` | `30.0` | Anthropic API timeout in seconds |
| `KB_BEDROCK_MODEL` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock cross-region inference profile |
| `KB_BEDROCK_REGION` | `us-east-1` | AWS region for Bedrock requests |
| `KB_BEDROCK_TIMEOUT` | `60.0` | Bedrock request timeout in seconds |
| `KB_AWS_PROFILE` | (unset) | AWS credentials profile for Bedrock. Falls back to `personal_kb_bedrock` profile if it exists, then default credential chain. |
| `KB_EXTRACTION_PROVIDER` | `anthropic` | LLM for graph enrichment during store/ingest: `anthropic`, `bedrock`, or `ollama` |
| `KB_QUERY_PROVIDER` | `anthropic` | LLM for query planning and synthesis: `anthropic`, `bedrock`, or `ollama` |

### Ingestion

| Variable | Default | Description |
|---|---|---|
| `KB_INGEST_MAX_FILE_SIZE` | `10485760` | Max file size in bytes for kb_ingest (default 10MB) |
| `KB_INGEST_CHUNK_SIZE` | `16000` | Chunk size in characters for large file processing |
| `KB_INGEST_CHUNK_OVERLAP` | `600` | Overlap in characters between adjacent chunks |
| `KB_AGENTIC_INGEST` | `TRUE` | Enable KB-aware deduplication during ingestion. Set `FALSE` to skip dedup. |
| `KB_INGEST_DEDUP_THRESHOLD` | `0.06` | RRF score threshold for near-duplicate detection |

### Agentic features

| Variable | Default | Description |
|---|---|---|
| `KB_AGENTIC_QUERY` | `TRUE` | Enable ReAct agent loop for `kb_ask` auto strategy. Set `FALSE` for latency-sensitive contexts. |
| `KB_AGENTIC_MAX_CALLS` | `4` | Max tool calls in the ReAct agent loop per query |
| `KB_AGENTIC_SYNTHESIS` | `TRUE` | Enable agentic retrieval + coverage check for `kb_summarize`. Set `FALSE` to use direct search. |

### Multi-user / attribution

| Variable | Default | Description |
|---|---|---|
| `KB_CONTRIBUTOR` | (unset) | Your name or identifier, attributed to all entries you store |
| `KB_TEAM` | (unset) | Team name for team-scoped knowledge bases |
| `KB_MANAGER` | (unset) | Set `TRUE` to enable maintenance tools (`kb_maintain`, `kb_bulk_update`) |
| `KB_INSTANCE_ROLE` | (unset) | `personal` or `team`. Prepends role-specific instructions and prefixes tool names. |

### Explorer web UI

| Variable | Default | Description |
|---|---|---|
| `KB_AUTO_EXPLORE` | `TRUE` | Auto-start the graph explorer web server on MCP startup |
| `KB_EXPLORE_PORT` | `8765` | Port for the explorer web UI |

### Safety

| Variable | Default | Description |
|---|---|---|
| `KB_SKIP_SAFETY` | (unset) | Set `TRUE` to bypass secret scanning on store and ingest. Use only for trusted content. |

### PostgreSQL connection pool

| Variable | Default | Description |
|---|---|---|
| `KB_PG_POOL_MIN` | `1` | Minimum Postgres connection pool size |
| `KB_PG_POOL_MAX` | `5` | Maximum Postgres connection pool size |
| `KB_PG_IAM_AUTH` | (unset) | Set `TRUE` for AWS RDS/Aurora IAM token authentication |
| `KB_PG_REGION` | `us-east-1` | AWS region for RDS IAM token signing |

### Logging

| Variable | Default | Description |
|---|---|---|
| `KB_LOG_LEVEL` | `WARNING` | Python logging level: DEBUG, INFO, WARNING, ERROR |

Logs go to `stderr` and to `~/.local/share/personal_kb/log.txt` (overwritten on each server start).

## Verification

### 1. Confirm server starts

```bash
# Run the server (will block, waiting for MCP stdio)
uv run personal-kb
# Expected: server starts without error; press Ctrl+C to stop
```

### 2. Check Ollama is detected

```bash
KB_LOG_LEVEL=INFO uv run personal-kb
# Expected log lines on stderr:
# INFO personal_kb.server Ollama available — vector search enabled
# INFO personal_kb.server Extraction LLM: anthropic
```

### 3. Run tests

```bash
uv run pytest -m "not eval"
# Expected: all tests pass, coverage ≥ 80%
```

### 4. Verify MCP tools are registered (from Claude Code)

After adding to MCP config, restart Claude Code and run:

```
kb_search(query="test")
```

Expected: empty results (not an error). If you see "Tool not found", check the MCP config path.

### 5. Verify vector search

```bash
ollama pull qwen3-embedding:0.6b  # if not already pulled
# Then in a session:
kb_store(short_title="test entry", long_title="Vector search test",
         knowledge_details="This entry tests that vector embeddings are working",
         entry_type="factual_reference")
kb_search(query="vector embedding test")
# Expect: kb-00001 returned with match_source=hybrid
```

## Troubleshooting

**"Ollama unavailable — vector search disabled"**
- Check Ollama is running: `curl http://localhost:11434`
- Check the model is pulled: `ollama list` should show `qwen3-embedding:0.6b`
- Check `KB_OLLAMA_URL` if using a non-default Ollama host

**"Extraction LLM not available (anthropic) — graph enrichment disabled"**
- Set `ANTHROPIC_API_KEY` in the server env block
- Or set `KB_EXTRACTION_PROVIDER=ollama` and ensure Ollama LLM is pulled

**Empty search results for recent entries**
- Vector search requires Ollama. Without it, only FTS (BM25) is used.
- Check `has_embedding=True` via `kb_get` on the entry.
- If `has_embedding=False`, run `kb_maintain(action="rebuild_embeddings")` after starting Ollama.

**"Database locked" errors on SQLite**
- Only one MCP server instance should access the same DB file at a time.
- WAL mode is enabled by default, allowing concurrent readers. Concurrent writers are not supported.
- Check for zombie server processes: `ps aux | grep personal-kb`

**PostgreSQL connection failures**
- Verify `KB_DATABASE_URL` DSN is correct: `postgresql://user:pass@host:5432/dbname`
- For IAM auth: ensure `KB_PG_IAM_AUTH=TRUE`, `KB_PG_REGION` is set, and boto3 credentials
  are available (`KB_AWS_PROFILE` or instance profile).
- Check pgvector extension is installed: `CREATE EXTENSION IF NOT EXISTS vector;`

**Ingest fails with "detect-secrets not installed"**
- `detect-secrets` and `scrubadub` are core dependencies since v0.60.0. Run `uv sync`.
- To bypass for trusted content: `KB_SKIP_SAFETY=TRUE`.

**Pre-commit hook fails with "files were modified by pre-commit"**
- Ensure hooks use `uv run --frozen` (already set in `.pre-commit-config.yaml`).
- If you added a local hook, add `--frozen` to its `uv run` invocation.

**Coverage below 80% on pre-push**
- Run `uv run pytest -m "not eval" --cov --cov-report=term-missing` locally.
- Missing coverage is shown in the term-missing report. Add tests for uncovered lines.

## Pointers

- `setup.sh` — automated macOS/Linux installation script
- `pyproject.toml` — Python version, dependency groups (aws, postgres, iam), entry points
- `src/personal_kb/config.py` — all env var getters with defaults and validation
- `CLAUDE.md` — complete environment variable table with descriptions
- `docs/team_setup_aws.md` — PostgreSQL + AWS Bedrock setup for team deployments
