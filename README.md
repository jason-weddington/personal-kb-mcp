# Personal Knowledge MCP Server

A persistent knowledge base for AI coding agents, exposed as an [MCP](https://modelcontextprotocol.io/) server. Agents store technical decisions, debugging insights, patterns, and facts — the server builds a knowledge graph automatically and answers natural language queries with cited, synthesized responses.

No installation needed — just add the MCP config below and your client handles the rest.

## Features

- **Hybrid search** — BM25 full-text search + vector similarity (via Ollama embeddings), fused with Reciprocal Rank Fusion
- **Knowledge graph** — Automatically built from entries (deterministic edges for tags/projects + LLM-extracted entities like tools, concepts, people)
- **Agentic queries** — ReAct agent loop that plans, executes, evaluates, and retries — resolves the right entries even when single-shot search ranks them poorly
- **Graph-aware queries** — 5 traversal strategies (auto, decision_trace, timeline, related, connection) with LLM query planning
- **Synthesized answers** — `kb_summarize` retrieves relevant entries and uses Claude Haiku to produce cited prose answers
- **File ingestion** — Bulk-import existing notes, code, and docs from disk with LLM-powered extraction
- **Multi-user attribution** — Server-side identity injection (`KB_CONTRIBUTOR`, `KB_TEAM`), per-entry attribution visible in all output, contributor/team search filters, audit trail for mutations
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

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jason-weddington/personal-kb-mcp.git[aws]", "personal-kb"],
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

Hybrid search combining BM25 full-text search with vector similarity (when Ollama is available). Returns compact summaries (no `knowledge_details`). Supports filtering by project, entry type, tags, contributor, and team. Results include confidence scores with staleness decay and `@contributor/team` attribution badges.

### `kb_get`

Retrieve full details for one or more entries by ID. Use after `kb_search` to read the complete `knowledge_details` of interesting results.

### `kb_ask`

Answer questions by traversing the knowledge graph. When an LLM is available, the default `auto` strategy uses a ReAct agent loop that can plan searches, explore the graph, evaluate results, and retry with different approaches — all within a hard cap of 4 tool calls. Strong initial results skip the agent entirely (0 LLM calls). Set `KB_AGENTIC_QUERY=FALSE` to fall back to single-shot query planning.

Strategies:

- **auto** — Agentic retrieval (default). The agent has access to hybrid search, graph neighbors, graph vocabulary, decision chains, and scope-based entry listing. It picks the right combination for each question.
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

### `kb_feedback`

Report when a KB query failed to help. Always available (not gated by `KB_MANAGER`). Three feedback types: `missing` (KB lacked needed knowledge), `unhelpful` (results existed but didn't help), `friction` (tool was awkward or slow).

### `kb_maintain`

Administrative operations (only available when `KB_MANAGER=TRUE`):

- `stats` — Database overview with counts
- `deactivate` / `reactivate` — Soft-delete and restore entries
- `rebuild_embeddings` — Re-embed entries (all or only missing)
- `rebuild_graph` — Full graph reconstruction from all active entries
- `purge_inactive` — Hard-delete entries inactive for N+ days
- `vacuum` — Optimize database (PRAGMA optimize + VACUUM)
- `entry_versions` — Show version history for an entry
- `list_feedback` — Recent agent feedback with optional type/date filters
- `summarize_feedback` — LLM-clustered summary of feedback themes
- `search_stats` — Search telemetry overview (total queries, zero-result rate, top missed queries)
- `list_contributors` — Contributor/team stats for active entries
- `list_audit` — Recent mutation events (create/update/deactivate/reactivate) with optional entry_id/date filters

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| **Core** | | |
| `KB_DATABASE_URL` | _(unset)_ | PostgreSQL URL — when set, uses Postgres instead of SQLite |
| `KB_DB_PATH` | `~/.local/share/personal_kb/knowledge.db` | SQLite database file path (ignored when `KB_DATABASE_URL` is set) |
| `KB_LOG_LEVEL` | `WARNING` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `KB_MANAGER` | _(unset)_ | Set to `TRUE` to enable `kb_maintain` and `kb_ingest` tools |
| `KB_INGEST_MAX_FILE_SIZE` | `512000` | Max file size in bytes for ingestion |
| `KB_AGENTIC_QUERY` | `TRUE` | Enable ReAct agent loop for `kb_ask` auto strategy |
| `KB_AGENTIC_MAX_CALLS` | `4` | Max tool calls in the agentic query loop |
| **Multi-user** | | |
| `KB_CONTRIBUTOR` | _(unset)_ | Your name — attached to entries, versions, search events, and audit trail |
| `KB_TEAM` | _(unset)_ | Your team — attached to entries alongside contributor |
| `KB_SKIP_SAFETY` | _(unset)_ | Set to `TRUE` to bypass secret scanning on store |
| `KB_PG_POOL_MIN` | `1` | Postgres connection pool minimum size |
| `KB_PG_POOL_MAX` | `5` | Postgres connection pool maximum size |
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
| **Aurora/RDS IAM auth** | | |
| `KB_PG_IAM_AUTH` | _(unset)_ | Set `TRUE` for RDS/Aurora IAM authentication |
| `KB_PG_REGION` | `us-east-1` | AWS region for RDS IAM token signing |
| **Provider selection** | | |
| `KB_EXTRACTION_PROVIDER` | `anthropic` | LLM for graph enrichment (`anthropic`, `bedrock`, or `ollama`) |
| `KB_QUERY_PROVIDER` | `anthropic` | LLM for query planning and synthesis (`anthropic`, `bedrock`, or `ollama`) |

> **Legacy SigV4 auth:** Bedrock also supports `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` for traditional IAM credentials. Bearer token auth (`AWS_BEARER_TOKEN_BEDROCK`) is the preferred method.

## Provider Architecture

The server uses two independent LLM slots, each configurable to use Anthropic (direct API), Bedrock (AWS-managed Claude), or Ollama (local):

- **Extraction LLM** (`KB_EXTRACTION_PROVIDER`) — Enriches the knowledge graph by extracting entities and relationships from stored entries.
- **Query LLM** (`KB_QUERY_PROVIDER`) — Plans graph queries from natural language questions and synthesizes answers in `kb_summarize`.

Both default to `anthropic`. You can mix providers (e.g., `bedrock` for extraction, `ollama` for queries). Vector embeddings always use Ollama and are independent of the provider settings.

## Team Use

Multiple people (or agents) can share a single knowledge base. Each contributor runs their own MCP server instance pointed at the same database, with `KB_CONTRIBUTOR` identifying who they are.

### Setup

Each team member sets their identity via environment variables in their MCP config:

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jason-weddington/personal-kb-mcp.git[postgres]", "personal-kb"],
      "env": {
        "KB_DATABASE_URL": "postgresql://user:pass@shared-host/team_kb",
        "KB_CONTRIBUTOR": "jason",
        "KB_TEAM": "platform",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

### What you get

- **Attribution** — Every entry, version, and search event records who created it. Search results show `@contributor/team` badges so you can see who wrote what.
- **Filtering** — `kb_search` accepts `contributor` and `team` parameters to scope results to a specific person or team.
- **Audit trail** — All mutations (create, update, deactivate, reactivate) are logged in the `audit_events` table. Use `kb_maintain list_audit` to review recent changes.
- **Sensitivity classification** — Entries can be tagged `internal`, `restricted`, or `public` via the `sensitivity` parameter on `kb_store`. Badges appear in output to signal handling expectations.
- **Secret scanning** — `kb_store` and `kb_store_batch` scan content for potential secrets (API keys, passwords) before storing. Override with `KB_SKIP_SAFETY=TRUE`.
- **Contributor stats** — `kb_maintain list_contributors` shows who has contributed what.

### Trust model and limitations

The multi-user features provide **attribution and visibility, not access control**. Understanding the trust model is important before deploying to a team:

- **Identity is environment-based, not authenticated.** `KB_CONTRIBUTOR` is set by each MCP server instance at startup. There is no login, no tokens, no verification. Anyone with database access can set any contributor name. This is appropriate for trusted teams where members configure their own environments honestly.
- **No read restrictions.** All entries are visible to all users regardless of contributor, team, or sensitivity classification. The `sensitivity` field is a label for human judgment — it does not hide or encrypt anything. A `restricted` entry is just as readable as a `public` one.
- **No write restrictions.** Any user can update or deactivate any entry. The `updated_by` field and audit trail record who did what, but nothing prevents the action.
- **Vector search ignores contributor/team filters.** The contributor and team filters apply to full-text search only. Vector similarity search returns all entries regardless of attribution. Filtered-out entries can still appear in results via the RRF fusion step. This is a known limitation of the current search architecture.
- **Audit trail is append-only but not tamper-proof.** Audit events are stored in the same database as everything else. Anyone with database access can modify or delete them. This is sufficient for "who did what" visibility, not for compliance or forensics.
- **Last-write-wins on concurrent edits.** If two contributors update the same entry simultaneously, the last write wins. There is no locking, merge, or conflict resolution. Version history preserves both changes, but only the latest version is active.

**Bottom line:** These features work well for a small trusted team sharing a knowledge base through separate MCP server instances, each configured with their own `KB_CONTRIBUTOR`. They are not designed for untrusted multi-tenant environments where users might act adversarially.

### Personal + team (dual-instance pattern)

Most team members want both a **shared team KB** (decisions, architecture, patterns) and a **personal KB** (dotfiles, shell aliases, workflow preferences). Run two MCP server instances with different names — the agent sees both and uses the server instructions to route entries correctly.

```json
{
  "mcpServers": {
    "team-kb": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jason-weddington/personal-kb-mcp.git[postgres]", "personal-kb"],
      "env": {
        "KB_DATABASE_URL": "postgresql://user:pass@shared-host/team_kb",
        "KB_CONTRIBUTOR": "jason",
        "KB_TEAM": "platform",
        "KB_INSTANCE_ROLE": "team",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    },
    "personal-kb": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jason-weddington/personal-kb-mcp.git", "personal-kb"],
      "env": {
        "KB_INSTANCE_ROLE": "personal",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

`KB_INSTANCE_ROLE` prepends a role-specific instruction to the server description:
- **`team`** — "This is the TEAM knowledge base — shared decisions, architecture, patterns, and conventions."
- **`personal`** — "This is your PERSONAL knowledge base — your config, dotfiles, workflow preferences, and private notes."

The MCP server name (`team-kb` vs `personal-kb`) plus the role instruction gives the agent enough signal to route stores and searches to the right instance. Environment variables in each `env` block are scoped to that server process — no collisions.

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
uvx --from "git+https://github.com/jason-weddington/personal-kb-mcp.git[postgres]" personal-kb
```

### Migrate your data

```bash
# Preview what will be migrated (read-only):
uv run python scripts/migrate_sqlite_to_pg.py --dry-run \
  ~/.local/share/personal_kb/knowledge.db

# Run the migration:
uv run python scripts/migrate_sqlite_to_pg.py \
  ~/.local/share/personal_kb/knowledge.db

# With attribution (stamps your name on migrated entries):
uv run python scripts/migrate_sqlite_to_pg.py \
  --contributor jason --team platform \
  ~/.local/share/personal_kb/knowledge.db
```

The target Postgres connection comes from environment variables — set `KB_DATABASE_URL` before running. For Aurora IAM auth, also set `KB_PG_IAM_AUTH=TRUE` and `KB_PG_REGION`.

The script copies all data tables, then rebuilds embeddings via Ollama automatically. In merge mode (target already has entries), source IDs are remapped to avoid collisions. Use `--skip-embeddings` to defer the re-embed step. See [docs/team_setup_aws.md](docs/team_setup_aws.md) for detailed reference.

### Switch your MCP config

Update your MCP client config to set `KB_DATABASE_URL`:

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jason-weddington/personal-kb-mcp.git[postgres]", "personal-kb"],
      "env": {
        "KB_DATABASE_URL": "postgresql://user:pass@localhost/my_kb",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

When `KB_DATABASE_URL` is set, the server uses PostgreSQL. When it's not set, it uses SQLite (the `KB_DB_PATH` file). You can switch back and forth — both backends are always available.

### Aurora Serverless with IAM auth

For AWS deployments where database passwords aren't acceptable, the server supports RDS/Aurora IAM authentication. Instead of a static password in the connection string, the server generates short-lived SigV4-signed tokens that are refreshed automatically on each new connection.

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jason-weddington/personal-kb-mcp.git[postgres,iam]", "personal-kb"],
      "env": {
        "KB_DATABASE_URL": "postgresql://myuser@aurora-cluster.cluster-xxx.us-east-1.rds.amazonaws.com:5432/my_kb",
        "KB_PG_IAM_AUTH": "TRUE",
        "KB_PG_REGION": "us-east-1",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Requirements:
- The `iam` optional dependency (`boto3`) — included via `personal-kb-mcp[iam]`
- AWS credentials available through the standard chain (environment variables, `~/.aws/credentials`, instance roles)
- The database user must have IAM authentication enabled in RDS/Aurora
- The connection URL should **not** include a password — the token factory provides it

The server uses `boto3.client('rds').generate_db_auth_token()` to sign tokens locally (no network call) and passes them to asyncpg's connection pool as a callable password, so tokens are refreshed on every new connection. SSL is enabled automatically — IAM auth requires TLS.

### If embeddings were skipped

Embeddings can't be copied between backends (sqlite-vec uses packed binary, pgvector uses native arrays), so the migration script re-embeds via Ollama. If Ollama wasn't running during migration, or you used `--skip-embeddings`, rebuild them manually:

```
kb_maintain rebuild_embeddings (force=True)
```

The KB works immediately without embeddings — you just won't get vector search results until the rebuild finishes. FTS and graph search work from the start.

### Keeping SQLite as a backup

The migration is additive — it doesn't modify your SQLite database. Your original file at `~/.local/share/personal_kb/knowledge.db` stays intact. To fall back, just remove `KB_DATABASE_URL` from your config.
