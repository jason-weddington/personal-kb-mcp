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

### One-liner setup

Installs all prerequisites (uv, Python 3.13, Ollama, embedding model), prompts for an optional Anthropic API key, and launches the knowledge explorer:

```bash
curl -fsSL https://raw.githubusercontent.com/jason-weddington/personal-kb-mcp/main/setup.sh | bash
```

### How local mode works (read this first)

The MCP server is a **thin HTTP client** of the `kb-service` web service.
In **local mode** the MCP server auto-spawns `kb-service` as a detached
singleton daemon on `127.0.0.1:8765` the first time a session opens, and
talks to it over loopback. Two things are required for that flow to work:

The `kb-service` console script ships with `personal-kb` (the `[local]` extra is kept for compatibility), so a plain `uvx` install has everything it needs.

**No configuration is needed for a local KB.** When `PERSONAL_KB_URL` is unset (or empty) the MCP server uses `http://127.0.0.1:8765`, sends the fixed non-secret API key `local-no-auth` (the spawned daemon runs with `KB_AUTH_MODE=none` and ignores it), and spawns the daemon on first use. The personal-kb-hook applies the same defaults but never spawns the daemon: if nothing is listening it stays silent. Set `PERSONAL_KB_URL` only to point at a hosted KB, and then `PERSONAL_KB_API_KEY` is required (a remote URL without a key is a startup error, never the sentinel). An explicit loopback URL on another port spawns the daemon on that port.

### With Anthropic (simplest)

Add this to your MCP client config — Claude Code (`~/.claude/mcp.json`), Claude Desktop (`claude_desktop_config.json`), etc.:

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jason-weddington/personal-kb-mcp.git[local]", "personal-kb"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

That's it. `uvx` installs `personal-kb` AND the bundled `kb-service`, the
lifespan auto-spawns the local daemon on first use, and the MCP server
connects to it over loopback.

### Fully local (Ollama, no API keys)

```json
{
  "mcpServers": {
    "personal-kb": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jason-weddington/personal-kb-mcp.git[local]", "personal-kb"],
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
      "args": ["--from", "git+https://github.com/jason-weddington/personal-kb-mcp.git[local,aws]", "personal-kb"],
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

Store or update a knowledge entry. Each entry has a short title, long title, full content, entry type, optional tags, and optional project reference. Updates create version records preserving full history. Graph edges and embeddings are built automatically on store. An optional `ttl` parameter (e.g. `'7d'`, `'24h'`, `'2w'`) sets an expiration date after which the entry is excluded from search results.

**Supersession.** When a new entry replaces older ones, declare it with `hints={"supersedes": ["kb-XXXXX", ...]}` (the hosted HTTP API also takes a top-level `supersedes` list, where an absent field, `[]` or `"none"` all mean "replaces nothing"). The hosted service checks every newly named target before writing: it must exist, be active, not be a `mental_map`, not be the entry itself, and not already supersede the writer. A rejected target fails the whole write with a 422 that names the problem. Each superseded entry's `superseded_by` then points at its newest superseder, and the field is maintained automatically when superseders are stored, updated or deactivated. On an update, supersedes targets are only ever added, never retracted. Updates and deactivations through the hosted service require a `change_reason` that says what changed and why; when you deactivate an entry because a newer one replaces it, also pass `superseded_by`.

### `kb_store_batch`

Store multiple entries in a single call (max 10). More efficient than repeated `kb_store` — uses a single LLM call for graph enrichment across all entries.

### `kb_search`

Hybrid search combining BM25 full-text search with vector similarity (when Ollama is available). Returns compact summaries (no `knowledge_details`). Supports filtering by project, entry type, tags, contributor, and team. Results include confidence scores with staleness decay and `@contributor/team` attribution badges. Set `include_expired=True` to include entries past their TTL expiration in results (excluded by default).

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

### `kb_ingest_url`

Ingest a web page into the knowledge base (only available when `KB_MANAGER=TRUE`). Uses [trafilatura](https://trafilatura.readthedocs.io/) for HTML-to-text extraction, then runs the same LLM summarize/extract pipeline as `kb_ingest`. Accepts `url`, `project_ref`, `tags`, and `dry_run`.

### `kb_explore`

Open an interactive graph visualization in your browser. The explorer auto-starts on MCP server startup (available at `http://localhost:8767` immediately, no tool call needed — use the tool to relaunch or change ports). Type a question in the search bar and watch the agent traverse the graph in real time via SSE streaming. Questions routed to `summarize` mode open a multi-turn chat panel where you can ask follow-up questions grounded in the retrieved KB entries, with iMessage-style message bubbles and clickable `[kb-XXXXX]` citations that fly to nodes in the graph. The chat panel supports write-back tools — update entries or ingest URLs directly from the conversation. The toolbar also provides file upload and multi-URL ingestion for bulk import without leaving the browser.

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

### `kb_map_eligibility`

Review per-project map eligibility across the whole KB: one line per project_ref with its effective verdict (computed or human override), mappable/hand-authored/ingested counts, existing mental-map count, top title prefix and evidence flags (`too_thin`, `ingest_corpus`, `journal`), plus a header of aggregate counts. Pass `project_ref` to narrow to a single project. This is the table to read before deciding which projects deserve a mental map — the review target is a project that is eligible AND has `maps == 0`.

### `kb_map_eligibility_override`

Set or clear a human map-eligibility override for one project_ref: `eligible=True/False` with a permanent `reason` forces the verdict in either direction until `clear=True` reverts it to the computed verdict. The nightly map-maintenance loop reads this table and never writes it — this tool is the human/agent-review path only.

## Mental maps — orientation nodes for your KB

A **mental map** (entry type `mental_map`) is the directory tier of your knowledge base. It is the only entry type that does not carry retrievable knowledge itself — instead, it points to *where* knowledge lives. Think of it as a small, curated index for a subsystem: "the ingestion flow lives across these five entries; auth lives across those three." Agents read a map to orient themselves, then `kb_get` the detail entries the map points to.

### When to author a map vs. a `factual_reference`

A `factual_reference` *is* the answer — a port number, a config threshold, an API signature, the name of a function. A `mental_map` *points to* answers — an orientation directory that says "for the auth subsystem, see kb-00310, kb-00312, kb-00318."

Rule of thumb: ask whether a reader would *act on the value directly* (fact) or *follow it to a source* (map). Counts of parts ("the pipeline has three stages") are pointers in disguise and belong in a map; the actual stage names, thresholds, or filenames belong in the linked factual entries a map points to.

### What a map must contain

Every `mental_map` must have **at least one outbound pointer**. A pointer is any of:

- a `kb-XXXXX` reference inside `knowledge_details`;
- a `related_entities` hint with either a dict carrying an `id`/`target` or a bare entry id string.

A `supersedes` hint is **not** a map pointer: a map cannot supersede anything, so the hosted service rejects a map whose only pointer is a `supersedes` hint. The local client-side check still counts `supersedes` / `superseded_by` until the MCP client catches up.

Tags, project refs, and person/tool hints **do not** count — those are categorization, not orientation. `kb_store` rejects an orphan map (zero pointers) on **create**, before any row is written, with an error: *"A mental_map entry requires at least one outbound pointer … A map with zero pointers is an orphan note, not a map."* As of 2026-09-19 the hosted service runs the same guard on **update**, evaluating the effective post-update body, so an update can no longer strip a map's last pointer; the local/no-auth kb-core path is still unguarded. Don't treat "the store would have caught it" as a substitute for checking your own map's pointers — the update guard is new, and every map authored before it predates that check.

### What a map should *not* contain (fact-free discipline)

A map is for **structure and relationships**, not retrievable values. `kb_store` runs a deterministic, advisory-only lint on every map body and flags content that looks like a fact — URLs, file paths, `ENV_VAR`-style tokens, dotted code identifiers (`module.func`), quoted literals, and config-like numerals (decimals, ≥4-digit integers, numerals adjacent to `=` or `:`). It also flags bodies longer than a compositional advisory budget (a fixed base allowance plus a per-pointer allowance for each distinct `kb-` reference, so well-pointed maps get more room for their glosses) with the note that an author should *cut orientation prose, not pointer glosses*.

The lint never blocks a store — every store succeeds and every warning starts with `Map lint (advisory):`. The signal is purely informational, pointing you at content that probably belongs in a `factual_reference` the map links to instead.

### Maps don't decay on a clock

Unlike the four value-bearing entry types — which lose confidence on an entry-type-specific half-life (90 d / 1 y / 2 y / 5 y) — a `mental_map` is exempt from confidence decay. It has no retrievable value of its own to go stale, and the access-aware self-heal (`last_accessed` resets the decay clock on `kb_get`) would just make a constantly-surfaced map look "fresh" while its pointers rotted. So maps don't decay at all.

Freshness for a map is instead **pointer-validity**, checked at retrieval time: when you `kb_get` a `mental_map`, the server resolves each outbound pointer and surfaces inline any target that is `superseded_by` another entry or has been deactivated. A stale map is one whose pointers no longer resolve, not one that hasn't been read recently.

### Discovering and surfacing maps

Maps are discoverable through two paired mechanisms:

- **Pull (in-process, default):** `kb_preflight(project_ref="...")` includes a `Maps:` section listing all of the project's mental maps (they're a small curated set, so there's no cap). An agent calls preflight at session start, sees which maps exist, and pulls full detail with `kb_get` only for the ones it judges relevant to the task. No hook, no extra config — this works out of the box on every install.
- **Push (opt-in CLI hook, below):** the `personal-kb-hook` console script proactively injects the same Maps directory into Claude Code's `SessionStart` and `UserPromptSubmit` hooks — for the cold-start case where the agent doesn't yet know which subsystem to ask about.

Both halves read from the same `mental_map` entries you author with `kb_store`; the push half is described in the next section.

## Surfacing maps via `personal-kb-hook` (opt-in)

The repo also ships a small stdlib-only CLI (`personal-kb-hook`) that you can wire into Claude Code's [hook system](https://docs.claude.com/en/docs/claude-code/hooks) to proactively surface the `mental_map` entries for the project you're working in. The hook itself never talks to the MCP server or the DB — it reads a denormalized JSONL index that the MCP server writes every time a `mental_map` is created, updated, or deactivated.

### Install

The hook lives in its **own standalone package** at `packages/personal-kb-hook/` with **zero third-party dependencies** — installing it does NOT drag in `fastmcp`, `anthropic`, `pymupdf`, etc. Install it directly from the git subdirectory:

```bash
uv tool install --from \
  "git+https://github.com/jason-weddington/personal-kb-mcp.git#subdirectory=packages/personal-kb-hook" \
  personal-kb-hook
```

This puts a single `personal-kb-hook` console script on your `PATH`. The tool venv contains exactly one package (`personal_kb_hook`) — nothing else. The hook *process* itself imports only the Python standard library and never opens the database — it reads the on-disk JSONL index — so it stays fast on the session hot path.

> The main `personal-kb` server package is installed separately (typically via `uvx --from "git+…" personal-kb` or `uv tool install --from "git+…" personal-kb`); it provides the MCP server that writes the on-disk maps index that the hook reads.

### The `.kb_project` convention

The hook resolves which project's maps to surface by walking up from the harness-provided `cwd` to the first `.kb_project` file it finds. That file is **one line** — the KB `project_ref` for the repo:

```text
# .kb_project — committed at the repo root
my-project
```

Commit one per repo so the hook works for every clone, every machine, every user — no per-user config required. Lines starting with `#` are comments; the first non-blank, non-comment line wins.

This repo's own `.kb_project` is just `personal-kb`.

### Wiring it into Claude Code

Add the hook to `~/.claude/settings.json` (or your project-level `.claude/settings.json`). The hook accepts `--format=text` (default — bare directory string) or `--format=claude-json` (the `hookSpecificOutput` envelope Claude Code understands).

```jsonc
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "personal-kb-hook --format=claude-json" }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "personal-kb-hook --format=claude-json" }
        ]
      }
    ]
  }
}
```

If you'd rather inject the bare directory string into the model context yourself, use `--format=text`:

```jsonc
{
  "hooks": {
    "SessionStart": [
      { "matcher": "", "hooks": [{ "type": "command", "command": "personal-kb-hook" }] }
    ],
    "UserPromptSubmit": [
      { "matcher": "", "hooks": [{ "type": "command", "command": "personal-kb-hook" }] }
    ]
  }
}
```

### What gets surfaced

When the hook fires, it:

1. Reads the JSON payload Claude Code writes to its stdin.
2. Walks up from `payload.cwd` to the first `.kb_project` (or stays silent if none).
3. Looks the resolved `project_ref` up in `~/.local/share/personal_kb/maps_index.jsonl` (the file the MCP server writes on mental_map store/update/deactivate).
4. Emits a single factual line like `Maps for my-project — [kb-00310] ingestion: Ingestion flow; [kb-00312] auth: Auth + sessions`.

The output is **factual**, never imperative — no "load", "use", "read" — so it never trips prompt-injection defenses.

A per-session scratch file under `~/.cache/personal_kb/` suppresses re-injection when the same project's same map ids have already been surfaced in this session. Scope drift (cwd → different repo → different `.kb_project`) re-emits. Compaction (`source=compact`) bypasses the suppression so the context is re-seeded.

Any error path — no `.kb_project`, no maps for the project, malformed stdin, write failure — exits 0 with no stdout. The hook never raises into the harness.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| **Thin client (always set)** | | |
| `PERSONAL_KB_URL` | `http://127.0.0.1:8765` | `kb-service` URL. Unset or empty means local mode (the MCP server auto-spawns a daemon on this port). Set it only for a hosted KB. |
| `PERSONAL_KB_API_KEY` | `local-no-auth` for loopback URLs; required otherwise | Bearer token for `kb-service`. Defaults to the sentinel only when the URL is loopback (the daemon ignores it). Remote mode: a per-machine API key minted from the service's Settings → API Keys pane. |
| **Core** | | |
| `KB_DATABASE_URL` | _(unset)_ | PostgreSQL URL — passed through to `kb-service` (the MCP server itself never opens a DB) |
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
| `KB_OLLAMA_KEEP_ALIVE` | `30m` | Per-request Ollama `keep_alive` sent with every embed call, so only the embedding model stays warm in VRAM |
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

The explorer auto-starts on both instances. Personal gets the default port (`8767`), team gets `8768`. Bookmark both — or use `kb_explore` to swap which KB is on which port at any time.

`KB_INSTANCE_ROLE` controls two things:

1. **Role-specific instructions** prepended to the server description:
   - **`team`** — "This is the TEAM knowledge base — shared decisions, architecture, patterns, and conventions."
   - **`personal`** — "This is your PERSONAL knowledge base — your config, dotfiles, workflow preferences, and private notes."

2. **Tool name prefixing** to avoid collisions when running two instances:
   - **`personal`** → `personal_kb_store`, `personal_kb_search`, etc.
   - **`team`** → `team_kb_store`, `team_kb_search`, etc.
   - **unset** → `kb_store`, `kb_search`, etc. (backwards-compatible default)

   This is essential for MCP clients (like Kiro) that don't namespace tools by server name — without prefixing, both instances would expose identically-named tools.

Environment variables in each `env` block are scoped to that server process — no collisions.

## Steering Your Agents

The MCP server ships with built-in instructions that teach agents *what* each tool does. But agents need additional guidance in your system prompt (e.g. `CLAUDE.md`, Cursor rules, Windsurf rules) to develop good KB habits — searching before acting, storing knowledge as they work, and not letting insights evaporate at session end.

Add something like this to your agent's system prompt:

```markdown
## Knowledge Base

A personal-kb MCP server is available. This is the primary place for
durable knowledge — not memory files or scratchpads.

### Query before you act — not after you fail
Search the KB BEFORE guessing or asking the user. Concrete triggers:

- Starting a task in an unfamiliar project → kb_search(project_ref="X")
- Debugging an error you haven't seen → kb_search the error message
- Making an architectural or library choice → kb_ask("decisions about X")
- Deployment, infrastructure, operational procedures → kb_search first

The cost of a failed kb_search is one second. The cost of not searching
is minutes of wasted fumbling.

### Capture while you still have context
Knowledge evaporates when a session ends. Store entries as you work,
not as an afterthought. If you just debugged something tricky, made a
decision, or discovered a non-obvious behavior — kb_store it immediately.

### What belongs in the KB
- Decisions and their rationale ("chose X because Y")
- Lessons learned from debugging, fixes with non-obvious root causes
- Patterns, conventions, project-specific gotchas
- Factual references: API behaviors, config values, version constraints

### What does NOT belong
- Trivial or temporary info (session state, in-progress work)
- Well-known public knowledge (stdlib docs, basic git usage)
- Duplicates — always kb_search before storing; use update_entry_id if
  an entry already exists

### Good practices
- Use tags for discoverability and project_ref for scoping
- Use hints to build the knowledge graph: {"supersedes": "kb-XXXXX"},
  {"tool": "sqlite"}, {"person": "jason"}
- Use kb_store_batch when capturing multiple related entries
- When a previous entry is wrong or outdated, update or deactivate it
```

The server instructions handle tool mechanics (which tool to use when, parameter formats, entry types). The system prompt guidance above handles *behavior* — when to search, when to store, and what's worth keeping. Both layers together produce agents that actively use the KB without being told.

## Development

```bash
git clone https://github.com/jason-weddington/personal-kb-mcp.git
cd personal-kb-mcp
uv sync

uv run pytest                    # run tests
uv run ruff check src/ tests/    # lint
uv run personal-kb               # run server directly
uv run personal-kb-web           # launch graph explorer web UI
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
