# Personal Knowledge MCP Server

## Quick Reference

- **Run tests**: `uv run pytest`
- **Lint**: `uv run ruff check src/ tests/`
- **Run server directly**: `uv run personal-kb`

## Architecture

- FastMCP async server with stdio transport
- SQLite + sqlite-vec (vector search) + FTS5 (full-text search)
- Ollama for local embeddings (graceful fallback when unavailable)
- All logging goes to stderr (stdout is reserved for MCP stdio transport)

## Key Conventions

- Entry IDs follow the format `kb-XXXXX` (zero-padded)
- All database operations use aiosqlite (async)
- Pydantic models in `src/personal_kb/models/`
- MCP tools in `src/personal_kb/tools/` (one file per tool)
- Tests mirror source structure under `tests/`

## Search Quality Eval

`tests/eval/` contains a regression framework with a controlled corpus (32 entries, 15 golden queries) and a `ControlledEmbedder` that makes vector search deterministic. Two baselines track quality at different layers:

| Baseline | File | Deterministic? | What it measures |
|----------|------|---------------|------------------|
| **Search baseline** | `tests/eval/baseline.json` | Yes (CI-safe) | Raw hybrid search ranking (FTS + vector RRF) |
| **Agent baseline** | `tests/eval/agent_baseline.json` | No (live LLM) | End-to-end agentic retrieval (search + graph + refinement) |

**Search baseline** (MRR=0.85, NDCG=0.89) — run for any change to ranking, RRF weights, decay, or score normalization:

1. Branch off main
2. Make your change
3. `uv run pytest tests/eval/test_baseline.py -s` — regenerates `baseline.json`
4. `git diff tests/eval/baseline.json` — see what moved
5. Commit the updated baseline alongside the code change

**Agent baseline** (MRR=1.00, NDCG=1.00) — run for any change to the agent loop, tool dispatch, fast-path threshold, or prompt. Requires `ANTHROPIC_API_KEY`:

1. `uv run pytest tests/eval/test_agent_baseline.py -s` — regenerates `agent_baseline.json`
2. Check scores AND `turns_used` — regressions may show as more LLM calls even if scores hold
3. Commit the updated baseline alongside the code change

Agent baseline tests are marked `@pytest.mark.eval` and excluded from the pre-push hook (they hit a live API and rewrite the baseline file). Run them manually.

**Both baselines matter.** Search quality affects the agent's fast-path (8/13 queries skip the LLM entirely). Agent quality catches regressions the search baseline misses — the 3 queries that were weak in search (q05, q06, q10) are all perfect with the agent.

## Roadmap

`ROADMAP.md` is a prioritized list of **problems worth solving**, not feature specs. Items describe the pain point and why it matters — the solution gets figured out when we pick it up. Keep it to one screenful. When we finish something, move it to Done as a one-liner and update the priorities. Don't prescribe implementation details in the roadmap; that's wasted effort when we can go from problem to shipped code in a single session.

This is a dogfooding project — we build the KB and use it in the same sessions. When you notice friction using the KB tools (wasted tokens, missing capabilities, awkward workflows), add the problem to ROADMAP.md under Next. You're the primary consumer of this tool; your perspective on what's painful matters.

## Documentation Workflow

Every new feature (not bug fixes) requires updating three things:

1. **`README.md`** — user-facing: getting started, feature overview, deciding whether to use the tool
2. **`how_it_works.md`** — technical deep dive: how things actually work in the code, for maintainers and contributors
3. **KB** (`kb_store`) — capture decisions, architecture, and non-obvious patterns for future sessions

**Process**: Use research agents (subagent_type `Explore`) to deep dive the codebase for exact function signatures, thresholds, data flow, and behavior. Parallelize with multiple agents when researching independent features. Write docs from the research reports — every statement must be verifiable against the code. Don't guess or paraphrase from memory; the agents have the source of truth.

## Commit Convention

This repo uses **conventional commits** enforced by a `commit-msg` hook.

Format: `type(optional-scope): description`

- `feat:` — new feature (bumps minor)
- `fix:` — bug fix (bumps patch)
- `chore:` — maintenance, deps, config (no bump)
- `docs:` — documentation only (no bump)
- `refactor:` — restructuring (no bump)
- `feat!:` or `fix!:` — breaking change (bumps major)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KB_DB_PATH` | `~/.local/share/personal_kb/knowledge.db` | Database file |
| `KB_OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `KB_EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Embedding model |
| `KB_EMBEDDING_DIM` | `1024` | Embedding vector dimensions |
| `KB_OLLAMA_TIMEOUT` | `10.0` | Ollama timeout (seconds) |
| `KB_OLLAMA_MODEL` | `qwen3:4b` | LLM model for Ollama generation |
| `KB_OLLAMA_LLM_TIMEOUT` | `120.0` | Ollama LLM timeout (seconds) |
| `ANTHROPIC_API_KEY` | (unset) | Anthropic API key (enrichment, planning, synthesis) |
| `KB_ANTHROPIC_MODEL` | `claude-haiku-4-5` | Anthropic model for planning/synthesis |
| `KB_ANTHROPIC_TIMEOUT` | `30.0` | Anthropic timeout (seconds) |
| `KB_BEDROCK_MODEL` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock model ID (cross-region inference profile) |
| `KB_BEDROCK_REGION` | `us-east-1` | AWS region for Bedrock |
| `KB_BEDROCK_TIMEOUT` | `30.0` | Bedrock timeout (seconds) |
| `KB_EXTRACTION_PROVIDER` | `anthropic` | LLM for graph enrichment (`anthropic`, `bedrock`, or `ollama`) |
| `KB_QUERY_PROVIDER` | `anthropic` | LLM for query planning/synthesis (`anthropic`, `bedrock`, or `ollama`) |
| `KB_MANAGER` | (unset) | Set `TRUE` for maintenance + ingestion tools |
| `KB_INGEST_MAX_FILE_SIZE` | `5242880` | Max file size in bytes for ingestion (5MB) |
| `KB_INGEST_CHUNK_SIZE` | `16000` | Chunk size in chars for large file ingestion |
| `KB_INGEST_CHUNK_OVERLAP` | `600` | Overlap in chars between adjacent chunks |
| `KB_AGENTIC_INGEST` | `TRUE` | Enable KB-aware dedup during ingestion |
| `KB_INGEST_DEDUP_THRESHOLD` | `0.06` | Hybrid search score threshold for dedup |
| `KB_AGENTIC_QUERY` | `TRUE` | Enable ReAct agent loop for kb_ask auto strategy |
| `KB_AGENTIC_MAX_CALLS` | `4` | Max tool calls in agentic query loop |
| `KB_AGENTIC_SYNTHESIS` | `TRUE` | Enable agentic retrieval + coverage check for kb_summarize |
| `KB_CONTRIBUTOR` | (unset) | Contributor name for entry attribution |
| `KB_TEAM` | (unset) | Team name for entry attribution |
| `KB_PG_POOL_MIN` | `1` | Postgres connection pool minimum size |
| `KB_PG_POOL_MAX` | `5` | Postgres connection pool maximum size |
| `KB_PG_IAM_AUTH` | (unset) | Set `TRUE` for RDS/Aurora IAM authentication |
| `KB_PG_REGION` | `us-east-1` | AWS region for RDS IAM token signing |
| `KB_SKIP_SAFETY` | (unset) | Set `TRUE` to bypass secret scanning on store |
| `KB_INSTANCE_ROLE` | (unset) | `personal` or `team` — prepends role-specific instructions and prefixes tool names (`personal` → `personal_kb_*`, `team` → `team_kb_*`) |
| `KB_LOG_LEVEL` | `WARNING` | Logging level |

## Agent Feedback Loop

Two layers close the feedback loop between agents and the KB maintainer:

**Search telemetry** (`search_events` table) — populated automatically inside `hybrid_search()`. Every query records `query_text`, `result_count`, `top_score`, and `match_source`. Zero token cost, zero agent cooperation needed.

**Agent feedback** (`agent_feedback` table + `kb_feedback` tool) — agent-initiated, structured, negative-only. Always-on (not manager-gated). Three feedback types: `missing` (KB lacked needed knowledge), `unhelpful` (results existed but didn't help), `friction` (tool was awkward or slow).

**Manager explore actions** (in `kb_maintain`, requires `KB_MANAGER=TRUE`):
- `list_feedback` — list recent feedback, filterable by `feedback_type` and `since`
- `summarize_feedback` — pipe feedback to query LLM for theme clustering, falls back to raw list
- `search_stats` — search telemetry overview: total queries, zero-result rate, avg top score, top missed queries
