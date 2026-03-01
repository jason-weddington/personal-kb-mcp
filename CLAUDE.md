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

`tests/eval/` contains a regression framework with a controlled corpus (32 entries, 15 golden queries) and a `ControlledEmbedder` that makes vector search deterministic.

**Baseline workflow** — when making ranking changes (RRF weights, decay formula, score normalization):

1. Branch off main
2. Make your change
3. `uv run pytest tests/eval/test_baseline.py -s` — regenerates `tests/eval/baseline.json`
4. `git diff tests/eval/baseline.json` — see what moved
5. Commit the updated baseline alongside the code change

Current weak spots: q05 (REST auth, MRR=0.33), q06 (CORS, MRR=0.25), q10 (encoding bug, MRR=0.50) — right entries found but ranked too low.

## Roadmap

`ROADMAP.md` is a prioritized list of **problems worth solving**, not feature specs. Items describe the pain point and why it matters — the solution gets figured out when we pick it up. Keep it to one screenful. When we finish something, move it to Done as a one-liner and update the priorities. Don't prescribe implementation details in the roadmap; that's wasted effort when we can go from problem to shipped code in a single session.

This is a dogfooding project — we build the KB and use it in the same sessions. When you notice friction using the KB tools (wasted tokens, missing capabilities, awkward workflows), add the problem to ROADMAP.md under Next. You're the primary consumer of this tool; your perspective on what's painful matters.

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
| `KB_INGEST_MAX_FILE_SIZE` | `512000` | Max file size in bytes for ingestion |
| `KB_LOG_LEVEL` | `WARNING` | Logging level |
