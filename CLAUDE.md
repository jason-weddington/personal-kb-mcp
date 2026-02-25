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
| `KB_LLM_MODEL` | `qwen3:4b` | LLM model for graph enrichment |
| `KB_LLM_TIMEOUT` | `120.0` | LLM generation timeout (seconds) |
| `KB_MANAGER` | (unset) | Set `TRUE` for maintenance tools |
| `KB_LOG_LEVEL` | `WARNING` | Logging level |
