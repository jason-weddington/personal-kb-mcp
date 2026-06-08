# Personal Knowledge MCP Server

## Quick Reference

- **Run tests**: `uv run pytest -m "not eval"` — the default for all iteration and CI. The `eval` marker covers the agent-baseline tests, which hit a **live Anthropic API** and rewrite baseline JSON files; **never run the bare `uv run pytest`** in a build/iteration loop (it's slow, nondeterministic, and dirties the tree). Run eval tests manually and deliberately — see "Search Quality Eval" below.
- **Lint**: `uv run ruff check src/ tests/`
- **Run server directly**: `uv run personal-kb`

## For headless build agents

- Iterate tests with `uv run pytest -m "not eval"` (see above). The pre-push hook already excludes eval; match it.
- **Commit and push your feature branch incrementally** — after each meaningful, green step — so partial work survives a dispatch timeout instead of being lost. Don't save the single push for the very end.
- **`sqlite-vec` may be missing in your sandbox.** If you see ~59 failures of the form `OperationalError: no such table: knowledge_vec`, that is the native `sqlite-vec` extension not being loaded in this environment — it is **environmental, not caused by your change** (confirm by stashing your work and seeing the same failures on the clean base). Do **not** try to fix them. Run the **targeted tests for your change** to prove correctness, then `git push --no-verify` with a comment listing exactly which failures are the environmental `knowledge_vec` ones so the reviewer can run the full gate locally where `sqlite-vec` is installed.
- **Once you've pushed your final branch and posted your completion comment, you are done — stop.** Do not keep trying to make the full in-sandbox suite pass; you'll just burn the dispatch budget to a timeout on environmental failures you can't fix.

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

## `personal-kb-hook` (CLI hook for mental_map surfacing)

A second console script — `personal-kb-hook` — ships alongside `personal-kb`
and `personal-kb-web`. It is wired into the harness's `SessionStart` and
`UserPromptSubmit` hooks and proactively surfaces a project's `mental_map`
directory into the model context. It is **stdlib-only**, **never touches the
DB**, and is silent on every error path.

**Install**: `uv tool install --from git+https://github.com/jason-weddington/personal-kb-mcp.git personal-kb-hook`.

**Wire it up**: in `~/.claude/settings.json`, add a hook entry for both
`SessionStart` and `UserPromptSubmit` invoking `personal-kb-hook --format=claude-json`
(see the README for the full snippet).

**`.kb_project` convention**: commit a one-line `.kb_project` file at each
repo root containing the KB `project_ref`. The hook walks up from the
session's `cwd` to that file to know which project's maps to surface. This
repo's own `.kb_project` is `personal-kb`.

The hook reads only `<KB_DB_PATH dir>/maps_index.jsonl`, which the MCP server
writes on `mental_map` create/update/deactivate. Per-session suppression
lives in `~/.cache/personal_kb/injected-<session_id>.json`.

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

## Branch, Merge & Release Policy

**The production boundary is GitHub, not `main`.** A ~60-person engineering org
runs this server by `uvx`-ing the package straight from the `github` remote
(`jason-weddington/personal-kb-mcp`). So **nothing reaches github except through
a deliberate, vetted release** — `./release.sh`. The two remotes have very
different trust levels:

| Remote | Host | Role | Push freely? |
|--------|------|------|--------------|
| `origin` | `git-host` (home lab) | Testing / backup | **Yes** — merge to `main` and push liberally |
| `github` | `github.com/jason-weddington/personal-kb-mcp` | **Production** (team `uvx`'s from it) | **No** — only via `./release.sh` |

### Day-to-day development (local, liberal)

1. Branch: `git checkout -b feat/...` (or `fix/`, `chore/`, `docs/`).
2. Code + commit on the branch.
3. Test: `uv run pytest` must pass (the pre-push hook runs the full suite + coverage ≥ 80%).
4. Squash-merge to `main`: `git checkout main && git merge --squash feat/... && git commit` (squash message must be a conventional commit — hook-enforced).
5. **Push to `origin` freely**: `git push origin main`. This lands on the home-lab VM for testing. No tags, **never `github`**.
6. Clean up: `git branch -D feat/...`.

Merging to `main` and pushing to `origin` no longer requires waiting for manual
testing — `main` accumulates verified-locally work between releases, and `origin`
is the home-lab testing target. The old "stop and wait before merging" gate has
moved to the **release** boundary below.

### Release (deliberate, promotes to github)

Run `./release.sh` only when local `main` is verified good and you intend to ship
to the team. It:

1. Asserts you're on `main` with a clean tree.
2. `uv run semantic-release version --no-push --no-vcs-release` — bumps version, updates `CHANGELOG.md` + `uv.lock`, and tags.
3. Pushes `main` + tags to **both** remotes: `origin` **and** `github`.
4. Runs `./deploy.sh` if one exists (none today — the team consumes via `uvx`, which is pull-based).

**Cutting a github release is the vetting checkpoint.** Confirm the work is good
before running `./release.sh`; that is the moment 60 people get the new code.

> Release machinery: `python-semantic-release` (dev dep) + `[tool.semantic_release]`
> in `pyproject.toml`. The legacy per-commit auto-release post-commit hook was
> removed (it bumped the version on every `main` commit, which is incompatible
> with liberal merges). `push` is **not** a valid semantic-release config key in
> v9+ — pushing is controlled by the `--no-push` flag in `release.sh`.

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
| `KB_BEDROCK_TIMEOUT` | `60.0` | Bedrock timeout (seconds) |
| `KB_AWS_PROFILE` | (unset) | AWS profile name for Bedrock credentials (uses boto3 credential chain). Falls back to `personal_kb_bedrock` profile if it exists |
| `KB_EXTRACTION_PROVIDER` | `anthropic` | LLM for graph enrichment (`anthropic`, `bedrock`, or `ollama`) |
| `KB_QUERY_PROVIDER` | `anthropic` | LLM for query planning/synthesis (`anthropic`, `bedrock`, or `ollama`) |
| `KB_MANAGER` | (unset) | Set `TRUE` for maintenance + ingestion tools |
| `KB_INGEST_MAX_FILE_SIZE` | `10485760` | Max file size in bytes for ingestion (10MB) |
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
| `KB_AUTO_EXPLORE` | `TRUE` | Auto-start explorer web server on MCP server startup |
| `KB_EXPLORE_PORT` | `8767` | Port for the explorer web server |
| `KB_LOG_LEVEL` | `WARNING` | Logging level |

## Agent Feedback Loop

Two layers close the feedback loop between agents and the KB maintainer:

**Search telemetry** (`search_events` table) — populated automatically inside `hybrid_search()`. Every query records `query_text`, `result_count`, `top_score`, and `match_source`. Zero token cost, zero agent cooperation needed.

**Agent feedback** (`agent_feedback` table + `kb_feedback` tool) — agent-initiated, structured, negative-only. Always-on (not manager-gated). Three feedback types: `missing` (KB lacked needed knowledge), `unhelpful` (results existed but didn't help), `friction` (tool was awkward or slow).

**Manager explore actions** (in `kb_maintain`, requires `KB_MANAGER=TRUE`):
- `list_feedback` — list recent feedback, filterable by `feedback_type` and `since`
- `summarize_feedback` — pipe feedback to query LLM for theme clustering, falls back to raw list
- `search_stats` — search telemetry overview: total queries, zero-result rate, avg top score, top missed queries
