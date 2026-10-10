# Personal Knowledge MCP Server

## North star and standing rules

- **North star: an agent that learns from experience.** Judge every KB proposal by whether it lowers the cross-session repeat-mistake rate and carries experience across harnesses, models and machines. Whispers, maps, preflight and the listener are delivery details, and harness-native memory is a channel to feed, not a competitor. Resume from `proposals/experience-loop-roadmap.md`. Decision: kb-03728.
- **Prove it in kb-bench before building more.** A memory or delivery feature earns its place with a controlled kb-bench result (an arm that isolates it), not design rationale or uncontrolled telemetry. As of 2026-10-10, eval coverage comes before any net-new memory feature: audit which shipped features lack kb-bench evidence, close those gaps, then continue (see the eval-coverage section of the roadmap).
- **No per-session budgets in KB delivery.** Jason's lead sessions run for weeks without clearing and are the main target of the experience loop. Never limit a gate, whisper, slice or failure-context injection to a count per session or once per session. Use rate limits (per turn, per hour) and re-arm on memory loss (SessionStart with source compact, resume or clear) or after a time window. Decision: kb-03729.

## Quick Reference

- **Run tests (main package)**: `uv run pytest -m "not eval"` — the default for all iteration and CI. The `eval` marker covers the agent-baseline tests, which hit a **live Anthropic API** and rewrite baseline JSON files; **never run the bare `uv run pytest`** in a build/iteration loop (it's slow, nondeterministic, and dirties the tree). Run eval tests manually and deliberately — see "Search Quality Eval" below.
- **Run tests (standalone `personal-kb-hook`)**: `(cd packages/personal-kb-hook && uv run --project ../.. pytest)`. The standalone hook lives in its own package and has its own test suite; both must be green.
- **Lint**: `uv run ruff check src/ tests/ packages/`
- **Type check**: `uv run mypy src/` for the main package; `(cd packages/personal-kb-hook && uv run --project ../.. mypy src/)` for the hook.
- **Run server directly**: `uv run personal-kb`

## For headless build agents

- Iterate tests with `uv run pytest -m "not eval"` (see above). The pre-push hook already excludes eval; match it.
- **Commit and push your feature branch incrementally** — after each meaningful, green step — so partial work survives a dispatch timeout instead of being lost. Don't save the single push for the very end.
- **`sqlite-vec` should work in your sandbox — expect to self-verify.** `sqlite-vec` is pinned to ≥0.1.9, where the native `vec0` extension loads on aarch64 as well as x86_64. So you are normally expected to **run the full suite, including eval tests that hit the `knowledge_vec` vector table, and verify your own change in-sandbox.** Quick probe if unsure: `uv run python -c "import sqlite_vec, sqlite3; db=sqlite3.connect(':memory:'); db.enable_load_extension(True); sqlite_vec.load(db); print(sqlite_vec.loadable_path())"`.
- **Fallback only if the probe shows `sqlite-vec` is genuinely absent here.** If you see ~59 failures of the form `OperationalError: no such table: knowledge_vec` AND the probe above fails, that is the extension not loading in this environment — **environmental, not caused by your change** (confirm by stashing your work and seeing the same failures on the clean base). Do **not** try to fix them. Run the **targeted tests for your change** to prove correctness, then `git push --no-verify` with a comment listing exactly which failures are the environmental `knowledge_vec` ones so the reviewer can run the full gate locally. In your completion comment, state whether `sqlite-vec` loaded and whether you self-verified or are deferring eval verification — that report keeps this guidance honest across hosts.
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

`personal-kb-hook` is a **separate, standalone, zero-dependency package**
that lives at `packages/personal-kb-hook/`. It is wired into the harness's
`SessionStart` and `UserPromptSubmit` hooks and proactively surfaces a
project's `mental_map` directory into the model context. It is
**stdlib-only**, **never touches the DB**, never imports the main
`personal_kb` package, and is silent on every error path.

**Install** (verified — produces a deps-free tool venv with only
`personal_kb_hook` in site-packages):

```
uv tool install --from \
  "git+https://github.com/jason-weddington/personal-kb-mcp.git#subdirectory=packages/personal-kb-hook" \
  personal-kb-hook
```

The MCP-side writer that populates the on-disk JSONL maps index that this
hook reads stays in the main package at
`src/personal_kb/maps_index_writer.py` — it imports `personal_kb.preflight`
and `personal_kb.db.backend`, which are server-only.

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

**Drift guard**: the on-disk path contract (maps-index location + scratch
file location) is duplicated between the main package
(`personal_kb.config.get_maps_index_path` /
`personal_kb.config.get_hook_scratch_path`) and the standalone hook package
(`personal_kb_hook.paths`). A test in the main repo
(`tests/test_path_drift_guard.py`) imports both implementations and asserts
they produce identical paths for representative inputs — including a custom
`KB_DB_PATH`. If you touch one side, touch the other and re-run that test.

**uv workspace**: the root `pyproject.toml` declares
`[tool.uv.workspace] members = ["packages/*"]` and pulls
`personal-kb-hook` in as a workspace dev dep, so `uv sync` installs the
standalone package editable into `.venv` (needed by the drift-guard test and
by the cross-package round-trip test in `tests/test_maps_index_writer.py`).

## Search Quality Eval

`tests/eval/` contains a regression framework with a controlled corpus (32 entries, 15 golden queries) and a `ControlledEmbedder` that makes vector search deterministic. Two baselines track quality at different layers:

| Baseline | File | Deterministic? | What it measures |
|----------|------|---------------|------------------|
| **Search baseline** | `tests/eval/baseline.json` | Yes (CI-safe) | Raw hybrid search ranking (FTS + vector RRF) |
| **Agent baseline** | `tests/eval/agent_baseline.json` | No (live LLM) | End-to-end agentic retrieval (search + graph + refinement) |

**Replay experiment** (does the KB stop repeat mistakes? KB off vs session-start slice vs soft gate) lives in `scripts/replay/` — see `scripts/replay/README.md`. Its outputs go to a private eval repo, never this one.

**Surprise detector eval** (precision/recall of the surprise-capture detector per shape on a private labelled set) lives in `scripts/surprise_eval/` — see `scripts/surprise_eval/README.md`. Its cases and outputs go to a private eval repo, never this one.

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

**The production boundary is the public GitHub repo, not `main`.** Users run this server by `uvx`-ing the package straight from `github.com/jason-weddington/personal-kb-mcp`, so nothing reaches that remote except through a deliberate, vetted release — `./release.sh`. The two remotes have very different trust levels:

| Remote | Host | Role | Push freely? |
|--------|------|------|--------------|
| `origin` | a private git host | Testing / backup | **Yes** — merge to `main` and push liberally |
| `github` | `github.com/jason-weddington/personal-kb-mcp` | **Public** (users `uvx` from it) | **No** — only via `./release.sh` |

The maintainer's own deploy and publish glue lives in a separate private ops repo, not in this tree.

### Day-to-day development (local, liberal)

1. Branch: `git checkout -b feat/...` (or `fix/`, `chore/`, `docs/`).
2. Code + commit on the branch.
3. Test: `uv run pytest -m "not eval"` must pass (the pre-push hook runs the suite + coverage ≥ 80%).
4. Squash-merge to `main`: `git checkout main && git merge --squash feat/... && git commit` (squash message must be a conventional commit — hook-enforced).
5. **Push to `origin` freely**: `git push origin main`. No tags, **never `github`** — a pre-push hook (`scripts/guard_github_push.sh`) refuses any push to `github` that does not come from `./release.sh`.
6. Clean up: `git branch -D feat/...`.

Docs-only pushes (`docs/**`, `proposals/**`, top-level `*.md`) skip the coverage and vitest pre-push gates automatically (`scripts/run_unless_docs_only.sh`), so `--no-verify` is never needed for them.

`main` accumulates verified-locally work between releases. The "stop and wait before merging" gate lives at the **release** boundary below.

### Release (deliberate, promotes to github)

Run `./release.sh` only when local `main` is verified good and you intend to ship
to users. It:

1. Asserts you're on `main` with a clean tree.
2. `uv run semantic-release version --no-push --no-vcs-release` — bumps version, updates `CHANGELOG.md` + `uv.lock`, and tags.
3. Builds `dist/` with exactly four wheels at the release version (`personal-kb`, `kb-core`, `personal-kb-web-service`, `personal-kb-hook`) and runs the optional publish hook (below). Then pushes `main` + tags to **both** remotes: `origin` **and** `github`.
4. Runs `./deploy.sh` if one exists (none in this tree — consumers use `uvx`, which is pull-based).

**Cutting a github release is the vetting checkpoint.** Confirm the work is good
before running `./release.sh`; that is the moment users get the new code.

**Optional publish hook.** An optional maintainer-local hook publishes the built artifacts before anything is pushed, so a remote never advertises a version whose artifacts did not ship. If an executable `./release.local.sh` exists (gitignored, never committed), `release.sh` runs it from the repo root, after the release commit and tag exist locally, as `./release.local.sh <version> <absolute-dist-dir>`. Exit 0 means the artifacts are published; any non-zero exit aborts the release. On any abort the local tag is deleted and nothing is pushed. With no hook present the release aborts as well, unless you pass `./release.sh --no-publish`, which prints a loud notice that no artifacts were published and continues to push.

> Release machinery: `python-semantic-release` (dev dep) + `[tool.semantic_release]`
> in `pyproject.toml`. The legacy per-commit auto-release post-commit hook was
> removed (it bumped the version on every `main` commit, which is incompatible
> with liberal merges). `push` is **not** a valid semantic-release config key in
> v9+ — pushing is controlled by the `--no-push` flag in `release.sh`.

### Pre-release verification

**Always run `scripts/smoke_install.sh` before publishing or releasing.** The
in-workspace test suite does NOT catch missing-runtime-dep / packaging breaks:
`uv sync` installs every workspace member editable, so a sibling package can
be absent from `[project] dependencies` while every test, ruff check, and
mypy pass — and the deploy artifact is silently broken. This is exactly how
the W6 deploy crashed with `ModuleNotFoundError: kb_core` despite 1089 green
tests (see KB kb-01738).

`scripts/smoke_install.sh` runs two checks:

- **Part A — kb-core wheel smoke** (automated): `uv build --package kb-core`,
  create a venv OUTSIDE the workspace, `pip install` the built wheel, and
  round-trip `create_sqlite → store → search`. Proves the published kb-core
  artifact stands on its own.
- **Part B — personal-kb deploy-path smoke** (release-time only — needs the
  release commit pushed first): `uvx --from "personal-kb @ git+ssh://...@<sha>" personal-kb`
  against a throwaway SQLite DB, asserting the FastMCP `Starting MCP server`
  banner with NO `ModuleNotFoundError`. See the script header for the exact
  invocation.

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
| `KB_OLLAMA_KEEP_ALIVE` | `30m` | Per-request Ollama `keep_alive` sent with every embed call, so only the embedding model stays warm in VRAM |
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
| `KB_NEAR_DUPLICATE_FLOOR` | `0.88` | Cosine similarity at or above which kb_store create returns a near-duplicate 409 (same project, active non-map entries) |
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
| `KB_SURPRISE_MIN_CONFIDENCE_SHAPE2` | `0.5` | Surprise-detector confidence floor for shape 2 (overrides the global value) |
| `KB_SURPRISE_MIN_CONFIDENCE_SHAPE3` | `0.7` | Surprise-detector confidence floor for shape 3 (overrides the global value) |
| `KB_SURPRISE_CRITIC_MODEL` | `claude-sonnet-5-5` | Model for the surprise-capture critic pass, which checks each drafted lesson against its evidence before any autonomous write or merge |
| `KB_SURPRISE_LESSON_TTL_DAYS` | `30` | Days an autonomous surprise-capture lesson lives after its last observation (integer 1..3650, anything else falls back to 30); a merge from a new session renews it and the third distinct session makes it permanent |
| `KB_SURPRISE_MIN_CONFIDENCE` | (unset) | Global surprise-detector floor override for both shapes; unset uses the per-shape defaults |
| `KB_SOFT_GATE_MAX_DENIES_PER_TURN` | `1` | Soft gate: most denies (or shadow `would_deny`) per turn; integer 1..1000, anything else falls back to the default with a warning |
| `KB_SOFT_GATE_MAX_DENIES_PER_HOUR` | `6` | Soft gate: most denies in any rolling 60 minutes; integer 1..1000, anything else falls back to the default with a warning |
| `KB_SOFT_GATE_REARM_HOURS` | `24` | Soft gate: hours a lesson stays quiet after it denies before it may deny again (compaction, resume or clear re-arms sooner); integer 1..1000, anything else falls back to the default with a warning |
| `KB_LOG_LEVEL` | `WARNING` | Logging level |

## Agent Feedback Loop

Two layers close the feedback loop between agents and the KB maintainer:

**Search telemetry** (`search_events` table) — populated automatically inside `hybrid_search()`. Every query records `query_text`, `result_count`, `top_score`, and `match_source`. Zero token cost, zero agent cooperation needed.

**Agent feedback** (`agent_feedback` table + `kb_feedback` tool) — agent-initiated, structured, negative-only. Always-on (not manager-gated). Three feedback types: `missing` (KB lacked needed knowledge), `unhelpful` (results existed but didn't help), `friction` (tool was awkward or slow).

**Manager explore actions** (in `kb_maintain`, requires `KB_MANAGER=TRUE`):
- `list_feedback` — list recent feedback, filterable by `feedback_type` and `since`
- `summarize_feedback` — pipe feedback to query LLM for theme clustering, falls back to raw list
- `search_stats` — search telemetry overview: total queries, zero-result rate, avg top score, top missed queries
