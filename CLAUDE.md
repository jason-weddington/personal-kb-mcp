# Personal Knowledge MCP Server

## North star and standing rules

- **North star: an agent that learns from experience.** Judge every KB proposal by whether it lowers the cross-session repeat-mistake rate and carries experience across harnesses, models and machines. Whispers, maps, preflight and the listener are delivery details, and harness-native memory is a channel to feed, not a competitor. Resume from `proposals/experience-loop-roadmap.md`. Decision: kb-03728.
- **Prove it in kb-bench before building more.** A memory or delivery feature earns its place with a controlled kb-bench result (an arm that isolates it), not design rationale or uncontrolled telemetry. As of 2026-10-10, eval coverage comes before any net-new memory feature: audit which shipped features lack kb-bench evidence, close those gaps, then continue (see the eval-coverage section of the roadmap). KB memory work follows the science loop in the applied-science repo: failure modes, hypotheses, pre-registered kb-bench tests, build only the winners (kb-03739).
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
- kb-service serves the same MCP tool set over streamable HTTP at `/mcp` (stateless, bearer API keys; `kb_ingest` is stdio-only). The stdio server is deprecated for one release of overlap.

## Key Conventions

- Entry IDs follow the format `kb-XXXXX` (zero-padded)
- All database operations use aiosqlite (async)
- Pydantic models in `src/personal_kb/models/`
- MCP tools in `src/personal_kb/tools/` (one file per tool)
- Tests mirror source structure under `tests/`
- During the stdio overlap every MCP tool exists twice (src/personal_kb/tools/ and packages/kb-service/src/kb_service/mcp_server/tools/); tests/test_mcp_http_parity.py enforces parity; change both together.

## `personal-kb-hook` (the harness hook)

`personal-kb-hook` is a separate, stdlib-only package at `packages/personal-kb-hook/`: one console script that Claude Code runs for six hook events (SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, PostToolUseFailure, Stop). It never imports the main package or touches a database: it talks to kb-service over HTTP (prevention slice and gate, maps index, failure events, turn digests, listener) and is silent on every error path, exiting 0. What each event does, the settings snippet and the scratch files are in `docs/intelligent-kb.md` section 3; keep that section current when the hook changes.

**Install**: `uv tool install` from this repo's `packages/personal-kb-hook` subdirectory (the README has the command). Production installs are pinned to the release tag by the maintainer's post-release hook.

**`.kb_project` convention**: commit a one-line `.kb_project` file at each repo root holding the KB `project_ref`. The hook walks up from the session's `cwd` to it to know which project's maps and lessons to surface. This repo's own `.kb_project` is `personal-kb`.

**uv workspace**: the root `pyproject.toml` declares `[tool.uv.workspace] members = ["packages/*"]`, so `uv sync` installs the hook package editable into `.venv`, and its own suite runs as described in Quick Reference.

## Search Quality Eval

`tests/eval/` contains a regression framework with a controlled corpus (58 entries, 36 queries; the committed search baseline scores 28 of them and the agent baseline 13) and a `ControlledEmbedder` that makes vector search deterministic. Two baselines track quality at different layers:

| Baseline | File | Deterministic? | What it measures |
|----------|------|---------------|------------------|
| **Search baseline** | `tests/eval/baseline.json` | Yes (CI-safe) | Raw hybrid search ranking (FTS + vector RRF) |
| **Agent baseline** | `tests/eval/agent_baseline.json` | No (live LLM) | End-to-end agentic retrieval (search + graph + refinement) |

**Replay experiment** (does the KB stop repeat mistakes? KB off vs session-start slice vs soft gate) lives in `scripts/replay/` — see `scripts/replay/README.md`. Its outputs go to a private eval repo, never this one.

**Surprise detector eval** (precision/recall of the surprise-capture detector per shape on a private labelled set) lives in `scripts/surprise_eval/` — see `scripts/surprise_eval/README.md`. Its cases and outputs go to a private eval repo, never this one.

**Search baseline** (MRR=0.91, NDCG=0.93 as of 2026-06-27) — run for any change to ranking, RRF weights, decay, or score normalization:

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

Two technical docs, one owner each. `how_it_works.md` covers the core KB: the entry model, storage, the write path, supersession and the near-duplicate guard, search and embeddings, the graph, kb_ask and kb_summarize, LLM providers and the llm_calls ledger, ingestion, mental_map as an entry type, preflight, kb-service, the web UI and both MCP surfaces. `docs/intelligent-kb.md` covers everything that fires without an agent asking, or exists to serve something that does: the hook's events, resolutions and trust, failure cues, turn digests, surprise capture, the write policy, the slice, the soft gate, failure context, maps push and tool inventory, the listener, map maintenance endpoints, harness mapping and contracts, measurement and capture cost. Rule of thumb: if it fires without an agent asking, it goes in `docs/intelligent-kb.md`.

**Same commit, not later.** A feature commit updates its owning doc in the same commit, and so does a fix that changes documented behaviour (a default, threshold, status code, env var, endpoint, hook event, table, prompt version). A groomed item names the doc in `files_to_modify` and carries an AC saying which section it updates. User-facing changes also update `README.md`, and decisions and non-obvious patterns go to the KB (`kb_store`).

**Enforced by `tests/test_docs_coverage.py`** (in the default suite): every kb-service route and every env var the shipped code reads must appear in one of the two docs. When it fails, document the thing; do not loosen the test.

**Process**: research the code with agents before writing (exact signatures, thresholds, data flow), write from their reports, and make every statement verifiable against the code. For a large refresh, the six-reader doc audit workflow (inventory, plan, write, adversarial verify, revise) is the model.

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

**Optional post-push hook.** If an executable `./release.post.local.sh` exists (gitignored, never committed), `release.sh` runs it after the release is pushed, as `./release.post.local.sh <version>`; use it for steps that need the pushed release, such as upgrading hosts that install from the git origin. A failure is reported loudly and makes release.sh exit 4, but nothing is rolled back because the release is already public.

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

The env var registry is the configuration reference in `how_it_works.md` (core settings) and the switch tables in `docs/intelligent-kb.md` (automation switches, with production defaults). `tests/test_docs_coverage.py` fails when the code reads a variable neither doc names, so add the row there in the same commit that adds the read.

## Agent Feedback Loop

Two layers close the feedback loop between agents and the KB maintainer:

**Search telemetry** (`search_events` table) — populated automatically inside `hybrid_search()`. Every query records `query_text`, `result_count`, `top_score`, and `match_source`. Zero token cost, zero agent cooperation needed.

**Agent feedback** (`agent_feedback` table + `kb_feedback` tool) — agent-initiated, structured, negative-only. Always-on (not manager-gated). Three feedback types: `missing` (KB lacked needed knowledge), `unhelpful` (results existed but didn't help), `friction` (tool was awkward or slow).

**Manager explore actions** (in `kb_maintain`, requires `KB_MANAGER=TRUE`):
- `list_feedback` — list recent feedback, filterable by `feedback_type` and `since`
- `summarize_feedback` — pipe feedback to query LLM for theme clustering, falls back to raw list
- `search_stats` — search telemetry overview: total queries, zero-result rate, avg top score, top missed queries

## Wire contracts (contracts/)

`contracts/` holds golden fixtures for `GET /api/kb/prevention` and `POST /api/kb/turn` that other harnesses vendor.

Any change to `PreventionResponse` or its nested models, `TurnDigestRequest`, the `TurnItem` union or its item models, `TurnDigestResponse`, the `/api/kb/prevention` query params, `TURN_DIGEST_MAX_BYTES` or `harness_tools.HARNESS_TOOL_MAPS`, including a docstring edit on any of these models, fails `test_contracts_service.py` until you run `uv run python packages/kb-service/scripts/gen_contracts.py`.

A new `TurnItem` kind also needs an example item in `_CLAUDE_CODE_ITEMS` or `_TALOS_ITEMS`.

A breaking change makes the generator refuse to write until you bump `CONTRACT_VERSIONS` (rule in `contracts/README.md`).

Never hand-edit the JSON.
