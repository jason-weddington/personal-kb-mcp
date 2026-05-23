# personal-kb — Testing Guide

## Testing Approach

The test suite is organized around three concerns:

1. **Unit tests** — isolated module-level correctness (search ranking, confidence decay, graph
   queries, tool handlers, LLM provider behavior)
2. **Integration tests** — multi-component flows against an in-memory SQLite backend
3. **Eval baselines** — quantitative search quality regression against a controlled corpus

The guiding principle: tests at each level are independent. Unit tests mock external dependencies
(Ollama, LLM providers) using `AsyncMock`. Integration tests use a real in-memory DB but mock
LLM calls. Eval tests use a deterministic embedder to make vector search reproducible in CI.

**Key invariant**: the pre-push hook runs the full unit+integration suite with coverage.
Eval tests (`@pytest.mark.eval`) are excluded from the pre-push hook because they hit a live
API or regenerate baseline files.

## Test Environments

### Development (local)

```bash
uv run pytest                           # all non-eval tests
uv run pytest -m "not eval" --cov       # with coverage report
uv run pytest tests/search/             # specific module
uv run pytest tests/tools/test_kb_search.py -v  # specific file
```

### CI (pre-push hook)

```bash
uv run --frozen pytest -m "not eval" --cov --cov-report=term-missing
```

Coverage threshold: **80%** (`fail_under = 80.0` in `pyproject.toml`). The pre-push hook fails
if coverage drops below this threshold. When you add tests that increase coverage, ratchet up
the `fail_under` value to lock in the gain.

Coverage omits files listed in `[tool.coverage.run] omit` in `pyproject.toml`.

### Eval (manual only)

Eval tests require live external services or regenerate baseline files. Run manually:

```bash
# Search quality baseline (deterministic, no API key needed)
uv run pytest tests/eval/test_baseline.py -s

# Agent quality baseline (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-ant-... uv run pytest tests/eval/test_agent_baseline.py -s
```

## Test Data

### Controlled corpus (`tests/eval/corpus.json`)

32 synthetic entries covering the test vocabulary. Fields per entry:
- `short_title`, `long_title`, `knowledge_details`
- `entry_type` (one of the four values)
- `tags`, `project_ref`
- `confidence_level` (default 0.9)
- `days_old` (backdating for confidence decay tests)
- `deactivate` (soft-delete flag for testing inactive entry filtering)
- `hints` (for graph edge tests)

### Golden queries (`tests/eval/queries.json`)

15 test queries, each with:
- `id` (q01–q15)
- Query text
- Expected top-k results (list of entry IDs)
- Relevance judgments (graded)

### ControlledEmbedder (`tests/eval/conftest.py`)

Makes vector search deterministic for CI. Key properties:
- Vectors registered by text prefix via `register(name, vector)`
- Unregistered text gets a stable hash-based fallback vector
- Vectors derived from `similarity_map.json` cluster definitions — entries and queries
  in the same cluster get aligned vectors so RRF ranking is predictable
- Methods: `embed(text)`, `store_embedding(entry_id, embedding)`,
  `search_similar(embedding, limit)`, `is_available()` (always returns True)
- Default dimension: 64 (lower than production 1024 — sufficient for ranking tests)

### Fixtures (`tests/conftest.py`)

Shared fixtures available across all tests:

| Fixture | Scope | Description |
|---|---|---|
| `db` | function | In-memory SQLite backend with full schema |
| `embedder` | function | ControlledEmbedder (or null embedder for non-search tests) |
| `store` | function | KnowledgeStore backed by in-memory db |
| `llm` | function | ScriptedLLM (deterministic, no API calls) for agent loop tests |

## Unit Tests

Tests live under `tests/` mirroring `src/personal_kb/`:

```
tests/
├── confidence/
│   └── test_decay.py          # compute_effective_confidence, staleness_warning
├── db/
│   ├── test_sqlite_backend.py # SQLite backend operations
│   └── test_postgres_backend.py # Postgres backend (skipped without asyncpg)
├── graph/
│   ├── test_agent.py          # ReAct loop tool calls, fast-path threshold
│   ├── test_enricher.py       # Entity extraction, fuzzy dedup
│   ├── test_builder.py        # Deterministic edge generation
│   ├── test_queries.py        # BFS, path-finding, supersedes chain
│   └── test_planner.py        # Query plan generation
├── ingest/
│   ├── test_ingester.py       # Pipeline: file validation, chunking, extraction
│   ├── test_chunker.py        # Markdown-aware chunking at H1/H2
│   ├── test_extractor.py      # LLM extraction output parsing
│   ├── test_safety.py         # Secret detection, PII redaction
│   └── test_dedup_agent.py    # Dedup threshold and LLM confirmation
├── llm/
│   ├── test_anthropic.py      # Client construction, generate() mocking
│   ├── test_bedrock.py        # Bedrock SDK mocking
│   └── test_ollama.py         # Ollama client, timeout handling
├── search/
│   ├── test_hybrid.py         # RRF fusion, score normalization, telemetry
│   ├── test_fts.py            # BM25 query construction, quote stripping
│   └── test_vector.py         # KNN search, graceful fallback
├── store/
│   └── test_knowledge_store.py # CRUD, ID generation, version increment
└── tools/
    ├── test_kb_search.py
    ├── test_kb_ask.py
    ├── test_kb_store.py
    ├── test_kb_get.py
    ├── test_kb_ingest.py
    ├── test_kb_summarize.py
    ├── test_kb_maintain.py
    └── ...
```

**Test style conventions**:
- `from __future__ import annotations` at top of every test file
- Test classes use `Test` prefix: `class TestHybridSearch:`
- All test methods annotated `-> None`
- `asyncio_mode = "auto"` in `pyproject.toml` — no `@pytest.mark.asyncio` decorator needed
- Docstrings on source modules and functions; not required on test functions
- D rules suppressed under `tests/` (no docstring requirement)

**Mocking pattern** for async LLM clients:

```python
from unittest.mock import AsyncMock, patch

async def test_enricher_entity_extraction() -> None:
    llm = AsyncMock()
    llm.generate.return_value = '{"entities": [{"name": "sqlite", "type": "tool"}]}'
    enricher = GraphEnricher(db, llm)
    ...
```

**Skipping optional dependency tests**:

```python
asyncpg = pytest.importorskip("asyncpg")  # skip entire module if asyncpg not installed
```

## Integration Tests

Location: `tests/integration/`

Integration tests exercise end-to-end flows — from tool call to DB query — using a real
in-memory SQLite DB but mocked LLM and Ollama:

- `test_store_search_cycle.py` — store an entry, search for it, retrieve by ID
- `test_ingest_pipeline.py` — ingest a markdown file, verify entries created
- `test_graph_traversal.py` — store linked entries, verify BFS finds them
- `test_agentic_query.py` — kb_ask with ScriptedLLM, verify tool call sequence

Integration tests are included in the normal `pytest` run (not marked `eval`).

## Test Automation

### Pre-push hook (CI gate)

Defined in `.pre-commit-config.yaml`:

```yaml
- id: coverage
  name: coverage
  entry: uv run --frozen pytest -m "not eval" --cov --cov-report=term-missing
  language: system
  pass_filenames: false
  always_run: true
  stages: [pre-push]
```

Runs on every `git push`. Fails if any test fails or coverage drops below 80%.

**Note**: uses `--frozen` to prevent uv from rebuilding the package mid-hook, which causes
spurious "files were modified" pre-commit failures.

### Other hooks (per-commit)

| Hook | Stage | Purpose |
|---|---|---|
| `trailing-whitespace`, `end-of-file-fixer` | pre-commit | Formatting |
| `check-yaml`, `check-toml` | pre-commit | Config validation |
| `check-added-large-files (--maxkb=1000)` | pre-commit | Prevent large binary commits |
| `gitleaks` | pre-commit | Secret detection in committed files |
| `ruff --fix` + `ruff-format` | pre-commit | Lint and format |
| `conventional-commit` | commit-msg | Enforce conventional commits on main only |
| `semantic-release` | post-commit | Auto-version on main (non-release commits) |
| `vulture (--min-confidence 80)` | pre-commit | Dead code detection |
| `mypy src/` | pre-commit | Type checking |

## Manual Testing

### Testing a new tool

1. `uv run personal-kb` to start the server (will wait for MCP stdio — kill with Ctrl+C after
   verifying startup logs).
2. From Claude Code: use the tool with various parameter combinations.
3. Check `~/.local/share/personal_kb/log.txt` for server-side logs.
4. Verify the DB state: `KB_LOG_LEVEL=DEBUG uv run personal-kb` for verbose output.

### Testing vector search

1. Start Ollama: `ollama serve`
2. Verify model: `ollama list` shows `qwen3-embedding:0.6b`
3. Store a test entry: `kb_store(...)` — check `has_embedding=True` in `kb_get` response
4. Search: `kb_search(query="...")` — `match_source` should be `hybrid` (not `fts`)

### Testing PostgreSQL backend

```bash
KB_DATABASE_URL=postgresql://user:pass@localhost/testdb \
KB_LOG_LEVEL=INFO \
uv run personal-kb
# Expected: "Connecting to PostgreSQL database"
```

For Postgres tests: `uv run pytest tests/db/test_postgres_backend.py -v`
(requires a live Postgres instance; skipped automatically if asyncpg not installed)

## Validation Criteria

A change is considered correct when:

1. `uv run pytest -m "not eval"` passes with ≥ 80% coverage
2. `uv run mypy src/` exits 0 (no type errors)
3. `uv run ruff check src/ tests/` exits 0 (no lint violations)
4. For search ranking changes: `tests/eval/test_baseline.py` regenerated and MRR/NDCG do not
   regress (mean_mrr ≥ 0.85, mean_ndcg_at_k ≥ 0.89)
5. For agent loop changes: `tests/eval/test_agent_baseline.py` regenerated and scores do not
   regress (MRR = 1.00, NDCG = 1.00) and `turns_used` does not increase

## Bug Reporting

When filing a bug:
- Minimum reproduction: Python version, relevant env vars, exact tool call, observed vs expected output
- For search quality bugs: run `kb_maintain(action="search_stats")` and include the output
- For agent loop bugs: set `KB_LOG_LEVEL=DEBUG` and include the log output
- Check `tests/eval/baseline.json` — if the query appears in the golden set, the eval baseline
  should catch regressions after your fix

## Regression Testing

### Search quality baseline

The `tests/eval/baseline.json` file records search quality metrics (MRR=0.85, NDCG=0.89)
against the controlled corpus. It must be regenerated and committed alongside any change to:
- RRF fusion weights or formula (`search/hybrid.py`)
- BM25 query construction (`search/fts.py`)
- Vector search ranking (`search/vector.py`)
- Confidence decay formula or half-lives (`confidence/decay.py`)
- Score normalization or thresholding

**To regenerate**:
```bash
uv run pytest tests/eval/test_baseline.py -s
git diff tests/eval/baseline.json   # review what moved
git add tests/eval/baseline.json    # commit with the code change
```

**Per-query metrics** in `baseline.json`: `mrr`, `recall_at_k`, `ndcg_at_k`, `top_k`,
`result_count`. A change that improves some queries while regressing others is a tradeoff —
check overall mean metrics.

### Agent quality baseline

The `tests/eval/agent_baseline.json` file records end-to-end agentic retrieval quality
(MRR=1.00, NDCG=1.00). It requires a live `ANTHROPIC_API_KEY` and must be regenerated
alongside any change to:
- The ReAct agent loop (`graph/agent.py`)
- Tool dispatch logic in the agent
- Fast-path threshold
- Query planner (`graph/planner.py`)

**To regenerate**:
```bash
ANTHROPIC_API_KEY=sk-ant-... uv run pytest tests/eval/test_agent_baseline.py -s
git diff tests/eval/agent_baseline.json   # verify scores and turns_used
```

**Watch `turns_used`**: regressions sometimes show as more LLM calls even when scores hold.
Increasing `turns_used` means the fast-path is firing less often — investigate why.

Both baselines are committed to the repository. The `@pytest.mark.eval` marker identifies them:

```python
@pytest.mark.eval
async def test_search_baseline() -> None:
    ...
```

Eval tests are excluded from `pytest -m "not eval"` (the pre-push hook invocation) and must
be run manually.

## Pointers

- `tests/eval/conftest.py` — `ControlledEmbedder`, corpus loading, DB setup for eval
- `tests/eval/baseline.json` — current search baseline (MRR=0.85, NDCG=0.89)
- `tests/eval/agent_baseline.json` — current agent baseline (MRR=1.00, NDCG=1.00)
- `tests/eval/corpus.json` — 32 controlled test entries
- `tests/eval/queries.json` — 15 golden queries with relevance judgments
- `tests/eval/metrics.py` — MRR, NDCG, recall@k calculation functions
- `tests/eval/similarity_map.json` — cluster definitions for ControlledEmbedder
- `pyproject.toml` — `[tool.pytest.ini_options]` (asyncio_mode, markers) and `[tool.coverage.report]`
- `.pre-commit-config.yaml` — pre-push coverage hook definition
- `CLAUDE.md` — Search Quality Eval section: when to run each baseline, what to commit
