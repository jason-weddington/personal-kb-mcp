# Roadmap

Problems worth solving, in priority order. Not specs — the "how" gets figured out when we build it.

## Positioning

**Your hard-won lessons aren't locked into any single agent platform.**

Developers jump between Claude Code, Codex, Gemini CLI, Kiro CLI, Cursor — whatever's best right now. MCP makes personal-kb agent-agnostic: your decisions, patterns, and debugging insights follow you. The KB compounds over time regardless of which tool you're using today.

## Now — Audit Fixes

Findings from the [March 2026 code audit](audit.md). Ordered by impact and effort.

### Quick wins (1-line fixes, high value)

- **Symlink protection in ingestion.** `Path.glob` follows symlinks; deny-list checks filename not target. A symlink `docs/notes.md -> /etc/shadow` passes all checks. Add `path.is_symlink()` guard. (audit H3)
- **URL ingestion content size limit.** `ingest_content()` has no max length — file path has 5MB cap but URL path has none. DoS via memory exhaustion. (audit H4)
- **Validate sensitivity enum.** Accepts arbitrary strings (`"BANANA"` renders as `[BANANA]` badge). Restrict to `Literal["internal", "restricted", "public"] | None`. (audit H6)
- **Filter deactivated entries from vector search.** `vector_search()` in both backends doesn't check `is_active`. Known issue, long-deferred. Join or post-filter. (audit M3)
- **Strip quotes from FTS query tokens.** Input containing `"` produces malformed FTS5 syntax, silently returns zero results. (audit M2)
- **Propagate scope in auto strategy.** `_auto_search_entries` accepts `scope` but never uses it — `kb_ask` with scope filter is silently ignored. (audit M11)

### Moderate effort

- **Atomic batch store.** If entry 5/10 fails, entries 1-4 are already committed. No rollback, no partial-success reporting. Wrap in transaction or catch per-entry and report. (audit H8)
- **Postgres transaction wrapper.** asyncpg auto-commits per `execute()` on separate pool connections. Multi-step ops (create_entry, delete_cascade) aren't atomic. Need a `transaction()` context manager. (audit M4)
- **String-literal-aware placeholder translation.** `_translate_placeholders` naively replaces all `?` including inside string literals. Latent bug — first SQL with a literal `?` gets silently corrupted on Postgres. (audit H5)
- **Config validation.** Bare `int()`/`float()` on env vars produces unhelpful errors. Unknown provider values silently return None. (audit M18)

### Ingestion hardening

- **Prompt injection via file content.** Ingested text is interpolated directly into LLM prompts. Malicious files can steer extraction. No easy fix — inherent to the architecture — but could add system prompt hardening, output validation, or content sandboxing. (audit H2)
- **No directory restriction on kb_ingest.** Any file matching extension allowlist anywhere on filesystem is ingestible. Glob patterns traverse freely. Consider base directory allowlist for shared deployments. (audit M1)
- **Extracted entries bypass secret scanning.** `_store_extracted_entry()` calls `create_entry()` directly, skipping `_check_secrets()`. (audit M8)
- **Orphaned state on re-ingestion failure.** Old entries deactivated before new extraction — if extraction fails, data is lost. Deactivate after success instead. (audit M7)
- **Warn at startup when safety deps missing.** `detect-secrets` and `scrubadub` are optional. When absent, secret/PII scanning silently passes all content with no warning. (audit M5)

### Design debt (lower urgency)

- **Shared LLM JSON parser.** The `_FENCE_RE → _JSON_OBJECT_RE → json.loads()` pattern is copy-pasted across 6 files with greedy regex that fails on multi-object responses. Extract to shared `parse_llm_json()` using `json.JSONDecoder().raw_decode()`. (audit M15, L22)
- **Deduplicate ingester code.** `ingest_file` and `ingest_content` share ~175 lines of nearly identical extraction/storage/graph logic. (audit M19)
- **Delete dead VersionStore.** Entirely unused — version history queried via raw SQL in `kb_maintain`. (audit L20)
- **Narrow `query_llm` type.** Typed as `object | None` with isinstance checks everywhere. Should be `LLMProvider | None`. (audit M20)
- **Agent tool call deduplication.** ReAct loop doesn't detect identical repeated calls. Each duplicate burns a turn. (audit M14)

## Later

- **Multi-user access control.** Current model has attribution but zero isolation — any contributor can read/modify/delete any entry. Fine for trusted teams. If multi-tenant isolation is needed: Postgres row-level security or app-level access checks. (audit H1)
- **Intra-cluster noise in search results.** Relative score thresholds cut cross-cluster noise but can't distinguish within a topic cluster. Needs semantic re-ranking: query-entity matching, personalized PageRank, or LLM-based re-scoring of top-k candidates.

## Done

- Wire update params through kb_store — `short_title`, `long_title`, `entry_type`, `project_ref`, `source_context` now applied on update instead of silently ignored. `entry_type` default changed to `None` to prevent overwriting. (audit H7)
- Fix conditional test assertions — `pytest.importorskip` for optional deps, unconditional asserts, meaningful threshold test.
- Multi-user Phase 2 & 3 — `@contributor/team` attribution badges in search output, contributor/team filters on kb_search, `list_contributors` maintain action, audit events table (`list_audit`), sensitivity field (internal/restricted/public classification).
- Multi-user Phase 1 — server-side identity injection (`KB_CONTRIBUTOR`, `KB_TEAM`), contributor/team/updated_by columns, atomic entry ID generation, advisory locks for Postgres schema migration, configurable pool sizes, deployment config table (embedding model consistency), secret scanning on kb_store path.

- Agent feedback loop — search telemetry (automatic in hybrid_search) + kb_feedback tool (missing/unhelpful/friction) + manager explore actions (list_feedback, summarize_feedback, search_stats).
- Agentic synthesis — kb_summarize retrieves via agent loop, coverage check fills gaps, structured entries for richer synthesis. Toggle: `KB_AGENTIC_SYNTHESIS=FALSE`.
- Agentic ingestion — markdown-aware chunking at H1/H2 headings, running context across chunks, KB-aware dedup agent (search + LLM confirm). Toggle: `KB_AGENTIC_INGEST=FALSE`.
- Agentic query planning — ReAct agent loop for kb_ask auto strategy. Fast-path skips LLM on strong results; 6 internal tools; ScriptedLLM for deterministic testing. Toggle: `KB_AGENTIC_QUERY=FALSE`.
- PostgreSQL backend — Database Protocol abstraction, pgvector + tsvector, migration script with auto-embed. SQLite remains default.
- Graph-boosted hybrid ranking — researched, rejected. Co-citation is query-agnostic; rewards connectivity not specificity. Net lateral on eval. Branch `feat/graph-proximity-rrf` preserved.
- Search quality eval framework — controlled corpus, golden queries, baseline snapshot (MRR=0.85, recall@5=1.0, NDCG@5=0.89).
- Entity dedup, access-aware decay, sparse graph hints — three graph quality improvements grounded in GraphRAG research.
- Improve tool descriptions for query tools — differentiate kb_search, kb_ask, kb_summarize; ungate kb_ingest with glob support.
- Amazon Bedrock LLM provider — BedrockLLMClient with async-native SDK, smithy-json newline workaround.
- Ollama, Anthropic, and Bedrock provider switching — configurable extraction and query LLM backends.
- Knowledge graph with LLM enrichment — entity extraction, relationship edges, graph traversal queries.
- File ingestion pipeline — deny-list, secrets scanning, PII redaction, LLM summarize + extract.
- Token efficiency — compact output, kb_get two-phase retrieval, kb_store_batch with batch enrichment.
- kb_get skips inactive entries — no stale knowledge leaking into agent context.
- Show long_title in compact search results — one extra line per result dramatically improves discoverability.
