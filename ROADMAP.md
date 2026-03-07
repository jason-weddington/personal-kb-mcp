# Roadmap

Problems worth solving, in priority order. Not specs — the "how" gets figured out when we build it.

## Positioning

**Your hard-won lessons aren't locked into any single agent platform.**

Developers jump between Claude Code, Codex, Gemini CLI, Kiro CLI, Cursor — whatever's best right now. MCP makes personal-kb agent-agnostic: your decisions, patterns, and debugging insights follow you. The KB compounds over time regardless of which tool you're using today.

## Now — Explorer Write-Back

The explorer is becoming a first-class human interface into the KB (see kb-00547). Humans and agents access the same KB, each through an interface suited to how they work. Agents create entries (schema-aware); humans correct, update, and ingest.

- **Chat agent tool dispatch.** Give the explorer chat agent internal tools for write operations: `update_entry`, `ingest_url`, `update_metadata`. ~150 LOC in chat.py. The chat session already has DB/embedder/LLM access. All safety checks (secret scanning) apply through the same code path.
- **Update entries via chat.** "That's outdated, update kb-00042 to say..." — user provides the correction, agent validates and writes. Low risk: existing schema/tags/hints preserved.
- **Ingest URLs via chat.** "Add this article to the KB: https://..." — agent fetches content, calls `FileIngester.ingest_content()`. kb_ingest is fully self-contained; no agent pre-processing needed.
- **Metadata tweaks via chat.** "Mark kb-00042 as expires in 30d", "add tag 'postgres' to kb-00042" — simple field updates, no re-embedding needed.
- **Entry creation stays in agent-land.** Do NOT add kb_store creation to the explorer. Schema-aware entry composition (EntryType, graph-aware tags, supersedes hints) requires the full KB context that MCP tool descriptions provide to Opus/Sonnet in Claude Code.

## Next — Audit Fixes

Findings from the [March 2026 code audit](audit.md). Ordered by impact and effort.

### Moderate effort

- **Postgres transaction wrapper.** asyncpg auto-commits per `execute()` on separate pool connections. Multi-step ops (create_entry, delete_cascade) aren't atomic. Need a `transaction()` context manager. (audit M4)
- **~~String-literal-aware placeholder translation.~~** (audit H5) → Done
- **Config validation.** Bare `int()`/`float()` on env vars produces unhelpful errors. Unknown provider values silently return None. (audit M18)

### Ingestion hardening

- **Prompt injection via file content.** Ingested text is interpolated directly into LLM prompts. Malicious files can steer extraction. No easy fix — inherent to the architecture — but could add system prompt hardening, output validation, or content sandboxing. (audit H2)
- **No directory restriction on kb_ingest.** Any file matching extension allowlist anywhere on filesystem is ingestible. Glob patterns traverse freely. Consider base directory allowlist for shared deployments. (audit M1)
- **~~Extracted entries bypass secret scanning.~~** Fixed: per-chunk secret scanning skips chunks with secrets before LLM extraction — secrets never reach the extractor. (audit M8)
- **~~Orphaned state on re-ingestion failure.~~** Fixed: deactivation moved to after successful extraction+storage. (audit M7)
- **~~Fail closed when safety deps missing.~~** Fixed: kb_ingest rejects with install instructions when detect-secrets or scrubadub not installed. KB_SKIP_SAFETY=TRUE overrides. (audit M5)

### Design debt (lower urgency)

- **Shared LLM JSON parser.** The `_FENCE_RE → _JSON_OBJECT_RE → json.loads()` pattern is copy-pasted across 6 files with greedy regex that fails on multi-object responses. Extract to shared `parse_llm_json()` using `json.JSONDecoder().raw_decode()`. (audit M15, L22)
- **Deduplicate ingester code.** `ingest_file` and `ingest_content` share ~175 lines of nearly identical extraction/storage/graph logic. (audit M19)
- **Delete dead VersionStore.** Entirely unused — version history queried via raw SQL in `kb_maintain`. (audit L20)
- **Narrow `query_llm` type.** Typed as `object | None` with isinstance checks everywhere. Should be `LLMProvider | None`. (audit M20)
- **Agent tool call deduplication.** ReAct loop doesn't detect identical repeated calls. Each duplicate burns a turn. (audit M14)

## Later

- **Conflict detection at write time.** Multiple contributors can store contradictory information with no signal. When a new entry is stored, the enricher should search for semantically similar entries and classify the relationship (supports, refines, conflicts_with, unrelated). `conflicts_with` edges get stored in the graph with the LLM's reasoning. At read time, formatters surface a `[CONFLICTING: kb-XXXXX]` badge so agents see both sides. Resolution uses the existing `supersedes` mechanism. Cost: one extra hybrid search + a few lines in the enricher prompt per store. Needs more design work before building — edge cases around context-dependent "conflicts" (different projects, different scopes) and how to avoid false positives.
- **Multi-user access control.** Current model has attribution but zero isolation — any contributor can read/modify/delete any entry. Fine for trusted teams. If multi-tenant isolation is needed: Postgres row-level security or app-level access checks. (audit H1)
- **Intra-cluster noise in search results.** Relative score thresholds cut cross-cluster noise but can't distinguish within a topic cluster. Needs semantic re-ranking: query-entity matching, personalized PageRank, or LLM-based re-scoring of top-k candidates.

## Done

- Explorer Chat — `generate_chat(messages)` on LLMProvider protocol + all 3 clients, `ChatSession` with token budget, `/api/chat/stream` SSE endpoint, iMessage-style chat UI with slide animation, clickable citations that fly to graph nodes.
- Graph Explorer Phase 1-2 + animated graph — `kb_explore` MCP tool, force-graph visualization, live web server (`personal-kb-web`), SSE streaming with agent traversal animation (node glow, particles, staggered reveal, progressive camera widening), query routing (explore/summarize), markdown rendering, info panel with on-demand entry details.
- Batch store partial failure reporting — per-entry try/except with clear error messages so agents can retry failed items. (audit H8)
- Audit quick wins (H3, H4, H6, M2, M3, M11) — symlink rejection, URL content size limit, sensitivity enum validation, FTS quote stripping, deactivated entry post-filter, scope propagation in auto strategy.
- String-literal-aware placeholder translation — `_translate_placeholders` now uses a quote-aware state machine instead of naive regex, preserving `?` inside SQL string literals. (audit H5)
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
