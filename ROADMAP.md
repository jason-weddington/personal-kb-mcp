# Roadmap

Problems worth solving, in priority order. Not specs — the "how" gets figured out when we build it.

## Positioning

**Your hard-won lessons aren't locked into any single agent platform.**

Developers jump between Claude Code, Codex, Gemini CLI, Kiro CLI, Cursor — whatever's best right now. MCP makes personal-kb agent-agnostic: your decisions, patterns, and debugging insights follow you. The KB compounds over time regardless of which tool you're using today.

## Now

- **Intra-cluster noise in search results.** Relative score thresholds cut cross-cluster noise (the "fullstack returns Docker entries" problem), but can't distinguish relevant from irrelevant entries within the same topic cluster. When 8 entries share a cluster, RRF ranks them all similarly — the right one is buried among its neighbors. Needs semantic re-ranking: query-entity matching, personalized PageRank seeded from query terms, or LLM-based re-scoring of the top-k candidates.

## Later

- **Single-user only.** No concept of who stored an entry or which agent session produced it. Multi-contributor support (attribution, provenance) is a prerequisite for team use — multiple developers and coding agents contributing to a shared KB.

## Done

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
