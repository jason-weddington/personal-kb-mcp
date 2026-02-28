# Roadmap

Problems worth solving, in priority order. Not specs — the "how" gets figured out when we build it.

## Positioning

**Your hard-won lessons aren't locked into any single agent platform.**

Developers jump between Claude Code, Codex, Gemini CLI, Kiro CLI, Cursor — whatever's best right now. MCP makes personal-kb agent-agnostic: your decisions, patterns, and debugging insights follow you. The KB compounds over time regardless of which tool you're using today.

## Now

**Agents default to kb_search and miss graph connections.** Despite improved tool descriptions, agents still hammer kb_search with keyword variations instead of using `kb_ask strategy:related` to traverse the knowledge graph. This means serendipitous discovery — the KB's most unique value — rarely happens. The solution isn't changing agent behavior — it's making the graph work *through* kb_search. Four steps, grounded in a survey of ~12 GraphRAG papers (see kb-00111, kb-00112):

1. **Entity deduplication in the enricher.** Nodes like `concept:async-io` and `technology:asyncio` fragment what should be a single entity. Check existing vocabulary via fuzzy match before creating new nodes. Prerequisite for everything below — graph quality determines graph signal quality. *(GraphRAG entity resolution, LightRAG — Size: S)*

2. **Graph-boosted hybrid ranking (3-signal RRF).** Add graph proximity as a third signal alongside BM25 and vector in `hybrid_search`. For each candidate, 2-hop BFS to count reachable co-candidates, weighted by inverse node degree. Every kb_search call becomes graph-aware — agents don't need to know about kb_ask. *(G-Retriever, Think-on-Graph, KG-RAG — Size: M)*

3. **Access-aware confidence decay.** Track when entries appear in results (`last_accessed`). Use `max(updated_at, last_accessed)` as the decay anchor so frequently-retrieved entries resist staleness. Currently, valuable old decisions decay even though agents keep finding them useful. *(MemoryBank, Generative Agents — Size: S)*

4. **Graph-hint annotations on sparse results.** When kb_search returns <3 results, append "Related via graph" hints from 1-hop BFS. Nudges agents toward graph connections exactly when keyword search is failing. *(Think-on-Graph, GraphRAG community summarization — Size: S)*

## Later

- **Knowledge lives on one machine.** SQLite is local-only. Working from a different machine means no KB. If the positioning promise is "your knowledge follows you across agents," it needs to follow you across machines too. A remote-capable backend (Turso, Postgres, or even SQLite over a sync protocol) would make the KB truly portable without giving up the simplicity of the current setup.

- **Single-user only.** No concept of who stored an entry or which agent session produced it. Multi-contributor support (attribution, provenance) is a prerequisite for team use — multiple developers and coding agents contributing to a shared KB.

## Done

- Improve tool descriptions for query tools — differentiate kb_search, kb_ask, kb_summarize; ungate kb_ingest with glob support.
- Amazon Bedrock LLM provider — BedrockLLMClient with async-native SDK, smithy-json newline workaround.
- Ollama, Anthropic, and Bedrock provider switching — configurable extraction and query LLM backends.
- Knowledge graph with LLM enrichment — entity extraction, relationship edges, graph traversal queries.
- File ingestion pipeline — deny-list, secrets scanning, PII redaction, LLM summarize + extract.
- Token efficiency — compact output, kb_get two-phase retrieval, kb_store_batch with batch enrichment.
- kb_get skips inactive entries — no stale knowledge leaking into agent context.
- Show long_title in compact search results — one extra line per result dramatically improves discoverability.
