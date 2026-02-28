# Roadmap

Problems worth solving, in priority order. Not specs — the "how" gets figured out when we build it.

## Now

- **Agents default to kb_search and miss graph connections.** Despite improved tool descriptions, agents still hammer kb_search with keyword variations instead of using `kb_ask strategy:related` to traverse the knowledge graph. This means serendipitous discovery — the KB's most unique value — rarely happens. The graph already connects related entries across projects and topics, but agents don't reach for it.

## Later

- **Knowledge lives on one machine.** SQLite is local-only. Working from a different machine means no KB. A remote-capable backend (Turso, Postgres, or even SQLite over a sync protocol) would make the KB portable without giving up the simplicity of the current setup.

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
