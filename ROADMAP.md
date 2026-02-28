# Roadmap

Problems worth solving, in priority order. Not specs — the "how" gets figured out when we build it.

## Now

- **No visibility into what the KB knows at a glance.** Starting a session, the agent has no idea what's in the KB without searching. A lightweight "what's in here?" summary (top entries by project, recent additions, graph stats) would help the agent decide whether to search or just proceed.

## Next

- **`kb_get` returns inactive entries without marking them.** Fetching a deactivated entry by ID returns full content with no indication it's inactive. This wastes tokens on obsolete information and can mislead the agent into using stale knowledge. Inactive entries should either be skipped by default or clearly marked.

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
