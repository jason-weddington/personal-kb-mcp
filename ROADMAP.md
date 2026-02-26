# Roadmap

## Near-term

- **Improve tool descriptions for query tools** — Agents default to `kb_search` (keyword lookup) when `kb_ask` (graph traversal + query planning) or `kb_summarize` (synthesized answers) would give better results. Review and improve tool descriptions and server instructions so any agent naturally picks the right tool based on the protocol alone. No out-of-band hacks — behavior must be driven by the MCP protocol layer.

## Medium-term

- **Multi-contributor support** — Add contributor metadata to entries (who stored it, which agent/session), enabling multi-user knowledge bases with attribution and provenance tracking.
- **Remote storage option** — Support a remote backend (e.g., PostgreSQL, Turso) as an alternative to local SQLite for cross-machine access and cloud deployment.

## Long-term

- **Team knowledge base** — Multiple developers and coding agents contributing to a shared KB. Collective experiential learning across a dev team.

## Completed

- **Amazon Bedrock LLM provider** — BedrockLLMClient using the async-native `aws-sdk-bedrock-runtime` SDK, with smithy-json newline workaround. (b0ccc29)
