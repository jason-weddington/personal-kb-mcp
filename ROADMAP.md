# Roadmap

## Near-term

(empty — pick from medium-term or add new items)

## Medium-term

- **Multi-contributor support** — Add contributor metadata to entries (who stored it, which agent/session), enabling multi-user knowledge bases with attribution and provenance tracking.
- **Remote storage option** — Support a remote backend (e.g., PostgreSQL, Turso) as an alternative to local SQLite for cross-machine access and cloud deployment.

## Long-term

- **Team knowledge base** — Multiple developers and coding agents contributing to a shared KB. Collective experiential learning across a dev team.

## Completed

- **Improve tool descriptions for query tools** — Rewrote MCP instructions and tool docstrings to differentiate kb_search (quick lookup), kb_ask (graph exploration), and kb_summarize (synthesized answers). Ungated kb_ingest with glob support.
- **Amazon Bedrock LLM provider** — BedrockLLMClient using the async-native `aws-sdk-bedrock-runtime` SDK, with smithy-json newline workaround. (b0ccc29)
