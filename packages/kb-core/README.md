# kb-core

**kb-core** is the channel-agnostic graph-RAG engine extracted from
[`personal-kb`](https://github.com/jason-weddington/personal-kb-mcp). It is
a small, async Python library that gives you:

- **Hybrid search** — BM25 full-text search (FTS5) fused with vector
  similarity via Reciprocal Rank Fusion
- **A knowledge graph** — deterministic edges (tags, projects,
  supersedes) plus optional LLM-extracted entities (tools, concepts,
  people)
- **Agentic retrieval** — a ReAct-style loop that plans, executes,
  evaluates, and refines search calls when single-shot retrieval is
  weak
- **Pluggable backends** — SQLite (`aiosqlite` + `sqlite-vec`) for
  embedded use, Postgres (`asyncpg`) for shared deployments
- **Pluggable providers** — Anthropic, AWS Bedrock, or Ollama for
  LLM-driven enrichment / planning / synthesis; Ollama (or any
  HTTP-reachable embedder) for vectors. Every provider is **optional**
  — base `kb-core` is FTS-only and zero-LLM by default.

`kb-core` reads **no environment variables**, has no global state, and
ships with `py.typed` — configuration is passed explicitly via
`KbConfig`. That makes it safe to embed in your own server, your own
CLI, or your own batch job without dragging the MCP transport along.

## Install

```bash
pip install kb-core                    # nucleus: SQLite + FTS + vectors
pip install "kb-core[anthropic]"       # + Anthropic provider
pip install "kb-core[aws]"             # + AWS Bedrock provider
pip install "kb-core[postgres,iam]"    # + Postgres backend (RDS IAM optional)
pip install "kb-core[ingest]"          # + HTML/PDF file ingestion
pip install "kb-core[safety]"          # + pre-store secret/PII scrubbers
pip install "kb-core[ollama]"          # marker extra (Ollama uses httpx only)
```

| Extra      | Pulls in                                  | Enables                                            |
|------------|-------------------------------------------|----------------------------------------------------|
| _(none)_   | `aiosqlite`, `sqlite-vec`, `pydantic`, `httpx` | SQLite backend, FTS5, hybrid search, graph queries |
| `anthropic`| `anthropic`                               | Anthropic LLM provider (enrichment / planning / synthesis) |
| `aws`      | `aws-sdk-bedrock-runtime`, `smithy-json`  | AWS Bedrock LLM provider                            |
| `postgres` | `asyncpg`                                 | Postgres backend                                    |
| `iam`      | `boto3`                                   | RDS / Aurora IAM auth token signing                 |
| `ingest`   | `trafilatura`, `pymupdf`                  | HTML and PDF text extraction for `kb_ingest`        |
| `safety`   | `detect-secrets`, `scrubadub`             | Pre-store secret + PII scrubbing                    |
| `ollama`   | _(none — uses `httpx` from nucleus)_      | Marker for consumers that target Ollama             |

Requires Python 3.13+.

## Quickstart

A 12-line round trip: open an empty SQLite database, store an entry,
search for it.

```python
import asyncio
from kb_core import create_sqlite
from kb_core.models.search import SearchQuery


async def main() -> None:
    async with await create_sqlite("/tmp/kb.db") as kb:
        await kb.store(
            short_title="kb-core ships py.typed",
            long_title="kb-core is a fully typed library",
            knowledge_details="Import kb_core and mypy sees inline types.",
        )
        results, _ = await kb.search(SearchQuery(query="py.typed"))
        for r in results:
            print(r.score, r.entry.short_title)


asyncio.run(main())
```

`create_sqlite` is the factory sugar — under the hood it builds a
`KbConfig` and opens every dependency (database, embedder, LLM clients)
the engine needs. Pass `embedding=EmbeddingConfig(...)` to opt into
vector search; omit it for FTS-only.

## Three consumption modes

`kb-core` is the engine; how you expose it is your choice.

- **Library** — import `KnowledgeBase` from your own Python code (the
  quickstart above). Best for batch jobs, scripts, and notebooks.
- **Microservice** — wrap the facade in your own HTTP / gRPC handler.
  `kb-core` is async-first and concurrency-safe, so a FastAPI app
  around it is a thin shim.
- **MCP server** — install
  [`personal-kb`](https://github.com/jason-weddington/personal-kb-mcp),
  which is the reference MCP channel built on `kb-core`. Same engine,
  agent-facing tools.

## How it works

Engine internals — schema, RRF fusion, decay literals, graph
primitives, the LLM provider roles — are documented in
[docs/how_it_works.md](docs/how_it_works.md).

## License

MIT — see [LICENSE](LICENSE).
