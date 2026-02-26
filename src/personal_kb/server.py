"""FastMCP server with lifespan management and tool registration."""

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from personal_kb.config import (
    get_db_path,
    get_embedding_dim,
    get_extraction_provider,
    get_log_level,
    get_query_provider,
    is_manager_mode,
)
from personal_kb.db.connection import create_connection
from personal_kb.graph.builder import GraphBuilder
from personal_kb.graph.enricher import GraphEnricher
from personal_kb.llm import AnthropicLLMClient, BedrockLLMClient
from personal_kb.llm.ollama import OllamaLLMClient
from personal_kb.llm.provider import LLMProvider
from personal_kb.search.embeddings import EmbeddingClient
from personal_kb.store.knowledge_store import KnowledgeStore
from personal_kb.tools.kb_ask import register_kb_ask
from personal_kb.tools.kb_ingest import register_kb_ingest
from personal_kb.tools.kb_maintain import register_kb_maintain
from personal_kb.tools.kb_search import register_kb_search
from personal_kb.tools.kb_store import register_kb_store
from personal_kb.tools.kb_summarize import register_kb_summarize


def _create_llm(provider: str) -> LLMProvider | None:
    """Create an LLM client for the given provider name."""
    if provider == "anthropic":
        if AnthropicLLMClient is not None:
            return AnthropicLLMClient()
        return None
    if provider == "bedrock":
        if BedrockLLMClient is not None:
            return BedrockLLMClient()
        return None
    if provider == "ollama":
        return OllamaLLMClient()
    return None


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage database connection and embedding client lifecycle."""
    # Configure logging to stderr (stdout is MCP stdio transport)
    logging.basicConfig(
        level=getattr(logging, get_log_level()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(__name__)

    db_path = get_db_path()
    logger.info("Opening database at %s", db_path)
    db = await create_connection(db_path, embedding_dim=get_embedding_dim())

    store = KnowledgeStore(db)
    embedder = EmbeddingClient(db)
    graph_builder = GraphBuilder(db)

    # Create LLM clients based on provider config
    extraction_provider = get_extraction_provider()
    query_provider = get_query_provider()

    extraction_llm = _create_llm(extraction_provider)
    query_llm = _create_llm(query_provider)

    graph_enricher: GraphEnricher | None = None
    if extraction_llm is not None:
        graph_enricher = GraphEnricher(db, extraction_llm)

    # Pre-check Ollama availability (non-blocking, just logs)
    ollama_ok = await embedder.is_available()
    if ollama_ok:
        logger.info("Ollama available — vector search enabled")
    else:
        logger.warning("Ollama unavailable — vector search disabled, FTS-only mode")

    if extraction_llm is not None:
        logger.info("Extraction LLM: %s", extraction_provider)
    else:
        logger.warning(
            "Extraction LLM not available (%s) — graph enrichment disabled", extraction_provider
        )

    if query_llm is not None:
        logger.info("Query LLM: %s", query_provider)
    else:
        logger.warning("Query LLM not available (%s) — query planning disabled", query_provider)

    try:
        yield {
            "db": db,
            "store": store,
            "embedder": embedder,
            "graph_builder": graph_builder,
            "llm_client": extraction_llm,
            "graph_enricher": graph_enricher,
            "query_llm": query_llm,
        }
    finally:
        if query_llm is not None:
            await query_llm.close()
        if extraction_llm is not None:
            await extraction_llm.close()
        await embedder.close()
        await db.close()
        logger.info("Database connection closed")


_INSTRUCTIONS = """\
As you work, proactively use this knowledge base. Don't wait to be asked.

CAPTURE knowledge when you encounter:
- Technical decisions and their rationale ("chose X because Y")
- Patterns, conventions, or architecture worth preserving
- Lessons learned from debugging, fixing issues, or trial-and-error
- Key facts: API behaviors, config values, version constraints, gotchas
- Anything the user explicitly asks you to remember

DON'T capture:
- Trivial or obvious information
- Temporary session context (what you're working on right now)
- Information already in an existing entry — update it instead

SEARCH before storing to avoid duplicates. If a relevant entry exists,
use update_entry_id to update it rather than creating a new one.

SEARCH at the start of tasks to recall relevant prior knowledge,
especially when working in a domain or project you've stored entries for.

Entry type guidance:
- factual_reference: version numbers, API details, config values
- decision: "chose X because Y" — rationale matters
- pattern_convention: coding standards, workflow preferences
- lesson_learned: hard-won debugging insights, things that surprised you

Use tags for discoverability. Use project_ref for project-specific
knowledge; omit it for knowledge that applies broadly.

Use hints to build the knowledge graph:
- {"supersedes": "kb-00042"} when replacing prior knowledge
- {"person": "jason"}, {"tool": "sqlite"} to link entities
- {"related_entities": [{"id": "kb-00003", "edge_type": "depends_on"}]}

Use kb_summarize for synthesized natural language answers with citations.
Use kb_ask for structured graph queries when you need raw data.
Use kb_search for keyword/semantic search.
"""


def create_server() -> FastMCP:
    """Create and configure the MCP server with all tools."""
    mcp = FastMCP(
        "personal-kb",
        instructions=_INSTRUCTIONS,
        lifespan=lifespan,
    )

    register_kb_store(mcp)
    register_kb_search(mcp)
    register_kb_ask(mcp)
    register_kb_summarize(mcp)

    if is_manager_mode():
        register_kb_maintain(mcp)
        register_kb_ingest(mcp)

    return mcp
