"""FastMCP server with lifespan management and tool registration."""

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from personal_kb.config import get_db_path, get_embedding_dim, get_log_level
from personal_kb.db.connection import create_connection
from personal_kb.graph.builder import GraphBuilder
from personal_kb.search.embeddings import EmbeddingClient
from personal_kb.store.knowledge_store import KnowledgeStore
from personal_kb.tools.kb_search import register_kb_search
from personal_kb.tools.kb_store import register_kb_store


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

    # Pre-check Ollama availability (non-blocking, just logs)
    ollama_ok = await embedder.is_available()
    if ollama_ok:
        logger.info("Ollama available — vector search enabled")
    else:
        logger.warning("Ollama unavailable — vector search disabled, FTS-only mode")

    try:
        yield {"db": db, "store": store, "embedder": embedder, "graph_builder": graph_builder}
    finally:
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

    return mcp
