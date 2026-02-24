"""FastMCP server with lifespan management and tool registration."""

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from personal_kb.config import get_db_path, get_embedding_dim, get_log_level
from personal_kb.db.connection import create_connection
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

    # Pre-check Ollama availability (non-blocking, just logs)
    ollama_ok = await embedder.is_available()
    if ollama_ok:
        logger.info("Ollama available — vector search enabled")
    else:
        logger.warning("Ollama unavailable — vector search disabled, FTS-only mode")

    try:
        yield {"db": db, "store": store, "embedder": embedder}
    finally:
        await embedder.close()
        await db.close()
        logger.info("Database connection closed")


def create_server() -> FastMCP:
    """Create and configure the MCP server with all tools."""
    mcp = FastMCP(
        "personal-kb",
        lifespan=lifespan,
    )

    register_kb_store(mcp)
    register_kb_search(mcp)

    return mcp
