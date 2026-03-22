"""FastMCP server with lifespan management and tool registration."""

import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from personal_kb.config import (
    get_contributor,
    get_database_url,
    get_db_path,
    get_embedding_dim,
    get_explore_port,
    get_extraction_provider,
    get_log_level,
    get_query_provider,
    get_team,
    is_auto_explore,
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
from personal_kb.tools.kb_explore import register_kb_explore
from personal_kb.tools.kb_feedback import register_kb_feedback
from personal_kb.tools.kb_get import register_kb_get
from personal_kb.tools.kb_ingest import register_kb_ingest
from personal_kb.tools.kb_ingest_url import register_kb_ingest_url
from personal_kb.tools.kb_list import (
    register_kb_list_contributors,
    register_kb_list_projects,
    register_kb_list_teams,
)
from personal_kb.tools.kb_maintain import register_kb_maintain
from personal_kb.tools.kb_preflight import register_kb_preflight
from personal_kb.tools.kb_search import register_kb_search
from personal_kb.tools.kb_store import register_kb_store
from personal_kb.tools.kb_store_batch import register_kb_store_batch
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


def _create_synthesis_llm(provider: str) -> LLMProvider | None:
    """Create a stronger LLM for human-facing synthesis (Sonnet 4.6)."""
    if provider == "anthropic":
        if AnthropicLLMClient is not None:
            from personal_kb.llm.anthropic import _SONNET_MODEL

            return AnthropicLLMClient(model_override=_SONNET_MODEL)
        return None
    if provider == "bedrock":
        if BedrockLLMClient is not None:
            from personal_kb.llm.bedrock import _SONNET_MODEL as _BR_SONNET

            return BedrockLLMClient(model_override=_BR_SONNET)
        return None
    # Ollama: no Sonnet equivalent, fall back to default
    return None


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage database connection and embedding client lifecycle."""
    # Configure logging to stderr (stdout is MCP stdio transport)
    log_level = getattr(logging, get_log_level())
    log_fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=log_level, format=log_fmt, stream=sys.stderr)

    # Also log to file (overwrite on each server start)
    log_dir = os.path.join(os.path.expanduser("~"), ".local", "share", "personal_kb")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.txt")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_fmt))
    logging.getLogger().addHandler(file_handler)

    logger = logging.getLogger(__name__)

    db_url = get_database_url()
    if db_url:
        logger.info("Connecting to PostgreSQL database")
    else:
        logger.info("Opening SQLite database at %s", get_db_path())
    db = await create_connection(embedding_dim=get_embedding_dim())

    # Check for accidental backend fallback (e.g. missing KB_DATABASE_URL)
    from personal_kb.config import check_backend_fallback

    check_backend_fallback()

    store = KnowledgeStore(db)
    embedder = EmbeddingClient(db)
    graph_builder = GraphBuilder(db)

    # Create LLM clients based on provider config
    extraction_provider = get_extraction_provider()
    query_provider = get_query_provider()

    extraction_llm = _create_llm(extraction_provider)
    query_llm = _create_llm(query_provider)

    # Stronger LLM for human-facing synthesis (web explorer, kb_summarize via browser)
    synthesis_llm = _create_synthesis_llm(query_provider)

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

    if synthesis_llm is not None:
        logger.info("Synthesis LLM: Sonnet 4.6 (%s)", query_provider)
    else:
        logger.info("Synthesis LLM: using query LLM (no Sonnet override available)")

    contributor = get_contributor()
    team = get_team()
    if contributor:
        logger.info("Contributor: %s, Team: %s", contributor, team or "(not set)")
    elif db_url:
        logger.warning(
            "KB_CONTRIBUTOR not set — entries will have no attribution. "
            "Set KB_CONTRIBUTOR for multi-user provenance."
        )

    # Auto-start explorer web server
    if is_auto_explore():
        from personal_kb.tools.kb_explore import start_explorer_server

        port = get_explore_port()
        started = await start_explorer_server(
            db,
            embedder,
            query_llm,
            synthesis_llm,
            store=store,
            graph_builder=graph_builder,
            graph_enricher=graph_enricher,
            extraction_llm=extraction_llm,
            contributor=contributor,
            team=team,
            port=port,
            kill_existing=False,
        )
        if started:
            logger.info("Explorer auto-started on http://127.0.0.1:%d", port)
        else:
            logger.info("Explorer auto-start skipped (port %d in use)", port)

    try:
        yield {
            "db": db,
            "store": store,
            "embedder": embedder,
            "graph_builder": graph_builder,
            "llm_client": extraction_llm,
            "graph_enricher": graph_enricher,
            "query_llm": query_llm,
            "synthesis_llm": synthesis_llm,
            "contributor": contributor,
            "team": team,
        }
    finally:
        if synthesis_llm is not None:
            await synthesis_llm.close()
        if query_llm is not None:
            await query_llm.close()
        if extraction_llm is not None:
            await extraction_llm.close()
        await embedder.close()
        await db.close()
        logger.info("Database connection closed")


_ROLE_PREFIXES = {
    "personal": (
        "This is your PERSONAL knowledge base — your config, dotfiles, workflow "
        "preferences, and private notes. Not for team-shared knowledge.\n\n"
    ),
    "team": (
        "This is the TEAM knowledge base — shared decisions, architecture, patterns, "
        "and conventions. Not for personal config or individual workflow notes.\n\n"
    ),
}

_INSTRUCTIONS = """\
This KB stores private context that you — an AI agent with public knowledge \
already memorized — would not otherwise have: project decisions, personal \
conventions, hard-won lessons, and domain-specific facts.

BEFORE YOU ACT — check the KB first:
The KB is your institutional memory. Search it BEFORE guessing, grepping, \
or asking the user:
- Deployment/infra questions → kb_search before SSH-ing or trying hostnames
- New project or unfamiliar codebase → kb_search(project_ref="X") for context
- Errors you haven't seen → kb_search the error message
- Architectural decisions → kb_ask("decisions about X")
- Operational procedures → kb_search before improvising
One failed kb_search costs a second. Not searching costs minutes of fumbling.

QUERYING — pick the right tool:
- kb_preflight: Get a project context primer at session start. Returns a \
compact table-of-contents of expiring entries, recent decisions/lessons, \
and active conventions. Use 'since' to narrow to a time window (e.g. '7d', \
'2w'). Call this when you start working on a project to see what's relevant.
- kb_search: Quick lookup by keywords or filters. Returns compact summaries \
(no details). Use for duplicate checking, finding entries, or filtering by \
tags/project/type.
- kb_get: Retrieve full details for specific entries by ID. Use after \
kb_search or kb_preflight to read the complete content of interesting results.
- kb_ask: Explore related knowledge via graph traversal. Use when you need \
to discover connections, trace decision history, or find everything related \
to a concept. Returns full details.
- kb_summarize: Get a synthesized natural-language answer with citations. \
Use when you need to answer a user question directly from the KB.

STORING — capture knowledge proactively:
- kb_store: Create or update a single entry.
- kb_store_batch: Create multiple entries in one call (max 10). More \
efficient — uses a single LLM call for graph enrichment.
- Entries are automatically attributed to the configured contributor \
and team — you do not need to specify who is storing.
- Technical decisions and their rationale ("chose X because Y")
- Patterns, conventions, or architecture worth preserving
- Lessons learned from debugging, fixing issues, or trial-and-error
- Key facts: API behaviors, config values, version constraints, gotchas
- Anything the user explicitly asks you to remember

DON'T capture trivial info, temporary session context, or duplicates. \
SEARCH before storing — if a relevant entry exists, use update_entry_id.

INGESTING — extend the KB from files or URLs:
- kb_ingest: Intelligent extraction from local files. An LLM reads the source \
and creates multiple properly structured KB entries (decisions, patterns, facts). \
Deduplicates against existing entries — safe to ingest overlapping files. \
Accepts file paths, directories, glob patterns (e.g. *.md, docs/**/*.txt).
- kb_ingest_url: Fetch a URL, extract article content from HTML, and ingest it. \
Handles boilerplate removal automatically — just provide the URL. \
If you already have the page content (e.g. from authenticated sites or WebFetch), \
pass it via the `content` parameter to skip fetching.

Entry types: factual_reference, decision, pattern_convention, lesson_learned.
Use tags for discoverability. Use project_ref for project-specific knowledge.

Use hints to build the knowledge graph:
- {"supersedes": "kb-00042"} when replacing prior knowledge
- {"person": "jason"}, {"tool": "sqlite"} to link entities
- {"related_entities": [{"id": "kb-00003", "edge_type": "depends_on"}]}

FEEDBACK — help improve the KB:
- kb_feedback: Call this whenever a KB query returned poor results (zero hits, \
irrelevant entries, missing knowledge). Takes 3 seconds, helps the human \
prioritize what to add next.
  - feedback_type: 'missing' (KB lacked needed knowledge), \
'unhelpful' (results existed but didn't help), 'friction' (tool was awkward)
  - Do NOT use for storing knowledge — use kb_store instead.
"""


_TOOL_BASES = [
    "store_batch",
    "store",
    "search",
    "get",
    "ask",
    "summarize",
    "ingest",
    "ingest_url",
    "explore",
    "feedback",
    "maintain",
    "list_projects",
    "list_contributors",
    "list_teams",
    "preflight",
]


_ROLE_PREFIXES_TOOL = {
    "personal": "personal_kb_",
    "team": "team_kb_",
}


def _get_tool_prefix() -> str:
    """Return the MCP tool name prefix based on KB_INSTANCE_ROLE.

    - role=personal → "personal_kb_"
    - role=team     → "team_kb_"
    - unset/empty   → "kb_"  (backwards-compatible default)
    """
    role = os.environ.get("KB_INSTANCE_ROLE", "").lower()
    return _ROLE_PREFIXES_TOOL.get(role, "kb_")


def _build_instructions(prefix: str) -> str:
    """Build server instructions, optionally prefixed by instance role."""
    role = os.environ.get("KB_INSTANCE_ROLE", "").lower()
    text = _ROLE_PREFIXES.get(role, "") + _INSTRUCTIONS
    if prefix != "kb_":
        for base in _TOOL_BASES:
            text = text.replace(f"kb_{base}", f"{prefix}{base}")
    return text


def create_server() -> FastMCP:
    """Create and configure the MCP server with all tools."""
    prefix = _get_tool_prefix()

    mcp = FastMCP(
        "personal-kb",
        instructions=_build_instructions(prefix),
        lifespan=lifespan,
    )

    register_kb_store(mcp, prefix)
    register_kb_store_batch(mcp, prefix)
    register_kb_search(mcp, prefix)
    register_kb_get(mcp, prefix)
    register_kb_ask(mcp, prefix)
    register_kb_summarize(mcp, prefix)
    register_kb_ingest(mcp, prefix)
    register_kb_ingest_url(mcp, prefix)
    register_kb_feedback(mcp, prefix)
    register_kb_preflight(mcp, prefix)
    register_kb_explore(mcp, prefix)

    if is_manager_mode():
        register_kb_maintain(mcp, prefix)

    register_kb_list_projects(mcp, prefix)
    if get_contributor():
        register_kb_list_contributors(mcp, prefix)
        register_kb_list_teams(mcp, prefix)

    return mcp
