"""FastMCP server with lifespan management and tool registration."""

import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP
from kb_core.knowledge_base import KnowledgeBase

from personal_kb import maps_index_writer
from personal_kb.config import (
    build_anthropic_config,
    build_bedrock_config,
    build_kb_config,
    build_ollama_provider_config,
    check_backend_fallback,
    get_contributor,
    get_database_url,
    get_db_path,
    get_explore_port,
    get_log_level,
    get_query_provider,
    is_auto_explore,
    is_manager_mode,
)
from personal_kb.llm import AnthropicLLMClient, BedrockLLMClient
from personal_kb.llm.ollama import OllamaLLMClient
from personal_kb.llm.provider import LLMProvider
from personal_kb.maps_index_writer import write_project_maps
from personal_kb.tools.kb_ask import register_kb_ask
from personal_kb.tools.kb_bulk_update import register_kb_bulk_update
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
    """Create an LLM client for the given provider name.

    Backwards-compat shim. The lifespan no longer calls this — it composes
    a :class:`~kb_core.config.KbConfig` and hands it to
    :meth:`~kb_core.knowledge_base.KnowledgeBase.create`. Still imported by
    :mod:`personal_kb.web.app` (its rewire onto the facade is W6b).
    """
    if provider == "anthropic":
        if AnthropicLLMClient is not None:
            return AnthropicLLMClient(build_anthropic_config())
        return None
    if provider == "bedrock":
        if BedrockLLMClient is not None:
            return BedrockLLMClient(build_bedrock_config())
        return None
    if provider == "ollama":
        return OllamaLLMClient(build_ollama_provider_config())
    return None


def _create_synthesis_llm(provider: str) -> LLMProvider | None:
    """Create a stronger LLM for human-facing synthesis (Sonnet 4.6).

    Backwards-compat shim with the same status as :func:`_create_llm`.
    """
    if provider == "anthropic":
        if AnthropicLLMClient is not None:
            from kb_core.llm.anthropic import _SONNET_MODEL

            return AnthropicLLMClient(build_anthropic_config(model=_SONNET_MODEL))
        return None
    if provider == "bedrock":
        if BedrockLLMClient is not None:
            from kb_core.llm.bedrock import _SONNET_MODEL as _BR_SONNET

            return BedrockLLMClient(build_bedrock_config(model=_BR_SONNET))
        return None
    # Ollama: no Sonnet equivalent, fall back to default
    return None


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage the :class:`KnowledgeBase` facade lifecycle.

    Builds a :class:`kb_core.config.KbConfig` from the channel-side env
    getters (:mod:`personal_kb.config`) and opens a
    :class:`~kb_core.knowledge_base.KnowledgeBase` over it. The facade
    owns the database, the embedder, the LLM clients, and the graph
    enricher; the channel just holds a reference. Maps-index startup
    rebuild + LISTEN/NOTIFY wiring use ``kb.db`` directly — the channel
    keeps those concerns since the on-disk JSONL path is env-driven and
    therefore server-side.
    """
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

    # Check for accidental backend fallback (e.g. missing KB_DATABASE_URL)
    check_backend_fallback()

    # Build the engine config + open the facade.
    kb_config = build_kb_config()

    # Ollama parity: the legacy ``_create_synthesis_llm`` returned ``None`` for
    # the "ollama" provider (no Sonnet equivalent). Preserve that — pass
    # ``synthesis_llm=None`` to the facade so the engine doesn't build a
    # separate synthesis client. Anthropic/Bedrock keep their Sonnet override
    # via :func:`build_provider_config`.
    query_provider = get_query_provider()
    if query_provider == "ollama":
        kb = await KnowledgeBase.create(kb_config, synthesis_llm=None)
    else:
        kb = await KnowledgeBase.create(kb_config)

    db = kb.db
    embedder = kb.embedder

    # Pre-check Ollama availability (non-blocking, just logs)
    ollama_ok = False
    if embedder is not None:
        try:
            ollama_ok = await embedder.is_available()  # type: ignore[attr-defined]
        except Exception:
            ollama_ok = False
    if ollama_ok:
        logger.info("Ollama available — vector search enabled")
    else:
        logger.warning("Ollama unavailable — vector search disabled, FTS-only mode")

    extraction_provider = kb.config.providers.extraction.provider
    if kb.extraction_llm is not None:
        logger.info("Extraction LLM: %s", extraction_provider)
    else:
        logger.warning(
            "Extraction LLM not available (%s) — graph enrichment disabled", extraction_provider
        )

    if kb.query_llm is not None:
        logger.info("Query LLM: %s", query_provider)
    else:
        logger.warning("Query LLM not available (%s) — query planning disabled", query_provider)

    if kb.synthesis_llm is not None:
        logger.info("Synthesis LLM: Sonnet 4.6 (%s)", query_provider)
    else:
        logger.info("Synthesis LLM: using query LLM (no Sonnet override available)")

    contributor = kb.config.attribution.contributor
    team = kb.config.attribution.team
    if contributor:
        logger.info("Contributor: %s, Team: %s", contributor, team or "(not set)")
    elif db_url:
        logger.warning(
            "KB_CONTRIBUTOR not set — entries will have no attribution. "
            "Set KB_CONTRIBUTOR for multi-user provenance."
        )

    # Startup rebuild: converge any drift from when this instance wasn't
    # running. Best-effort — a failure here must not abort startup.
    try:
        await maps_index_writer.rebuild_all_projects(db, team=team)
    except Exception:
        logger.warning("Startup maps_index rebuild failed", exc_info=True)

    # Live-refresh listener: peer instances NOTIFY 'kb_maps_changed' after
    # they write; we re-render this project's line via write_project_maps,
    # and on every (re)connect we do a full rebuild to re-sync any events
    # missed while disconnected. SQLite returns a no-op teardown.
    async def _on_change(project_ref: str) -> None:
        try:
            await write_project_maps(db, project_ref, team=team)
        except Exception:
            logger.warning(
                "maps_index on_change refresh failed for %s",
                project_ref,
                exc_info=True,
            )

    async def _on_reconnect() -> None:
        try:
            await maps_index_writer.rebuild_all_projects(db, team=team)
        except Exception:
            logger.warning("maps_index on_reconnect rebuild failed", exc_info=True)

    try:
        listener_teardown = await db.start_maps_listener(
            on_change=_on_change,
            on_reconnect=_on_reconnect,
        )
    except Exception:
        logger.warning("Starting maps_index listener failed", exc_info=True)

        async def listener_teardown() -> None:
            return None

    # Auto-start explorer web server
    if is_auto_explore():
        from personal_kb.tools.kb_explore import start_explorer_server

        port = get_explore_port()
        started = await start_explorer_server(
            db,
            embedder,
            kb.query_llm,
            kb.synthesis_llm,
            store=kb.store,
            graph_builder=kb.graph_builder,
            graph_enricher=kb.graph_enricher,
            extraction_llm=kb.extraction_llm,
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
        yield {"kb": kb}
    finally:
        # Tear down the maps listener before closing the DB so its dedicated
        # asyncpg connection (Postgres) is closed cleanly. Best-effort.
        try:
            await listener_teardown()
        except Exception:
            logger.warning("maps_index listener teardown failed", exc_info=True)
        # The facade owns everything it built (DB, embedder, LLMs) — close()
        # releases them in reverse construction order, swallowing teardown
        # failures.
        await kb.close()
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
    "bulk_update",
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
        register_kb_bulk_update(mcp, prefix)

    register_kb_list_projects(mcp, prefix)
    if get_contributor():
        register_kb_list_contributors(mcp, prefix)
        register_kb_list_teams(mcp, prefix)

    return mcp
