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
    get_log_level,
    get_personal_kb_url,
    is_manager_mode,
)
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
from personal_kb.tools.kb_map_eligibility import (
    register_kb_map_eligibility,
    register_kb_map_eligibility_override,
)
from personal_kb.tools.kb_preflight import register_kb_preflight
from personal_kb.tools.kb_search import register_kb_search
from personal_kb.tools.kb_store import register_kb_store
from personal_kb.tools.kb_store_batch import register_kb_store_batch
from personal_kb.tools.kb_summarize import register_kb_summarize


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage the backend lifecycle — HTTP-only path with optional local daemon.

    An unset/empty ``PERSONAL_KB_URL`` means local mode (``LOCAL_KB_URL``).
    The MCP server opens an :class:`HttpBackend` against that URL in every mode. When the
    URL targets a loopback host (``127.0.0.1`` / ``localhost``), the
    lifespan runs :func:`ensure_daemon` as a pre-step — spawning a
    detached, singleton ``kb-service`` daemon if ``/api/health`` is
    unhealthy. For a remote (non-loopback) URL, no spawn occurs.

    The daemon outlives the MCP session: the ``finally`` block closes
    ONLY the HTTP client, never the daemon process.
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

    kb_service_url = get_personal_kb_url()
    if not os.environ.get("PERSONAL_KB_URL"):
        logger.info("PERSONAL_KB_URL not set; using local default %s", kb_service_url)

    from personal_kb.backend import HttpBackend
    from personal_kb.config import get_personal_kb_api_key
    from personal_kb.daemon import ensure_daemon, is_loopback_url

    api_key = get_personal_kb_api_key()
    if api_key is None:
        msg = (
            f"PERSONAL_KB_API_KEY is not set but PERSONAL_KB_URL {kb_service_url!r} "
            "is not a local URL. Set PERSONAL_KB_API_KEY for the hosted KB."
        )
        raise RuntimeError(msg)

    # Loopback URLs trigger the spawn pre-step.  Remote URLs go straight
    # to HttpBackend.open() — no daemon, no pidfile, no health poll.
    if is_loopback_url(kb_service_url):
        logger.info(
            "Local mode — ensuring kb-service daemon at %s before connecting",
            kb_service_url,
        )
        await ensure_daemon(kb_service_url)

    logger.info("Opening HttpBackend at %s", kb_service_url)
    backend = HttpBackend(base_url=kb_service_url, api_key=api_key)
    await backend.open()
    skew_note: str | None = None
    try:
        from personal_kb.version_skew import check_version_skew

        skew_note = await check_version_skew(kb_service_url)
    except Exception:
        logger.debug("version skew check failed", exc_info=True)
    try:
        # No 'kb' object — every backend operation goes through HTTP.
        yield {"backend": backend, "version_skew_note": skew_note}
    finally:
        # Close ONLY the backend.  The daemon (if we spawned one) outlives
        # this session — it serves future MCP processes too.
        await backend.close()


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

supersedes is a required kb_store parameter: the kb ids this entry replaces, or "none".
Use hints to build the knowledge graph:
- {"person": "jason"}, {"tool": "sqlite"} to link entities
- {"related_entities": [{"id": "kb-00003", "edge_type": "depends_on"}]}
- {"resolution": {"corrected_fact": "...", "cue": \
{"tool": "Bash", "target_class": "git remote"}}} records a corrected belief \
(see kb_store hints)

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
    register_kb_map_eligibility(mcp, prefix)
    register_kb_map_eligibility_override(mcp, prefix)

    if is_manager_mode():
        register_kb_maintain(mcp, prefix)
        register_kb_bulk_update(mcp, prefix)

    register_kb_list_projects(mcp, prefix)
    if get_contributor():
        register_kb_list_contributors(mcp, prefix)
        register_kb_list_teams(mcp, prefix)

    return mcp
