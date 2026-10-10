"""FastMCP server factory for the /mcp endpoint.

The tool set, names, schemas and instructions are twins of
``personal_kb.server.create_server`` (``tests/test_mcp_http_parity.py``
enforces it), except that ``{prefix}ingest`` is never registered: it reads
files on the machine running the tool, which over /mcp is the service host.
Use ``{prefix}ingest_url`` with ``content`` instead.

No lifespan: the kb-service lifespan owns every resource; tools reach it
through the per-request ``InProcessBackend`` (``context.backend_for_request``).
"""

from fastmcp import FastMCP

from kb_service.config import get_contributor, get_instance_role, is_manager_mode
from kb_service.mcp_server.observability import McpCallLogMiddleware
from kb_service.mcp_server.surface_filter import SurfaceToolFilterMiddleware
from kb_service.mcp_server.tools.kb_ask import register_kb_ask
from kb_service.mcp_server.tools.kb_bulk_update import register_kb_bulk_update
from kb_service.mcp_server.tools.kb_explore import register_kb_explore
from kb_service.mcp_server.tools.kb_feedback import register_kb_feedback
from kb_service.mcp_server.tools.kb_get import register_kb_get
from kb_service.mcp_server.tools.kb_ingest_url import register_kb_ingest_url
from kb_service.mcp_server.tools.kb_list import (
    register_kb_list_contributors,
    register_kb_list_projects,
    register_kb_list_teams,
)
from kb_service.mcp_server.tools.kb_maintain import register_kb_maintain
from kb_service.mcp_server.tools.kb_map_eligibility import (
    register_kb_map_eligibility,
    register_kb_map_eligibility_override,
)
from kb_service.mcp_server.tools.kb_preflight import register_kb_preflight
from kb_service.mcp_server.tools.kb_search import register_kb_search
from kb_service.mcp_server.tools.kb_store import register_kb_store
from kb_service.mcp_server.tools.kb_store_batch import register_kb_store_batch
from kb_service.mcp_server.tools.kb_summarize import register_kb_summarize

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
    role = get_instance_role()
    return _ROLE_PREFIXES_TOOL.get(role, "kb_")


def _build_instructions(prefix: str) -> str:
    """Build server instructions, optionally prefixed by instance role."""
    role = get_instance_role()
    text = _ROLE_PREFIXES.get(role, "") + _INSTRUCTIONS
    if prefix != "kb_":
        for base in _TOOL_BASES:
            text = text.replace(f"kb_{base}", f"{prefix}{base}")
    return text


def create_mcp_server() -> FastMCP:
    """Create the /mcp FastMCP server with the HTTP-served tool set."""
    prefix = _get_tool_prefix()

    mcp = FastMCP(
        "personal-kb",
        instructions=_build_instructions(prefix),
        middleware=[McpCallLogMiddleware(), SurfaceToolFilterMiddleware(prefix)],
    )

    register_kb_store(mcp, prefix)
    register_kb_store_batch(mcp, prefix)
    register_kb_search(mcp, prefix)
    register_kb_get(mcp, prefix)
    register_kb_ask(mcp, prefix)
    register_kb_summarize(mcp, prefix)
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
