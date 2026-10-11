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
This KB holds private context you lack: project decisions, conventions, hard-won \
lessons and domain facts.

SEARCH BEFORE YOU ACT. Check the KB before guessing, grepping or asking the user: \
infra details before SSH or trying hostnames; kb_search(project_ref="X") in an \
unfamiliar project; an error you have not seen; prior decisions (kb_ask). A miss \
costs a second; not searching costs minutes.

QUERY
- kb_preflight: project primer at session start (recent decisions, lessons, \
conventions).
- kb_search: keyword and filter lookup with compact results; also your duplicate \
check.
- kb_get: full text of entries by id.
- kb_ask: graph traversal for connections and decision history.
- kb_summarize: a synthesized answer with citations.

STORE what a future agent would need: decisions with their rationale, conventions, \
debugging lessons, non-obvious facts, anything the user asks you to remember. Skip \
trivia and session state; search first and update with update_entry_id.
- kb_store, or kb_store_batch for up to 10 related entries.
- supersedes is a required kb_store parameter: the ids this entry replaces, or \
"none".
- entry_type: factual_reference, decision, pattern_convention or lesson_learned; add \
tags and project_ref.
- hints link the graph: {"tool": "sqlite"}, {"related_entities": [{"id": "kb-00003", \
"edge_type": "depends_on"}]}.
- A corrected belief gets {"resolution": {"corrected_fact": "...", "wrong_belief": \
"...", "cue": {"tool": "Bash", "target_class": "git remote"}}}. Only entries with a \
resolution are pushed to future sessions before the mistake repeats.
- kb_ingest_url: ingest a web page (or pass content you already fetched).

kb_feedback: report an empty or unhelpful result or an awkward tool (missing, \
unhelpful, friction).
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
