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


def stdio_deprecation_note(kb_service_url: str, *, api_key: str) -> str | None:
    """Return the stdio-server deprecation note for a hosted KB, else None.

    Local-daemon installs (loopback URL with the sentinel key) get no note:
    HTTP has no daemon auto-start yet, so the stdio server is still their only
    zero-setup path. A hosted KB on loopback (real key) does get it. The note
    names the ``$PERSONAL_KB_API_KEY`` variable literally and never its value.
    """
    from personal_kb.daemon import manages_local_daemon

    if manages_local_daemon(kb_service_url, api_key):
        return None
    url = kb_service_url.rstrip("/")
    name = "team-kb" if os.environ.get("KB_INSTANCE_ROLE", "").lower() == "team" else "personal-kb"
    return (
        "DEPRECATED: the personal-kb stdio MCP server will be removed in a future "
        "release. Connect Claude Code to kb-service over HTTP instead: "
        f"claude mcp add --transport http {name} {url}/mcp "
        '--header "Authorization: Bearer $PERSONAL_KB_API_KEY"'
    )


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage the backend lifecycle — HTTP-only path with optional local daemon.

    An unset/empty ``PERSONAL_KB_URL`` means local mode (``LOCAL_KB_URL``).
    The MCP server opens an :class:`HttpBackend` against that URL in every mode. When the
    URL targets a loopback host (``127.0.0.1`` / ``localhost``) AND the key is
    the local sentinel, the lifespan runs :func:`ensure_daemon` as a pre-step —
    spawning a detached, singleton ``kb-service`` daemon if ``/api/health`` is
    unhealthy. A remote URL, or a loopback URL with a real key (a hosted KB on
    this machine or behind an SSH tunnel), never spawns or stops anything.

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
    from personal_kb.daemon import ensure_daemon, manages_local_daemon

    api_key = get_personal_kb_api_key()
    if api_key is None:
        msg = (
            f"PERSONAL_KB_API_KEY is not set but PERSONAL_KB_URL {kb_service_url!r} "
            "is not a local URL. Set PERSONAL_KB_API_KEY for the hosted KB."
        )
        raise RuntimeError(msg)

    # Only the local daemon (loopback + sentinel key) gets the spawn pre-step.
    # Remote URLs and hosted-on-loopback KBs go straight to HttpBackend.open()
    # — no daemon, no pidfile, no health poll, and never a shutdown.
    if manages_local_daemon(kb_service_url, api_key):
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

        skew_note = await check_version_skew(kb_service_url, api_key=api_key)
    except Exception:
        logger.debug("version skew check failed", exc_info=True)
    note = stdio_deprecation_note(kb_service_url, api_key=api_key)
    if note is not None:
        logger.warning("%s", note)
    try:
        # No 'kb' object — every backend operation goes through HTTP.
        yield {
            "backend": backend,
            "version_skew_note": skew_note,
            "deprecation_note": note,
        }
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
- lesson_learned requires the resolution parameter: {"wrong_belief": "...", \
"corrected_fact": "..."}, or "none" if no belief was corrected. Add a cue only when \
one Bash command caused the mistake. Only resolutions are pushed to future sessions.
- kb_ingest: extract entries from local files, directories or globs, deduplicated \
against the KB.
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
