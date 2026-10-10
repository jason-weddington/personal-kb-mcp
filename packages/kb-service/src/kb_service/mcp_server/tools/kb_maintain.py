"""kb_maintain MCP tool (HTTP twin of ``personal_kb.tools.kb_maintain``).

Only deactivate, reactivate and reconcile_supersession run over HTTP; every
other action keeps the stdio HTTP-mode error string.
"""

import logging
import re
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from kb_service.mcp_server import context
from kb_service.mcp_server.errors import BackendHttpError, map_error

logger = logging.getLogger(__name__)

_ACTIONS = {
    "stats",
    "deactivate",
    "reactivate",
    "rebuild_embeddings",
    "rebuild_graph",
    "purge_inactive",
    "vacuum",
    "entry_versions",
    "list_feedback",
    "summarize_feedback",
    "search_stats",
    "list_contributors",
    "list_audit",
    "reconcile_supersession",
}

_MAINTAIN_DESCRIPTION = (
    "Administrative maintenance operations for the knowledge base.\n"
    "\n"
    "Requires KB_MANAGER=TRUE environment variable.\n"
    "\n"
    "Actions:\n"
    "- stats: Database overview (entry counts, graph stats, embeddings)\n"
    "- deactivate: Soft-delete an entry (requires entry_id; over HTTP also requires\n"
    "  change_reason, the reason it is retired; superseded_by is the entry that "
    "replaces it)\n"
    "- reactivate: Undo deactivation (requires entry_id)\n"
    "- rebuild_embeddings: Re-embed entries (force=True for all)\n"
    "- rebuild_graph: Full graph reconstruction from all active entries\n"
    "- purge_inactive: Hard-delete entries inactive for N+ days (requires "
    "confirm=True)\n"
    "- vacuum: Optimize database (PRAGMA optimize + VACUUM)\n"
    "- entry_versions: Show version history (requires entry_id)\n"
    "- list_feedback: List recent agent feedback (optional: feedback_type, since)\n"
    "- summarize_feedback: LLM-clustered summary of feedback themes (optional: "
    "since)\n"
    "- search_stats: Search telemetry overview (optional: since)\n"
    "- list_contributors: Show contributor/team stats for active entries\n"
    "- list_audit: Recent audit events (optional: entry_id, since)\n"
    "- reconcile_supersession: Heal drifted superseded_by values (idempotent; admin "
    "over HTTP)"
)


_ACTION_DESCRIPTION = (
    "Maintenance action: stats, deactivate, reactivate, "
    "rebuild_embeddings, rebuild_graph, purge_inactive, vacuum, "
    "entry_versions, list_feedback, summarize_feedback, search_stats, "
    "list_contributors, list_audit, reconcile_supersession"
)


def _render_reconcile(
    edges_added: int,
    set_count: int,
    cleared_count: int,
    changed: list[tuple[object, ...]],
) -> str:
    """Render a supersession reconcile report."""
    lines = [
        f"Supersession reconcile: edges_added={edges_added} set={set_count} "
        f"cleared={cleared_count}"
    ]
    for target, old, new in changed:
        lines.append(f"  drift {target}: {old!r} -> {new!r}")
    return "\n".join(lines)


def register_kb_maintain(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_maintain tool with the MCP server."""

    @mcp.tool(name=f"{prefix}maintain", description=_MAINTAIN_DESCRIPTION)
    async def kb_maintain(
        action: Annotated[
            str,
            Field(description=_ACTION_DESCRIPTION),
        ],
        entry_id: Annotated[
            str | None,
            Field(description="Required for deactivate, reactivate, entry_versions"),
        ] = None,
        days_inactive: Annotated[
            int,
            Field(description="For purge_inactive: min days since deactivation", ge=1),
        ] = 90,
        force: Annotated[
            bool,
            Field(
                description="For rebuild_embeddings: re-embed ALL (not just missing)"
            ),
        ] = False,
        confirm: Annotated[
            bool,
            Field(description="Required True for purge_inactive"),
        ] = False,
        feedback_type: Annotated[
            str | None,
            Field(
                description="For list_feedback: filter by type "
                "(missing, unhelpful, friction)"
            ),
        ] = None,
        since: Annotated[
            str | None,
            Field(
                description="ISO date for list_feedback/summarize_feedback/search_stats"
            ),
        ] = None,
        change_reason: Annotated[
            str | None,
            Field(
                description="For deactivate: why the entry is being retired (required)"
            ),
        ] = None,
        superseded_by: Annotated[
            str | None,
            Field(
                description="For deactivate: ID of the entry that replaces this one "
                "(kb-NNNNN)"
            ),
        ] = None,
    ) -> str:
        """Administrative maintenance operations for the knowledge base."""
        if action not in _ACTIONS:
            return f"Unknown action '{action}'. Use: {', '.join(sorted(_ACTIONS))}"

        backend = context.backend_for_request()

        if action == "deactivate":
            if not entry_id:
                return "Error: entry_id is required for deactivate action."
            if not change_reason or not change_reason.strip():
                return "Error: change_reason is required for deactivate action."
            if superseded_by is not None and not re.fullmatch(
                r"kb-\d{5}", superseded_by
            ):
                return (
                    f"Error: superseded_by '{superseded_by}' is not a valid entry ID"
                    " (expected kb-NNNNN)."
                )
            try:
                entry = await backend.deactivate(
                    entry_id, change_reason=change_reason, superseded_by=superseded_by
                )
            except BackendHttpError as exc:
                return map_error(exc)
            return f"Deactivated entry {entry.id}: {entry.short_title}"
        elif action == "reactivate":
            if not entry_id:
                return "Error: entry_id is required for reactivate action."
            try:
                entry = await backend.reactivate(entry_id)
            except BackendHttpError as exc:
                return map_error(exc)
            return f"Reactivated entry {entry.id}: {entry.short_title}"
        elif action == "reconcile_supersession":
            try:
                data = await backend.reconcile_supersession()
            except BackendHttpError as exc:
                return map_error(exc)
            return _render_reconcile(
                data.get("edges_added", 0),
                data.get("set_count", 0),
                data.get("cleared_count", 0),
                [tuple(c) for c in data.get("changed", [])],
            )
        return (
            f"Error: action {action} is not supported in HTTP mode"
            " — run it on the KB service host."
        )
