"""kb_get MCP tool (HTTP twin of ``personal_kb.tools.kb_get``)."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from kb_core.formatting import format_entry_full, format_result_list
from pydantic import Field

from kb_service.mcp_server import context

logger = logging.getLogger(__name__)

_MAX_IDS = 20


def _get_description(prefix: str) -> str:
    """Build kb_get description with correct tool name cross-references."""
    return (
        "Retrieve full details for one or more knowledge entries by ID.\n\n"
        f"Use after {prefix}search to get the full content of interesting results. "
        f"{prefix}search returns compact summaries; {prefix}get returns the complete "
        "knowledge_details for entries you want to read in full."
    )


def _render_pointer_rot(pointer_rot_pairs: list[tuple[str, str | None]]) -> str | None:
    """Render a pointer-rot block from ``(target_id, superseded_by)`` pairs."""
    if not pointer_rot_pairs:
        return None
    lines = ["  Pointer-rot:"]
    for target_id, superseded_by in pointer_rot_pairs:
        if superseded_by is not None:
            lines.append(f"    [{target_id}] superseded by [{superseded_by}]")
        else:
            lines.append(f"    [{target_id}] deactivated")
    return "\n".join(lines)


def register_kb_get(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_get tool with the MCP server."""

    @mcp.tool(name=f"{prefix}get", description=_get_description(prefix))
    async def kb_get(
        entry_id: Annotated[
            str | list[str],
            Field(description="Single entry ID or list of IDs (max 20)"),
        ],
    ) -> str:
        """Retrieve full details for one or more knowledge entries by ID."""
        backend = context.backend_for_request()

        ids = [entry_id] if isinstance(entry_id, str) else list(entry_id)

        if len(ids) > _MAX_IDS:
            return f"Error: Maximum {_MAX_IDS} IDs per request (got {len(ids)})."

        entries_data = await backend.get_entries(ids)

        found_titles = {
            eid: e.short_title for eid, e, _ in entries_data if e is not None
        }
        missing = sorted(
            {
                e.superseded_by
                for _, e, _ in entries_data
                if e is not None and e.superseded_by
            }
            - set(found_titles)
        )
        if missing:
            for mid, me, _ in await backend.get_entries(missing):
                if me is not None:
                    found_titles[mid] = me.short_title

        formatted: list[str] = []
        for eid, entry, rot_pairs in entries_data:
            if entry is None:
                formatted.append(f"[{eid}] not found")
            else:
                rendered = format_entry_full(entry, mark_superseded=False)
                sid = entry.superseded_by
                if sid:
                    if sid in found_titles:
                        banner = f"SUPERSEDED by {sid} — {found_titles[sid]}"
                    else:
                        banner = f"SUPERSEDED by {sid}"
                        logger.warning(
                            "supersession-read invariant_breach op=kb_get target=%s "
                            "superseded_by=%s superseder=not_found_or_inactive",
                            eid,
                            sid,
                        )
                    rendered = f"{banner}\n{rendered}"
                note = _render_pointer_rot(rot_pairs)
                if note is not None:
                    rendered = f"{rendered}\n{note}"
                formatted.append(rendered)

        return format_result_list(formatted)
