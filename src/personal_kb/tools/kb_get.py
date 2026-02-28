"""kb_get MCP tool — full entry retrieval by ID."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.tools.formatters import format_entry_full, format_result_list

logger = logging.getLogger(__name__)

_MAX_IDS = 20


def register_kb_get(mcp: FastMCP) -> None:
    """Register the kb_get tool with the MCP server."""

    @mcp.tool()
    async def kb_get(
        entry_id: Annotated[
            str | list[str],
            Field(description="Single entry ID or list of IDs (max 20)"),
        ],
        ctx: Context | None = None,
    ) -> str:
        """Retrieve full details for one or more knowledge entries by ID.

        Use after kb_search to get the full content of interesting results.
        kb_search returns compact summaries; kb_get returns the complete
        knowledge_details for entries you want to read in full.
        """
        if ctx is None:
            raise RuntimeError("Context not injected")

        from personal_kb.db.queries import get_entry, touch_accessed

        lifespan = ctx.lifespan_context
        db = lifespan["db"]

        # Normalize to list
        ids = [entry_id] if isinstance(entry_id, str) else list(entry_id)

        if len(ids) > _MAX_IDS:
            return f"Error: Maximum {_MAX_IDS} IDs per request (got {len(ids)})."

        formatted: list[str] = []
        accessed_ids: list[str] = []
        for eid in ids:
            entry = await get_entry(db, eid)
            if entry is None or not entry.is_active:
                formatted.append(f"[{eid}] not found")
            else:
                formatted.append(format_entry_full(entry))
                accessed_ids.append(eid)

        if accessed_ids:
            await touch_accessed(db, accessed_ids)

        return format_result_list(formatted)
