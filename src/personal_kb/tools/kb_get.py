"""kb_get MCP tool — full entry retrieval by ID."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.db.backend import Database
from personal_kb.db.queries import get_entry
from personal_kb.graph.queries import _KB_ID_RE, get_neighbors
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.tools.formatters import format_entry_full, format_result_list

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


async def _pointer_rot_note(db: Database, entry: KnowledgeEntry) -> str | None:
    """Render the inline pointer-rot note for a mental_map entry, or None.

    For mental_map entries only, resolves OUTBOUND graph edges via
    ``get_neighbors(db, entry.id, direction="outgoing")`` (no edge_types
    filter — all outgoing edges, of which only kb-id targets are treated as
    pointers). A target is "rotted" if EITHER ``target.superseded_by is not
    None`` OR ``target.is_active is False``. When >=1 target is rotted, we
    return a multi-line block:

        ``  Pointer-rot:``
        ``    [kb-XXXXX] superseded by [kb-YYYYY]``   (if superseded)
        ``    [kb-XXXXX] deactivated``                (if deactivated-only)

    Lines are sorted by ascending target id. When superseded AND deactivated,
    the SUPERSEDED form wins (it names the actionable replacement). For all
    other entry types the function returns None without issuing any extra DB
    lookups, so non-map ``kb_get`` output is byte-identical to before.
    """
    if entry.entry_type != EntryType.MENTAL_MAP:
        return None
    if entry.id is None:
        return None

    neighbors = await get_neighbors(db, entry.id, direction="outgoing")
    # Dedupe targets: a single target reached by multiple edge_types still
    # surfaces once (sorted by id keeps the ordering stable).
    seen: set[str] = set()
    rotted: list[tuple[str, str | None]] = []  # (target_id, superseded_by)
    for target_id, _edge_type, _direction in neighbors:
        if target_id in seen:
            continue
        if not _KB_ID_RE.match(target_id):
            continue
        seen.add(target_id)
        # IMPORTANT: do not reuse kb_get's top-level "not is_active → not found"
        # short-circuit — a deactivated target is precisely the rot signal we
        # want to surface. get_entry has no is_active filter (db/queries.py:120).
        target = await get_entry(db, target_id)
        if target is None:
            continue
        if target.superseded_by is not None:
            rotted.append((target_id, target.superseded_by))
        elif not target.is_active:
            rotted.append((target_id, None))

    if not rotted:
        return None

    rotted.sort(key=lambda pair: pair[0])
    lines = ["  Pointer-rot:"]
    for target_id, superseded_by in rotted:
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
        ctx: Context | None = None,
    ) -> str:
        """Retrieve full details for one or more knowledge entries by ID."""
        if ctx is None:
            raise RuntimeError("Context not injected")

        from personal_kb.db.queries import touch_accessed
        from personal_kb.tools._lifespan import kb_from_lifespan

        kb = kb_from_lifespan(ctx.lifespan_context)
        db = kb.db

        # Normalize to list
        ids = [entry_id] if isinstance(entry_id, str) else list(entry_id)

        if len(ids) > _MAX_IDS:
            return f"Error: Maximum {_MAX_IDS} IDs per request (got {len(ids)})."

        formatted: list[str] = []
        accessed_ids: list[str] = []
        for eid in ids:
            entry = await kb.get(eid)
            if entry is None or not entry.is_active:
                formatted.append(f"[{eid}] not found")
            else:
                rendered = format_entry_full(entry)
                note = await _pointer_rot_note(db, entry)
                if note is not None:
                    rendered = f"{rendered}\n{note}"
                formatted.append(rendered)
                accessed_ids.append(eid)

        if accessed_ids:
            await touch_accessed(db, accessed_ids)

        return format_result_list(formatted)
