"""kb_bulk_update MCP tool (HTTP twin of ``personal_kb.tools.kb_bulk_update``)."""

import logging
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from kb_service.mcp_server import context
from kb_service.mcp_server.errors import BackendHttpError, map_error

logger = logging.getLogger(__name__)


def _format_diff(before: Any, after: Any, field: str) -> str:
    """Format a single field change."""
    b = getattr(before, field)
    a = getattr(after, field)
    if b == a:
        return ""
    return f"  {field}: {b!r} → {a!r}"


def _format_result(results: list[tuple[Any, Any]], dry_run: bool) -> str:
    """Format bulk update results for MCP response."""
    if not results:
        return "No entries matched the filters (or no changes needed)."

    mode = "DRY RUN — " if dry_run else ""
    lines = [f"{mode}{len(results)} entries {'would be ' if dry_run else ''}updated:\n"]

    fields = ("project_ref", "entry_type", "confidence_level", "tags", "team")
    for before, after in results:
        diffs = [_format_diff(before, after, f) for f in fields]
        diffs = [d for d in diffs if d]
        lines.append(f"  {before.id} (v{before.version} → v{after.version})")
        lines.extend(diffs)

    return "\n".join(lines)


def _bulk_update_description(prefix: str) -> str:
    return (
        "Apply metadata changes to multiple entries matching filter criteria.\n\n"
        "Filters: contributor, team, project_ref (None matches unset), "
        "entry_type, tags, entry_ids.\n\n"
        "Updates: project_ref, entry_type, confidence_level, "
        "tags_add (list), tags_remove (list), team (str).\n\n"
        "Always use dry_run=true first to preview. "
        "Each updated entry gets a version bump and audit event."
    )


def register_kb_bulk_update(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_bulk_update tool with the MCP server."""

    @mcp.tool(
        name=f"{prefix}bulk_update",
        description=_bulk_update_description(prefix),
    )
    async def kb_bulk_update(
        filters: Annotated[
            dict[str, object],
            Field(
                description=(
                    "Filter criteria to select entries. Keys: "
                    "contributor (str), team (str), project_ref (str or null), "
                    "entry_type (str), tags (list[str]), entry_ids (list[str])."
                ),
            ),
        ],
        updates: Annotated[
            dict[str, object],
            Field(
                description=(
                    "Metadata changes to apply. Keys: "
                    "project_ref (str), entry_type (str), "
                    "confidence_level (float), "
                    "tags_add (list[str]), tags_remove (list[str]), "
                    "team (str)."
                ),
            ),
        ],
        dry_run: Annotated[
            bool,
            Field(
                description="Preview changes without persisting. Always try this first."
            ),
        ] = True,
    ) -> str:
        """Apply metadata changes to multiple entries matching filter criteria.

        Filters select which entries to update. Updates specify what to change.
        Use dry_run=true to preview before committing.
        """
        if not filters:
            return (
                "Error: At least one filter is required to prevent accidental mass "
                "updates."
            )

        if not updates:
            return "Error: No updates specified."

        backend = context.backend_for_request()

        try:
            results = await backend.bulk_update(
                filters=filters,
                updates=updates,
                dry_run=dry_run,
            )
        except Exception as exc:
            if isinstance(exc, BackendHttpError):
                return map_error(exc)
            logger.warning("Bulk update failed", exc_info=True)
            return f"Error: {exc}"

        return _format_result(results, dry_run)
