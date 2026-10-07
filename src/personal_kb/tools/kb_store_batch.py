"""kb_store_batch MCP tool — create multiple knowledge entries in one call."""

import logging
from datetime import UTC, datetime
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.confidence.decay import compute_effective_confidence
from personal_kb.config import is_safety_skip
from personal_kb.ingest.safety import detect_secrets_in_content
from personal_kb.models.entry import EntryType
from personal_kb.tools.formatters import format_entry_compact, format_result_list
from personal_kb.tools.map_lint import lint_map_body
from personal_kb.tools.ttl import compute_expires_at

logger = logging.getLogger(__name__)

_MAX_BATCH = 10

_REQUIRED_FIELDS = {"short_title", "long_title", "knowledge_details"}


async def batch_store_entries(
    entries: list[dict[str, Any]],
    lifespan: dict[str, Any],
) -> str:
    """Core batch store logic, testable without MCP context."""
    from personal_kb.tools._lifespan import backend_from_lifespan

    if len(entries) > _MAX_BATCH:
        return f"Error: Maximum {_MAX_BATCH} entries per batch (got {len(entries)})."

    if not entries:
        return "Error: entries list is empty."

    # Validate required fields
    for i, entry_dict in enumerate(entries):
        missing = _REQUIRED_FIELDS - set(entry_dict.keys())
        if missing:
            return f"Error: entry {i} missing required fields: {', '.join(sorted(missing))}"

    # Validate sensitivity values
    from personal_kb.tools.kb_store import _VALID_SENSITIVITY

    for i, entry_dict in enumerate(entries):
        sens = entry_dict.get("sensitivity")
        if sens is not None and sens not in _VALID_SENSITIVITY:
            valid = ", ".join(sorted(_VALID_SENSITIVITY))
            return f'Error: entry {i} has invalid sensitivity "{sens}". Must be one of: {valid}'

    # Secret scanning — reject entire batch if any entry has secrets
    if not is_safety_skip():
        for i, entry_dict in enumerate(entries):
            details = str(entry_dict.get("knowledge_details", ""))
            secrets = detect_secrets_in_content(details)
            if secrets:
                types = ", ".join(secrets)
                return (
                    f"Error: Potential secrets detected in entry {i} ({types}). "
                    "Remove sensitive values before storing. "
                    "Set KB_SKIP_SAFETY=TRUE to override."
                )

    backend = backend_from_lifespan(lifespan)
    is_http = backend.is_remote

    # TTL pre-validation: entries with bad TTL go to the client-side failed list
    # and are EXCLUDED from the backend call (both modes).
    valid_entries: list[dict[str, Any]] = []
    client_failed: list[tuple[int, str, str]] = []
    for i, entry_dict in enumerate(entries):
        raw_ttl = entry_dict.get("ttl")
        if raw_ttl:
            try:
                compute_expires_at(str(raw_ttl))
            except ValueError as exc:
                title = str(entry_dict.get("short_title", f"entry {i}"))
                client_failed.append((i, title, str(exc)))
                logger.warning("Invalid TTL for entry %d (%s): %s", i, title, exc)
                continue
        valid_entries.append(entry_dict)

    # Call the backend
    from personal_kb.backend.http import BackendHttpError, _map_error

    try:
        created, backend_failed = await backend.store_batch(valid_entries)
    except BackendHttpError as e:
        return _map_error(e, "")

    # Merge failures: client-side (TTL) + backend-side (per-entry DB errors in local mode)
    all_failed = client_failed + backend_failed

    # All entries failed
    if all_failed and not created:
        lines = [f"Batch failed: all {len(all_failed)} entries failed."]
        for _idx, title, err in all_failed:
            lines.append(f"  Entry {_idx} ({title}): {err}")
        return "\n".join(lines)

    # Re-fetch entries to get updated state (embedding flag) — local mode only.
    # In HTTP mode the returned entries are already fully hydrated.
    now = datetime.now(UTC)
    formatted: list[str] = []
    if not is_http:
        from personal_kb.tools._lifespan import kb_from_lifespan

        kb = kb_from_lifespan(lifespan)
        store = kb.knowledge_store
        for entry in created:
            refreshed = await store.get_entry(entry.id) or entry
            anchor = refreshed.updated_at or refreshed.created_at or now
            eff = compute_effective_confidence(
                refreshed.confidence_level,
                refreshed.entry_type,
                anchor,
            )
            block = f"Created {refreshed.id} (v{refreshed.version})\n" + format_entry_compact(
                refreshed, eff
            )
            if refreshed.entry_type == EntryType.MENTAL_MAP and refreshed.knowledge_details:
                warnings = lint_map_body(refreshed.knowledge_details)
                if warnings:
                    block += "\n" + "\n".join(warnings)
            formatted.append(block)
    else:
        for entry in created:
            anchor = entry.updated_at or entry.created_at or now
            eff = compute_effective_confidence(
                entry.confidence_level,
                entry.entry_type,
                anchor,
            )
            block = f"Created {entry.id} (v{entry.version})\n" + format_entry_compact(entry, eff)
            if entry.entry_type == EntryType.MENTAL_MAP and entry.knowledge_details:
                warnings = lint_map_body(entry.knowledge_details)
                if warnings:
                    block += "\n" + "\n".join(warnings)
            formatted.append(block)

    # Build header
    # In HTTP mode: server-side failures have no per-entry detail.
    # Compute total failures = client-side + server-side (for HTTP: inferred from requested count).
    if is_http:
        server_requested = len(valid_entries)
        server_created = len(created)
        server_failed_count = server_requested - server_created
        total_failed = len(client_failed) + server_failed_count
    else:
        total_failed = len(all_failed)

    header = f"Batch: {len(created)} entries created"
    if total_failed:
        header += f", {total_failed} failed"

    result = format_result_list(formatted, header=header)

    # Append per-entry failure detail (local-mode backend failures + client-side TTL failures).
    # HTTP mode server-side failures are not expanded here.
    if all_failed:
        fail_lines = ["", "Failed entries (retry these):"]
        for idx, title, err in all_failed:
            fail_lines.append(f"  Entry {idx} ({title}): {err}")
        result += "\n".join(fail_lines)

    if not is_http:
        from personal_kb.config import get_backend_warning

        warning = get_backend_warning()
        if warning:
            result = f"{warning}\n\n{result}"

    return result


def _store_batch_description(prefix: str) -> str:
    """Build kb_store_batch description with correct tool name cross-references."""
    return (
        "Store multiple knowledge entries in a single call.\n\n"
        f"More efficient than calling {prefix}store repeatedly — uses a single LLM "
        "call for graph enrichment across all entries.\n\n"
        "Each entry dict requires: short_title, long_title, knowledge_details. "
        "Optional fields: entry_type (default: factual_reference), project_ref, "
        "source_context, confidence_level (default: 0.9), tags, hints, ttl."
    )


def register_kb_store_batch(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_store_batch tool with the MCP server."""

    @mcp.tool(name=f"{prefix}store_batch", description=_store_batch_description(prefix))
    async def kb_store_batch(
        entries: Annotated[
            list[dict[str, object]],
            Field(
                description=(
                    "List of entry dicts (max 10). Each requires: "
                    "short_title, long_title, knowledge_details. "
                    "Optional: entry_type, project_ref, source_context, "
                    "confidence_level, tags, hints."
                ),
            ),
        ],
        ctx: Context | None = None,
    ) -> str:
        """Store multiple knowledge entries in a single call."""
        if ctx is None:
            raise RuntimeError("Context not injected")

        return await batch_store_entries(entries, ctx.lifespan_context)
