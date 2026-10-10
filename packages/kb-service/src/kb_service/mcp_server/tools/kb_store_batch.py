"""kb_store_batch MCP tool (HTTP twin of ``personal_kb.tools.kb_store_batch``)."""

import logging
import re
from datetime import UTC, datetime
from typing import Annotated, Any

from fastmcp import FastMCP
from kb_core.confidence.decay import compute_effective_confidence
from kb_core.formatting import format_entry_compact, format_result_list
from kb_core.ingest.safety import detect_secrets_in_content
from kb_core.models.entry import EntryType
from kb_core.ttl import compute_expires_at
from pydantic import Field

from kb_service.config import is_safety_skip
from kb_service.mcp_server import context
from kb_service.mcp_server.errors import BackendHttpError, map_error
from kb_service.mcp_server.tools.kb_store import (
    _VALID_SENSITIVITY,
    _log_decision,
    _validate_distinct_from,
    _validate_hints_supersedes_conflict,
    _validate_supersedes,
)
from kb_service.mcp_server.tools.map_lint import lint_map_body

logger = logging.getLogger(__name__)

_MAX_BATCH = 10

_REQUIRED_FIELDS = {"short_title", "long_title", "knowledge_details", "supersedes"}

_ENTRIES_DESCRIPTION = (
    "List of entry dicts (max 10). "
    "Required keys: short_title, long_title, knowledge_details, "
    'supersedes (list of kb-XXXXX ids or "none"). '
    "Optional: entry_type, project_ref, source_context, "
    "confidence_level, tags, hints, sensitivity, ttl, distinct_from. "
    "hints may carry a resolution object "
    "(see the store tool's hints description)."
)


async def batch_store_entries(entries: list[dict[str, Any]]) -> str:
    """Core batch store logic (validation, one backend call, rendering)."""
    if len(entries) > _MAX_BATCH:
        return f"Error: Maximum {_MAX_BATCH} entries per batch (got {len(entries)})."

    if not entries:
        return "Error: entries list is empty."

    for i, entry_dict in enumerate(entries):
        missing = _REQUIRED_FIELDS - set(entry_dict.keys())
        if missing:
            return (
                f"Error: entry {i} missing required fields: "
                f"{', '.join(sorted(missing))}"
            )

    # Validate supersedes / distinct_from (whole batch is rejected on any failure)
    for i, entry_dict in enumerate(entries):
        sup = entry_dict["supersedes"]
        checks = (
            (_validate_supersedes(sup), None),
            (
                _validate_hints_supersedes_conflict(sup, entry_dict.get("hints")),
                "hints_conflict",
            ),
            (_validate_distinct_from(entry_dict.get("distinct_from")), "bad_shape"),
        )
        for err, rule in checks:
            if err:
                if rule is None:
                    rule = "empty_list" if "ambiguous" in err else "bad_shape"
                _log_decision("store_batch", "create", "rejected", rule, sup)
                return f"Error: entry {i}: " + err.removeprefix("Error: ")
        _log_decision("store_batch", "create", "accepted", "none", sup)

    for i, entry_dict in enumerate(entries):
        sens = entry_dict.get("sensitivity")
        if sens is not None and sens not in _VALID_SENSITIVITY:
            valid = ", ".join(sorted(_VALID_SENSITIVITY))
            return (
                f'Error: entry {i} has invalid sensitivity "{sens}". '
                f"Must be one of: {valid}"
            )

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

    backend = context.backend_for_request()

    # TTL pre-validation: entries with bad TTL go to the client-side failed list
    # and are EXCLUDED from the backend call.
    valid_entries: list[dict[str, Any]] = []
    original_index: list[int] = []  # valid_entries position -> caller's position
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
        original_index.append(i)

    try:
        created, backend_failed, superseded_ids = await backend.store_batch(
            valid_entries
        )
    except BackendHttpError as e:
        mapped = map_error(e)
        return re.sub(
            r"\bentry (\d+)\b",
            lambda m: (
                f"entry {original_index[int(m.group(1))]}"
                if int(m.group(1)) < len(original_index)
                else m.group(0)
            ),
            mapped,
        )

    backend_failed = [
        (original_index[idx] if 0 <= idx < len(original_index) else idx, t, err)
        for idx, t, err in backend_failed
    ]

    all_failed = client_failed + backend_failed

    if all_failed and not created:
        lines = [f"Batch failed: all {len(all_failed)} entries failed."]
        for _idx, title, err in all_failed:
            lines.append(f"  Entry {_idx} ({title}): {err}")
        return "\n".join(lines)

    # The returned entries are already fully hydrated.
    now = datetime.now(UTC)
    formatted: list[str] = []
    for pos, entry in enumerate(created):
        anchor = entry.updated_at or entry.created_at or now
        eff = compute_effective_confidence(
            entry.confidence_level,
            entry.entry_type,
            anchor,
        )
        block = f"Created {entry.id} (v{entry.version})\n" + format_entry_compact(
            entry, eff
        )
        if pos < len(superseded_ids) and superseded_ids[pos]:
            block += "\nSupersedes: " + ", ".join(superseded_ids[pos])
        if entry.entry_type == EntryType.MENTAL_MAP and entry.knowledge_details:
            warnings = lint_map_body(entry.knowledge_details)
            if warnings:
                block += "\n" + "\n".join(warnings)
        formatted.append(block)

    # Server-side failures have no per-entry detail: infer them from counts.
    server_failed_count = len(valid_entries) - len(created)
    total_failed = len(client_failed) + server_failed_count

    header = f"Batch: {len(created)} entries created"
    if total_failed:
        header += f", {total_failed} failed"

    result = format_result_list(formatted, header=header)

    if all_failed:
        fail_lines = ["", "Failed entries (retry these):"]
        for idx, title, err in all_failed:
            fail_lines.append(f"  Entry {idx} ({title}): {err}")
        result += "\n".join(fail_lines)

    return result


def _store_batch_description(prefix: str) -> str:
    """Build kb_store_batch description with correct tool name cross-references."""
    return (
        "Store multiple knowledge entries in a single call.\n\n"
        f"More efficient than calling {prefix}store repeatedly — uses a single LLM "
        "call for graph enrichment across all entries.\n\n"
        "Each entry dict requires: short_title, long_title, knowledge_details, "
        'supersedes (list of kb-XXXXX ids this entry replaces, or "none"). '
        "Optional fields: entry_type (default: factual_reference), project_ref, "
        "source_context, confidence_level (default: 0.9), tags, hints, ttl, "
        "distinct_from. "
        f"hints may carry a resolution object; see {prefix}store."
    )


def register_kb_store_batch(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_store_batch tool with the MCP server."""

    @mcp.tool(name=f"{prefix}store_batch", description=_store_batch_description(prefix))
    async def kb_store_batch(
        entries: Annotated[
            list[dict[str, object]],
            Field(description=_ENTRIES_DESCRIPTION),
        ],
    ) -> str:
        """Store multiple knowledge entries in a single call."""
        return await batch_store_entries(entries)
