"""kb_store MCP tool — create and update knowledge entries."""

import logging
import re
from datetime import UTC, datetime
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.confidence.decay import compute_effective_confidence
from personal_kb.config import is_safety_skip
from personal_kb.graph.builder import _as_list
from personal_kb.ingest.safety import detect_secrets_in_content
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.tools.formatters import format_entry_compact
from personal_kb.tools.map_lint import lint_map_body
from personal_kb.tools.ttl import compute_expires_at

logger = logging.getLogger(__name__)

_VALID_SENSITIVITY = {"internal", "restricted", "public"}

_KB_ID_RE = re.compile(r"kb-\d{5}")

ORPHAN_MAP_ERROR = (
    "Error: A mental_map entry requires at least one outbound pointer "
    "(a kb-XXXXX reference in knowledge_details, or a supersedes/related_entities hint). "
    "A map with zero pointers is an orphan note, not a map."
)


def _validate_sensitivity(sensitivity: str | None) -> str | None:
    """Return an error string if sensitivity is invalid, None if OK."""
    if sensitivity is not None and sensitivity not in _VALID_SENSITIVITY:
        valid = ", ".join(sorted(_VALID_SENSITIVITY))
        return f'Error: Invalid sensitivity "{sensitivity}". Must be one of: {valid}'
    return None


def _mental_map_has_pointer(
    knowledge_details: str,
    hints: dict[str, object] | None,
    superseded_by: str | None = None,
) -> bool:
    """Return True if a mental_map entry has at least one outbound pointer.

    Closed checklist mirroring graph/builder.py's edge-producing logic exactly
    (builder.py:52-86), so a future builder change is the only place this can
    diverge. An outbound pointer exists iff ANY of:
      (a) a ``kb-XXXXX`` reference appears in knowledge_details;
      (b) a ``supersedes`` hint contains a ``kb-XXXXX`` id;
      (c) ``superseded_by`` is a non-empty string;
      (d) a ``related_entities`` hint contains a dict with a non-empty
          ``id``/``target`` OR a bare non-empty string.
    Tag/project/person/tool hints do NOT count.
    """
    # (a) kb-XXXXX reference in knowledge_details (mirrors builder.py:69 finditer)
    if knowledge_details and _KB_ID_RE.search(knowledge_details):
        return True

    h = hints or {}

    # (b) supersedes hint (mirrors builder.py:52-54 fullmatch)
    for target in _as_list(h.get("supersedes")):
        if isinstance(target, str) and _KB_ID_RE.fullmatch(target):
            return True

    # (c) superseded_by reversed edge (mirrors builder.py:63-65)
    if isinstance(superseded_by, str) and superseded_by:
        return True

    # (d) related_entities — dict id/target OR bare non-empty str (mirrors builder.py:77-86)
    for rel in _as_list(h.get("related_entities")):
        if isinstance(rel, dict):
            ref = rel.get("id") or rel.get("target")
            if isinstance(ref, str) and ref:
                return True
        elif isinstance(rel, str) and rel:
            return True

    return False


def format_store_result(
    entry: KnowledgeEntry,
    is_update: bool = False,
    include_backend_warning: bool = True,
) -> str:
    """Format the result of a store operation for the MCP response.

    Pass ``include_backend_warning=False`` in HTTP mode to suppress the
    SQLite-fallback warning (the remote service is not SQLite).
    """
    action = "Updated" if is_update else "Created"
    anchor = entry.updated_at or entry.created_at or datetime.now(UTC)
    eff = compute_effective_confidence(entry.confidence_level, entry.entry_type, anchor)
    compact = format_entry_compact(entry, eff)
    line = f"{action} {entry.id} (v{entry.version})\n{compact}"
    if not entry.has_embedding:
        line += "\n  Note: Entry will be embedded when Ollama is available"
    if include_backend_warning:
        from personal_kb.config import get_backend_warning

        warning = get_backend_warning()
        if warning:
            line = f"{warning}\n\n{line}"
    return line


def _prepend_map_advisories(
    result: str,
    warnings: list[str],
    include_backend_warning: bool = True,
) -> str:
    """Insert advisory mental_map lint lines into a store result.

    Advisories land ABOVE the Created/Updated compact block but BELOW any
    backend warning that ``format_store_result`` already prepended.
    The store always succeeds; these warnings are informational only.
    """
    if not warnings:
        return result
    advisory = "\n".join(warnings)
    if include_backend_warning:
        from personal_kb.config import get_backend_warning

        backend = get_backend_warning()
        if backend and result.startswith(backend):
            rest = result[len(backend) :].lstrip("\n")
            return f"{backend}\n\n{advisory}\n\n{rest}"
    return f"{advisory}\n\n{result}"


def register_kb_store(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_store tool with the MCP server."""

    @mcp.tool(name=f"{prefix}store")
    async def kb_store(
        short_title: Annotated[str, Field(description="Brief identifier for the entry")] = "",
        long_title: Annotated[str, Field(description="Descriptive title")] = "",
        knowledge_details: Annotated[
            str, Field(description="Full content of the knowledge entry")
        ] = "",
        entry_type: Annotated[
            EntryType | None,
            Field(
                description=(
                    "factual_reference, decision, pattern_convention, lesson_learned, "
                    "mental_map: structural orientation node — pointers/relationships only, "
                    "no retrievable values"
                )
            ),
        ] = None,
        project_ref: Annotated[
            str | None, Field(description="Project tag/category for filtering")
        ] = None,
        source_context: Annotated[
            str | None,
            Field(description="Where this knowledge came from"),
        ] = None,
        confidence_level: Annotated[
            float,
            Field(
                description=(
                    "Initial confidence score (0.0-1.0). "
                    "Decays over time based on entry_type half-life: "
                    "factual_reference 90d, decision 1y, pattern_convention 2y, lesson_learned 5y, "
                    "mental_map: no decay (exempt). "
                    "Lower for uncertain info, higher for verified facts. Default 0.9"
                ),
                ge=0.0,
                le=1.0,
            ),
        ] = 0.9,
        tags: Annotated[
            list[str] | None, Field(description="Freeform tags for categorization")
        ] = None,
        hints: Annotated[
            dict[str, object] | None,
            Field(description="Structured hints for graph building (supersedes, related_entities)"),
        ] = None,
        update_entry_id: Annotated[
            str | None,
            Field(description="ID of existing entry to update (e.g. kb-00042)"),
        ] = None,
        sensitivity: Annotated[
            str | None,
            Field(
                description=(
                    "Sensitivity classification: internal, restricted, public, or None. "
                    "Classification only — no enforcement."
                ),
            ),
        ] = None,
        ttl: Annotated[
            str | None,
            Field(
                description=(
                    "Time-to-live (e.g. '7d', '24h', '2w'). "
                    "Entry excluded from search after expiry. "
                    "Use for time-bounded knowledge like project status."
                ),
            ),
        ] = None,
        deactivate_entry_id: Annotated[
            str | None,
            Field(
                description=(
                    "ID of entry to deactivate (soft-delete). "
                    "Removes from search results and graph. Reversible via kb_maintain."
                ),
            ),
        ] = None,
        change_reason: Annotated[
            str | None,
            Field(description="Reason for update or deactivation"),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Store or update a knowledge entry in the personal knowledge base.

        Creates a new entry or updates an existing one. Every update creates a version
        record preserving the full history. Entries are automatically indexed for
        full-text search and (when Ollama is available) vector search.

        For metadata-only updates (tags, project_ref, sensitivity, entry_type, etc.),
        pass update_entry_id with the fields to change — knowledge_details is optional.
        This avoids pulling and rewriting the full entry content.

        Use deactivate_entry_id to soft-delete incorrect or obsolete entries.

        Use entry_type to classify the knowledge:
        - factual_reference: version numbers, API endpoints, config values
        - decision: "chose X because Y" — history is critical
        - pattern_convention: coding standards, workflow preferences
        - lesson_learned: mistakes, debugging insights
        - mental_map: structural orientation node — pointers/relationships only,
          no retrievable values; requires at least one outbound pointer
        """
        from personal_kb.tools._lifespan import backend_from_lifespan

        if ctx is None:
            raise RuntimeError("Context not injected")

        backend = backend_from_lifespan(ctx.lifespan_context)
        is_http = backend.is_remote

        # --- Deactivate path ---
        if deactivate_entry_id:
            try:
                entry = await backend.deactivate(deactivate_entry_id)
            except Exception as e:
                from personal_kb.backend.http import BackendHttpError

                if isinstance(e, BackendHttpError):
                    from personal_kb.backend.http import _map_error

                    return _map_error(e, "")
                return f"Error: {e}"

            reason = f" ({change_reason})" if change_reason else ""
            return f"Deactivated entry {entry.id}: {entry.short_title}{reason}"

        # --- Update path ---
        if update_entry_id:
            # Validate sensitivity
            sens_err = _validate_sensitivity(sensitivity)
            if sens_err:
                return sens_err
            # Secret scanning on content if provided
            if knowledge_details:
                secret_err = _check_secrets(knowledge_details)
                if secret_err:
                    return secret_err
            # Validate TTL (for early error feedback; the raw ttl is passed to backend)
            if ttl:
                try:
                    compute_expires_at(ttl)
                except ValueError as e:
                    return f"Error: {e}"

            try:
                _action, entry = await backend.store(
                    short_title=short_title,
                    long_title=long_title,
                    knowledge_details=knowledge_details,
                    entry_type=entry_type,
                    project_ref=project_ref,
                    source_context=source_context,
                    confidence_level=confidence_level,
                    tags=tags,
                    hints=hints,
                    sensitivity=sensitivity,
                    ttl=ttl,
                    update_entry_id=update_entry_id,
                    change_reason=change_reason,
                )
            except Exception as e:
                from personal_kb.backend.http import BackendHttpError

                if isinstance(e, BackendHttpError):
                    from personal_kb.backend.http import _map_error

                    return _map_error(e, "")
                return f"Error: {e}"

            result = format_store_result(entry, is_update=True, include_backend_warning=not is_http)
            # Advisory mental_map lint — gate on the re-fetched entry's type
            if entry.entry_type == EntryType.MENTAL_MAP and knowledge_details:
                result = _prepend_map_advisories(
                    result,
                    lint_map_body(knowledge_details),
                    include_backend_warning=not is_http,
                )
            return result

        # --- Create path ---
        if not short_title or not long_title or not knowledge_details:
            return (
                "Error: short_title, long_title, and knowledge_details "
                "are required when creating a new entry."
            )

        # Validate sensitivity
        sens_err = _validate_sensitivity(sensitivity)
        if sens_err:
            return sens_err

        # Secret scanning
        secret_err = _check_secrets(knowledge_details)
        if secret_err:
            return secret_err

        if entry_type is None:
            entry_type = EntryType.FACTUAL_REFERENCE

        # Mental maps are orientation nodes defined by their pointers. Reject a
        # zero-pointer map BEFORE create_entry so no orphan row/version is written.
        if entry_type == EntryType.MENTAL_MAP and not _mental_map_has_pointer(
            knowledge_details, hints
        ):
            return ORPHAN_MAP_ERROR

        # Validate TTL (for early error feedback; the raw ttl is passed to backend)
        if ttl:
            try:
                compute_expires_at(ttl)
            except ValueError as e:
                return f"Error: {e}"

        try:
            _action, entry = await backend.store(
                short_title=short_title,
                long_title=long_title,
                knowledge_details=knowledge_details,
                entry_type=entry_type,
                project_ref=project_ref,
                source_context=source_context,
                confidence_level=confidence_level,
                tags=tags,
                hints=hints,
                sensitivity=sensitivity,
                ttl=ttl,
                change_reason=change_reason,
            )
        except Exception as e:
            from personal_kb.backend.http import BackendHttpError

            if isinstance(e, BackendHttpError):
                from personal_kb.backend.http import _map_error

                return _map_error(e, "")
            return f"Error: {e}"

        result = format_store_result(entry, is_update=False, include_backend_warning=not is_http)
        # Advisory mental_map lint
        if entry.entry_type == EntryType.MENTAL_MAP and knowledge_details:
            result = _prepend_map_advisories(
                result,
                lint_map_body(knowledge_details),
                include_backend_warning=not is_http,
            )
        return result


def _check_secrets(content: str) -> str | None:
    """Return an error message if secrets are detected, None otherwise."""
    if is_safety_skip():
        return None
    secrets = detect_secrets_in_content(content)
    if secrets:
        types = ", ".join(secrets)
        return (
            f"Error: Potential secrets detected ({types}). "
            "Remove sensitive values before storing. "
            "Set KB_SKIP_SAFETY=TRUE to override."
        )
    return None
