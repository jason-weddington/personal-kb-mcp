"""kb_store MCP tool — create and update knowledge entries."""

import logging
import re
from datetime import UTC, datetime
from typing import Annotated, Literal

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
    "(a kb-XXXXX reference in knowledge_details, or a related_entities hint). "
    "A map with zero pointers is an orphan note, not a map."
)

SUPERSEDES_DESCRIPTION = (
    'Entries this one replaces: a list of kb-XXXXX ids, or the literal "none". '
    "On create, each listed entry gets superseded_by set to the new entry. "
    "On update, listed ids are added to the entry's supersedes set; ids are never removed. "
    'Must be "none" when deactivate_entry_id is set.'
)
DISTINCT_FROM_DESCRIPTION = (
    "kb-XXXXX ids of existing entries this new entry is deliberately distinct from. "
    "Recorded server-side; a near-duplicate rejection for the same pair is not raised again. "
    "Create only."
)
SUPERSEDED_BY_DESCRIPTION = (
    "With deactivate_entry_id: the kb-XXXXX id of the active entry that replaces "
    "the deactivated one; it records the supersedes edge and sets superseded_by."
)

_CHANGE_REASON_ERROR = (
    "Error: change_reason is required when updating or deactivating an entry: "
    "say what changed and why."
)


def _validate_supersedes(value: object) -> str | None:
    """Return an error string if *value* is not a valid supersedes, None if OK."""
    if value == "none" and isinstance(value, str):
        return None
    if isinstance(value, list):
        if not value:
            return (
                'Error: supersedes=[] is ambiguous; pass "none" when this entry replaces nothing.'
            )
        if all(isinstance(v, str) and _KB_ID_RE.fullmatch(v) for v in value):
            return None
    return (
        'Error: supersedes must be a list of kb-XXXXX ids or the literal "none" (got '
        + repr(value)
        + ")."
    )


def _validate_distinct_from(value: object) -> str | None:
    """Return an error string if *value* is not a valid distinct_from, None if OK."""
    if value is None:
        return None
    if isinstance(value, list) and all(
        isinstance(v, str) and _KB_ID_RE.fullmatch(v) for v in value
    ):
        return None
    return "Error: distinct_from must be a list of kb-XXXXX ids (got " + repr(value) + ")."


def _validate_hints_supersedes_conflict(supersedes: object, hints: object) -> str | None:
    """Reject supersedes="none" combined with a non-empty hints.supersedes."""
    if (
        isinstance(supersedes, str)
        and supersedes == "none"
        and isinstance(hints, dict)
        and hints.get("supersedes")
    ):
        return (
            'Error: supersedes="none" conflicts with hints.supersedes; list the ids in supersedes.'
        )
    return None


def _log_decision(op: str, path: str, outcome: str, rule: str, value: object) -> None:
    logger.info(
        "supersession-client op=%s path=%s outcome=%s rule=%s value=%r",
        op,
        path,
        outcome,
        rule,
        value,
    )


def _reject(path: str, rule: str, value: object, message: str, op: str = "store") -> str:
    _log_decision(op, path, "rejected", rule, value)
    return message


def _supersedes_rule(message: str) -> str:
    return "empty_list" if "ambiguous" in message else "bad_shape"


def _validate_sensitivity(sensitivity: str | None) -> str | None:
    """Return an error string if sensitivity is invalid, None if OK."""
    if sensitivity is not None and sensitivity not in _VALID_SENSITIVITY:
        valid = ", ".join(sorted(_VALID_SENSITIVITY))
        return f'Error: Invalid sensitivity "{sensitivity}". Must be one of: {valid}'
    return None


def _mental_map_has_pointer(
    knowledge_details: str,
    hints: dict[str, object] | None,
) -> bool:
    """Return True if a mental_map entry has at least one outbound pointer.

    Closed checklist mirroring graph/builder.py's edge-producing logic exactly
    (builder.py:52-86), so a future builder change is the only place this can
    diverge. An outbound pointer exists iff ANY of:
      (a) a ``kb-XXXXX`` reference appears in knowledge_details;
      (b) a ``related_entities`` hint contains a dict with a non-empty
          ``id``/``target`` OR a bare non-empty string.
    Tag/project/person/tool hints do NOT count.
    """
    # (a) kb-XXXXX reference in knowledge_details (mirrors builder.py:69 finditer)
    if knowledge_details and _KB_ID_RE.search(knowledge_details):
        return True

    h = hints or {}

    # (b) related_entities — dict id/target OR bare non-empty str (mirrors builder.py:77-86)
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
        supersedes: Annotated[
            list[str] | Literal["none"], Field(description=SUPERSEDES_DESCRIPTION)
        ],
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
            Field(
                description="Structured hints for graph building (related_entities, person, tool)"
            ),
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
            Field(
                description=(
                    "Required when update_entry_id or deactivate_entry_id is set: "
                    "what changed and why. Recorded in the version history."
                )
            ),
        ] = None,
        distinct_from: Annotated[
            list[str] | None, Field(description=DISTINCT_FROM_DESCRIPTION)
        ] = None,
        superseded_by: Annotated[str | None, Field(description=SUPERSEDED_BY_DESCRIPTION)] = None,
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
            if change_reason is None or not change_reason.strip():
                return _reject(
                    "deactivate", "change_reason_missing", supersedes, _CHANGE_REASON_ERROR
                )
            if supersedes != "none":
                return _reject(
                    "deactivate",
                    "bad_shape",
                    supersedes,
                    "Error: supersedes does not apply to deactivate_entry_id; "
                    'pass "none" and name the replacing entry with superseded_by.',
                )
            if superseded_by is not None and not (
                isinstance(superseded_by, str) and _KB_ID_RE.fullmatch(superseded_by)
            ):
                return _reject(
                    "deactivate",
                    "bad_shape",
                    supersedes,
                    "Error: superseded_by must be a kb-XXXXX id (got " + repr(superseded_by) + ").",
                )
            if distinct_from:
                return _reject(
                    "deactivate",
                    "distinct_from_misplaced",
                    supersedes,
                    "Error: distinct_from applies to create only.",
                )
            _log_decision("store", "deactivate", "accepted", "none", supersedes)
            try:
                entry = await backend.deactivate(
                    deactivate_entry_id,
                    change_reason=change_reason,
                    superseded_by=superseded_by,
                )
            except Exception as e:
                from personal_kb.backend.http import BackendHttpError

                if isinstance(e, BackendHttpError):
                    from personal_kb.backend.http import _map_error

                    return _map_error(e, "")
                return f"Error: {e}"

            by = f"; superseded by {superseded_by}" if superseded_by else ""
            return f"Deactivated entry {entry.id}: {entry.short_title} ({change_reason}){by}"

        # --- Update path ---
        if update_entry_id:
            if change_reason is None or not change_reason.strip():
                return _reject("update", "change_reason_missing", supersedes, _CHANGE_REASON_ERROR)
            sup_err = _validate_supersedes(supersedes)
            if sup_err:
                return _reject("update", _supersedes_rule(sup_err), supersedes, sup_err)
            conflict_err = _validate_hints_supersedes_conflict(supersedes, hints)
            if conflict_err:
                return _reject("update", "hints_conflict", supersedes, conflict_err)
            if superseded_by is not None:
                return _reject(
                    "update",
                    "superseded_by_misplaced",
                    supersedes,
                    "Error: superseded_by applies to deactivate_entry_id only.",
                )
            if distinct_from:
                return _reject(
                    "update",
                    "distinct_from_misplaced",
                    supersedes,
                    "Error: distinct_from applies to create only.",
                )
            _log_decision("store", "update", "accepted", "none", supersedes)
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
                _action, entry, ids = await backend.store(
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
                    supersedes=supersedes,
                )
            except Exception as e:
                from personal_kb.backend.http import BackendHttpError

                if isinstance(e, BackendHttpError):
                    from personal_kb.backend.http import _map_error

                    return _map_error(e, "")
                return f"Error: {e}"

            if ids is None and isinstance(supersedes, list) and supersedes:
                logger.warning(
                    "supersession-client mismatch op=update sent=%r superseded_ids=%r "
                    "(server skew or regression)",
                    supersedes,
                    ids,
                )
            result = format_store_result(entry, is_update=True, include_backend_warning=not is_http)
            if ids:
                result += "\nSupersedes: " + ", ".join(ids)
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
            return _reject(
                "create",
                "missing_field",
                supersedes,
                "Error: short_title, long_title, and knowledge_details "
                "are required when creating a new entry.",
            )
        sup_err = _validate_supersedes(supersedes)
        if sup_err:
            return _reject("create", _supersedes_rule(sup_err), supersedes, sup_err)
        conflict_err = _validate_hints_supersedes_conflict(supersedes, hints)
        if conflict_err:
            return _reject("create", "hints_conflict", supersedes, conflict_err)
        if superseded_by is not None:
            return _reject(
                "create",
                "superseded_by_misplaced",
                supersedes,
                "Error: superseded_by applies to deactivate_entry_id only.",
            )
        df_err = _validate_distinct_from(distinct_from)
        if df_err:
            return _reject("create", "bad_shape", supersedes, df_err)
        _log_decision("store", "create", "accepted", "none", supersedes)

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
            _action, entry, ids = await backend.store(
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
                supersedes=supersedes,
                distinct_from=distinct_from or None,
            )
        except Exception as e:
            from personal_kb.backend.http import BackendHttpError

            if isinstance(e, BackendHttpError):
                from personal_kb.backend.http import _map_error

                return _map_error(e, "")
            return f"Error: {e}"

        if (
            isinstance(supersedes, list)
            and supersedes
            and (ids is None or set(ids) != set(supersedes))
        ):
            logger.warning(
                "supersession-client mismatch op=create sent=%r superseded_ids=%r "
                "(server skew or regression)",
                supersedes,
                ids,
            )
        result = format_store_result(entry, is_update=False, include_backend_warning=not is_http)
        if ids:
            result += "\nSupersedes: " + ", ".join(ids)
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
