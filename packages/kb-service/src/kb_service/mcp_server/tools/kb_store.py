"""kb_store MCP tool (HTTP twin of ``personal_kb.tools.kb_store``)."""

import logging
import re
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastmcp import FastMCP
from kb_core.confidence.decay import compute_effective_confidence
from kb_core.formatting import format_entry_compact
from kb_core.graph.builder import _as_list
from kb_core.ingest.safety import detect_secrets_in_content
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.ttl import compute_expires_at
from pydantic import Field

from kb_service.config import is_safety_skip
from kb_service.mcp_server import context
from kb_service.mcp_server.backend import QueuedStore
from kb_service.mcp_server.errors import BackendHttpError, map_error
from kb_service.mcp_server.observability import _principal_fields
from kb_service.mcp_server.tools.map_lint import lint_map_body

logger = logging.getLogger(__name__)

HINTS_DESCRIPTION = (
    "Structured hints for graph building (related_entities, person, tool). "
    "Optional 'resolution' object records a corrected belief that is delivered "
    "to future sessions: corrected_fact (required str), wrong_belief (str), "
    "evidence (str), cue {tool, target_class, args_prefix} where a Bash "
    "target_class is the two-word normalized command class such as 'git remote' "
    "and args_prefix (optional str) narrows the match to commands whose "
    "arguments start with it, provenance {capture: deliberate|autonomous, "
    "grounding: observed|asserted, event_id}, observed_sessions (int >= 1), "
    "scope (project|global; global applies in every project). The server "
    "defaults capture to 'deliberate' and grounding to 'asserted' and rejects "
    "a malformed resolution with 422. "
    "Prefer the top-level resolution parameter, which lesson_learned requires; "
    "passing both is an error."
)

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
    "On update, listed ids are added to the entry's supersedes set; "
    "ids are never removed. "
    'Must be "none" when deactivate_entry_id is set.'
)
DISTINCT_FROM_DESCRIPTION = (
    "kb-XXXXX ids of existing entries this new entry is deliberately distinct "
    "from. "
    "Recorded server-side; a near-duplicate rejection for the same pair is not "
    "raised again. "
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

_STORE_DESCRIPTION = (
    "Store or update a knowledge entry in the personal knowledge base.\n"
    "\n"
    "Creates a new entry or updates an existing one. Every update creates a version\n"
    "record preserving the full history. Entries are automatically indexed for\n"
    "full-text search and (when Ollama is available) vector search.\n"
    "\n"
    "For metadata-only updates (tags, project_ref, sensitivity, entry_type, etc.),\n"
    "pass update_entry_id with the fields to change — knowledge_details is "
    "optional.\n"
    "This avoids pulling and rewriting the full entry content.\n"
    "\n"
    "Use deactivate_entry_id to soft-delete incorrect or obsolete entries.\n"
    "\n"
    "A lesson_learned create requires resolution: the wrong belief and the corrected\n"
    'fact, or "none" when the lesson is not a corrected belief. The KB pushes only\n'
    "resolutions to future sessions (the session-start gotcha list and the pre-command"
    "\ngate), so a lesson stored without one is found only if someone "
    "searches for it.\n"
    "\n"
    "Use entry_type to classify the knowledge:\n"
    "- factual_reference: version numbers, API endpoints, config values\n"
    '- decision: "chose X because Y" — history is critical\n'
    "- pattern_convention: coding standards, workflow preferences\n"
    "- lesson_learned: mistakes, debugging insights\n"
    "- mental_map: structural orientation node — pointers/relationships only,\n"
    "  no retrievable values; requires at least one outbound pointer"
)


_ENTRY_TYPE_DESCRIPTION = (
    "factual_reference, decision, pattern_convention, lesson_learned, "
    "mental_map: structural orientation node — pointers/relationships only, "
    "no retrievable values"
)

_CONFIDENCE_DESCRIPTION = (
    "Initial confidence score (0.0-1.0). "
    "Decays over time based on entry_type half-life: "
    "factual_reference 90d, decision 1y, pattern_convention 2y, "
    "lesson_learned 5y, "
    "mental_map: no decay (exempt). "
    "Lower for uncertain info, higher for verified facts. Default 0.9"
)


def _validate_supersedes(value: object) -> str | None:
    """Return an error string if *value* is not a valid supersedes, None if OK."""
    if value == "none" and isinstance(value, str):
        return None
    if isinstance(value, list):
        if not value:
            return (
                'Error: supersedes=[] is ambiguous; pass "none" when this entry '
                "replaces nothing."
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
    return (
        "Error: distinct_from must be a list of kb-XXXXX ids (got " + repr(value) + ")."
    )


def _validate_hints_supersedes_conflict(
    supersedes: object, hints: object
) -> str | None:
    """Reject supersedes="none" combined with a non-empty hints.supersedes."""
    if (
        isinstance(supersedes, str)
        and supersedes == "none"
        and isinstance(hints, dict)
        and hints.get("supersedes")
    ):
        return (
            'Error: supersedes="none" conflicts with hints.supersedes; '
            "list the ids in supersedes."
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


def _reject(
    path: str, rule: str, value: object, message: str, op: str = "store"
) -> str:
    _log_decision(op, path, "rejected", rule, value)
    return message


def _supersedes_rule(message: str) -> str:
    return "empty_list" if "ambiguous" in message else "bad_shape"


RESOLUTION_DESCRIPTION = (
    "Required when creating a lesson_learned entry; optional otherwise. "
    'Either the literal "none" (the lesson is not a corrected belief) '
    "or an object: wrong_belief (required str: what was believed that "
    "turned out wrong), corrected_fact (required str: what is actually "
    "true), evidence (str), cue {tool, target_class, args_prefix} where "
    "a Bash target_class is the two-word normalized command class such "
    "as 'git push' and args_prefix (optional str) narrows the match to "
    "commands whose arguments start with it, scope (project|global; "
    "global applies in every project). A Bash cue with a two-word "
    "target_class can arm the pre-command gate, which denies a matching "
    "command once per lesson per day, so add a cue only when one "
    "specific command caused the mistake and set args_prefix to narrow "
    "it. The KB pushes only resolutions to future sessions (the "
    "session-start gotcha list and, for a Bash cue, the pre-command "
    "gate); a lesson stored without one is found only by search. It is "
    "stored as hints.resolution, so do not also pass hints.resolution. "
    'On update an object replaces the stored resolution and "none" '
    "leaves it unchanged. Not allowed with deactivate_entry_id."
)

RESOLUTION_REQUIRED_ERROR = (
    "Error: a lesson_learned entry requires resolution: an object "
    '{"wrong_belief": "...", "corrected_fact": "..."} naming the belief '
    "that turned out wrong and the fact that replaced it (optional "
    'evidence, cue, scope), or the literal "none" when the lesson is '
    "not a corrected belief. The KB pushes only resolutions to future "
    "sessions (the session-start gotcha list and the pre-command gate); "
    "a lesson stored without one is found only by search."
)
_RESOLUTION_CONFLICT_ERROR = (
    "Error: resolution and hints.resolution are both set; pass it once, as resolution."
)
_RESOLUTION_DEACTIVATE_ERROR = (
    "Error: resolution does not apply to deactivate_entry_id."
)
_RESOLUTION_KEYS = {"wrong_belief", "corrected_fact", "evidence", "cue", "scope"}
_RESOLUTION_CUE_KEYS = {"tool", "target_class", "args_prefix"}
_CUE_OBJECT_ERROR = (
    "Error: resolution.cue must be an object with non-blank tool and "
    "target_class strings."
)


def _non_blank(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _validate_resolution(value: object) -> tuple[str, str] | None:
    """Return ``(check, message)`` if *value* is not a valid resolution, None if OK."""
    if not isinstance(value, dict) and not (isinstance(value, str) and value == "none"):
        return (
            "not_object",
            'Error: resolution must be an object or the literal "none" (got '
            + repr(value)
            + ").",
        )
    if not isinstance(value, dict):
        return None
    unknown = set(value) - _RESOLUTION_KEYS
    if unknown:
        return (
            "unknown_keys",
            "Error: resolution has unknown keys "
            + repr(sorted(unknown))
            + "; allowed: wrong_belief, corrected_fact, evidence, cue, scope "
            "(provenance and observed_sessions are server-managed).",
        )
    if not _non_blank(value.get("wrong_belief")):
        return (
            "wrong_belief",
            "Error: resolution.wrong_belief is required and must be a non-blank "
            "string.",
        )
    if not _non_blank(value.get("corrected_fact")):
        return (
            "corrected_fact",
            "Error: resolution.corrected_fact is required and must be a non-blank "
            "string.",
        )
    if "evidence" in value and not isinstance(value["evidence"], str):
        return ("evidence", "Error: resolution.evidence must be a string.")
    if "cue" in value:
        cue = value["cue"]
        if not isinstance(cue, dict):
            return ("cue_not_object", _CUE_OBJECT_ERROR)
        cue_unknown = set(cue) - _RESOLUTION_CUE_KEYS
        if cue_unknown:
            return (
                "cue_unknown_keys",
                "Error: resolution.cue has unknown keys "
                + repr(sorted(cue_unknown))
                + "; allowed: tool, target_class, args_prefix.",
            )
        if not _non_blank(cue.get("tool")) or not _non_blank(cue.get("target_class")):
            return ("cue_fields", _CUE_OBJECT_ERROR)
        if "args_prefix" in cue and not _non_blank(cue["args_prefix"]):
            return (
                "args_prefix",
                "Error: resolution.cue.args_prefix must be a non-blank string.",
            )
    if "scope" in value and value["scope"] not in ("project", "global"):
        return ("scope", 'Error: resolution.scope must be "project" or "global".')
    return None


def _has_hints_resolution(hints: object) -> bool:
    return isinstance(hints, dict) and "resolution" in hints


def _resolution_mode(value: object, hints: object) -> str:
    """How the caller supplied a resolution: object, none, invalid, hints or absent."""
    if isinstance(value, dict):
        return "object"
    if isinstance(value, str) and value == "none":
        return "none"
    if value is not None:
        return "invalid"
    if _has_hints_resolution(hints):
        return "hints"
    return "absent"


def _entry_type_label(entry_type: object) -> object:
    return entry_type.value if isinstance(entry_type, EntryType) else entry_type


def _log_resolution_decision(
    op: str,
    path: str,
    outcome: str,
    rule: str,
    check: str,
    value: object,
    hints: object,
    entry_type: object,
) -> None:
    logger.info(
        "resolution-client op=%s path=%s outcome=%s rule=%s check=%s mode=%s "
        "entry_type=%s "
        "user_id=%s key_id=%s",
        op,
        path,
        outcome,
        rule,
        check,
        _resolution_mode(value, hints),
        _entry_type_label(entry_type),
        *_principal_fields()[:2],
    )


def _log_resolution_stored(
    op: str,
    path: str,
    mode: str,
    entry: KnowledgeEntry,
) -> None:
    """Log what the server returned; warn when a resolution was dropped."""
    stored = "resolution" in entry.hints
    logger.info(
        "resolution-client stored op=%s path=%s mode=%s entry_id=%s entry_type=%s "
        "resolution_stored=%s user_id=%s key_id=%s",
        op,
        path,
        mode,
        entry.id,
        _entry_type_label(entry.entry_type),
        stored,
        *_principal_fields()[:2],
    )
    if mode in ("object", "hints") and not stored:
        logger.warning(
            "resolution-client mismatch op=%s path=%s entry_id=%s mode=%s "
            "(server skew or regression)",
            op,
            path,
            entry.id,
            mode,
        )


def _map_resolution(
    resolution: object, hints: dict[str, object] | None
) -> dict[str, object] | None:
    """Fold a resolution object into a copy of hints; "none" leaves hints as passed."""
    if isinstance(resolution, dict):
        return {**(hints or {}), "resolution": dict(resolution)}
    return hints


def _check_resolution(
    op: str,
    path: str,
    resolution: object,
    hints: object,
    entry_type: object,
    *,
    require: bool,
) -> str | None:
    """Run the resolution checks; return an error message or None."""

    def fail(rule: str, check: str, message: str) -> str:
        _log_resolution_decision(
            op, path, "rejected", rule, check, resolution, hints, entry_type
        )
        return message

    if resolution is not None:
        shape = _validate_resolution(resolution)
        if shape:
            return fail("bad_shape", shape[0], shape[1])
        if (
            isinstance(resolution, dict)
            and hints is not None
            and not isinstance(hints, dict)
        ):
            return fail(
                "bad_shape",
                "hints_not_object",
                "Error: hints must be an object when resolution is set (got "
                + repr(hints)
                + ").",
            )
        if _has_hints_resolution(hints):
            return fail("hints_conflict", "none", _RESOLUTION_CONFLICT_ERROR)
    elif (
        require
        and entry_type == EntryType.LESSON_LEARNED
        and not _has_hints_resolution(hints)
    ):
        return fail("required", "none", RESOLUTION_REQUIRED_ERROR)
    _log_resolution_decision(
        op, path, "accepted", "none", "none", resolution, hints, entry_type
    )
    return None


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

    An outbound pointer exists iff a ``kb-XXXXX`` reference appears in
    knowledge_details, or a ``related_entities`` hint contains a dict with a
    non-empty ``id``/``target`` or a bare non-empty string.
    """
    if knowledge_details and _KB_ID_RE.search(knowledge_details):
        return True

    h = hints or {}

    for rel in _as_list(h.get("related_entities")):
        if isinstance(rel, dict):
            ref = rel.get("id") or rel.get("target")
            if isinstance(ref, str) and ref:
                return True
        elif isinstance(rel, str) and rel:
            return True

    return False


def format_queued_store(q: QueuedStore) -> str:
    """Render a create the KB service's write policy queued as a candidate."""
    if q.capture_mode == "on":
        return (
            f"Queued as candidate {q.candidate_id} for review (write policy:"
            f" {q.surface} surface). Not in the KB yet: the candidate pipeline's"
            " distiller and critic decide whether it is written."
        )
    return (
        f"Queued as candidate {q.candidate_id} (write policy: {q.surface} surface)."
        f" Not in the KB: capture mode is {q.capture_mode}, so it is recorded for"
        " audit only."
    )


def format_store_result(entry: KnowledgeEntry, is_update: bool = False) -> str:
    """Format the result of a store operation for the MCP response."""
    action = "Updated" if is_update else "Created"
    anchor = entry.updated_at or entry.created_at or datetime.now(UTC)
    eff = compute_effective_confidence(entry.confidence_level, entry.entry_type, anchor)
    compact = format_entry_compact(entry, eff)
    line = f"{action} {entry.id} (v{entry.version})\n{compact}"
    if not entry.has_embedding:
        line += "\n  Note: Entry will be embedded when Ollama is available"
    return line


def _prepend_map_advisories(result: str, warnings: list[str]) -> str:
    """Insert advisory mental_map lint lines above a store result."""
    if not warnings:
        return result
    advisory = "\n".join(warnings)
    return f"{advisory}\n\n{result}"


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


def register_kb_store(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_store tool with the MCP server."""

    @mcp.tool(name=f"{prefix}store", description=_STORE_DESCRIPTION)
    async def kb_store(
        supersedes: Annotated[
            list[str] | Literal["none"], Field(description=SUPERSEDES_DESCRIPTION)
        ],
        short_title: Annotated[
            str, Field(description="Brief identifier for the entry")
        ] = "",
        long_title: Annotated[str, Field(description="Descriptive title")] = "",
        knowledge_details: Annotated[
            str, Field(description="Full content of the knowledge entry")
        ] = "",
        entry_type: Annotated[
            EntryType | None, Field(description=_ENTRY_TYPE_DESCRIPTION)
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
            Field(description=_CONFIDENCE_DESCRIPTION, ge=0.0, le=1.0),
        ] = 0.9,
        tags: Annotated[
            list[str] | None, Field(description="Freeform tags for categorization")
        ] = None,
        hints: Annotated[
            dict[str, object] | None,
            Field(description=HINTS_DESCRIPTION),
        ] = None,
        update_entry_id: Annotated[
            str | None,
            Field(description="ID of existing entry to update (e.g. kb-00042)"),
        ] = None,
        sensitivity: Annotated[
            str | None,
            Field(
                description=(
                    "Sensitivity classification: internal, restricted, public, "
                    "or None. "
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
                    "Removes from search results and graph. "
                    "Reversible via kb_maintain."
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
        superseded_by: Annotated[
            str | None, Field(description=SUPERSEDED_BY_DESCRIPTION)
        ] = None,
        resolution: Annotated[
            dict[str, object] | Literal["none"] | None,
            Field(description=RESOLUTION_DESCRIPTION),
        ] = None,
    ) -> str:
        """Store or update a knowledge entry in the knowledge base."""
        backend = context.backend_for_request()

        # --- Deactivate path ---
        if deactivate_entry_id:
            if change_reason is None or not change_reason.strip():
                return _reject(
                    "deactivate",
                    "change_reason_missing",
                    supersedes,
                    _CHANGE_REASON_ERROR,
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
                    "Error: superseded_by must be a kb-XXXXX id (got "
                    + repr(superseded_by)
                    + ").",
                )
            if distinct_from:
                return _reject(
                    "deactivate",
                    "distinct_from_misplaced",
                    supersedes,
                    "Error: distinct_from applies to create only.",
                )
            if resolution is not None:
                _log_resolution_decision(
                    "store",
                    "deactivate",
                    "rejected",
                    "misplaced",
                    "none",
                    resolution,
                    hints,
                    None,
                )
                return _RESOLUTION_DEACTIVATE_ERROR
            _log_decision("store", "deactivate", "accepted", "none", supersedes)
            try:
                entry = await backend.deactivate(
                    deactivate_entry_id,
                    change_reason=change_reason,
                    superseded_by=superseded_by,
                )
            except Exception as e:
                if isinstance(e, BackendHttpError):
                    return map_error(e)
                return f"Error: {e}"

            by = f"; superseded by {superseded_by}" if superseded_by else ""
            return (
                f"Deactivated entry {entry.id}: {entry.short_title} "
                f"({change_reason}){by}"
            )

        # --- Update path ---
        if update_entry_id:
            if change_reason is None or not change_reason.strip():
                return _reject(
                    "update", "change_reason_missing", supersedes, _CHANGE_REASON_ERROR
                )
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
            res_err = _check_resolution(
                "store", "update", resolution, hints, entry_type, require=False
            )
            if res_err:
                return res_err
            res_mode = _resolution_mode(resolution, hints)
            hints = _map_resolution(resolution, hints)
            sens_err = _validate_sensitivity(sensitivity)
            if sens_err:
                return sens_err
            if knowledge_details:
                secret_err = _check_secrets(knowledge_details)
                if secret_err:
                    return secret_err
            if ttl:
                try:
                    compute_expires_at(ttl)
                except ValueError as e:
                    return f"Error: {e}"

            try:
                res = await backend.store(
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
                if isinstance(e, BackendHttpError):
                    return map_error(e)
                return f"Error: {e}"

            if isinstance(res, QueuedStore):
                return (
                    "Error: the KB service queued an update as candidate"
                    f" {res.candidate_id}; updates are never queued."
                )
            _action, entry, ids = res
            _log_resolution_stored("store", "update", res_mode, entry)
            if ids is None and isinstance(supersedes, list) and supersedes:
                logger.warning(
                    "supersession-client mismatch op=update sent=%r superseded_ids=%r "
                    "(server skew or regression)",
                    supersedes,
                    ids,
                )
            result = format_store_result(entry, is_update=True)
            if ids:
                result += "\nSupersedes: " + ", ".join(ids)
            if entry.entry_type == EntryType.MENTAL_MAP and knowledge_details:
                result = _prepend_map_advisories(
                    result, lint_map_body(knowledge_details)
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
        res_err = _check_resolution(
            "store", "create", resolution, hints, entry_type, require=True
        )
        if res_err:
            return res_err
        res_mode = _resolution_mode(resolution, hints)
        hints = _map_resolution(resolution, hints)

        sens_err = _validate_sensitivity(sensitivity)
        if sens_err:
            return sens_err

        secret_err = _check_secrets(knowledge_details)
        if secret_err:
            return secret_err

        if entry_type is None:
            entry_type = EntryType.FACTUAL_REFERENCE

        # Reject a zero-pointer map before any write.
        if entry_type == EntryType.MENTAL_MAP and not _mental_map_has_pointer(
            knowledge_details, hints
        ):
            return ORPHAN_MAP_ERROR

        if ttl:
            try:
                compute_expires_at(ttl)
            except ValueError as e:
                return f"Error: {e}"

        try:
            res = await backend.store(
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
            if isinstance(e, BackendHttpError):
                return map_error(e)
            return f"Error: {e}"

        if isinstance(res, QueuedStore):
            return format_queued_store(res)
        _action, entry, ids = res
        _log_resolution_stored("store", "create", res_mode, entry)
        hint_sup = hints.get("supersedes") if isinstance(hints, dict) else None
        expected_sup: set[str] = (
            set(supersedes) if isinstance(supersedes, list) else set()
        )
        if isinstance(hint_sup, list):
            expected_sup |= {h for h in hint_sup if isinstance(h, str)}
        elif isinstance(hint_sup, str) and hint_sup != "none":
            expected_sup.add(hint_sup)
        if (
            isinstance(supersedes, list)
            and supersedes
            and (ids is None or set(ids) != expected_sup)
        ):
            logger.warning(
                "supersession-client mismatch op=create sent=%r superseded_ids=%r "
                "(server skew or regression)",
                supersedes,
                ids,
            )
        result = format_store_result(entry, is_update=False)
        if ids:
            result += "\nSupersedes: " + ", ".join(ids)
        if entry.entry_type == EntryType.MENTAL_MAP and knowledge_details:
            result = _prepend_map_advisories(result, lint_map_body(knowledge_details))
        return result
