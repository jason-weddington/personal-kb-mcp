"""KB-domain request/response Pydantic models for the write API surface.

Used by ``kb_write_routes.py`` and extended by sibling P2 items.  Response
models embed ``kb_core.models.entry.KnowledgeEntry`` directly so the P5 thin
client can reconstruct engine models losslessly from JSON; do NOT use
``kb_core.formatting`` anywhere in this service.
"""

from typing import Any, Literal

from kb_core.models.entry import EntryType, KnowledgeEntry
from pydantic import BaseModel, Field


class StoreRequest(BaseModel):
    """Request body for ``POST /api/kb/store`` (create or update).

    When ``update_entry_id`` is None the create path is taken.  When it is set
    the update path is taken and ``short_title``/``long_title``/
    ``knowledge_details`` become optional (empty string means no-change).
    """

    short_title: str = ""
    long_title: str = ""
    knowledge_details: str = ""
    entry_type: EntryType | None = None
    project_ref: str | None = None
    source_context: str | None = None
    confidence_level: float | None = Field(None, ge=0.0, le=1.0)
    tags: list[str] | None = None
    hints: dict[str, Any] | None = None
    sensitivity: Literal["internal", "restricted", "public"] | None = None
    ttl: str | None = None
    update_entry_id: str | None = None
    change_reason: str | None = None
    supersedes: list[str] | Literal["none"] | None = Field(
        None,
        description=(
            "Entry ids this entry replaces. Absent (older clients), [] and the"
            " literal 'none' all mean no supersession."
        ),
    )
    distinct_from: list[str] | None = Field(
        None,
        description=(
            "Ids of existing entries this NEW entry is deliberately distinct from;"
            " resolves a near-duplicate 409. Recorded on the new entry as"
            " hints.distinct_from."
        ),
    )


class StoreResponse(BaseModel):
    """Response from ``POST /api/kb/store``."""

    action: Literal["created", "updated"]
    entry: KnowledgeEntry
    superseded_ids: list[str] = Field(
        default_factory=list,
        description=(
            "create: all validated targets; update: targets newly added by this request"
        ),
    )


class StoreBatchEntry(BaseModel):
    """Single entry in a ``store_batch`` request.

    All three text fields are required (``min_length=1``) and are validated by
    Pydantic before the endpoint body is processed.  ``contributor`` / ``team``
    are intentionally absent — the server injects attribution server-side.
    """

    short_title: str = Field(..., min_length=1)
    long_title: str = Field(..., min_length=1)
    knowledge_details: str = Field(..., min_length=1)
    entry_type: EntryType = EntryType.FACTUAL_REFERENCE
    project_ref: str | None = None
    source_context: str | None = None
    confidence_level: float = Field(0.9, ge=0.0, le=1.0)
    tags: list[str] | None = None
    hints: dict[str, Any] | None = None
    sensitivity: Literal["internal", "restricted", "public"] | None = None
    ttl: str | None = None
    supersedes: list[str] | Literal["none"] | None = Field(
        None,
        description=(
            "Entry ids this entry replaces. Absent (older clients), [] and the"
            " literal 'none' all mean no supersession."
        ),
    )
    distinct_from: list[str] | None = Field(
        None,
        description=(
            "Ids of existing entries this NEW entry is deliberately distinct from;"
            " resolves a near-duplicate 409. Recorded on the new entry as"
            " hints.distinct_from."
        ),
    )


class StoreBatchRequest(BaseModel):
    """Request body for ``POST /api/kb/store_batch``.

    ``entries`` must have 1-10 elements; outside that range Pydantic raises a
    ``RequestValidationError`` (422 list-envelope, not the custom-rule 422).
    """

    entries: list[StoreBatchEntry] = Field(..., min_length=1, max_length=10)


class StoreBatchResponse(BaseModel):
    """Response from ``POST /api/kb/store_batch``.

    ``created`` may be shorter than ``requested`` when the engine skips
    per-entry runtime failures (logged-and-skipped facade behaviour).
    """

    requested: int
    created: list[KnowledgeEntry]


class DeactivateRequest(BaseModel):
    """Optional body for ``POST /api/kb/entries/{id}/deactivate``.

    Both fields default to ``None`` on purpose: the route returns its own
    readable 422 for a missing ``change_reason`` rather than pydantic's
    list envelope.
    """

    change_reason: str | None = None
    superseded_by: str | None = None


class EntryActionResponse(BaseModel):
    """Response from deactivate / reactivate endpoints."""

    entry: KnowledgeEntry


class BulkUpdateRequest(BaseModel):
    """Request body for ``POST /api/kb/bulk_update``.

    ``dry_run`` defaults to ``True`` — callers must explicitly set it to
    ``False`` to persist changes, mirroring the MCP channel's safety default.
    """

    filters: dict[str, Any]
    updates: dict[str, Any]
    dry_run: bool = True


class BulkUpdatePair(BaseModel):
    """Before/after snapshot pair from a bulk_update operation."""

    before: KnowledgeEntry
    after: KnowledgeEntry


class BulkUpdateResponse(BaseModel):
    """Response from ``POST /api/kb/bulk_update``."""

    dry_run: bool
    count: int
    results: list[BulkUpdatePair]


class FeedbackRequest(BaseModel):
    """Request body for ``POST /api/kb/feedback``.

    ``feedback_type`` is a Literal — an unknown value causes Pydantic to
    produce a ``RequestValidationError`` (422) before the endpoint runs.
    """

    feedback_type: Literal["missing", "unhelpful", "friction"]
    tool_name: str | None = None
    query_or_params: str | None = None
    detail: str | None = None


class FeedbackResponse(BaseModel):
    """Response from ``POST /api/kb/feedback``."""

    status: Literal["recorded"]
    feedback_type: str
