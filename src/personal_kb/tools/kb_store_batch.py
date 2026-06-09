"""kb_store_batch MCP tool — create multiple knowledge entries in one call."""

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.confidence.decay import compute_effective_confidence
from personal_kb.config import is_safety_skip
from personal_kb.ingest.safety import detect_secrets_in_content
from personal_kb.maps_index_writer import write_project_maps
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.tools.formatters import format_entry_compact, format_result_list
from personal_kb.tools.map_lint import lint_map_body
from personal_kb.tools.ttl import compute_expires_at

if TYPE_CHECKING:
    from personal_kb.graph.builder import GraphBuilder
    from personal_kb.graph.enricher import GraphEnricher
    from personal_kb.store.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

_MAX_BATCH = 10

_REQUIRED_FIELDS = {"short_title", "long_title", "knowledge_details"}


async def batch_store_entries(
    entries: list[dict[str, Any]],
    lifespan: dict[str, Any],
) -> str:
    """Core batch store logic, testable without MCP context."""
    from personal_kb.tools._lifespan import kb_from_lifespan

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

    # Per-entry failure tracking + maps-index refresh stay channel-side; the
    # facade's store_batch swallows failures and doesn't expose them, and
    # write_project_maps is env-driven (server-only). Reach for store /
    # embedder / graph_builder / graph_enricher / db through the facade so
    # there's a single source of truth even though the loop is local.
    kb = kb_from_lifespan(lifespan)
    store: KnowledgeStore = kb.knowledge_store
    embedder = kb.embedder
    graph_builder: GraphBuilder = kb.graph_builder
    graph_enricher: GraphEnricher | None = kb.graph_enricher
    db = kb.db
    contributor = kb.config.attribution.contributor
    team = kb.config.attribution.team

    created: list[KnowledgeEntry] = []
    failed: list[tuple[int, str, str]] = []  # (index, short_title, error)
    for i, entry_dict in enumerate(entries):
        entry_type = EntryType(entry_dict.get("entry_type", "factual_reference"))
        confidence = float(entry_dict.get("confidence_level", 0.9))
        tags = entry_dict.get("tags")
        hints = entry_dict.get("hints")

        sensitivity = entry_dict.get("sensitivity")

        # Compute expires_at from TTL if provided
        expires_at = None
        raw_ttl = entry_dict.get("ttl")
        if raw_ttl:
            try:
                expires_at = compute_expires_at(str(raw_ttl))
            except ValueError as exc:
                title = str(entry_dict.get("short_title", f"entry {i}"))
                failed.append((i, title, str(exc)))
                logger.warning("Invalid TTL for entry %d (%s): %s", i, title, exc)
                continue

        try:
            entry = await store.create_entry(
                short_title=entry_dict["short_title"],
                long_title=entry_dict["long_title"],
                knowledge_details=entry_dict["knowledge_details"],
                entry_type=entry_type,
                project_ref=entry_dict.get("project_ref"),
                source_context=entry_dict.get("source_context"),
                confidence_level=confidence,
                tags=list(tags) if tags else None,
                hints=dict(hints) if hints else None,
                contributor=contributor,
                team=team,
                sensitivity=str(sensitivity) if sensitivity else None,  # type: ignore[arg-type]  # validated by tool
                expires_at=expires_at,
            )
        except Exception as exc:
            title = str(entry_dict.get("short_title", f"entry {i}"))
            failed.append((i, title, str(exc)))
            logger.warning("Failed to create entry %d (%s): %s", i, title, exc)
            continue

        # Embed. ``embedder`` is the Embedder Protocol on kb_core; the concrete
        # EmbeddingClient carries ``store_embedding``. Duck-type via getattr so
        # a future plain-Protocol embedder degrades gracefully.
        if embedder:
            try:
                embedding = await embedder.embed(entry.embedding_text)
                if embedding is not None:
                    store_embedding = getattr(embedder, "store_embedding", None)
                    if callable(store_embedding):
                        await store_embedding(entry.id, embedding)
                        await store.mark_embedding(entry.id, True)
            except Exception:
                logger.warning("Failed to embed entry %s", entry.id, exc_info=True)

        # Build deterministic graph
        try:
            await graph_builder.build_for_entry(entry)
        except Exception:
            logger.warning("Failed to build graph for %s", entry.id, exc_info=True)

        created.append(entry)

    # Batch enrichment — single LLM call
    if graph_enricher and created:
        try:
            await graph_enricher.enrich_batch(created)
        except Exception:
            logger.warning("Batch enrichment failed", exc_info=True)

    # Refresh the on-disk maps index once per distinct project_ref that
    # received a mental_map. Best-effort: a writer failure must not fail
    # the batch (mirror the graph try/except wrapper above).
    map_projects: set[str] = {
        entry.project_ref
        for entry in created
        if entry.entry_type == EntryType.MENTAL_MAP and entry.project_ref
    }
    for project_ref in map_projects:
        try:
            await write_project_maps(db, project_ref, team=team)
        except Exception:
            logger.warning(
                "Failed to refresh maps index for project %s", project_ref, exc_info=True
            )
        try:
            await db.notify_maps_changed(project_ref)
        except Exception:
            logger.warning(
                "Failed to NOTIFY kb_maps_changed for project %s",
                project_ref,
                exc_info=True,
            )

    # Re-fetch entries to get updated state (embedding flag)
    now = datetime.now(UTC)
    formatted: list[str] = []
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
        # Advisory mental_map lint, attributed to this specific entry's block.
        # Never fails or skips the entry; purely informational.
        if refreshed.entry_type == EntryType.MENTAL_MAP and refreshed.knowledge_details:
            warnings = lint_map_body(refreshed.knowledge_details)
            if warnings:
                block += "\n" + "\n".join(warnings)
        formatted.append(block)

    # Build header with failure details
    if failed and not created:
        lines = [f"Batch failed: all {len(failed)} entries failed."]
        for idx, title, err in failed:
            lines.append(f"  Entry {idx} ({title}): {err}")
        return "\n".join(lines)

    header = f"Batch: {len(created)} entries created"
    if failed:
        header += f", {len(failed)} failed"

    result = format_result_list(formatted, header=header)

    if failed:
        fail_lines = ["", "Failed entries (retry these):"]
        for idx, title, err in failed:
            fail_lines.append(f"  Entry {idx} ({title}): {err}")
        result += "\n".join(fail_lines)

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
