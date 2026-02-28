"""kb_store_batch MCP tool — create multiple knowledge entries in one call."""

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.confidence.decay import compute_effective_confidence
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.tools.formatters import format_entry_compact, format_result_list

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
    if len(entries) > _MAX_BATCH:
        return f"Error: Maximum {_MAX_BATCH} entries per batch (got {len(entries)})."

    if not entries:
        return "Error: entries list is empty."

    # Validate required fields
    for i, entry_dict in enumerate(entries):
        missing = _REQUIRED_FIELDS - set(entry_dict.keys())
        if missing:
            return f"Error: entry {i} missing required fields: {', '.join(sorted(missing))}"

    store: KnowledgeStore = lifespan["store"]
    embedder = lifespan["embedder"]
    graph_builder: GraphBuilder = lifespan["graph_builder"]
    graph_enricher: GraphEnricher | None = lifespan.get("graph_enricher")

    created: list[KnowledgeEntry] = []
    for entry_dict in entries:
        entry_type = EntryType(entry_dict.get("entry_type", "factual_reference"))
        confidence = float(entry_dict.get("confidence_level", 0.9))
        tags = entry_dict.get("tags")
        hints = entry_dict.get("hints")

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
        )

        # Embed
        if embedder:
            try:
                embedding = await embedder.embed(entry.embedding_text)
                if embedding is not None:
                    await embedder.store_embedding(entry.id, embedding)
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
        formatted.append(
            f"Created {refreshed.id} (v{refreshed.version})\n"
            + format_entry_compact(refreshed, eff)
        )

    return format_result_list(formatted, header=f"Batch: {len(created)} entries created")


def register_kb_store_batch(mcp: FastMCP) -> None:
    """Register the kb_store_batch tool with the MCP server."""

    @mcp.tool()
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
        """Store multiple knowledge entries in a single call.

        More efficient than calling kb_store repeatedly — uses a single LLM
        call for graph enrichment across all entries.

        Each entry dict requires: short_title, long_title, knowledge_details.
        Optional fields: entry_type (default: factual_reference), project_ref,
        source_context, confidence_level (default: 0.9), tags, hints.
        """
        if ctx is None:
            raise RuntimeError("Context not injected")

        return await batch_store_entries(entries, ctx.lifespan_context)
