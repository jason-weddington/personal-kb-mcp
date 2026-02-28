"""kb_store MCP tool — create and update knowledge entries."""

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.confidence.decay import compute_effective_confidence
from personal_kb.graph.builder import GraphBuilder
from personal_kb.graph.enricher import GraphEnricher
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.search.embeddings import EmbeddingClient
from personal_kb.store.knowledge_store import KnowledgeStore
from personal_kb.tools.formatters import format_entry_compact

logger = logging.getLogger(__name__)


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


def register_kb_store(mcp: FastMCP) -> None:
    """Register the kb_store tool with the MCP server."""

    @mcp.tool()
    async def kb_store(
        short_title: Annotated[str, Field(description="Brief identifier for the entry")] = "",
        long_title: Annotated[str, Field(description="Descriptive title")] = "",
        knowledge_details: Annotated[
            str, Field(description="Full content of the knowledge entry")
        ] = "",
        entry_type: Annotated[
            EntryType,
            Field(description="factual_reference, decision, pattern_convention, lesson_learned"),
        ] = EntryType.FACTUAL_REFERENCE,
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
                    "factual_reference 90d, decision 1y, pattern_convention 2y, lesson_learned 5y. "
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

        Use deactivate_entry_id to soft-delete incorrect or obsolete entries.

        Use entry_type to classify the knowledge:
        - factual_reference: version numbers, API endpoints, config values
        - decision: "chose X because Y" — history is critical
        - pattern_convention: coding standards, workflow preferences
        - lesson_learned: mistakes, debugging insights
        """
        if ctx is None:
            raise RuntimeError("Context not injected")
        lifespan = ctx.lifespan_context
        store: KnowledgeStore = lifespan["store"]
        embedder: EmbeddingClient = lifespan["embedder"]
        graph_builder: GraphBuilder = lifespan["graph_builder"]
        db = lifespan["db"]

        graph_enricher: GraphEnricher | None = lifespan.get("graph_enricher")

        # --- Deactivate path ---
        if deactivate_entry_id:
            try:
                entry = await store.deactivate_entry(deactivate_entry_id)
            except ValueError as e:
                return f"Error: {e}"
            # Remove outgoing graph edges
            await db.execute(
                "DELETE FROM graph_edges WHERE source = ?",
                (deactivate_entry_id,),
            )
            await db.commit()
            reason = f" ({change_reason})" if change_reason else ""
            return f"Deactivated entry {entry.id}: {entry.short_title}{reason}"

        # --- Update path ---
        if update_entry_id:
            if not knowledge_details:
                return "Error: knowledge_details is required when updating."
            entry = await store.update_entry(
                entry_id=update_entry_id,
                knowledge_details=knowledge_details,
                change_reason=change_reason,
                confidence_level=confidence_level,
                tags=tags,
                hints=hints,
            )
            # Re-embed updated entry
            if embedder:
                await _embed_entry(embedder, store, entry)
            await _build_graph(graph_builder, entry)
            await _enrich_graph(graph_enricher, entry)
            entry = await store.get_entry(entry.id) or entry
            return format_store_result(entry, is_update=True)

        # --- Create path ---
        if not short_title or not long_title or not knowledge_details:
            return (
                "Error: short_title, long_title, and knowledge_details "
                "are required when creating a new entry."
            )

        entry = await store.create_entry(
            short_title=short_title,
            long_title=long_title,
            knowledge_details=knowledge_details,
            entry_type=entry_type,
            project_ref=project_ref,
            source_context=source_context,
            confidence_level=confidence_level,
            tags=tags,
            hints=hints,
        )

        # Embed new entry
        if embedder:
            await _embed_entry(embedder, store, entry)
        await _build_graph(graph_builder, entry)
        await _enrich_graph(graph_enricher, entry)
        entry = await store.get_entry(entry.id) or entry

        return format_store_result(entry, is_update=False)


async def _build_graph(graph_builder: GraphBuilder, entry: KnowledgeEntry) -> None:
    """Build graph edges for an entry, logging failures without raising."""
    try:
        await graph_builder.build_for_entry(entry)
    except Exception:
        logger.warning("Failed to build graph for entry %s", entry.id, exc_info=True)


async def _enrich_graph(enricher: GraphEnricher | None, entry: KnowledgeEntry) -> None:
    """Attempt to enrich graph via LLM, logging failures without raising."""
    if enricher is None:
        return
    try:
        await enricher.enrich_entry(entry)
    except Exception:
        logger.warning("Failed to enrich graph for entry %s", entry.id, exc_info=True)


async def _embed_entry(
    embedder: EmbeddingClient, store: KnowledgeStore, entry: KnowledgeEntry
) -> None:
    """Attempt to embed an entry, logging failures without raising."""
    try:
        embedding = await embedder.embed(entry.embedding_text)
        if embedding is not None:
            await embedder.store_embedding(entry.id, embedding)
            await store.mark_embedding(entry.id, True)
    except Exception:
        logger.warning("Failed to embed entry %s", entry.id, exc_info=True)
