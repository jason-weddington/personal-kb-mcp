"""kb_store MCP tool — create and update knowledge entries."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.search.embeddings import EmbeddingClient
from personal_kb.store.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


def format_store_result(entry: KnowledgeEntry, is_update: bool = False) -> str:
    """Format the result of a store operation for the MCP response."""
    action = "Updated" if is_update else "Created"
    lines = [
        f"{action} entry {entry.id} (v{entry.version})",
        f"  Title: {entry.short_title}",
        f"  Type: {entry.entry_type.value}",
    ]
    if entry.project_ref:
        lines.append(f"  Project: {entry.project_ref}")
    if entry.tags:
        lines.append(f"  Tags: {', '.join(entry.tags)}")
    if not entry.has_embedding:
        lines.append("  Note: Entry will be embedded when Ollama is available")
    return "\n".join(lines)


def register_kb_store(mcp: FastMCP) -> None:
    """Register the kb_store tool with the MCP server."""

    @mcp.tool()
    async def kb_store(
        short_title: Annotated[str, Field(description="Brief identifier for the entry")],
        long_title: Annotated[str, Field(description="Descriptive title")],
        knowledge_details: Annotated[str, Field(description="Full content of the knowledge entry")],
        entry_type: Annotated[
            EntryType,
            Field(description="factual_reference, decision, pattern_convention, lesson_learned"),
        ],
        project_ref: Annotated[
            str | None, Field(description="Project tag/category for filtering")
        ] = None,
        source_context: Annotated[
            str | None,
            Field(description="Where this knowledge came from"),
        ] = None,
        confidence_level: Annotated[
            float,
            Field(description="Initial confidence score (0.0-1.0)", ge=0.0, le=1.0),
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
        change_reason: Annotated[
            str | None, Field(description="Reason for update (required when updating)")
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Store or update a knowledge entry in the personal knowledge base.

        Creates a new entry or updates an existing one. Every update creates a version
        record preserving the full history. Entries are automatically indexed for
        full-text search and (when Ollama is available) vector search.

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

        if update_entry_id:
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
            entry = await store.get_entry(entry.id) or entry
            return format_store_result(entry, is_update=True)

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
        entry = await store.get_entry(entry.id) or entry

        return format_store_result(entry, is_update=False)


async def _embed_entry(
    embedder: EmbeddingClient, store: KnowledgeStore, entry: KnowledgeEntry
) -> None:
    """Attempt to embed an entry, logging failures without raising."""
    try:
        text = f"{entry.short_title} {entry.long_title} {entry.knowledge_details}"
        embedding = await embedder.embed(text)
        if embedding is not None:
            await embedder.store_embedding(entry.id, embedding)
            await store.mark_embedding(entry.id, True)
    except Exception:
        logger.warning("Failed to embed entry %s", entry.id, exc_info=True)
