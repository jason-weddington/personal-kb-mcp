"""kb_ingest MCP tool — ingest files from disk into the knowledge base."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.ingest.ingester import FileIngester, FileResult, IngestResult

if TYPE_CHECKING:
    from personal_kb.graph.builder import GraphBuilder
    from personal_kb.graph.enricher import GraphEnricher
    from personal_kb.llm.provider import LLMProvider
    from personal_kb.search.embeddings import EmbeddingClient
    from personal_kb.store.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


def _format_file_result(r: FileResult) -> str:
    """Format a single file result."""
    line = f"  {r.action}: {r.path}"
    if r.reason:
        line += f" — {r.reason}"
    if r.entry_count > 0:
        line += f" ({r.entry_count} entries)"
    if r.entry_ids:
        line += f" [{', '.join(r.entry_ids)}]"
    return line


def _format_ingest_result(result: IngestResult, dry_run: bool) -> str:
    """Format directory ingestion results."""
    prefix = "[DRY RUN] " if dry_run else ""
    lines = [f"{prefix}Ingestion complete\n"]
    lines.append(
        f"Files: {result.total_files} total, "
        f"{result.ingested} ingested, "
        f"{result.skipped} skipped, "
        f"{result.flagged} flagged, "
        f"{result.unchanged} unchanged, "
        f"{result.errors} errors"
    )
    lines.append(f"Entries: {result.entries_created} created\n")

    # Show details for non-trivial results
    for r in result.file_results:
        if r.action != "skipped":
            lines.append(_format_file_result(r))

    # Show skipped files summarized
    skipped = [r for r in result.file_results if r.action == "skipped"]
    if skipped:
        lines.append(f"\n  ({len(skipped)} files skipped — unsupported type or deny-list)")

    return "\n".join(lines)


def register_kb_ingest(mcp: FastMCP) -> None:
    """Register the kb_ingest tool with the MCP server."""

    @mcp.tool()
    async def kb_ingest(
        file_path: Annotated[
            str,
            Field(description="Absolute path to a file or directory to ingest"),
        ],
        project_ref: Annotated[
            str | None,
            Field(description="Project tag for extracted entries"),
        ] = None,
        dry_run: Annotated[
            bool,
            Field(description="Analyze files without storing entries"),
        ] = False,
        recursive: Annotated[
            bool,
            Field(description="Recurse into subdirectories"),
        ] = True,
        ctx: Context | None = None,
    ) -> str:
        """Ingest files from disk into the knowledge base.

        Reads files, runs safety checks (deny-list, secret detection, PII redaction),
        uses an LLM to summarize and extract structured knowledge entries.

        Files become note nodes in the knowledge graph, with extracted entries linked
        back to their source via extracted_from edges.

        Requires KB_MANAGER=TRUE environment variable.

        Supports: .md, .txt, .py, .js, .ts, .yaml, .json, .toml, and many more text formats.
        Skips: binaries, images, archives, keys, .env files, and other sensitive formats.
        """
        if ctx is None:
            raise RuntimeError("Context not injected")

        lifespan = ctx.lifespan_context
        db = lifespan["db"]
        store: KnowledgeStore = lifespan["store"]
        embedder: EmbeddingClient = lifespan["embedder"]
        graph_builder: GraphBuilder = lifespan["graph_builder"]
        graph_enricher: GraphEnricher | None = lifespan.get("graph_enricher")
        query_llm: LLMProvider | None = lifespan.get("query_llm")

        if query_llm is None:
            return "Error: No LLM available for ingestion. Configure an LLM provider."

        ingester = FileIngester(
            db=db,
            store=store,
            embedder=embedder,
            graph_builder=graph_builder,
            graph_enricher=graph_enricher,
            llm=query_llm,
        )

        target = Path(file_path).expanduser().resolve()

        if not target.exists():
            return f"Error: Path does not exist: {target}"

        if target.is_file():
            file_result = await ingester.ingest_file(
                target,
                project_ref=project_ref,
                base_dir=target.parent,
                dry_run=dry_run,
            )
            prefix = "[DRY RUN] " if dry_run else ""
            line = f"{prefix}{_format_file_result(file_result)}"
            if file_result.summary:
                line += f"\n  Summary: {file_result.summary}"
            return line

        if target.is_dir():
            dir_result = await ingester.ingest_directory(
                target,
                project_ref=project_ref,
                recursive=recursive,
                dry_run=dry_run,
            )
            return _format_ingest_result(dir_result, dry_run)

        return f"Error: {target} is not a file or directory."
