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

_GLOB_CHARS = set("*?[")


def _is_glob(path: str) -> bool:
    """Return True if the path contains glob metacharacters."""
    return bool(_GLOB_CHARS.intersection(path))


def _tally_result(result: IngestResult, file_result: FileResult) -> None:
    """Update IngestResult counters from a single FileResult."""
    if file_result.action == "ingested":
        result.ingested += 1
        result.entries_created += file_result.entry_count
    elif file_result.action == "skipped":
        result.skipped += 1
    elif file_result.action == "flagged":
        result.flagged += 1
    elif file_result.action == "error":
        result.errors += 1
    elif file_result.action == "unchanged":
        result.unchanged += 1
    elif file_result.action == "dry_run":
        result.ingested += 1  # Count as would-be-ingested
        result.entries_created += file_result.entry_count


def _format_file_result(r: FileResult) -> str:
    """Format a single file result."""
    line = f"  {r.action}: {r.path}"
    if r.reason:
        line += f" — {r.reason}"
    if r.entry_count > 0:
        line += f" ({r.entry_count} entries)"
    if r.chunks_processed > 1 or r.chunks_skipped > 0:
        chunk_info = f"{r.chunks_processed} chunks"
        if r.chunks_skipped > 0:
            chunk_info += f", {r.chunks_skipped} skipped"
        line += f" [{chunk_info}]"
    elif r.entry_ids:
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


def register_kb_ingest(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_ingest tool with the MCP server."""

    @mcp.tool(name=f"{prefix}ingest")
    async def kb_ingest(
        path: Annotated[
            str,
            Field(
                default="",
                description=(
                    "File, directory, or glob pattern to ingest. "
                    "Accepts absolute paths, relative paths, ~ paths, "
                    "and glob patterns (e.g. *.md, docs/**/*.txt). "
                    "Not required when content is provided."
                ),
            ),
        ] = "",
        content: Annotated[
            str | None,
            Field(
                description=(
                    "Pre-fetched content to ingest (e.g. from a URL). "
                    "When provided, source_url is required and path is ignored."
                ),
            ),
        ] = None,
        source_url: Annotated[
            str | None,
            Field(
                description=(
                    "Source URL for attribution and dedup when ingesting "
                    "pre-fetched content. Required when content is provided."
                ),
            ),
        ] = None,
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
            Field(description="Recurse into subdirectories (for directory paths)"),
        ] = True,
        ctx: Context | None = None,
    ) -> str:
        """Ingest files from disk or pre-fetched content into the KB.

        This is intelligent extraction, not raw dumping. An LLM reads the source,
        identifies distinct knowledge entries (decisions, patterns, facts, lessons),
        and creates properly titled/typed/tagged entries — typically several per file.
        New entries are deduplicated against the existing KB, so it is safe to ingest
        files that overlap with what's already stored. Re-ingesting the same file
        cleanly replaces its old entries.

        Also runs secret detection and PII redaction before extraction.

        Two modes:
        - File mode: provide path (file, directory, or glob). Runs deny-list and
          extension checks. Supports .md, .txt, .py, .js, .ts, .yaml, .json, .toml, etc.
        - Content mode: provide content + source_url. Skips filesystem checks.
          Use for pre-fetched web pages, wiki articles, or any text with a URL.
        """
        if ctx is None:
            raise RuntimeError("Context not injected")

        # Validate parameters
        if content is not None and not source_url:
            return "Error: source_url is required when content is provided."
        if content is None and not path:
            return "Error: Either path or content (with source_url) must be provided."

        lifespan = ctx.lifespan_context
        db = lifespan["db"]
        store: KnowledgeStore = lifespan["store"]
        embedder: EmbeddingClient = lifespan["embedder"]
        graph_builder: GraphBuilder = lifespan["graph_builder"]
        graph_enricher: GraphEnricher | None = lifespan.get("graph_enricher")
        query_llm: LLMProvider | None = lifespan.get("query_llm")

        if query_llm is None:
            return "Error: No LLM available for ingestion. Configure an LLM provider."

        # Construct dedup agent if agentic ingest is enabled
        dedup_agent = None
        if not dry_run:
            from personal_kb.config import is_agentic_ingest

            if is_agentic_ingest():
                from personal_kb.ingest.dedup_agent import DedupAgent

                dedup_agent = DedupAgent(db=db, embedder=embedder, llm=query_llm)

        contributor: str | None = lifespan.get("contributor")
        team: str | None = lifespan.get("team")

        ingester = FileIngester(
            db=db,
            store=store,
            embedder=embedder,
            graph_builder=graph_builder,
            graph_enricher=graph_enricher,
            llm=query_llm,
            dedup_agent=dedup_agent,
            contributor=contributor,
            team=team,
        )

        # Content mode: ingest pre-fetched content with URL attribution
        if content is not None:
            file_result = await ingester.ingest_content(
                content,
                source_url,  # type: ignore[arg-type]  # validated above
                project_ref=project_ref,
                dry_run=dry_run,
            )
            prefix = "[DRY RUN] " if dry_run else ""
            line = f"{prefix}{_format_file_result(file_result)}"
            if file_result.summary:
                line += f"\n  Summary: {file_result.summary}"
            return line

        # Glob pattern: expand and ingest each matched file
        if _is_glob(path):
            base = Path.cwd()
            matched = sorted(f for f in base.glob(path) if f.is_file() and not f.is_symlink())
            if not matched:
                return f"Error: No files matched pattern: {path}"

            result = IngestResult()
            for file_path in matched:
                result.total_files += 1
                file_result = await ingester.ingest_file(
                    file_path,
                    project_ref=project_ref,
                    base_dir=base,
                    dry_run=dry_run,
                )
                result.file_results.append(file_result)
                _tally_result(result, file_result)

            return _format_ingest_result(result, dry_run)

        # Single file or directory
        target = Path(path).expanduser().resolve()

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
