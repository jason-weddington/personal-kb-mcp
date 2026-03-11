"""kb_ingest_url MCP tool — ingest a URL into the knowledge base."""

import logging
from typing import TYPE_CHECKING, Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.ingest.ingester import FileIngester
from personal_kb.tools.kb_ingest import _check_safety_deps, _format_file_result

if TYPE_CHECKING:
    from personal_kb.graph.builder import GraphBuilder
    from personal_kb.graph.enricher import GraphEnricher
    from personal_kb.llm.provider import LLMProvider
    from personal_kb.search.embeddings import EmbeddingClient
    from personal_kb.store.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


def register_kb_ingest_url(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_ingest_url tool with the MCP server."""

    @mcp.tool(name=f"{prefix}ingest_url")
    async def kb_ingest_url(
        url: Annotated[
            str,
            Field(
                description="URL to fetch and ingest into the KB.",
            ),
        ],
        content: Annotated[
            str | None,
            Field(
                description=(
                    "Pre-fetched content for the URL. When provided, "
                    "skips fetching and HTML extraction — ingests this "
                    "text directly. Use when you already have the page "
                    "content (e.g. from authenticated sites, WebFetch, "
                    "or JavaScript-rendered pages)."
                ),
            ),
        ] = None,
        project_ref: Annotated[
            str | None,
            Field(description="Project tag for extracted entries"),
        ] = None,
        dry_run: Annotated[
            bool,
            Field(description="Analyze content without storing entries"),
        ] = False,
        ctx: Context | None = None,
    ) -> str:
        """Ingest a URL's content into the KB.

        By default, fetches the page and extracts article content from HTML
        (strips navigation, ads, and boilerplate). If ``content`` is provided,
        skips fetching and uses the supplied text directly — useful for
        authenticated pages, intranet sites, or JS-rendered SPAs where the
        agent has already retrieved the content.

        Runs the standard ingestion pipeline: PII redaction, secret scanning,
        LLM extraction, and deduplication.

        """
        if ctx is None:
            raise RuntimeError("Context not injected")

        if not url:
            return "Error: url is required."

        # Fail closed: require safety deps for secret/PII scanning
        from personal_kb.config import is_safety_skip

        if not is_safety_skip():
            missing = _check_safety_deps()
            if missing:
                return (
                    f"Error: Safety dependencies not installed: {', '.join(missing)}. "
                    "Secret and PII scanning cannot run without them.\n\n"
                    "Install with:\n"
                    "  uv sync --extra safety\n\n"
                    "To bypass (not recommended): set KB_SKIP_SAFETY=TRUE"
                )

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

        if content is not None:
            # Agent provided pre-fetched content — skip fetch/extraction
            file_result = await ingester._ingest_content(
                content,
                url,
                project_ref=project_ref,
                dry_run=dry_run,
            )
        else:
            file_result = await ingester.ingest_url(
                url,
                project_ref=project_ref,
                dry_run=dry_run,
            )

        dry_prefix = "[DRY RUN] " if dry_run else ""
        line = f"{dry_prefix}{_format_file_result(file_result)}"
        if file_result.summary:
            line += f"\n  Summary: {file_result.summary}"
        return line
