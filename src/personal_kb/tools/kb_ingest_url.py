"""kb_ingest_url MCP tool — ingest a URL into the knowledge base."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.tools.kb_ingest import _check_safety_deps, _format_file_result

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

        Two modes:
        - **url only**: fetches the page and extracts article text from HTML.
          Works for public sites; will fail on internal/corporate sites behind
          SSO, OAuth, or VPN — use ``content`` for those.
        - **url + content**: skips fetching; ingests the supplied text directly.
          Use when you already have the content (e.g. from an internal wiki
          fetched via another tool).

        Runs the standard ingestion pipeline: PII redaction, secret scanning,
        LLM extraction, and deduplication.

        """
        from personal_kb.tools._lifespan import backend_from_lifespan

        if ctx is None:
            raise RuntimeError("Context not injected")

        if not url:
            return "Error: url is required."

        backend = backend_from_lifespan(ctx.lifespan_context)

        if backend.is_remote:
            try:
                file_result = await backend.ingest_url(url, content, project_ref, dry_run)
            except Exception as exc:
                from personal_kb.backend.http import BackendHttpError, _map_error

                if isinstance(exc, BackendHttpError):
                    return _map_error(exc, "")
                return f"Error: {exc}"
            dry_prefix = "[DRY RUN] " if dry_run else ""
            line = f"{dry_prefix}{_format_file_result(file_result)}"
            if file_result.summary:
                line += f"\n  Summary: {file_result.summary}"
            return line

        # Local mode — require safety deps for secret/PII scanning
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

        from personal_kb.tools._lifespan import kb_from_lifespan

        kb = kb_from_lifespan(ctx.lifespan_context)
        if kb.extraction_llm is None and kb.query_llm is None:
            return "Error: No LLM available for ingestion. Configure an LLM provider."
        if kb.embedder is None:
            return "Error: No embedder configured. The ingest pipeline requires embeddings."

        try:
            if content is not None:
                # Agent provided pre-fetched content — skip fetch/extraction
                file_result = await kb.ingest_url_content(
                    content,
                    url,
                    project_ref=project_ref,
                    dry_run=dry_run,
                )
            else:
                file_result = await kb.ingest_url(
                    url,
                    project_ref=project_ref,
                    dry_run=dry_run,
                )
        except RuntimeError as exc:
            return f"Error: {exc}"

        dry_prefix = "[DRY RUN] " if dry_run else ""
        line = f"{dry_prefix}{_format_file_result(file_result)}"
        if file_result.summary:
            line += f"\n  Summary: {file_result.summary}"
        return line
