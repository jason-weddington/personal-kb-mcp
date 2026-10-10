"""kb_ingest_url MCP tool (HTTP twin of ``personal_kb.tools.kb_ingest_url``)."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from kb_core.ingest.ingester import FileResult
from pydantic import Field

from kb_service.mcp_server import context
from kb_service.mcp_server.errors import BackendHttpError, map_error

logger = logging.getLogger(__name__)

_INGEST_URL_DESCRIPTION = (
    "Ingest a URL's content into the KB.\n"
    "\n"
    "Two modes:\n"
    "- **url only**: fetches the page and extracts article text from HTML.\n"
    "  Works for public sites; will fail on internal/corporate sites behind\n"
    "  SSO, OAuth, or VPN — use ``content`` for those.\n"
    "- **url + content**: skips fetching; ingests the supplied text directly.\n"
    "  Use when you already have the content (e.g. from an internal wiki\n"
    "  fetched via another tool).\n"
    "\n"
    "Runs the standard ingestion pipeline: PII redaction, secret scanning,\n"
    "LLM extraction, and deduplication."
)


_CONTENT_DESCRIPTION = (
    "Pre-fetched content for the URL. When provided, "
    "skips fetching and HTML extraction — ingests this "
    "text directly. Use when you already have the page "
    "content (e.g. from authenticated sites, WebFetch, "
    "or JavaScript-rendered pages)."
)


def _format_file_result(r: FileResult) -> str:
    """Format a single file result (private copy of the stdio formatter)."""
    line = f"  {r.action}: {r.path}"
    if r.reason:
        line += f" — {r.reason}"
    if r.entry_count > 0:
        line += f" ({r.entry_count} entries)"
    if r.chunks_processed > 1 or r.chunks_skipped > 0 or r.chunks_flagged > 0:
        chunk_info = f"{r.chunks_processed} chunks"
        if r.chunks_skipped > 0:
            chunk_info += f", {r.chunks_skipped} deduped"
        if r.chunks_flagged > 0:
            chunk_info += f", {r.chunks_flagged} redacted (secrets)"
        line += f" [{chunk_info}]"
    elif r.entry_ids:
        line += f" [{', '.join(r.entry_ids)}]"
    return line


def register_kb_ingest_url(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_ingest_url tool with the MCP server."""

    @mcp.tool(name=f"{prefix}ingest_url", description=_INGEST_URL_DESCRIPTION)
    async def kb_ingest_url(
        url: Annotated[
            str,
            Field(
                description="URL to fetch and ingest into the KB.",
            ),
        ],
        content: Annotated[
            str | None,
            Field(description=_CONTENT_DESCRIPTION),
        ] = None,
        project_ref: Annotated[
            str | None,
            Field(description="Project tag for extracted entries"),
        ] = None,
        dry_run: Annotated[
            bool,
            Field(description="Analyze content without storing entries"),
        ] = False,
    ) -> str:
        """Ingest a URL's content into the KB."""
        if not url:
            return "Error: url is required."

        backend = context.backend_for_request()

        try:
            file_result = await backend.ingest_url(url, content, project_ref, dry_run)
        except Exception as exc:
            if isinstance(exc, BackendHttpError):
                return map_error(exc)
            return f"Error: {exc}"
        dry_prefix = "[DRY RUN] " if dry_run else ""
        line = f"{dry_prefix}{_format_file_result(file_result)}"
        if file_result.summary:
            line += f"\n  Summary: {file_result.summary}"
        return line
