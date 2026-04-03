"""File ingestion orchestrator — reads files, runs safety, extracts entries."""

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from personal_kb.config import get_ingest_max_file_size
from personal_kb.db.backend import Database
from personal_kb.graph.builder import GraphBuilder
from personal_kb.graph.enricher import GraphEnricher
from personal_kb.ingest.chunker import chunk_content
from personal_kb.ingest.extractor import ExtractedEntry, extract_entries, summarize_file
from personal_kb.ingest.safety import detect_secrets_in_content
from personal_kb.llm.provider import LLMProvider
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.search.embeddings import EmbeddingClient
from personal_kb.store.knowledge_store import KnowledgeStore

if TYPE_CHECKING:
    from personal_kb.ingest.dedup_agent import DedupAgent

logger = logging.getLogger(__name__)

# Optional async callback for progress events during ingestion
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]

# Extensions we can meaningfully ingest as text
_ALLOWED_EXTENSIONS: set[str] = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".org",
    ".adoc",
    ".tex",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".rb",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".swift",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".json",
    ".xml",
    ".html",
    ".css",
    ".scss",
    ".sql",
    ".r",
    ".R",
    ".jl",
    ".lua",
    ".vim",
    ".el",
    ".clj",
    ".ex",
    ".exs",
    ".erl",
    ".hs",
    ".ml",
    ".nix",
    ".tf",
    ".Dockerfile",
    ".Makefile",
    ".pdf",
}

# Also allow files with no extension that have known names
_ALLOWED_NAMES: set[str] = {
    "Dockerfile",
    "Makefile",
    "Rakefile",
    "Gemfile",
    "Procfile",
    "README",
    "CHANGELOG",
    "LICENSE",
    "NOTES",
}


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    import pymupdf

    doc = pymupdf.open(str(path))  # type: ignore[no-untyped-call]
    try:
        pages = []
        for page in doc:  # type: ignore[attr-defined]
            text = page.get_text()
            if text.strip():
                pages.append(text)
        return "\n\n".join(pages)
    finally:
        doc.close()  # type: ignore[no-untyped-call]


@dataclass
class FileResult:
    """Result of ingesting a single file."""

    path: str
    action: str  # "ingested", "skipped", "flagged", "error", "unchanged", "dry_run"
    reason: str | None = None
    entry_count: int = 0
    entry_ids: list[str] = field(default_factory=list)
    summary: str | None = None
    chunks_processed: int = 0
    chunks_skipped: int = 0
    chunks_flagged: int = 0


@dataclass
class IngestResult:
    """Result of ingesting a directory."""

    total_files: int = 0
    ingested: int = 0
    skipped: int = 0
    flagged: int = 0
    errors: int = 0
    unchanged: int = 0
    entries_created: int = 0
    file_results: list[FileResult] = field(default_factory=list)


class FileIngester:
    """Orchestrates file ingestion: safety checks, LLM extraction, and storage."""

    def __init__(
        self,
        db: Database,
        store: KnowledgeStore,
        embedder: EmbeddingClient,
        graph_builder: GraphBuilder,
        graph_enricher: GraphEnricher | None,
        llm: LLMProvider,
        dedup_agent: "DedupAgent | None" = None,
        contributor: str | None = None,
        team: str | None = None,
    ) -> None:
        """Initialize with all required dependencies."""
        self._db = db
        self._store = store
        self._embedder = embedder
        self._graph_builder = graph_builder
        self._graph_enricher = graph_enricher
        self._llm = llm
        self._dedup_agent = dedup_agent
        self._contributor = contributor
        self._team = team

    async def _emit(
        self,
        cb: ProgressCallback | None,
        event_type: str,
        source: str,
        **extra: object,
    ) -> None:
        """Emit a progress event if callback is set."""
        if cb:
            await cb({"type": event_type, "source": source, **extra})

    async def ingest_file(
        self,
        path: Path,
        *,
        project_ref: str | None = None,
        base_dir: Path | None = None,
        display_name: str | None = None,
        dry_run: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> FileResult:
        """Ingest a single file through the full pipeline.

        Handles filesystem-specific checks (deny-list, extension, symlink,
        size), then delegates to _run_pipeline() for extraction and storage.

        If display_name is set, it overrides the filename used for note nodes
        and ingested_files records — useful when the file on disk is a temp
        file (e.g. browser uploads) but the original name should be preserved.
        """
        rel_path = display_name or (str(path.relative_to(base_dir)) if base_dir else path.name)

        # 1. Deny-list check (security boundary — before extension check)
        from personal_kb.ingest.safety import check_deny_list

        denied = check_deny_list(path)
        if denied:
            return FileResult(
                path=rel_path,
                action="skipped",
                reason=f"Matches deny-list pattern: {denied}",
            )

        # 2. Check extension
        if not _is_allowed_file(path):
            return FileResult(
                path=rel_path,
                action="skipped",
                reason=f"Unsupported file type: {path.suffix or path.name}",
            )

        # 2b. Reject symlinks (could escape deny-list via indirection)
        if path.is_symlink():
            return FileResult(
                path=rel_path,
                action="skipped",
                reason="Symlinks not allowed",
            )

        # 3. Check file size
        try:
            file_size = path.stat().st_size
        except OSError as e:
            return FileResult(path=rel_path, action="error", reason=str(e))

        max_size = get_ingest_max_file_size()
        if file_size > max_size:
            return FileResult(
                path=rel_path,
                action="skipped",
                reason=f"File too large: {file_size:,} bytes (max {max_size:,})",
            )

        # 4. Read content
        try:
            if path.suffix.lower() == ".pdf":
                content = _read_pdf(path)
            else:
                content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return FileResult(path=rel_path, action="error", reason=str(e))
        except ImportError:
            return FileResult(
                path=rel_path,
                action="error",
                reason="pymupdf not installed — run: uv pip install pymupdf",
            )

        # 5. Compute hash, skip if unchanged
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        existing = await self._get_ingested_file(rel_path)
        if existing and existing["content_hash"] == content_hash and existing["is_active"]:
            return FileResult(path=rel_path, action="unchanged")

        # 6. PII redaction (file-level — transforms content, non-blocking)
        from personal_kb.ingest.safety import redact_pii

        content, pii_redactions = redact_pii(content)

        return await self._run_pipeline(
            content=content,
            source=rel_path,
            content_hash=content_hash,
            pii_redactions=pii_redactions,
            existing=existing,
            file_size=file_size,
            file_extension=path.suffix,
            note_node_id=f"note:{rel_path}",
            note_props={"path": rel_path},
            project_ref=project_ref,
            dry_run=dry_run,
            progress_callback=progress_callback,
        )

    async def ingest_url(
        self,
        url: str,
        *,
        project_ref: str | None = None,
        dry_run: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> FileResult:
        """Fetch a URL, extract clean content, and ingest it.

        Steps:
        1. Fetch HTML via httpx
        2. Extract article content via trafilatura
        3. Safety checks (PII, secrets)
        4. LLM extraction pipeline (same as file ingestion)
        """
        import httpx

        from personal_kb.ingest.html_extract import extract_content

        # 1. Fetch URL
        try:
            _headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:137.0) "
                    "Gecko/20100101 Firefox/137.0"
                ),
            }
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=30.0, headers=_headers
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                raw_html = resp.text
        except httpx.HTTPError as e:
            reason = f"Failed to fetch: {e}"
            await self._emit(
                progress_callback,
                "ingest_error",
                url,
                error=reason,
            )
            return FileResult(
                path=url,
                action="error",
                reason=reason,
            )

        # 2. Extract clean content from HTML
        content = extract_content(raw_html, url=url)
        if not content:
            reason = (
                "Could not extract content (trafilatura returned"
                " empty). The page may require JavaScript or"
                " have no article content."
            )
            await self._emit(
                progress_callback,
                "ingest_error",
                url,
                error=reason,
            )
            return FileResult(
                path=url,
                action="error",
                reason=reason,
            )

        return await self._ingest_content(
            content,
            url,
            project_ref=project_ref,
            dry_run=dry_run,
            progress_callback=progress_callback,
        )

    async def ingest_text(
        self,
        content: str,
        source_name: str,
        *,
        project_ref: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> FileResult:
        """Ingest raw text content (e.g. from a file upload in the explorer).

        Validates extension from source_name against the allowlist and checks
        content size, then delegates to _ingest_content().
        """
        ext = Path(source_name).suffix.lower()
        name = Path(source_name).name
        if ext not in _ALLOWED_EXTENSIONS and name not in _ALLOWED_NAMES:
            reason = f"Unsupported file type: {ext or name}"
            await self._emit(
                progress_callback,
                "ingest_error",
                source_name,
                error=reason,
            )
            return FileResult(
                path=source_name,
                action="skipped",
                reason=reason,
            )

        content_bytes = len(content.encode("utf-8"))
        max_size = get_ingest_max_file_size()
        if content_bytes > max_size:
            reason = f"File too large: {content_bytes:,} bytes (max {max_size:,})"
            await self._emit(
                progress_callback,
                "ingest_error",
                source_name,
                error=reason,
            )
            return FileResult(
                path=source_name,
                action="skipped",
                reason=reason,
            )

        return await self._ingest_content(
            content,
            source_name,
            project_ref=project_ref,
            progress_callback=progress_callback,
        )

    async def _ingest_content(
        self,
        content: str,
        source_url: str,
        *,
        project_ref: str | None = None,
        dry_run: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> FileResult:
        """Ingest clean text content with URL attribution.

        Handles content-specific checks (hash, PII, size limit), then
        delegates to _run_pipeline() for extraction and storage.
        """
        # 1. Compute hash, skip if unchanged
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        existing = await self._get_ingested_file(source_url)
        if existing and existing["content_hash"] == content_hash and existing["is_active"]:
            return FileResult(path=source_url, action="unchanged")

        # 2. PII redaction (content-level — transforms content, non-blocking)
        from personal_kb.ingest.safety import redact_pii

        content, pii_redactions = redact_pii(content)

        # 2b. Content size limit (mirrors file size check in ingest_file)
        content_bytes = len(content.encode("utf-8"))
        max_size = get_ingest_max_file_size()
        if content_bytes > max_size:
            return FileResult(
                path=source_url,
                action="skipped",
                reason=f"Content too large: {content_bytes:,} bytes (max {max_size:,})",
            )

        # Derive extension from URL path, or empty string
        url_path = urlparse(source_url).path
        file_extension = Path(url_path).suffix if url_path else ""

        return await self._run_pipeline(
            content=content,
            source=source_url,
            content_hash=content_hash,
            pii_redactions=pii_redactions,
            existing=existing,
            file_size=content_bytes,
            file_extension=file_extension,
            note_node_id=f"note:{source_url}",
            note_props={"url": source_url},
            project_ref=project_ref,
            dry_run=dry_run,
            progress_callback=progress_callback,
        )

    async def _run_pipeline(
        self,
        *,
        content: str,
        source: str,
        content_hash: str,
        pii_redactions: list[str],
        existing: dict[str, object] | None,
        file_size: int,
        file_extension: str,
        note_node_id: str,
        note_props: dict[str, str],
        project_ref: str | None = None,
        dry_run: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> FileResult:
        """Shared extraction/storage pipeline for file and content ingestion.

        Both ingest_file() and _ingest_content() delegate here after their
        own preamble (filesystem checks vs content checks).
        """
        if dry_run:
            summary = await summarize_file(self._llm, source, content)
            chunks = chunk_content(content)
            dry_entries: list[ExtractedEntry] = []
            dry_prev: list[str] = []
            dry_flagged = 0
            for chunk in chunks:
                secrets = detect_secrets_in_content(chunk.text)
                if secrets:
                    dry_flagged += 1
                    logger.info("Chunk flagged for secrets in %s: %s", source, secrets)
                    continue
                entries = await extract_entries(
                    self._llm,
                    source,
                    chunk.text,
                    previously_extracted=dry_prev or None,
                    chunk_heading=chunk.heading,
                )
                dry_entries.extend(entries)
                dry_prev.extend(e.short_title for e in entries)
            return FileResult(
                path=source,
                action="dry_run",
                entry_count=len(dry_entries),
                summary=summary,
                chunks_processed=len(chunks) - dry_flagged,
                chunks_flagged=dry_flagged,
            )

        # 1. Summarize
        pc = progress_callback
        await self._emit(pc, "ingest_summarizing", source)
        summary = await summarize_file(self._llm, source, content)
        if summary is None:
            await self._emit(
                pc,
                "ingest_error",
                source,
                error="LLM unavailable for summarization",
            )
            return FileResult(
                path=source,
                action="error",
                reason="LLM unavailable for summarization",
            )

        # 2. Chunk content
        chunks = chunk_content(content)
        await self._emit(
            pc,
            "ingest_start",
            source,
            total_chunks=len(chunks),
        )

        # 3. Extract entries per chunk with running context
        all_extracted: list[ExtractedEntry] = []
        previously_extracted: list[str] = []
        chunks_skipped = 0
        chunks_flagged = 0
        for ci, chunk in enumerate(chunks):
            # Secret scan per-chunk — skip chunks with secrets
            secrets = detect_secrets_in_content(chunk.text)
            if secrets:
                chunks_flagged += 1
                logger.info(
                    "Chunk flagged for secrets in %s: %s",
                    source,
                    secrets,
                )
                continue

            # Dedup check (if agent provided)
            if self._dedup_agent:
                dedup_result = await self._dedup_agent.check_chunk(
                    chunk,
                    previously_extracted,
                )
                if dedup_result.action == "skip":
                    chunks_skipped += 1
                    continue
                if dedup_result.action == "partial":
                    combined_context = previously_extracted + dedup_result.existing_titles
                else:
                    combined_context = previously_extracted
            else:
                combined_context = previously_extracted

            await self._emit(
                pc,
                "ingest_chunk_start",
                source,
                chunk_index=ci,
                total_chunks=len(chunks),
            )

            extracted = await extract_entries(
                self._llm,
                source,
                chunk.text,
                previously_extracted=combined_context or None,
                chunk_heading=chunk.heading,
            )
            all_extracted.extend(extracted)
            previously_extracted.extend(e.short_title for e in extracted)

            await self._emit(
                pc,
                "ingest_chunk_done",
                source,
                chunk_index=ci,
                total_chunks=len(chunks),
                entries_extracted=len(extracted),
            )

        # 4. Store entries, batch embed, and batch enrich
        entry_ids: list[str] = []
        stored_entries: list[KnowledgeEntry] = []
        async with self._db.transaction():
            for ext_entry in all_extracted:
                entry = await self._store_extracted_entry(ext_entry, project_ref, source)
                if entry:
                    entry_ids.append(entry.id)
                    stored_entries.append(entry)

        # Batch embedding — single Ollama call for all entries
        if stored_entries:
            try:
                texts = [e.embedding_text for e in stored_entries]
                embeddings = await self._embedder.embed_batch(texts)
                if embeddings is not None:
                    pairs = list(zip([e.id for e in stored_entries], embeddings, strict=True))
                    await self._embedder.store_embeddings(pairs)
                    for entry in stored_entries:
                        await self._store.mark_embedding(entry.id, True)
            except Exception:
                logger.warning("Batch embedding failed for %s", source, exc_info=True)

        # Batch enrichment — single LLM call for all entries
        if self._graph_enricher and stored_entries:
            try:
                await self._graph_enricher.enrich_batch(stored_entries)
            except Exception:
                logger.warning("Batch enrichment failed for %s", source, exc_info=True)
            finally:
                self._graph_enricher.clear_vocab_cache()

        # Deactivate old entries only after new ones are stored successfully
        if existing:
            await self._deactivate_old_entries(existing)

        # 5. Create note node and edges
        full_props = {**note_props, "summary": summary}
        await self._upsert_note_node(note_node_id, full_props)
        for eid in entry_ids:
            await self._add_extracted_from_edge(eid, note_node_id)
        await self._db.commit()

        # 6. Record in ingested_files
        now = datetime.now(UTC).isoformat()
        if existing:
            await self._db.execute(
                "UPDATE ingested_files SET content_hash = ?, note_node_id = ?, "
                "entry_ids = ?, summary = ?, file_size = ?, file_extension = ?, "
                "project_ref = ?, redactions = ?, contributor = ?, "
                "updated_at = ?, is_active = 1 "
                "WHERE relative_path = ?",
                (
                    content_hash,
                    note_node_id,
                    json.dumps(entry_ids),
                    summary,
                    file_size,
                    file_extension,
                    project_ref,
                    json.dumps(pii_redactions),
                    self._contributor,
                    now,
                    source,
                ),
            )
        else:
            await self._db.execute(
                "INSERT INTO ingested_files "
                "(relative_path, content_hash, note_node_id, entry_ids, summary, "
                "file_size, file_extension, project_ref, redactions, contributor, "
                "ingested_at, updated_at, is_active) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (
                    source,
                    content_hash,
                    note_node_id,
                    json.dumps(entry_ids),
                    summary,
                    file_size,
                    file_extension,
                    project_ref,
                    json.dumps(pii_redactions),
                    self._contributor,
                    now,
                    now,
                ),
            )
        await self._db.commit()

        result = FileResult(
            path=source,
            action="ingested",
            entry_count=len(entry_ids),
            entry_ids=entry_ids,
            summary=summary,
            chunks_processed=len(chunks) - chunks_skipped - chunks_flagged,
            chunks_skipped=chunks_skipped,
            chunks_flagged=chunks_flagged,
        )
        await self._emit(
            pc,
            "ingest_done",
            source,
            action="ingested",
            entry_count=len(entry_ids),
            entry_ids=entry_ids,
        )
        return result

    async def ingest_directory(
        self,
        dir_path: Path,
        *,
        project_ref: str | None = None,
        recursive: bool = True,
        dry_run: bool = False,
    ) -> IngestResult:
        """Ingest all eligible files from a directory."""
        result = IngestResult()

        if not dir_path.is_dir():
            result.errors = 1
            result.file_results.append(
                FileResult(
                    path=str(dir_path),
                    action="error",
                    reason="Not a directory",
                )
            )
            return result

        # Collect files
        pattern = "**/*" if recursive else "*"
        files = sorted(f for f in dir_path.glob(pattern) if f.is_file() and not f.is_symlink())

        for file_path in files:
            result.total_files += 1
            file_result = await self.ingest_file(
                file_path,
                project_ref=project_ref,
                base_dir=dir_path,
                dry_run=dry_run,
            )
            result.file_results.append(file_result)

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

        return result

    async def _get_ingested_file(self, rel_path: str) -> dict[str, object] | None:
        """Look up a previously ingested file by relative path."""
        cursor = await self._db.execute(
            "SELECT * FROM ingested_files WHERE relative_path = ?",
            (rel_path,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def _deactivate_old_entries(self, record: dict[str, object]) -> None:
        """Deactivate entries from a previous ingestion of this file."""
        raw_ids = record.get("entry_ids", "[]")
        try:
            old_ids = json.loads(str(raw_ids))
        except (json.JSONDecodeError, TypeError):
            old_ids = []

        for eid in old_ids:
            if not isinstance(eid, str):
                continue
            try:
                await self._store.deactivate_entry(eid)
                # Remove graph edges
                await self._db.execute("DELETE FROM graph_edges WHERE source = ?", (eid,))
            except ValueError:
                logger.warning("Could not deactivate old entry %s", eid)

        # Remove old note node edges
        old_node_id = record.get("note_node_id")
        if isinstance(old_node_id, str):
            await self._db.execute(
                "DELETE FROM graph_edges WHERE source = ? OR target = ?",
                (old_node_id, old_node_id),
            )

        await self._db.commit()

    async def _store_extracted_entry(
        self,
        ext: ExtractedEntry,
        project_ref: str | None,
        source_path: str,
    ) -> KnowledgeEntry | None:
        """Store a single extracted entry through the full kb_store pipeline."""
        try:
            entry_type = EntryType(ext.entry_type)
        except ValueError:
            entry_type = EntryType.FACTUAL_REFERENCE

        try:
            entry = await self._store.create_entry(
                short_title=ext.short_title,
                long_title=ext.long_title,
                knowledge_details=ext.knowledge_details,
                entry_type=entry_type,
                project_ref=project_ref,
                source_context=f"Ingested from {source_path}",
                tags=ext.tags,
                contributor=self._contributor,
                team=self._team,
            )
        except Exception:
            logger.warning("Failed to create entry from %s", source_path, exc_info=True)
            return None

        # Build deterministic graph edges (tags, projects, supersedes, text refs)
        try:
            await self._graph_builder.build_for_entry(entry)
        except Exception:
            logger.warning("Failed to build graph for %s", entry.id, exc_info=True)

        # Embedding and LLM enrichment are handled in batch after all entries
        return entry

    async def _upsert_note_node(self, node_id: str, props: dict[str, str]) -> None:
        """Create or update a note node in the graph."""
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            """INSERT INTO graph_nodes (node_id, node_type, properties, created_at)
               VALUES (?, 'note', ?, ?)
               ON CONFLICT(node_id) DO UPDATE SET
                   properties = excluded.properties""",
            (node_id, json.dumps(props), now),
        )

    async def _add_extracted_from_edge(self, entry_id: str, note_node_id: str) -> None:
        """Add an extracted_from edge from an entry to its source note node."""
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            """INSERT INTO graph_edges
               (source, target, edge_type, properties, created_at)
               VALUES (?, ?, 'extracted_from', '{}', ?)
               ON CONFLICT (source, target, edge_type) DO NOTHING""",
            (entry_id, note_node_id, now),
        )


def _is_allowed_file(path: Path) -> bool:
    """Check if a file is in the allowlist for ingestion."""
    if path.name in _ALLOWED_NAMES:
        return True
    return path.suffix.lower() in _ALLOWED_EXTENSIONS
