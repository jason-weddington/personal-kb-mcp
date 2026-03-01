"""File ingestion orchestrator — reads files, runs safety, extracts entries."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from personal_kb.config import get_ingest_max_file_size
from personal_kb.db.backend import Database
from personal_kb.graph.builder import GraphBuilder
from personal_kb.graph.enricher import GraphEnricher
from personal_kb.ingest.extractor import ExtractedEntry, extract_entries, summarize_file
from personal_kb.ingest.safety import SafetyResult, run_safety_pipeline
from personal_kb.llm.provider import LLMProvider
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.search.embeddings import EmbeddingClient
from personal_kb.store.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

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


@dataclass
class FileResult:
    """Result of ingesting a single file."""

    path: str
    action: str  # "ingested", "skipped", "flagged", "error", "unchanged", "dry_run"
    reason: str | None = None
    entry_count: int = 0
    entry_ids: list[str] = field(default_factory=list)
    summary: str | None = None


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
    ) -> None:
        """Initialize with all required dependencies."""
        self._db = db
        self._store = store
        self._embedder = embedder
        self._graph_builder = graph_builder
        self._graph_enricher = graph_enricher
        self._llm = llm

    async def ingest_file(
        self,
        path: Path,
        *,
        project_ref: str | None = None,
        base_dir: Path | None = None,
        dry_run: bool = False,
    ) -> FileResult:
        """Ingest a single file through the full pipeline.

        Steps:
        1. Check deny-list (security boundary)
        2. Check extension allowlist
        3. Check file size
        4. Read content
        5. Compute hash, skip if unchanged
        6. Run safety pipeline (secrets, PII)
        7. Summarize file (LLM call 1)
        8. Extract entries (LLM call 2)
        9. Store entries through kb_store pipeline
        10. Create note node and edges
        11. Record in ingested_files table
        """
        rel_path = str(path.relative_to(base_dir)) if base_dir else path.name

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
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return FileResult(path=rel_path, action="error", reason=str(e))

        # 5. Compute hash, skip if unchanged
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        existing = await self._get_ingested_file(rel_path)
        if existing and existing["content_hash"] == content_hash and existing["is_active"]:
            return FileResult(path=rel_path, action="unchanged")

        # 6. Safety pipeline (secrets + PII — deny-list already checked above)
        safety: SafetyResult = run_safety_pipeline(path, content)
        if safety.action == "skip":
            return FileResult(
                path=rel_path,
                action="skipped",
                reason=safety.reason,
            )
        if safety.action == "flag":
            return FileResult(
                path=rel_path,
                action="flagged",
                reason=safety.reason,
            )

        # Use safety-processed content (may have PII redacted)
        content = safety.content

        if dry_run:
            # Still run LLM calls for preview
            summary = await summarize_file(self._llm, rel_path, content)
            entries = await extract_entries(self._llm, rel_path, content)
            return FileResult(
                path=rel_path,
                action="dry_run",
                entry_count=len(entries),
                summary=summary,
            )

        # Handle re-ingestion: deactivate old entries
        if existing:
            await self._deactivate_old_entries(existing)

        # 6. Summarize
        summary = await summarize_file(self._llm, rel_path, content)
        if summary is None:
            return FileResult(
                path=rel_path,
                action="error",
                reason="LLM unavailable for summarization",
            )

        # 7. Extract entries
        extracted = await extract_entries(self._llm, rel_path, content)

        # 8. Store entries through the full pipeline
        entry_ids: list[str] = []
        for ext_entry in extracted:
            entry = await self._store_extracted_entry(ext_entry, project_ref, rel_path)
            if entry:
                entry_ids.append(entry.id)

        # 9. Create note node and edges
        note_node_id = f"note:{rel_path}"
        await self._create_note_node(note_node_id, rel_path, summary)
        for eid in entry_ids:
            await self._add_extracted_from_edge(eid, note_node_id)
        await self._db.commit()

        # 10. Record in ingested_files
        now = datetime.now(UTC).isoformat()
        if existing:
            await self._db.execute(
                "UPDATE ingested_files SET content_hash = ?, note_node_id = ?, "
                "entry_ids = ?, summary = ?, file_size = ?, file_extension = ?, "
                "project_ref = ?, redactions = ?, updated_at = ?, is_active = 1 "
                "WHERE relative_path = ?",
                (
                    content_hash,
                    note_node_id,
                    json.dumps(entry_ids),
                    summary,
                    file_size,
                    path.suffix,
                    project_ref,
                    json.dumps(safety.redactions),
                    now,
                    rel_path,
                ),
            )
        else:
            await self._db.execute(
                "INSERT INTO ingested_files "
                "(relative_path, content_hash, note_node_id, entry_ids, summary, "
                "file_size, file_extension, project_ref, redactions, ingested_at, "
                "updated_at, is_active) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (
                    rel_path,
                    content_hash,
                    note_node_id,
                    json.dumps(entry_ids),
                    summary,
                    file_size,
                    path.suffix,
                    project_ref,
                    json.dumps(safety.redactions),
                    now,
                    now,
                ),
            )
        await self._db.commit()

        return FileResult(
            path=rel_path,
            action="ingested",
            entry_count=len(entry_ids),
            entry_ids=entry_ids,
            summary=summary,
        )

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
        files = sorted(f for f in dir_path.glob(pattern) if f.is_file())

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
            )
        except Exception:
            logger.warning("Failed to create entry from %s", source_path, exc_info=True)
            return None

        # Embed
        try:
            embedding = await self._embedder.embed(entry.embedding_text)
            if embedding is not None:
                await self._embedder.store_embedding(entry.id, embedding)
                await self._store.mark_embedding(entry.id, True)
        except Exception:
            logger.warning("Failed to embed entry %s", entry.id, exc_info=True)

        # Build graph
        try:
            await self._graph_builder.build_for_entry(entry)
        except Exception:
            logger.warning("Failed to build graph for %s", entry.id, exc_info=True)

        # Enrich graph
        if self._graph_enricher:
            try:
                await self._graph_enricher.enrich_entry(entry)
            except Exception:
                logger.warning("Failed to enrich graph for %s", entry.id, exc_info=True)

        return entry

    async def _create_note_node(self, node_id: str, rel_path: str, summary: str) -> None:
        """Create or update a note node in the graph."""
        now = datetime.now(UTC).isoformat()
        props = json.dumps({"path": rel_path, "summary": summary})
        await self._db.execute(
            """INSERT INTO graph_nodes (node_id, node_type, properties, created_at)
               VALUES (?, 'note', ?, ?)
               ON CONFLICT(node_id) DO UPDATE SET
                   properties = excluded.properties""",
            (node_id, props, now),
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
