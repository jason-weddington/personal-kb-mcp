"""Tests for the file ingestion orchestrator."""

import json
from pathlib import Path

import pytest
import pytest_asyncio

from personal_kb.db.connection import create_connection
from personal_kb.graph.builder import GraphBuilder
from personal_kb.graph.enricher import GraphEnricher
from personal_kb.ingest.ingester import FileIngester, _is_allowed_file
from personal_kb.store.knowledge_store import KnowledgeStore
from tests.conftest import FakeEmbedder, FakeLLM


@pytest_asyncio.fixture
async def ingester_deps():
    """Create all dependencies for FileIngester with in-memory DB."""
    db = await create_connection(":memory:")
    store = KnowledgeStore(db)
    embedder = FakeEmbedder(db)
    graph_builder = GraphBuilder(db)
    llm = FakeLLM()
    enricher = GraphEnricher(db, llm)
    yield {
        "db": db,
        "store": store,
        "embedder": embedder,
        "graph_builder": graph_builder,
        "enricher": enricher,
        "llm": llm,
    }
    await db.close()


def _make_llm_with_responses(summary: str, entries: list[dict]) -> FakeLLM:
    """Create a FakeLLM that returns summary first, then entries JSON."""

    class SequenceLLM(FakeLLM):
        def __init__(self, responses):
            super().__init__()
            self._responses = list(responses)
            self._call_index = 0

        async def generate(self, prompt, *, system=None):
            self.last_prompt = prompt
            self.last_system = system
            self.generate_count += 1
            if self._call_index < len(self._responses):
                resp = self._responses[self._call_index]
                self._call_index += 1
                return resp
            return None

    return SequenceLLM([summary, json.dumps(entries)])


class TestIsAllowedFile:
    def test_allows_markdown(self):
        assert _is_allowed_file(Path("notes.md")) is True

    def test_allows_python(self):
        assert _is_allowed_file(Path("script.py")) is True

    def test_allows_txt(self):
        assert _is_allowed_file(Path("notes.txt")) is True

    def test_allows_dockerfile(self):
        assert _is_allowed_file(Path("Dockerfile")) is True

    def test_allows_makefile(self):
        assert _is_allowed_file(Path("Makefile")) is True

    def test_rejects_unknown(self):
        assert _is_allowed_file(Path("file.xyz")) is False

    def test_allows_yaml(self):
        assert _is_allowed_file(Path("config.yaml")) is True

    def test_allows_json(self):
        assert _is_allowed_file(Path("data.json")) is True

    def test_case_insensitive_extension(self):
        assert _is_allowed_file(Path("README.MD")) is True


class TestIngestFile:
    async def test_skips_unsupported_extension(self, ingester_deps, tmp_path):
        f = tmp_path / "image.xyz"
        f.write_text("data")
        deps = ingester_deps
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            deps["enricher"],
            deps["llm"],
        )
        result = await ingester.ingest_file(f, base_dir=tmp_path)
        assert result.action == "skipped"
        assert "Unsupported" in result.reason

    async def test_skips_large_file(self, ingester_deps, tmp_path, monkeypatch):
        f = tmp_path / "big.md"
        f.write_text("x" * 100)
        monkeypatch.setenv("KB_INGEST_MAX_FILE_SIZE", "50")
        deps = ingester_deps
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            deps["enricher"],
            deps["llm"],
        )
        result = await ingester.ingest_file(f, base_dir=tmp_path)
        assert result.action == "skipped"
        assert "too large" in result.reason

    async def test_skips_deny_listed_file(self, ingester_deps, tmp_path):
        f = tmp_path / "secret.pem"
        f.write_text("cert data")
        deps = ingester_deps
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            deps["enricher"],
            deps["llm"],
        )
        result = await ingester.ingest_file(f, base_dir=tmp_path)
        assert result.action == "skipped"
        assert "deny-list" in result.reason

    async def test_ingests_markdown_file(self, ingester_deps, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Python Async\n\nUseful patterns for async programming.")

        entries_json = [
            {
                "short_title": "async patterns",
                "long_title": "Python async programming patterns",
                "knowledge_details": "Useful patterns for async.",
                "entry_type": "pattern_convention",
                "tags": ["python", "async"],
            }
        ]
        deps = ingester_deps
        llm = _make_llm_with_responses("Notes about Python async patterns.", entries_json)
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        result = await ingester.ingest_file(f, base_dir=tmp_path, project_ref="test")
        assert result.action == "ingested"
        assert result.entry_count == 1
        assert len(result.entry_ids) == 1
        assert result.summary == "Notes about Python async patterns."

        # Verify note node was created
        cursor = await deps["db"].execute(
            "SELECT * FROM graph_nodes WHERE node_id = ?",
            ("note:notes.md",),
        )
        node = await cursor.fetchone()
        assert node is not None
        assert node["node_type"] == "note"

        # Verify extracted_from edge
        cursor = await deps["db"].execute(
            "SELECT * FROM graph_edges WHERE edge_type = 'extracted_from'"
        )
        edge = await cursor.fetchone()
        assert edge is not None
        assert edge["target"] == "note:notes.md"

        # Verify ingested_files record
        cursor = await deps["db"].execute(
            "SELECT * FROM ingested_files WHERE relative_path = 'notes.md'"
        )
        record = await cursor.fetchone()
        assert record is not None
        assert record["is_active"] == 1
        assert record["project_ref"] == "test"

    async def test_skips_unchanged_file(self, ingester_deps, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Static content")

        entries_json = [
            {
                "short_title": "test",
                "long_title": "Test entry",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        deps = ingester_deps
        llm = _make_llm_with_responses("Summary.", entries_json)
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        # First ingest
        result1 = await ingester.ingest_file(f, base_dir=tmp_path)
        assert result1.action == "ingested"

        # Second ingest — same content
        llm2 = _make_llm_with_responses("Summary.", entries_json)
        ingester2 = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm2,
        )
        result2 = await ingester2.ingest_file(f, base_dir=tmp_path)
        assert result2.action == "unchanged"

    async def test_reingests_changed_file(self, ingester_deps, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Version 1")

        entries_v1 = [
            {
                "short_title": "v1 entry",
                "long_title": "Version 1 entry",
                "knowledge_details": "V1 details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        deps = ingester_deps
        llm1 = _make_llm_with_responses("V1 summary.", entries_v1)
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm1,
        )
        result1 = await ingester.ingest_file(f, base_dir=tmp_path)
        assert result1.action == "ingested"
        old_entry_id = result1.entry_ids[0]

        # Modify file
        f.write_text("# Version 2 — updated content")

        entries_v2 = [
            {
                "short_title": "v2 entry",
                "long_title": "Version 2 entry",
                "knowledge_details": "V2 details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        llm2 = _make_llm_with_responses("V2 summary.", entries_v2)
        ingester2 = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm2,
        )
        result2 = await ingester2.ingest_file(f, base_dir=tmp_path)
        assert result2.action == "ingested"
        assert result2.entry_ids[0] != old_entry_id

        # Old entry should be deactivated
        old_entry = await deps["store"].get_entry(old_entry_id)
        assert old_entry is not None
        assert old_entry.is_active is False

    async def test_dry_run_no_storage(self, ingester_deps, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Dry run test")

        entries_json = [
            {
                "short_title": "test",
                "long_title": "Test",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        deps = ingester_deps
        llm = _make_llm_with_responses("Summary.", entries_json)
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        result = await ingester.ingest_file(f, base_dir=tmp_path, dry_run=True)
        assert result.action == "dry_run"
        assert result.entry_count == 1
        assert result.summary == "Summary."

        # Nothing should be stored
        cursor = await deps["db"].execute("SELECT COUNT(*) FROM knowledge_entries")
        row = await cursor.fetchone()
        assert row[0] == 0

        cursor = await deps["db"].execute("SELECT COUNT(*) FROM ingested_files")
        row = await cursor.fetchone()
        assert row[0] == 0

    async def test_error_on_llm_unavailable(self, ingester_deps, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Content")

        deps = ingester_deps
        llm = FakeLLM(response=None)
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            deps["enricher"],
            llm,
        )

        result = await ingester.ingest_file(f, base_dir=tmp_path)
        assert result.action == "error"
        assert "LLM" in result.reason

    async def test_secrets_skip_chunk_not_file(self, ingester_deps, tmp_path, monkeypatch):
        """File with secrets in one chunk still extracts from clean chunks."""
        pytest.importorskip("detect_secrets")
        # Force small chunk size so headings split into separate chunks
        monkeypatch.setenv("KB_INGEST_CHUNK_SIZE", "60")
        monkeypatch.setenv("KB_INGEST_CHUNK_OVERLAP", "0")
        deps = ingester_deps
        entries_json = [
            {
                "short_title": "arch-decision",
                "long_title": "Architecture decision",
                "knowledge_details": "We chose X because Y",
                "entry_type": "decision",
                "tags": "architecture",
            }
        ]
        llm = _make_llm_with_responses("A mixed doc", entries_json)
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], llm),
            llm,
        )

        f = tmp_path / "mixed.md"
        f.write_text(
            '# Architecture\n\nWe chose X because Y.\n\n# Credentials\n\npassword = "hunter2"\n'
        )

        result = await ingester.ingest_file(f, base_dir=tmp_path)
        assert result.action == "ingested"
        assert result.entry_count == 1
        assert result.chunks_flagged == 1
        assert result.chunks_processed == 1


class TestIngestDirectory:
    async def test_ingests_multiple_files(self, ingester_deps, tmp_path):
        (tmp_path / "a.md").write_text("# File A")
        (tmp_path / "b.txt").write_text("File B content")
        (tmp_path / "c.png").write_bytes(b"fake image")

        entries_json = [
            {
                "short_title": "entry",
                "long_title": "An entry",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        deps = ingester_deps
        llm = _make_llm_with_responses("Summary.", entries_json)

        # Use a multi-call LLM that can handle multiple files
        class MultiLLM(FakeLLM):
            def __init__(self):
                super().__init__()
                self._call_index = 0
                self._responses = [
                    "Summary A.",
                    json.dumps(entries_json),
                    "Summary B.",
                    json.dumps(entries_json),
                ]

            async def generate(self, prompt, *, system=None):
                self.generate_count += 1
                if self._call_index < len(self._responses):
                    resp = self._responses[self._call_index]
                    self._call_index += 1
                    return resp
                return None

        llm = MultiLLM()
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        result = await ingester.ingest_directory(tmp_path, recursive=False)
        assert result.total_files == 3
        assert result.ingested == 2  # a.md and b.txt
        assert result.skipped == 1  # c.png

    async def test_recursive_ingestion(self, ingester_deps, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "root.md").write_text("# Root")
        (subdir / "nested.md").write_text("# Nested")

        entries_json = [
            {
                "short_title": "entry",
                "long_title": "An entry",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]

        class MultiLLM(FakeLLM):
            def __init__(self):
                super().__init__()
                self._call_index = 0
                self._responses = [
                    "Summary 1.",
                    json.dumps(entries_json),
                    "Summary 2.",
                    json.dumps(entries_json),
                ]

            async def generate(self, prompt, *, system=None):
                self.generate_count += 1
                if self._call_index < len(self._responses):
                    resp = self._responses[self._call_index]
                    self._call_index += 1
                    return resp
                return None

        deps = ingester_deps
        llm = MultiLLM()
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        result = await ingester.ingest_directory(tmp_path, recursive=True)
        assert result.ingested == 2

    async def test_non_recursive_skips_subdirs(self, ingester_deps, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "root.md").write_text("# Root")
        (subdir / "nested.md").write_text("# Nested")

        entries_json = [
            {
                "short_title": "entry",
                "long_title": "An entry",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        deps = ingester_deps
        llm = _make_llm_with_responses("Summary.", entries_json)
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        result = await ingester.ingest_directory(tmp_path, recursive=False)
        assert result.ingested == 1  # Only root.md

    async def test_error_on_nonexistent_dir(self, ingester_deps):
        deps = ingester_deps
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            deps["enricher"],
            deps["llm"],
        )
        result = await ingester.ingest_directory(Path("/nonexistent"))
        assert result.errors == 1


class TestSymlinkProtection:
    async def test_ingest_file_skips_symlink(self, ingester_deps, tmp_path):
        real = tmp_path / "real.md"
        real.write_text("# Real file")
        link = tmp_path / "link.md"
        link.symlink_to(real)

        deps = ingester_deps
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            deps["enricher"],
            deps["llm"],
        )
        result = await ingester.ingest_file(link, base_dir=tmp_path)
        assert result.action == "skipped"
        assert "Symlinks" in result.reason

    async def test_ingest_directory_skips_symlinks(self, ingester_deps, tmp_path):
        real = tmp_path / "real.md"
        real.write_text("# Real file")
        link = tmp_path / "link.md"
        link.symlink_to(real)

        entries_json = [
            {
                "short_title": "entry",
                "long_title": "An entry",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        deps = ingester_deps
        llm = _make_llm_with_responses("Summary.", entries_json)
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        result = await ingester.ingest_directory(tmp_path, recursive=False)
        # Only real.md should be processed (symlink filtered in glob)
        assert result.total_files == 1
        assert result.ingested == 1


class TestIngestContent:
    async def test_ingests_url_content(self, ingester_deps):
        entries_json = [
            {
                "short_title": "wiki patterns",
                "long_title": "Wiki patterns for team docs",
                "knowledge_details": "Patterns from the wiki.",
                "entry_type": "pattern_convention",
                "tags": ["wiki"],
            }
        ]
        deps = ingester_deps
        llm = _make_llm_with_responses("Summary of wiki page.", entries_json)
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        url = "https://wiki.example.com/team/patterns"
        result = await ingester._ingest_content(
            "# Team Patterns\n\nUseful patterns from the wiki.",
            url,
            project_ref="test",
        )
        assert result.action == "ingested"
        assert result.entry_count == 1
        assert len(result.entry_ids) == 1
        assert result.path == url

        # Verify source_context includes URL
        entry = await deps["store"].get_entry(result.entry_ids[0])
        assert entry is not None
        assert url in entry.source_context

    async def test_url_dedup_unchanged(self, ingester_deps):
        entries_json = [
            {
                "short_title": "test",
                "long_title": "Test entry",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        deps = ingester_deps
        url = "https://wiki.example.com/page"
        content = "# Static content from URL"

        llm1 = _make_llm_with_responses("Summary.", entries_json)
        ingester1 = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm1,
        )
        result1 = await ingester1._ingest_content(content, url)
        assert result1.action == "ingested"

        # Second ingest — same content → unchanged
        llm2 = _make_llm_with_responses("Summary.", entries_json)
        ingester2 = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm2,
        )
        result2 = await ingester2._ingest_content(content, url)
        assert result2.action == "unchanged"

    async def test_url_reingestion(self, ingester_deps):
        entries_v1 = [
            {
                "short_title": "v1",
                "long_title": "V1 entry",
                "knowledge_details": "V1 details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        deps = ingester_deps
        url = "https://wiki.example.com/evolving"

        llm1 = _make_llm_with_responses("V1 summary.", entries_v1)
        ingester1 = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm1,
        )
        result1 = await ingester1._ingest_content("# V1 content", url)
        assert result1.action == "ingested"
        old_entry_id = result1.entry_ids[0]

        # Re-ingest with different content
        entries_v2 = [
            {
                "short_title": "v2",
                "long_title": "V2 entry",
                "knowledge_details": "V2 details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        llm2 = _make_llm_with_responses("V2 summary.", entries_v2)
        ingester2 = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm2,
        )
        result2 = await ingester2._ingest_content("# V2 updated content", url)
        assert result2.action == "ingested"
        assert result2.entry_ids[0] != old_entry_id

        # Old entry should be deactivated
        old_entry = await deps["store"].get_entry(old_entry_id)
        assert old_entry is not None
        assert old_entry.is_active is False

    async def test_url_dry_run(self, ingester_deps):
        entries_json = [
            {
                "short_title": "test",
                "long_title": "Test",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        deps = ingester_deps
        llm = _make_llm_with_responses("Summary.", entries_json)
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        result = await ingester._ingest_content(
            "# Dry run content",
            "https://wiki.example.com/dry",
            dry_run=True,
        )
        assert result.action == "dry_run"
        assert result.entry_count == 1
        assert result.summary == "Summary."

        # Nothing stored
        cursor = await deps["db"].execute("SELECT COUNT(*) FROM knowledge_entries")
        row = await cursor.fetchone()
        assert row[0] == 0

        cursor = await deps["db"].execute("SELECT COUNT(*) FROM ingested_files")
        row = await cursor.fetchone()
        assert row[0] == 0

    async def test_url_secrets_chunk_skipped(self, ingester_deps):
        """Chunks with secrets are skipped, not the entire file."""
        pytest.importorskip("detect_secrets")
        deps = ingester_deps
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            deps["enricher"],
            deps["llm"],
        )

        result = await ingester._ingest_content(
            'password = "hunter2"',
            "https://wiki.example.com/secrets",
        )
        # Single-chunk content where that chunk has secrets → ingested with 0 entries
        assert result.action == "ingested"
        assert result.entry_count == 0
        assert result.chunks_flagged == 1

    async def test_url_secrets_partial_extraction(self, ingester_deps, monkeypatch):
        """Multi-chunk content: clean chunks extracted, secret chunks skipped."""
        pytest.importorskip("detect_secrets")
        monkeypatch.setenv("KB_INGEST_CHUNK_SIZE", "80")
        monkeypatch.setenv("KB_INGEST_CHUNK_OVERLAP", "0")
        deps = ingester_deps
        entries_json = [
            {
                "short_title": "clean-fact",
                "long_title": "A clean fact",
                "knowledge_details": "Safe content",
                "entry_type": "factual_reference",
                "tags": "test",
            }
        ]
        llm = _make_llm_with_responses("A doc with mixed content", entries_json)
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], llm),
            llm,
        )

        content = (
            "# Clean Section\n\n"
            "This is perfectly safe content about architecture decisions.\n\n"
            "# Secrets Section\n\n"
            'password = "hunter2"\n'
        )
        result = await ingester._ingest_content(
            content,
            "https://wiki.example.com/mixed",
        )
        assert result.action == "ingested"
        assert result.entry_count == 1
        assert result.chunks_flagged == 1
        assert result.chunks_processed == 1

    async def test_url_note_node(self, ingester_deps):
        entries_json = [
            {
                "short_title": "node test",
                "long_title": "Note node test",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        deps = ingester_deps
        url = "https://wiki.example.com/node-test"
        llm = _make_llm_with_responses("Node test summary.", entries_json)
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        await ingester._ingest_content("# Node test\n\nContent.", url)

        # Verify note node with URL in properties
        cursor = await deps["db"].execute(
            "SELECT * FROM graph_nodes WHERE node_id = ?",
            (f"note:{url}",),
        )
        node = await cursor.fetchone()
        assert node is not None
        assert node["node_type"] == "note"
        props = json.loads(node["properties"])
        assert props["url"] == url
        assert "summary" in props

    async def test_url_extracted_from_edge(self, ingester_deps):
        entries_json = [
            {
                "short_title": "edge test",
                "long_title": "Edge test entry",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        deps = ingester_deps
        url = "https://wiki.example.com/edge-test"
        llm = _make_llm_with_responses("Edge test summary.", entries_json)
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        result = await ingester._ingest_content("# Edge test\n\nContent.", url)

        # Verify extracted_from edge points to note:{url}
        cursor = await deps["db"].execute(
            "SELECT * FROM graph_edges WHERE edge_type = 'extracted_from' AND target = ?",
            (f"note:{url}",),
        )
        edge = await cursor.fetchone()
        assert edge is not None
        assert edge["source"] == result.entry_ids[0]

    async def test_url_content_too_large(self, ingester_deps, monkeypatch):
        monkeypatch.setenv("KB_INGEST_MAX_FILE_SIZE", "50")
        deps = ingester_deps
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            deps["enricher"],
            deps["llm"],
        )

        result = await ingester._ingest_content(
            "x" * 100,
            "https://wiki.example.com/large",
        )
        assert result.action == "skipped"
        assert "too large" in result.reason
