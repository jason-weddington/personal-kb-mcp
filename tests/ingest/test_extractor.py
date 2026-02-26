"""Tests for the LLM file extractor."""

import json

from personal_kb.ingest.extractor import (
    ExtractedEntry,
    _is_code_file,
    _is_prose_file,
    _parse_entries,
    extract_entries,
    summarize_file,
)
from tests.conftest import FakeLLM


class TestIsCodeFile:
    def test_python_is_code(self):
        assert _is_code_file("src/main.py") is True

    def test_javascript_is_code(self):
        assert _is_code_file("app.js") is True

    def test_rust_is_code(self):
        assert _is_code_file("lib.rs") is True

    def test_shell_is_code(self):
        assert _is_code_file("setup.sh") is True

    def test_markdown_is_not_code(self):
        assert _is_code_file("README.md") is False

    def test_yaml_is_not_code(self):
        assert _is_code_file("config.yaml") is False

    def test_txt_is_not_code(self):
        assert _is_code_file("notes.txt") is False

    def test_case_insensitive(self):
        assert _is_code_file("Main.PY") is True

    def test_nested_path(self):
        assert _is_code_file("src/personal_kb/tools/kb_store.py") is True


class TestIsProseFile:
    def test_markdown_is_prose(self):
        assert _is_prose_file("README.md") is True

    def test_txt_is_prose(self):
        assert _is_prose_file("notes.txt") is True

    def test_rst_is_prose(self):
        assert _is_prose_file("docs/guide.rst") is True

    def test_html_is_prose(self):
        assert _is_prose_file("page.html") is True

    def test_python_is_not_prose(self):
        assert _is_prose_file("main.py") is False

    def test_yaml_is_not_prose(self):
        assert _is_prose_file("config.yaml") is False

    def test_json_is_not_prose(self):
        assert _is_prose_file("data.json") is False

    def test_case_insensitive(self):
        assert _is_prose_file("NOTES.TXT") is True


class TestSummarizeFile:
    async def test_returns_summary(self):
        llm = FakeLLM(response="This file contains notes about Python async patterns.")
        result = await summarize_file(llm, "notes.md", "# Async Patterns\n...")
        assert result == "This file contains notes about Python async patterns."
        assert llm.generate_count == 1

    async def test_returns_none_when_unavailable(self):
        llm = FakeLLM(available=False)
        result = await summarize_file(llm, "notes.md", "content")
        assert result is None
        assert llm.generate_count == 0

    async def test_returns_none_on_llm_failure(self):
        llm = FakeLLM(response=None)
        result = await summarize_file(llm, "notes.md", "content")
        assert result is None

    async def test_passes_file_path_in_prompt(self):
        llm = FakeLLM(response="summary")
        await summarize_file(llm, "docs/architecture.md", "content")
        assert "docs/architecture.md" in llm.last_prompt

    async def test_truncates_long_content(self):
        llm = FakeLLM(response="summary")
        long_content = "x" * 200_000
        await summarize_file(llm, "big.md", long_content)
        # Prompt should be truncated, not the full 200K
        assert len(llm.last_prompt) < 150_000

    async def test_uses_code_supplement_for_code_files(self):
        llm = FakeLLM(response="summary")
        await summarize_file(llm, "main.py", "print('hello')")
        assert "SOURCE CODE file" in llm.last_system

    async def test_no_code_supplement_for_docs(self):
        llm = FakeLLM(response="summary")
        await summarize_file(llm, "notes.md", "# Notes")
        assert "SOURCE CODE file" not in llm.last_system

    async def test_uses_prose_supplement_for_markdown(self):
        llm = FakeLLM(response="summary")
        await summarize_file(llm, "notes.md", "# Notes")
        assert "NOTES or DOCUMENTATION" in llm.last_system

    async def test_uses_prose_supplement_for_txt(self):
        llm = FakeLLM(response="summary")
        await summarize_file(llm, "ideas.txt", "Some ideas")
        assert "NOTES or DOCUMENTATION" in llm.last_system

    async def test_no_prose_supplement_for_config(self):
        llm = FakeLLM(response="summary")
        await summarize_file(llm, "config.yaml", "key: value")
        assert "NOTES or DOCUMENTATION" not in llm.last_system
        assert "SOURCE CODE file" not in llm.last_system


class TestExtractEntries:
    async def test_audience_framing_in_base_prompt(self):
        llm = FakeLLM(response="[]")
        await extract_entries(llm, "config.yaml", "key: value")
        assert "memorized" in llm.last_system
        assert "doesn't exist online" in llm.last_system

    async def test_extracts_entries(self):
        entries = [
            {
                "short_title": "test entry",
                "long_title": "A test knowledge entry",
                "knowledge_details": "Details about testing.",
                "entry_type": "factual_reference",
                "tags": ["testing"],
            }
        ]
        llm = FakeLLM(response=json.dumps(entries))
        result = await extract_entries(llm, "test.md", "# Testing")
        assert len(result) == 1
        assert result[0].short_title == "test entry"
        assert result[0].entry_type == "factual_reference"
        assert result[0].tags == ["testing"]

    async def test_returns_empty_when_unavailable(self):
        llm = FakeLLM(available=False)
        result = await extract_entries(llm, "test.md", "content")
        assert result == []

    async def test_returns_empty_on_llm_failure(self):
        llm = FakeLLM(response=None)
        result = await extract_entries(llm, "test.md", "content")
        assert result == []

    async def test_returns_empty_on_empty_array(self):
        llm = FakeLLM(response="[]")
        result = await extract_entries(llm, "test.md", "content")
        assert result == []

    async def test_uses_code_supplement_for_code_files(self):
        llm = FakeLLM(response="[]")
        await extract_entries(llm, "utils.py", "def foo(): pass")
        assert "SOURCE CODE file" in llm.last_system
        assert "COMMENTS and ANNOTATIONS" in llm.last_system

    async def test_no_code_supplement_for_docs(self):
        llm = FakeLLM(response="[]")
        await extract_entries(llm, "guide.md", "# Guide")
        assert "SOURCE CODE file" not in llm.last_system

    async def test_uses_prose_supplement_for_markdown(self):
        llm = FakeLLM(response="[]")
        await extract_entries(llm, "guide.md", "# Guide")
        assert "NOTES or DOCUMENTATION" in llm.last_system
        assert "original arguments" in llm.last_system

    async def test_no_prose_supplement_for_config(self):
        llm = FakeLLM(response="[]")
        await extract_entries(llm, "settings.toml", "[section]")
        assert "NOTES or DOCUMENTATION" not in llm.last_system
        assert "SOURCE CODE file" not in llm.last_system


class TestParseEntries:
    def test_parses_valid_json(self):
        data = [
            {
                "short_title": "test",
                "long_title": "Test entry",
                "knowledge_details": "Details",
                "entry_type": "decision",
                "tags": ["python"],
            }
        ]
        result = _parse_entries(json.dumps(data))
        assert len(result) == 1
        assert isinstance(result[0], ExtractedEntry)
        assert result[0].entry_type == "decision"

    def test_strips_markdown_fences(self):
        data = [
            {
                "short_title": "test",
                "long_title": "Test",
                "knowledge_details": "Details",
                "entry_type": "lesson_learned",
                "tags": [],
            }
        ]
        raw = f"```json\n{json.dumps(data)}\n```"
        result = _parse_entries(raw)
        assert len(result) == 1

    def test_skips_invalid_entry_type(self):
        data = [
            {
                "short_title": "test",
                "long_title": "Test",
                "knowledge_details": "Details",
                "entry_type": "invalid_type",
                "tags": [],
            }
        ]
        result = _parse_entries(json.dumps(data))
        assert len(result) == 0

    def test_skips_missing_fields(self):
        data = [
            {
                "short_title": "test",
                # missing long_title, knowledge_details, entry_type
            }
        ]
        result = _parse_entries(json.dumps(data))
        assert len(result) == 0

    def test_caps_at_max_entries(self):
        data = [
            {
                "short_title": f"entry-{i}",
                "long_title": f"Entry {i}",
                "knowledge_details": f"Details {i}",
                "entry_type": "factual_reference",
                "tags": [],
            }
            for i in range(15)
        ]
        result = _parse_entries(json.dumps(data))
        assert len(result) == 10

    def test_handles_malformed_json(self):
        result = _parse_entries("not json at all")
        assert result == []

    def test_handles_non_list_response(self):
        result = _parse_entries('{"key": "value"}')
        assert result == []

    def test_lowercases_tags(self):
        data = [
            {
                "short_title": "test",
                "long_title": "Test",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": ["Python", "ASYNC"],
            }
        ]
        result = _parse_entries(json.dumps(data))
        assert result[0].tags == ["python", "async"]

    def test_handles_non_list_tags(self):
        data = [
            {
                "short_title": "test",
                "long_title": "Test",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": "not a list",
            }
        ]
        result = _parse_entries(json.dumps(data))
        assert result[0].tags == []

    def test_finds_array_in_surrounding_text(self):
        data = [
            {
                "short_title": "test",
                "long_title": "Test",
                "knowledge_details": "Details",
                "entry_type": "pattern_convention",
                "tags": [],
            }
        ]
        raw = f"Here are the entries:\n{json.dumps(data)}\nDone!"
        result = _parse_entries(raw)
        assert len(result) == 1
