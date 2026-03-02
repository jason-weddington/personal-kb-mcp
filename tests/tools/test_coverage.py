"""Tests for the coverage assessment module."""

import pytest

from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.tools.coverage import _parse_coverage_response, assess_coverage
from tests.conftest import FakeLLM


def _make_entry(
    entry_id: str = "kb-00001", title: str = "Test", details: str = "Details"
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        short_title=title,
        long_title=title,
        knowledge_details=details,
        entry_type=EntryType.FACTUAL_REFERENCE,
    )


# --- _parse_coverage_response ---


def test_parse_no_gaps():
    raw = '{"has_gaps": false, "suggested_query": null, "reason": "entries cover the topic"}'
    result = _parse_coverage_response(raw)
    assert not result.has_gaps
    assert result.suggested_query is None
    assert "cover" in result.reason


def test_parse_with_gaps():
    raw = '{"has_gaps": true, "suggested_query": "sqlite WAL mode", "reason": "missing"}'
    result = _parse_coverage_response(raw)
    assert result.has_gaps
    assert result.suggested_query == "sqlite WAL mode"


def test_parse_markdown_fence():
    raw = '```json\n{"has_gaps": true, "suggested_query": "async io", "reason": "gap"}\n```'
    result = _parse_coverage_response(raw)
    assert result.has_gaps
    assert result.suggested_query == "async io"


def test_parse_malformed_json():
    raw = "not json at all"
    result = _parse_coverage_response(raw)
    assert not result.has_gaps
    assert "malformed" in result.reason


def test_parse_invalid_json():
    raw = '{"has_gaps": true, broken}'
    result = _parse_coverage_response(raw)
    assert not result.has_gaps
    assert "malformed" in result.reason


def test_parse_not_dict():
    raw = "[1, 2, 3]"
    result = _parse_coverage_response(raw)
    assert not result.has_gaps


# --- assess_coverage ---


@pytest.mark.asyncio
async def test_assess_no_gaps():
    llm = FakeLLM(response='{"has_gaps": false, "suggested_query": null, "reason": "all covered"}')
    entries = [(_make_entry(), "search match")]
    result = await assess_coverage(llm, "test question", entries)
    assert not result.has_gaps


@pytest.mark.asyncio
async def test_assess_gap_found():
    llm = FakeLLM(response='{"has_gaps": true, "suggested_query": "WAL mode", "reason": "missing"}')
    entries = [(_make_entry(), "search match")]
    result = await assess_coverage(llm, "how does WAL work?", entries)
    assert result.has_gaps
    assert result.suggested_query == "WAL mode"


@pytest.mark.asyncio
async def test_assess_llm_returns_none():
    llm = FakeLLM(response=None)
    entries = [(_make_entry(), "search match")]
    result = await assess_coverage(llm, "question", entries)
    assert not result.has_gaps
    assert "no response" in result.reason


@pytest.mark.asyncio
async def test_assess_llm_unavailable():
    llm = FakeLLM(available=False)
    entries = [(_make_entry(), "search match")]
    result = await assess_coverage(llm, "question", entries)
    assert not result.has_gaps


@pytest.mark.asyncio
async def test_assess_includes_question_and_entries_in_prompt():
    llm = FakeLLM(response='{"has_gaps": false, "suggested_query": null, "reason": "ok"}')
    entries = [(_make_entry(title="SQLite WAL"), "search match")]
    await assess_coverage(llm, "my specific question", entries)
    assert llm.last_prompt is not None
    assert "my specific question" in llm.last_prompt
    assert "SQLite WAL" in llm.last_prompt


@pytest.mark.asyncio
async def test_assess_graceful_on_exception():
    """Any exception should return has_gaps=False."""

    class BrokenLLM:
        async def is_available(self) -> bool:
            return True

        async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
            raise RuntimeError("LLM exploded")

        async def close(self) -> None:
            pass

    entries = [(_make_entry(), "search match")]
    result = await assess_coverage(BrokenLLM(), "question", entries)  # type: ignore[arg-type]
    assert not result.has_gaps
    assert "failed" in result.reason
