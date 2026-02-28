"""Tests for compact output formatters."""

from datetime import UTC, datetime

from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.tools.formatters import (
    format_entry_compact,
    format_entry_full,
    format_entry_header,
    format_entry_meta,
    format_result_list,
)


def _make_entry(**kwargs) -> KnowledgeEntry:
    defaults = {
        "id": "kb-00001",
        "short_title": "Test Entry",
        "long_title": "A test entry",
        "knowledge_details": "Some important details",
        "entry_type": EntryType.FACTUAL_REFERENCE,
        "tags": ["python", "sqlite"],
        "project_ref": "personal-kb",
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


# --- format_entry_header ---


def test_header_basic():
    entry = _make_entry()
    result = format_entry_header(entry, 0.85)
    assert result == "[kb-00001] factual_reference | Test Entry (85%)"


def test_header_decision_type():
    entry = _make_entry(entry_type=EntryType.DECISION, short_title="Chose X")
    result = format_entry_header(entry, 0.92)
    assert result == "[kb-00001] decision | Chose X (92%)"


def test_header_zero_confidence():
    entry = _make_entry()
    result = format_entry_header(entry, 0.0)
    assert "(0%)" in result


# --- format_entry_meta ---


def test_meta_tags_and_project():
    entry = _make_entry()
    result = format_entry_meta(entry)
    assert "#python #sqlite" in result
    assert "personal-kb" in result


def test_meta_no_tags():
    entry = _make_entry(tags=[])
    result = format_entry_meta(entry)
    assert result == "personal-kb"


def test_meta_no_project():
    entry = _make_entry(project_ref=None)
    result = format_entry_meta(entry)
    assert result == "#python #sqlite"


def test_meta_no_tags_no_project():
    entry = _make_entry(tags=[], project_ref=None)
    result = format_entry_meta(entry)
    assert result == ""


def test_meta_staleness():
    entry = _make_entry()
    result = format_entry_meta(entry, stale_warning="Stale entry")
    assert "[STALE]" in result


def test_meta_staleness_no_tags_no_project():
    entry = _make_entry(tags=[], project_ref=None)
    result = format_entry_meta(entry, stale_warning="Stale entry")
    assert result == "[STALE]"


# --- format_entry_compact ---


def test_compact_with_meta():
    entry = _make_entry()
    result = format_entry_compact(entry, 0.85)
    assert "[kb-00001]" in result
    assert "#python" in result
    assert "Some important details" not in result  # no details in compact


def test_compact_no_meta():
    entry = _make_entry(tags=[], project_ref=None)
    result = format_entry_compact(entry, 0.85)
    assert "\n" not in result  # single line, no meta


def test_compact_with_staleness():
    entry = _make_entry()
    result = format_entry_compact(entry, 0.3, stale_warning="Stale")
    assert "[STALE]" in result


# --- format_entry_full ---


def test_full_with_context():
    entry = _make_entry()
    result = format_entry_full(entry, context="search match", effective_confidence=0.9)
    assert "[kb-00001]" in result
    assert "\u21b3 search match" in result
    assert "Some important details" in result


def test_full_without_context():
    entry = _make_entry()
    result = format_entry_full(entry, effective_confidence=0.9)
    assert "\u21b3" not in result
    assert "Some important details" in result


def test_full_auto_computes_confidence():
    """When no effective_confidence given, it's computed from the entry."""
    entry = _make_entry()
    result = format_entry_full(entry)
    # Should not crash, and should contain a percentage
    assert "%" in result


def test_full_includes_staleness():
    entry = _make_entry()
    result = format_entry_full(
        entry,
        effective_confidence=0.3,
        stale_warning="Stale factual_reference entry",
    )
    assert "[STALE]" in result


# --- format_result_list ---


def test_result_list_empty():
    result = format_result_list([])
    assert result == "No results found."


def test_result_list_with_entries():
    entries = ["[kb-00001] factual_reference | A (90%)", "[kb-00002] decision | B (85%)"]
    result = format_result_list(entries, header="Search results")
    assert "2 result(s)" in result
    assert "Search results" in result
    assert "[kb-00001]" in result
    assert "[kb-00002]" in result


def test_result_list_with_note():
    entries = ["[kb-00001] factual_reference | A (90%)"]
    result = format_result_list(entries, note="FTS-only")
    assert "Note: FTS-only" in result


def test_result_list_no_header():
    entries = ["[kb-00001] factual_reference | A (90%)"]
    result = format_result_list(entries)
    assert "1 result(s)" in result
