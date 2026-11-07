"""Tests for tag normalization on the entry model."""

import pytest

from kb_core.models.entry import EntryType, KnowledgeEntry, normalize_tags


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["archival evidence"], ["archival-evidence"]),
        (["  a   b\tc "], ["a-b-c"]),
        (["x", "x", "  ", "", "y"], ["x", "y"]),
        (["a b", "a-b"], ["a-b"]),
        (["Foo", "foo"], ["Foo", "foo"]),
    ],
)
def test_normalize_tags(raw: list[str], expected: list[str]) -> None:
    assert normalize_tags(raw) == expected


def test_entry_validator_normalizes_tags() -> None:
    entry = KnowledgeEntry(
        id="kb-00001",
        short_title="s",
        long_title="l",
        knowledge_details="d",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["archival evidence"],
    )
    assert entry.tags == ["archival-evidence"]
