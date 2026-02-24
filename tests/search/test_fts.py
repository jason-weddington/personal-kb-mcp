"""Tests for FTS5 search."""

import pytest

from personal_kb.models.entry import EntryType
from personal_kb.search.fts import fts_search


@pytest.mark.asyncio
async def test_fts_basic_search(db, store):
    await store.create_entry(
        short_title="Python asyncio",
        long_title="Guide to Python asyncio patterns",
        knowledge_details="Use async/await for concurrent IO operations.",
        entry_type=EntryType.PATTERN_CONVENTION,
        tags=["python", "async"],
    )
    await store.create_entry(
        short_title="Rust ownership",
        long_title="Rust ownership model",
        knowledge_details="Rust uses ownership and borrowing for memory safety.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["rust"],
    )

    results = await fts_search(db, "python asyncio")
    assert len(results) >= 1
    assert results[0][0] == "kb-00001"


@pytest.mark.asyncio
async def test_fts_project_filter(db, store):
    await store.create_entry(
        short_title="API endpoint",
        long_title="User API endpoint",
        knowledge_details="GET /api/users returns user list",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="project-a",
    )
    await store.create_entry(
        short_title="API config",
        long_title="API configuration",
        knowledge_details="API runs on port 8080",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="project-b",
    )

    results = await fts_search(db, "API", project_ref="project-a")
    assert len(results) >= 1
    assert all(r[0] == "kb-00001" for r in results)


@pytest.mark.asyncio
async def test_fts_type_filter(db, store):
    await store.create_entry(
        short_title="Decision about DB",
        long_title="Chose SQLite for storage",
        knowledge_details="SQLite chosen for simplicity and single-file deployment.",
        entry_type=EntryType.DECISION,
    )
    await store.create_entry(
        short_title="SQLite fact",
        long_title="SQLite version info",
        knowledge_details="SQLite 3.45.0 supports JSONB.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    results = await fts_search(db, "SQLite", entry_type="decision")
    assert len(results) >= 1
    assert results[0][0] == "kb-00001"


@pytest.mark.asyncio
async def test_fts_empty_query(db, store):
    results = await fts_search(db, "")
    assert results == []


@pytest.mark.asyncio
async def test_fts_no_matches(db, store):
    await store.create_entry(
        short_title="Unrelated",
        long_title="Unrelated entry",
        knowledge_details="Nothing about the query term.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    results = await fts_search(db, "xyznonexistent")
    assert results == []
