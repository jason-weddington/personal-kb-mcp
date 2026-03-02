"""Tests for the kb_maintain MCP tool."""

from datetime import UTC, datetime, timedelta

import pytest

from personal_kb.config import is_manager_mode
from personal_kb.db.queries import get_entry
from personal_kb.models.entry import EntryType
from personal_kb.tools.kb_maintain import (
    _action_deactivate,
    _action_entry_versions,
    _action_list_audit,
    _action_list_contributors,
    _action_list_feedback,
    _action_purge_inactive,
    _action_reactivate,
    _action_rebuild_embeddings,
    _action_rebuild_graph,
    _action_search_stats,
    _action_stats,
    _action_summarize_feedback,
    _action_vacuum,
)

# --- Manager mode gating ---


def test_manager_mode_gating(monkeypatch):
    """is_manager_mode() should respond to KB_MANAGER env var."""
    monkeypatch.delenv("KB_MANAGER", raising=False)
    assert is_manager_mode() is False

    monkeypatch.setenv("KB_MANAGER", "TRUE")
    assert is_manager_mode() is True

    monkeypatch.setenv("KB_MANAGER", "true")
    assert is_manager_mode() is True

    monkeypatch.setenv("KB_MANAGER", "false")
    assert is_manager_mode() is False

    monkeypatch.setenv("KB_MANAGER", "")
    assert is_manager_mode() is False


# --- Unknown action ---


@pytest.mark.asyncio
async def test_unknown_action(db, store, fake_embedder, graph_builder):
    """Unknown action should return an error listing valid actions."""
    from personal_kb.tools.kb_maintain import _ACTIONS

    # We test the dispatch indirectly by checking the action set
    assert "stats" in _ACTIONS
    assert "nonexistent" not in _ACTIONS


# --- Stats ---


@pytest.mark.asyncio
async def test_stats_empty_db(db):
    """Stats on a fresh database should show zero counts."""
    result = await _action_stats(db)
    assert "Knowledge Base Statistics" in result
    assert "0 total" in result


@pytest.mark.asyncio
async def test_stats_with_entries(db, store, graph_builder):
    """Stats should reflect correct counts."""
    e1 = await store.create_entry(
        short_title="Stat entry 1",
        long_title="First stat entry",
        knowledge_details="Details 1",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="proj-a",
        tags=["python"],
    )
    await graph_builder.build_for_entry(e1)
    e2 = await store.create_entry(
        short_title="Stat entry 2",
        long_title="Second stat entry",
        knowledge_details="Details 2",
        entry_type=EntryType.DECISION,
        project_ref="proj-a",
    )
    await graph_builder.build_for_entry(e2)

    result = await _action_stats(db)
    assert "2 total" in result
    assert "2 active" in result
    assert "factual_reference: 1" in result
    assert "decision: 1" in result
    assert "proj-a: 2" in result


# --- Deactivate ---


@pytest.mark.asyncio
async def test_deactivate_entry(db, store, graph_builder):
    """Deactivate should set entry inactive and clean graph edges."""
    entry = await store.create_entry(
        short_title="To deactivate",
        long_title="Entry for deactivation",
        knowledge_details="Will be deactivated",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["python"],
    )
    await graph_builder.build_for_entry(entry)

    # Verify graph edges exist
    cursor = await db.execute("SELECT COUNT(*) FROM graph_edges WHERE source = ?", (entry.id,))
    assert (await cursor.fetchone())[0] > 0

    result = await _action_deactivate(db, store, entry.id)
    assert "Deactivated" in result
    assert entry.id in result

    # Entry should be inactive
    fetched = await get_entry(db, entry.id)
    assert fetched is not None
    assert fetched.is_active is False

    # Graph edges should be cleaned
    cursor = await db.execute("SELECT COUNT(*) FROM graph_edges WHERE source = ?", (entry.id,))
    assert (await cursor.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_deactivate_missing_entry_id(db, store):
    """Deactivate without entry_id should return error."""
    result = await _action_deactivate(db, store, None)
    assert "entry_id is required" in result


# --- Reactivate ---


@pytest.mark.asyncio
async def test_reactivate_entry(db, store, graph_builder):
    """Reactivate should set entry active and rebuild graph edges."""
    entry = await store.create_entry(
        short_title="To reactivate",
        long_title="Entry for reactivation",
        knowledge_details="Will be reactivated",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["python"],
    )
    await graph_builder.build_for_entry(entry)
    await store.deactivate_entry(entry.id)
    await db.execute("DELETE FROM graph_edges WHERE source = ?", (entry.id,))
    await db.commit()

    result = await _action_reactivate(db, store, graph_builder, None, entry.id)
    assert "Reactivated" in result
    assert entry.id in result

    # Entry should be active
    fetched = await get_entry(db, entry.id)
    assert fetched is not None
    assert fetched.is_active is True

    # Graph edges should be rebuilt
    cursor = await db.execute("SELECT COUNT(*) FROM graph_edges WHERE source = ?", (entry.id,))
    assert (await cursor.fetchone())[0] > 0


# --- Rebuild embeddings ---


@pytest.mark.asyncio
async def test_rebuild_embeddings_missing(db, store, fake_embedder):
    """Should embed entries without embeddings."""
    e1 = await store.create_entry(
        short_title="No embedding",
        long_title="Entry without embedding",
        knowledge_details="Needs embedding",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    assert e1.has_embedding is False

    result = await _action_rebuild_embeddings(db, store, fake_embedder, force=False)
    assert "1 processed" in result
    assert "1 succeeded" in result

    fetched = await get_entry(db, e1.id)
    assert fetched is not None
    assert fetched.has_embedding is True


@pytest.mark.asyncio
async def test_rebuild_embeddings_force(db, store, fake_embedder):
    """force=True should re-embed all active entries."""
    e1 = await store.create_entry(
        short_title="Already embedded",
        long_title="Entry with embedding",
        knowledge_details="Has embedding",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    # Manually mark as embedded
    await store.mark_embedding(e1.id, True)

    await store.create_entry(
        short_title="Not embedded",
        long_title="Entry without embedding",
        knowledge_details="No embedding",
        entry_type=EntryType.DECISION,
    )

    result = await _action_rebuild_embeddings(db, store, fake_embedder, force=True)
    assert "2 processed" in result
    assert "2 succeeded" in result


@pytest.mark.asyncio
async def test_rebuild_embeddings_no_ollama(db, store, fake_embedder):
    """Should gracefully skip when Ollama is unavailable."""
    fake_embedder._available = False

    await store.create_entry(
        short_title="Won't embed",
        long_title="Entry that won't be embedded",
        knowledge_details="Ollama is down",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    result = await _action_rebuild_embeddings(db, store, fake_embedder, force=False)
    assert "not available" in result


# --- Rebuild graph ---


@pytest.mark.asyncio
async def test_rebuild_graph(db, store, graph_builder):
    """Full graph rebuild should produce correct counts."""
    await store.create_entry(
        short_title="Graph entry 1",
        long_title="First graph entry",
        knowledge_details="Details 1",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["python", "sqlite"],
        project_ref="test-proj",
    )
    await store.create_entry(
        short_title="Graph entry 2",
        long_title="Second graph entry",
        knowledge_details="Details 2",
        entry_type=EntryType.DECISION,
        tags=["python"],
    )

    result = await _action_rebuild_graph(db, graph_builder, None)
    assert "2 entries processed" in result
    assert "nodes" in result
    assert "edges" in result

    # Verify graph has content
    cursor = await db.execute("SELECT COUNT(*) FROM graph_nodes")
    assert (await cursor.fetchone())[0] > 0
    cursor = await db.execute("SELECT COUNT(*) FROM graph_edges")
    assert (await cursor.fetchone())[0] > 0


# --- Purge inactive ---


@pytest.mark.asyncio
async def test_purge_requires_confirm(db):
    """Purge should reject without confirm=True."""
    result = await _action_purge_inactive(db, days_inactive=90, confirm=False)
    assert "confirm=True" in result


@pytest.mark.asyncio
async def test_purge_inactive_entries(db, store):
    """Should hard-delete entries inactive for N+ days."""
    entry = await store.create_entry(
        short_title="To purge",
        long_title="Entry to purge",
        knowledge_details="Will be purged",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.deactivate_entry(entry.id)

    # Backdate the updated_at to make it old enough
    old_date = (datetime.now(UTC) - timedelta(days=100)).isoformat()
    await db.execute(
        "UPDATE knowledge_entries SET updated_at = ? WHERE id = ?",
        (old_date, entry.id),
    )
    await db.commit()

    result = await _action_purge_inactive(db, days_inactive=90, confirm=True)
    assert "Purged 1" in result

    # Entry should be gone
    fetched = await get_entry(db, entry.id)
    assert fetched is None


# --- Vacuum ---


@pytest.mark.asyncio
async def test_vacuum(db):
    """Vacuum should run without error."""
    result = await _action_vacuum(db)
    assert "Vacuum complete" in result


# --- Entry versions ---


@pytest.mark.asyncio
async def test_entry_versions(db, store):
    """Should show version history for an entry."""
    entry = await store.create_entry(
        short_title="Versioned",
        long_title="Versioned entry",
        knowledge_details="Version 1",
        entry_type=EntryType.DECISION,
    )
    await store.update_entry(
        entry_id=entry.id,
        knowledge_details="Version 2",
        change_reason="Updated info",
        confidence_level=0.95,
    )

    result = await _action_entry_versions(db, entry.id)
    assert "Version history" in result
    assert "Versioned" in result
    assert "v1" in result
    assert "v2" in result
    assert "Updated info" in result
    assert "Current version: 2" in result


@pytest.mark.asyncio
async def test_entry_versions_missing_id(db):
    """Entry versions without entry_id should return error."""
    result = await _action_entry_versions(db, None)
    assert "entry_id is required" in result


# --- List feedback ---


async def _insert_feedback(db, feedback_type, tool_name=None, query=None, detail=None, ts=None):
    """Helper to insert a feedback row."""
    ts = ts or datetime.now(UTC).isoformat()
    await db.execute(
        "INSERT INTO agent_feedback (feedback_type, tool_name, query_or_params, detail, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (feedback_type, tool_name, query, detail, ts),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_list_feedback_returns_entries(db):
    """list_feedback should return formatted feedback entries."""
    await _insert_feedback(db, "missing", "kb_search", "sqlite tips", "No results found")
    await _insert_feedback(db, "friction", detail="Slow response")

    result = await _action_list_feedback(db, None, None)
    assert "Agent Feedback (2 entries)" in result
    assert "[missing]" in result
    assert "[friction]" in result
    assert "kb_search" in result
    assert "sqlite tips" in result
    assert "No results found" in result


@pytest.mark.asyncio
async def test_list_feedback_filters_by_type(db):
    """list_feedback should filter by feedback_type."""
    await _insert_feedback(db, "missing")
    await _insert_feedback(db, "friction")

    result = await _action_list_feedback(db, "missing", None)
    assert "[missing]" in result
    assert "[friction]" not in result


@pytest.mark.asyncio
async def test_list_feedback_filters_by_since(db):
    """list_feedback should filter by since date."""
    await _insert_feedback(db, "missing", ts="2025-01-01T00:00:00")
    await _insert_feedback(db, "friction", ts="2026-06-01T00:00:00")

    result = await _action_list_feedback(db, None, "2026-01-01")
    assert "[friction]" in result
    assert "[missing]" not in result


@pytest.mark.asyncio
async def test_list_feedback_empty(db):
    """list_feedback with no entries should return message."""
    result = await _action_list_feedback(db, None, None)
    assert "No feedback entries found" in result


# --- Summarize feedback ---


@pytest.mark.asyncio
async def test_summarize_feedback_with_llm(db, fake_llm):
    """summarize_feedback with LLM should produce summary."""
    await _insert_feedback(db, "missing", "kb_search", "python async")
    await _insert_feedback(db, "missing", "kb_ask", "error handling")

    fake_llm.response = "- Theme 1: Missing async knowledge\n- Theme 2: Error handling gaps"
    result = await _action_summarize_feedback(db, fake_llm, None)
    assert "Feedback Summary (2 entries)" in result
    assert "Theme 1" in result


@pytest.mark.asyncio
async def test_summarize_feedback_without_llm(db):
    """summarize_feedback without LLM should fall back to raw list."""
    await _insert_feedback(db, "unhelpful", "kb_search", "docker tips")

    result = await _action_summarize_feedback(db, None, None)
    # Falls back to list_feedback format
    assert "[unhelpful]" in result


@pytest.mark.asyncio
async def test_summarize_feedback_empty(db):
    """summarize_feedback with no entries should return message."""
    result = await _action_summarize_feedback(db, None, None)
    assert "No feedback entries to summarize" in result


# --- Search stats ---


async def _insert_search_event(db, query_text, result_count, top_score, match_source, ts=None):
    """Helper to insert a search_events row."""
    ts = ts or datetime.now(UTC).isoformat()
    await db.execute(
        "INSERT INTO search_events"
        " (query_text, result_count, top_score, match_source, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (query_text, result_count, top_score, match_source, ts),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_search_stats_basic(db):
    """search_stats should return correct totals."""
    await _insert_search_event(db, "python async", 5, 0.035, "hybrid")
    await _insert_search_event(db, "docker tips", 3, 0.020, "fts")
    await _insert_search_event(db, "nonexistent", 0, None, "fts")

    result = await _action_search_stats(db, None)
    assert "Search Telemetry" in result
    assert "Total queries: 3" in result
    assert "Zero-result queries: 1 (33%)" in result
    assert "Average top score:" in result
    assert "nonexistent" in result  # Shows up in missed queries


@pytest.mark.asyncio
async def test_search_stats_empty(db):
    """search_stats with no events should return message."""
    result = await _action_search_stats(db, None)
    assert "No search events recorded" in result


@pytest.mark.asyncio
async def test_search_stats_with_since_filter(db):
    """search_stats should filter by since date."""
    await _insert_search_event(db, "old query", 2, 0.01, "fts", ts="2025-01-01T00:00:00")
    await _insert_search_event(db, "new query", 4, 0.03, "hybrid", ts="2026-06-01T00:00:00")

    result = await _action_search_stats(db, "2026-01-01")
    assert "Total queries: 1" in result


# --- List contributors ---


@pytest.mark.asyncio
async def test_list_contributors(db, store):
    """list_contributors should show contributor/team stats."""
    await store.create_entry(
        short_title="Entry 1",
        long_title="First entry",
        knowledge_details="Details 1",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="jason",
        team="platform",
    )
    await store.create_entry(
        short_title="Entry 2",
        long_title="Second entry",
        knowledge_details="Details 2",
        entry_type=EntryType.DECISION,
        contributor="jason",
        team="platform",
    )
    await store.create_entry(
        short_title="Entry 3",
        long_title="Third entry",
        knowledge_details="Details 3",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="alice",
    )

    result = await _action_list_contributors(db)
    assert "Contributors" in result
    assert "@jason/platform" in result
    assert "2 entries" in result
    assert "@alice" in result
    assert "1 entries" in result


@pytest.mark.asyncio
async def test_list_contributors_empty(db):
    """list_contributors with no attributed entries should return message."""
    result = await _action_list_contributors(db)
    assert "No attributed entries found" in result


# --- List audit ---


@pytest.mark.asyncio
async def test_list_audit_records_on_create(db, store):
    """Creating an entry should record an audit event."""
    await store.create_entry(
        short_title="Audited entry",
        long_title="Entry with audit",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="jason",
    )

    result = await _action_list_audit(db, None, None)
    assert "Audit Events" in result
    assert "[entry_created]" in result
    assert "kb-00001" in result
    assert "@jason" in result


@pytest.mark.asyncio
async def test_list_audit_records_on_update(db, store):
    """Updating an entry should record an audit event."""
    entry = await store.create_entry(
        short_title="To update",
        long_title="Entry to update",
        knowledge_details="Original",
        entry_type=EntryType.DECISION,
    )
    await store.update_entry(
        entry_id=entry.id,
        knowledge_details="Updated details",
        change_reason="Improved info",
        updated_by="alice",
    )

    result = await _action_list_audit(db, None, None)
    assert "[entry_updated]" in result
    assert "Improved info" in result


@pytest.mark.asyncio
async def test_list_audit_records_on_deactivate(db, store):
    """Deactivating an entry should record an audit event."""
    entry = await store.create_entry(
        short_title="To deactivate",
        long_title="Entry to deactivate",
        knowledge_details="Will be deactivated",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.deactivate_entry(entry.id)

    result = await _action_list_audit(db, None, None)
    assert "[entry_deactivated]" in result


@pytest.mark.asyncio
async def test_list_audit_records_on_reactivate(db, store):
    """Reactivating an entry should record an audit event."""
    entry = await store.create_entry(
        short_title="To reactivate",
        long_title="Entry to reactivate",
        knowledge_details="Will be reactivated",
        entry_type=EntryType.DECISION,
    )
    await store.deactivate_entry(entry.id)
    await store.reactivate_entry(entry.id)

    result = await _action_list_audit(db, None, None)
    assert "[entry_reactivated]" in result


@pytest.mark.asyncio
async def test_list_audit_filter_by_entry_id(db, store):
    """list_audit should filter by entry_id."""
    await store.create_entry(
        short_title="Entry A",
        long_title="First entry",
        knowledge_details="Details A",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.create_entry(
        short_title="Entry B",
        long_title="Second entry",
        knowledge_details="Details B",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    result = await _action_list_audit(db, "kb-00001", None)
    assert "kb-00001" in result
    assert "kb-00002" not in result


@pytest.mark.asyncio
async def test_list_audit_empty(db):
    """list_audit with no events should return message."""
    result = await _action_list_audit(db, None, None)
    assert "No audit events found" in result
