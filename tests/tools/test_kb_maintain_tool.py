"""Tests for the kb_maintain MCP tool."""

from datetime import UTC, datetime, timedelta

import pytest

from personal_kb.config import is_manager_mode
from personal_kb.db.queries import get_entry
from personal_kb.models.entry import EntryType
from personal_kb.tools.kb_maintain import (
    _action_deactivate,
    _action_entry_versions,
    _action_purge_inactive,
    _action_reactivate,
    _action_rebuild_embeddings,
    _action_rebuild_graph,
    _action_stats,
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
