"""Tests for discovery tools: list_projects, list_contributors, list_teams."""

import pytest

from personal_kb.models.entry import EntryType
from personal_kb.tools.kb_list import (
    list_contributors_logic,
    list_projects_logic,
    list_teams_logic,
)

# --- list_projects ---


@pytest.mark.asyncio
async def test_list_projects_empty(db):
    """Empty DB returns 'No projects found.'"""
    result = await list_projects_logic(db)
    assert result == "No projects found."


@pytest.mark.asyncio
async def test_list_projects_with_entries(db, store):
    """Projects listed with counts, sorted descending."""
    for _ in range(3):
        await store.create_entry(
            short_title="A",
            long_title="A entry",
            knowledge_details="details",
            entry_type=EntryType.FACTUAL_REFERENCE,
            project_ref="alpha",
        )
    await store.create_entry(
        short_title="B",
        long_title="B entry",
        knowledge_details="details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="beta",
    )

    result = await list_projects_logic(db)
    lines = result.strip().split("\n")
    assert len(lines) == 2
    assert lines[0] == "alpha (3 entries)"
    assert lines[1] == "beta (1 entries)"


@pytest.mark.asyncio
async def test_list_projects_excludes_null(db, store):
    """Entries without project_ref are not listed."""
    await store.create_entry(
        short_title="No proj",
        long_title="No project",
        knowledge_details="details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    result = await list_projects_logic(db)
    assert result == "No projects found."


@pytest.mark.asyncio
async def test_list_projects_excludes_inactive(db, store):
    """Inactive entries are not counted."""
    entry = await store.create_entry(
        short_title="Gone",
        long_title="Deactivated",
        knowledge_details="details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="gone-proj",
    )
    await store.deactivate_entry(entry.id)

    result = await list_projects_logic(db)
    assert result == "No projects found."


# --- list_contributors ---


@pytest.mark.asyncio
async def test_list_contributors_empty(db):
    """Empty DB returns 'No contributors found.'"""
    result = await list_contributors_logic(db)
    assert result == "No contributors found."


@pytest.mark.asyncio
async def test_list_contributors_with_entries(db, store):
    """Contributors listed with counts, sorted descending."""
    for _ in range(2):
        await store.create_entry(
            short_title="A",
            long_title="A entry",
            knowledge_details="details",
            entry_type=EntryType.FACTUAL_REFERENCE,
            contributor="alice",
        )
    await store.create_entry(
        short_title="B",
        long_title="B entry",
        knowledge_details="details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="bob",
    )

    result = await list_contributors_logic(db)
    lines = result.strip().split("\n")
    assert len(lines) == 2
    assert lines[0] == "alice (2 entries)"
    assert lines[1] == "bob (1 entries)"


# --- list_teams ---


@pytest.mark.asyncio
async def test_list_teams_empty(db):
    """Empty DB returns 'No teams found.'"""
    result = await list_teams_logic(db)
    assert result == "No teams found."


@pytest.mark.asyncio
async def test_list_teams_with_entries(db, store):
    """Teams listed with counts, sorted descending."""
    for _ in range(2):
        await store.create_entry(
            short_title="A",
            long_title="A entry",
            knowledge_details="details",
            entry_type=EntryType.FACTUAL_REFERENCE,
            team="eng",
        )
    await store.create_entry(
        short_title="B",
        long_title="B entry",
        knowledge_details="details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        team="ops",
    )

    result = await list_teams_logic(db)
    lines = result.strip().split("\n")
    assert len(lines) == 2
    assert lines[0] == "eng (2 entries)"
    assert lines[1] == "ops (1 entries)"


# --- Registration gating ---


@pytest.mark.asyncio
async def test_list_projects_always_registered(monkeypatch):
    """list_projects is registered without KB_CONTRIBUTOR."""
    monkeypatch.delenv("KB_CONTRIBUTOR", raising=False)

    from personal_kb.server import create_server

    server = create_server()
    tools = {t.name for t in await server.list_tools()}
    assert "kb_list_projects" in tools


@pytest.mark.asyncio
async def test_contributor_tools_gated(monkeypatch):
    """list_contributors and list_teams require KB_CONTRIBUTOR."""
    monkeypatch.delenv("KB_CONTRIBUTOR", raising=False)

    from personal_kb.server import create_server

    server = create_server()
    tools = {t.name for t in await server.list_tools()}
    assert "kb_list_contributors" not in tools
    assert "kb_list_teams" not in tools


@pytest.mark.asyncio
async def test_contributor_tools_registered_with_contributor(monkeypatch):
    """list_contributors and list_teams registered when KB_CONTRIBUTOR set."""
    monkeypatch.setenv("KB_CONTRIBUTOR", "alice")

    from personal_kb.server import create_server

    server = create_server()
    tools = {t.name for t in await server.list_tools()}
    assert "kb_list_contributors" in tools
    assert "kb_list_teams" in tools
