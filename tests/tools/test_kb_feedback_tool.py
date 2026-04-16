"""Tests for the kb_feedback tool."""

import pytest

from personal_kb.tools.kb_feedback import submit_feedback
from personal_kb.tools.kb_maintain import _action_list_feedback


@pytest.mark.asyncio
async def test_submit_feedback_missing(db):
    """submit_feedback with 'missing' type inserts a row."""
    result = await submit_feedback(db, "missing", "kb_search", "sqlite tips", "No results")
    assert "Feedback recorded" in result
    assert "missing" in result

    cursor = await db.execute("SELECT * FROM agent_feedback")
    rows = await cursor.fetchall()
    assert len(rows) == 1
    assert rows[0]["feedback_type"] == "missing"
    assert rows[0]["tool_name"] == "kb_search"
    assert rows[0]["query_or_params"] == "sqlite tips"
    assert rows[0]["detail"] == "No results"


@pytest.mark.asyncio
async def test_submit_feedback_unhelpful(db):
    """submit_feedback with 'unhelpful' type works."""
    result = await submit_feedback(db, "unhelpful", "kb_ask", "docker networking")
    assert "Feedback recorded" in result
    assert "unhelpful" in result

    cursor = await db.execute("SELECT * FROM agent_feedback")
    rows = await cursor.fetchall()
    assert len(rows) == 1
    assert rows[0]["feedback_type"] == "unhelpful"


@pytest.mark.asyncio
async def test_submit_feedback_friction(db):
    """submit_feedback with 'friction' type works."""
    result = await submit_feedback(db, "friction", detail="Too many results to parse")
    assert "Feedback recorded" in result
    assert "friction" in result


@pytest.mark.asyncio
async def test_submit_feedback_invalid_type(db):
    """submit_feedback with invalid type returns error message."""
    result = await submit_feedback(db, "bad_type")
    assert "Invalid feedback_type" in result
    assert "bad_type" in result
    assert "friction" in result  # Lists valid types

    # No row should be inserted
    cursor = await db.execute("SELECT COUNT(*) FROM agent_feedback")
    row = await cursor.fetchone()
    assert row[0] == 0


@pytest.mark.asyncio
async def test_submit_feedback_optional_fields_none(db):
    """Optional fields can all be None."""
    result = await submit_feedback(db, "missing")
    assert "Feedback recorded" in result

    cursor = await db.execute("SELECT * FROM agent_feedback")
    rows = await cursor.fetchall()
    assert len(rows) == 1
    assert rows[0]["tool_name"] is None
    assert rows[0]["query_or_params"] is None
    assert rows[0]["detail"] is None


@pytest.mark.asyncio
async def test_submit_feedback_with_contributor(db):
    """Feedback records contributor when provided."""
    await submit_feedback(db, "missing", "kb_search", "test query", contributor="jason")
    cursor = await db.execute("SELECT contributor FROM agent_feedback")
    row = await cursor.fetchone()
    assert row["contributor"] == "jason"


@pytest.mark.asyncio
async def test_submit_feedback_contributor_none_by_default(db):
    """Feedback contributor is NULL when not provided."""
    await submit_feedback(db, "friction")
    cursor = await db.execute("SELECT contributor FROM agent_feedback")
    row = await cursor.fetchone()
    assert row["contributor"] is None


@pytest.mark.asyncio
async def test_submit_feedback_created_at_set(db):
    """created_at should be populated automatically."""
    await submit_feedback(db, "missing")

    cursor = await db.execute("SELECT created_at FROM agent_feedback")
    row = await cursor.fetchone()
    assert row["created_at"] is not None
    assert "T" in row["created_at"]  # ISO format


@pytest.mark.asyncio
async def test_submit_feedback_with_team(db):
    """Feedback records team when provided."""
    await submit_feedback(db, "missing", "kb_search", "test query", team="platform")
    cursor = await db.execute("SELECT team FROM agent_feedback")
    row = await cursor.fetchone()
    assert row["team"] == "platform"


@pytest.mark.asyncio
async def test_submit_feedback_team_none_by_default(db):
    """Feedback team is NULL when not provided."""
    await submit_feedback(db, "friction")
    cursor = await db.execute("SELECT team FROM agent_feedback")
    row = await cursor.fetchone()
    assert row["team"] is None


@pytest.mark.asyncio
async def test_submit_feedback_contributor_and_team_together(db):
    """Both contributor and team are stored correctly on the same row."""
    await submit_feedback(
        db, "unhelpful", "kb_ask", "docker networking", contributor="jason", team="platform"
    )
    cursor = await db.execute("SELECT contributor, team FROM agent_feedback")
    row = await cursor.fetchone()
    assert row["contributor"] == "jason"
    assert row["team"] == "platform"


# --- AC6: list_feedback attribution badge tests ---


@pytest.mark.asyncio
async def test_list_feedback_badge_contributor_and_team(db):
    """list_feedback shows @contributor/team badge when both are set."""
    await submit_feedback(
        db,
        "missing",
        "kb_search",
        "sqlite tips",
        "No results for my topic",
        contributor="jason",
        team="infra",
    )
    result = await _action_list_feedback(db, None, None)
    assert "@jason/infra" in result


@pytest.mark.asyncio
async def test_list_feedback_badge_contributor_only(db):
    """list_feedback shows @contributor badge when only contributor is set."""
    await submit_feedback(
        db,
        "unhelpful",
        "kb_ask",
        "docker tips",
        contributor="alice",
    )
    result = await _action_list_feedback(db, None, None)
    assert "@alice" in result
    # Ensure there's no trailing slash (no team)
    assert "@alice/" not in result


@pytest.mark.asyncio
async def test_list_feedback_no_badge_when_no_contributor(db):
    """list_feedback shows no badge when contributor is not set."""
    await submit_feedback(db, "friction", detail="Too slow")
    result = await _action_list_feedback(db, None, None)
    assert "@" not in result
