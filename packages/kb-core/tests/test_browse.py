"""Hermetic SQLite suite for kb_core.browse.

Seeds the shared corpus from ``browse_fixture.py`` with direct INSERTs
(never ``kb.store()``) and asserts the shared cases, plus field values,
paging, parameter errors and the skip rules.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from browse_fixture import (
    BROWSE_CASES,
    BROWSE_EDGE_INSERT_SQL,
    BROWSE_ENTRY_INSERT_SQL,
    BROWSE_EXPECTED_DEGREES,
    BROWSE_EXPECTED_PROJECTS,
    BROWSE_NODE_INSERT_SQL,
    BROWSE_NOW,
    BROWSE_SEED_EDGES,
    BROWSE_SEED_ENTRIES,
    BROWSE_SEED_NODES,
)
from kb_core import browse, create_sqlite
from kb_core.browse import BrowsePage, BrowseParamError, BrowseRow, ProjectNotFound, ProjectPage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from kb_core.db.backend import Database
    from kb_core.knowledge_base import KnowledgeBase


async def seed_browse(db: Database) -> None:
    """Insert the shared browse corpus: nodes, then edges, then entries."""
    for node in BROWSE_SEED_NODES:
        await db.execute(BROWSE_NODE_INSERT_SQL, node)
    for edge in BROWSE_SEED_EDGES:
        await db.execute(BROWSE_EDGE_INSERT_SQL, edge)
    for entry in BROWSE_SEED_ENTRIES:
        await db.execute(BROWSE_ENTRY_INSERT_SQL, entry)
    await db.commit()


async def run_case(db: Database, fn: str, kwargs: dict[str, Any]) -> BrowsePage | ProjectPage:
    """Call one browse function the way the shared cases expect."""
    if fn == "browse_projects":
        return await browse.browse_projects(db, **kwargs)
    result: BrowsePage = await getattr(browse, fn)(db, **kwargs, now=BROWSE_NOW)
    return result


def keys_of(page: BrowsePage | ProjectPage) -> list[str]:
    """Return ids (entry pages) or project_refs (project pages) in order."""
    if isinstance(page, ProjectPage):
        return [p.project_ref for p in page.items]
    return [r.id for r in page.items]


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[KnowledgeBase]:
    """A SQLite KB seeded with the shared browse corpus."""
    kb = await create_sqlite(tmp_path / "kb.db")
    await seed_browse(kb.db)
    try:
        yield kb
    finally:
        await kb.close()


def _row(page: BrowsePage, entry_id: str) -> BrowseRow:
    return next(r for r in page.items if r.id == entry_id)


@pytest.mark.parametrize(
    ("name", "fn", "kwargs", "expected"), BROWSE_CASES, ids=[c[0] for c in BROWSE_CASES]
)
async def test_browse_cases(
    kb: KnowledgeBase, name: str, fn: str, kwargs: dict[str, Any], expected: list[str]
) -> None:
    """Every shared case returns exactly the expected ids or project_refs."""
    page = await run_case(kb.db, fn, kwargs)
    assert keys_of(page) == expected, name


async def test_top_degree_signal_values(kb: KnowledgeBase) -> None:
    """Top by degree reports the graph degree, a self-loop counted once."""
    page = await browse.browse_top(kb.db, now=BROWSE_NOW)
    assert [r.signal_value for r in page.items] == BROWSE_EXPECTED_DEGREES


async def test_projects_summaries(kb: KnowledgeBase) -> None:
    """Project summaries match counts, type order and last activity."""
    page = await browse.browse_projects(kb.db)
    got = [
        (p.project_ref, p.entry_count, list(p.counts_by_type.items()), p.last_activity.isoformat())
        for p in page.items
    ]
    expected = [(r, n, list(c.items()), ts) for r, n, c, ts in BROWSE_EXPECTED_PROJECTS]
    assert got == expected


async def test_recent_scanned_and_skipped(kb: KnowledgeBase) -> None:
    """The entry SELECT returns only active, non-superseded rows."""
    page = await browse.browse_recent(kb.db, now=BROWSE_NOW)
    assert page.scanned == 9
    assert page.skipped == 0


async def test_derived_fields(kb: KnowledgeBase) -> None:
    """Effective confidence, half-life, anchor age and expiry are derived per row."""
    stale = await browse.browse_stale(kb.db, now=BROWSE_NOW)
    r3 = _row(stale, "kb-90003")
    assert r3.effective_confidence == round(0.5 * 2 ** (-365 / 90), 4)
    assert r3.half_life_days == 90.0
    assert r3.anchor_age_days == 365.0
    assert r3.expired is False
    r6 = _row(stale, "kb-90006")
    assert r6.effective_confidence == round(0.25 * 2 ** (-30 / 1825), 4)
    assert r6.half_life_days == 1825.0

    alpha = await browse.browse_project(kb.db, "p-alpha", now=BROWSE_NOW)
    r5 = _row(alpha, "kb-90005")
    assert r5.effective_confidence == 1.0
    assert r5.half_life_days is None

    recent = await browse.browse_recent(kb.db, now=BROWSE_NOW)
    r9 = _row(recent, "kb-90009")
    assert r9.expired is True
    assert r9.created_at == datetime(2026, 6, 15, 10, 15, 16, tzinfo=UTC)
    assert _row(recent, "kb-90011").created_at == datetime(2026, 6, 9, 23, 0, tzinfo=UTC)
    assert _row(recent, "kb-90010").expired is False


async def test_paging(kb: KnowledgeBase) -> None:
    """Offset cursors page through the ordered rows."""
    expected = BROWSE_CASES[0][3]
    first = await browse.browse_recent(kb.db, limit=3, now=BROWSE_NOW)
    assert [r.id for r in first.items] == expected[:3]
    assert first.next_cursor == "3"
    assert first.total == 9

    last = await browse.browse_recent(kb.db, limit=3, cursor="6", now=BROWSE_NOW)
    assert [r.id for r in last.items] == ["kb-90003", "kb-90004", "kb-90005"]
    assert last.next_cursor is None

    beyond = await browse.browse_recent(kb.db, cursor="9", now=BROWSE_NOW)
    assert beyond.items == []
    assert beyond.next_cursor is None
    assert beyond.total == 9


@pytest.mark.parametrize("cursor", ["abc", "-1", "٣"])
async def test_invalid_cursor(kb: KnowledgeBase, cursor: str) -> None:
    """Non-ASCII-decimal cursors raise BrowseParamError."""
    with pytest.raises(BrowseParamError):
        await browse.browse_recent(kb.db, cursor=cursor, now=BROWSE_NOW)


@pytest.mark.parametrize("limit", [0, 101])
async def test_invalid_limit(kb: KnowledgeBase, limit: int) -> None:
    """Limits outside 1..100 raise BrowseParamError."""
    with pytest.raises(BrowseParamError):
        await browse.browse_recent(kb.db, limit=limit, now=BROWSE_NOW)
    with pytest.raises(BrowseParamError):
        await browse.browse_projects(kb.db, limit=limit)


async def test_unknown_project(kb: KnowledgeBase) -> None:
    """A project with no active entries raises ProjectNotFound."""
    with pytest.raises(ProjectNotFound):
        await browse.browse_project(kb.db, "p-none", now=BROWSE_NOW)


async def test_unknown_type_is_skipped(kb: KnowledgeBase, caplog: pytest.LogCaptureFixture) -> None:
    """An unknown entry type is skipped, counted and logged once."""
    await kb.db.execute(
        BROWSE_ENTRY_INSERT_SQL,
        (
            "kb-90099",
            "p-alpha",
            "Bogus",
            "Bogus (long)",
            "Details",
            "bogus_type",
            1.0,
            "2026-06-30T23:00:00+00:00",
            "2026-06-30T23:00:00+00:00",
            None,
            None,
            None,
            1,
        ),
    )
    await kb.db.commit()
    caplog.set_level(logging.WARNING, logger="kb_core.browse")
    page = await browse.browse_recent(kb.db, now=BROWSE_NOW)
    assert page.total == 9
    assert "kb-90099" not in [r.id for r in page.items]
    assert page.skipped == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "browse-skip" in warnings[0].getMessage()
    assert "kb-90099" in warnings[0].getMessage()


async def test_bad_timestamp_is_skipped(kb: KnowledgeBase) -> None:
    """An unparseable timestamp is skipped without raising."""
    await kb.db.execute(
        BROWSE_ENTRY_INSERT_SQL,
        (
            "kb-90098",
            "p-alpha",
            "Bad ts",
            "Bad ts (long)",
            "Details",
            "factual_reference",
            1.0,
            "2026-06-30T23:00:00+00:00",
            "2026-06-30T23:00:00+00:00",
            "not-a-date",
            None,
            None,
            1,
        ),
    )
    await kb.db.commit()
    page = await browse.browse_recent(kb.db, now=BROWSE_NOW)
    assert page.total == 9
    assert "kb-90098" not in [r.id for r in page.items]
    assert page.skipped == 1
