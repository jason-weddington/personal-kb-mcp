"""A stored hints.resolution survives hints-only updates (shallow hint merge)."""

from __future__ import annotations

from typing import Any

import pytest

from kb_core import create_sqlite
from kb_core.models.entry import EntryType
from kb_core.supersession import add_supersedes_hint

pytestmark = pytest.mark.asyncio

RESOLUTION = {
    "corrected_fact": "use the script",
    "cue": {"tool": "Bash", "target_class": "git remote"},
    "provenance": {"capture": "deliberate", "grounding": "asserted"},
}


@pytest.fixture
async def kb_entry(tmp_path: Any) -> Any:
    kb = await create_sqlite(tmp_path / "kb.db")
    entry = await kb.store(
        short_title="res",
        long_title="Resolution carrier",
        knowledge_details="Body.",
        entry_type=EntryType.PATTERN_CONVENTION,
        hints={"resolution": RESOLUTION},
        enrich=False,
    )
    try:
        yield kb, entry
    finally:
        await kb.close()


async def test_hints_only_update_keeps_resolution(kb_entry: Any) -> None:
    kb, entry = kb_entry
    await kb.update(entry.id, hints={"tags_note": 1}, change_reason="r", enrich=False)
    got = await kb.get(entry.id)
    assert got.hints["resolution"] == RESOLUTION
    assert got.hints["tags_note"] == 1


async def test_supersedes_hint_keeps_resolution(kb_entry: Any) -> None:
    kb, entry = kb_entry
    other = await kb.store(
        short_title="old",
        long_title="Old entry",
        knowledge_details="Old.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        enrich=False,
    )
    await add_supersedes_hint(kb.db, entry.id, other.id)
    await kb.db.commit()
    got = await kb.get(entry.id)
    assert got.hints["resolution"] == RESOLUTION
    assert got.hints["supersedes"] == [other.id]
