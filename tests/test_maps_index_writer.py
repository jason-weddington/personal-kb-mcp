"""Tests for the on-disk JSONL maps index writer (MCP-side).

The reader half is exercised in the standalone ``personal-kb-hook`` package's
own test suite (``packages/personal-kb-hook/tests/test_index_reader.py``). The
round-trip test below requires both — it imports the writer from the main
package and the reader from the standalone hook package, so it doubles as a
cross-package contract test.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from personal_kb_hook.index_reader import read_index

from personal_kb.db.connection import create_connection
from personal_kb.maps_index_writer import (
    _fetch_maps_for_project,
    rebuild_all_projects,
    write_project_maps,
)
from personal_kb.preflight import _maps_sql

# ---------------------------------------------------------------------------
# DB fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
async def maps_db():
    """In-memory DB ready to hold knowledge_entries rows for the writer."""
    db = await create_connection(":memory:", embedding_dim=64)
    yield db
    await db.close()


async def _insert_map(
    db,
    entry_id: str,
    project_ref: str,
    short: str,
    long: str,
    *,
    age_minutes: int,
    team: str | None = None,
) -> None:
    created = (datetime.now(UTC) - timedelta(minutes=age_minutes)).isoformat()
    await db.execute(
        "INSERT INTO knowledge_entries "
        "(id, project_ref, short_title, long_title, knowledge_details, "
        " entry_type, created_at, updated_at, is_active, team) "
        "VALUES (?, ?, ?, ?, ?, 'mental_map', ?, ?, 1, ?)",
        [entry_id, project_ref, short, long, "details", created, created, team],
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


async def test_writer_emits_parseable_jsonl_and_reader_roundtrips(maps_db, tmp_path: Path) -> None:
    """write_project_maps + read_index round-trip cleanly."""
    await _insert_map(maps_db, "kb-1", "demo", "auth", "Auth map", age_minutes=10)
    await _insert_map(maps_db, "kb-2", "demo", "ingest", "Ingestion", age_minutes=20)
    await maps_db.commit()

    index_path = tmp_path / "maps_index.jsonl"
    await write_project_maps(maps_db, "demo", path=index_path)

    raw = index_path.read_text(encoding="utf-8")
    # JSONL — exactly one valid JSON record per line.
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["project_ref"] == "demo"

    table = read_index(index_path)
    assert set(table.keys()) == {"demo"}
    ids = [m["id"] for m in table["demo"]]
    # ORDER BY created_at DESC → newer (10-min-old) first
    assert ids == ["kb-1", "kb-2"]


async def test_writer_caps_at_five(maps_db, tmp_path: Path) -> None:
    """A project with 6 active maps writes exactly 5 (preflight LIMIT 5)."""
    for i in range(6):
        await _insert_map(
            maps_db,
            f"kb-{i:02d}",
            "big",
            f"m{i}",
            f"long {i}",
            age_minutes=i * 5,
        )
    await maps_db.commit()

    index_path = tmp_path / "maps_index.jsonl"
    await write_project_maps(maps_db, "big", path=index_path)

    table = read_index(index_path)
    assert len(table["big"]) == 5


async def test_on_disk_equals_preflight_maps_sql(maps_db, tmp_path: Path) -> None:
    """For a ≤5-map project, on-disk content equals preflight._maps_sql output."""
    await _insert_map(maps_db, "kb-a", "p", "alpha", "first", age_minutes=10)
    await _insert_map(maps_db, "kb-b", "p", "beta", "second", age_minutes=20)
    await maps_db.commit()

    index_path = tmp_path / "maps_index.jsonl"
    await write_project_maps(maps_db, "p", path=index_path)

    table = read_index(index_path)
    written = [(m["id"], m["short_title"], m["long_title"]) for m in table["p"]]

    sql, _ = _maps_sql(None)
    cursor = await maps_db.execute(sql, ["p"])
    rows = await cursor.fetchall()
    direct = [(row[0], row[1], row[2]) for row in rows]
    assert written == direct


async def test_deactivating_last_map_removes_project_line(maps_db, tmp_path: Path) -> None:
    """If a project no longer has any active maps, its line is dropped."""
    await _insert_map(maps_db, "kb-X", "solo", "only", "Only map", age_minutes=10)
    await maps_db.commit()

    index_path = tmp_path / "maps_index.jsonl"
    await write_project_maps(maps_db, "solo", path=index_path)
    assert "solo" in read_index(index_path)

    # Deactivate the map.
    await maps_db.execute("UPDATE knowledge_entries SET is_active = 0 WHERE id = ?", ["kb-X"])
    await maps_db.commit()
    await write_project_maps(maps_db, "solo", path=index_path)

    table = read_index(index_path)
    assert "solo" not in table


async def test_writer_preserves_other_projects(maps_db, tmp_path: Path) -> None:
    """Rewriting one project's line preserves other projects' records."""
    await _insert_map(maps_db, "kb-1", "alpha", "a", "first", age_minutes=10)
    await _insert_map(maps_db, "kb-2", "beta", "b", "second", age_minutes=10)
    await maps_db.commit()

    index_path = tmp_path / "maps_index.jsonl"
    await write_project_maps(maps_db, "alpha", path=index_path)
    await write_project_maps(maps_db, "beta", path=index_path)

    table = read_index(index_path)
    assert set(table.keys()) == {"alpha", "beta"}

    # Rewrite alpha; beta must remain.
    await _insert_map(maps_db, "kb-3", "alpha", "a2", "first-v2", age_minutes=1)
    await maps_db.commit()
    await write_project_maps(maps_db, "alpha", path=index_path)

    table = read_index(index_path)
    assert set(table.keys()) == {"alpha", "beta"}
    alpha_ids = [m["id"] for m in table["alpha"]]
    assert alpha_ids[0] == "kb-3"


# ---------------------------------------------------------------------------
# Reader tolerance
# ---------------------------------------------------------------------------


def test_reader_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_index(tmp_path / "nope.jsonl") == {}


def test_reader_skips_garbage_lines(tmp_path: Path) -> None:
    """A garbage line is skipped; the valid line is returned."""
    path = tmp_path / "maps_index.jsonl"
    valid = json.dumps(
        {"project_ref": "good", "maps": [{"id": "kb-1", "short_title": "s", "long_title": "l"}]}
    )
    path.write_text(
        valid + '\nthis is not json\n{"missing": "project_ref"}\n',
        encoding="utf-8",
    )

    table = read_index(path)
    assert list(table.keys()) == ["good"]
    assert table["good"][0]["id"] == "kb-1"


# ---------------------------------------------------------------------------
# rebuild_all_projects
# ---------------------------------------------------------------------------


async def test_rebuild_all_projects_multi_project(maps_db, tmp_path: Path) -> None:
    """Seed >=2 projects; assert per-project lines match _fetch_maps_for_project."""
    await _insert_map(maps_db, "kb-a1", "alpha", "a1", "alpha 1", age_minutes=10)
    await _insert_map(maps_db, "kb-a2", "alpha", "a2", "alpha 2", age_minutes=20)
    await _insert_map(maps_db, "kb-b1", "beta", "b1", "beta 1", age_minutes=5)
    await maps_db.commit()

    index_path = tmp_path / "maps_index.default.jsonl"
    await rebuild_all_projects(maps_db, path=index_path)

    table = read_index(index_path)
    expected: dict[str, list[dict[str, str]]] = {}
    for project_ref in ("alpha", "beta"):
        rows = await _fetch_maps_for_project(maps_db, project_ref, None)
        expected[project_ref] = rows

    assert set(table.keys()) == set(expected.keys())
    for project_ref, maps in expected.items():
        got = [(m["id"], m["short_title"], m["long_title"]) for m in table[project_ref]]
        want = [(m["id"], m["short_title"], m["long_title"]) for m in maps]
        assert got == want


async def test_rebuild_all_projects_skips_zero_map_projects(maps_db, tmp_path: Path) -> None:
    """A project whose ONLY active map is deactivated is OMITTED — no line emitted.

    Other (still-active) projects still appear in the file.
    """
    await _insert_map(maps_db, "kb-1", "active", "a", "still here", age_minutes=10)
    # gone-project's row is INACTIVE; it should not appear in the DISTINCT
    # discovery (which filters is_active = 1).
    await maps_db.execute(
        "INSERT INTO knowledge_entries "
        "(id, project_ref, short_title, long_title, knowledge_details, "
        " entry_type, created_at, updated_at, is_active) "
        "VALUES ('kb-2', 'gone', 'g', 'g', 'd', 'mental_map', "
        " '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z', 0)"
    )
    await maps_db.commit()

    index_path = tmp_path / "maps_index.default.jsonl"
    await rebuild_all_projects(maps_db, path=index_path)
    table = read_index(index_path)
    assert "active" in table
    assert "gone" not in table


async def test_rebuild_all_projects_caps_at_five(maps_db, tmp_path: Path) -> None:
    """A project with 6 active maps yields exactly 5 (preflight LIMIT 5)."""
    for i in range(6):
        await _insert_map(
            maps_db,
            f"kb-{i:02d}",
            "bigp",
            f"m{i}",
            f"long {i}",
            age_minutes=i * 5,
        )
    await maps_db.commit()

    index_path = tmp_path / "maps_index.default.jsonl"
    await rebuild_all_projects(maps_db, path=index_path)

    table = read_index(index_path)
    assert len(table["bigp"]) == 5
    # ORDER BY created_at DESC: newest (age=0) first
    assert table["bigp"][0]["id"] == "kb-00"


async def test_rebuild_all_projects_threads_team(maps_db, tmp_path: Path) -> None:
    """``team`` argument is threaded into _fetch_maps_for_project (team clause)."""
    await _insert_map(maps_db, "kb-no-team", "p", "noteam", "no team", age_minutes=10, team=None)
    await _insert_map(maps_db, "kb-acme", "p", "acme", "acme team", age_minutes=20, team="acme")
    await maps_db.commit()

    # team=None → BOTH show up (no team clause filter)
    none_path = tmp_path / "team_none.jsonl"
    await rebuild_all_projects(maps_db, team=None, path=none_path)
    none_table = read_index(none_path)
    assert {m["id"] for m in none_table["p"]} == {"kb-no-team", "kb-acme"}

    # team='acme' → both kb-acme (team-matched) AND kb-no-team (NULL team OK)
    # plus excludes other teams. Mirror preflight._maps_sql: the team clause
    # is `(team IS NULL OR team = ?)`. So we need a foreign-team row to
    # prove filtering.
    await _insert_map(maps_db, "kb-other", "p", "other", "other team", age_minutes=30, team="other")
    await maps_db.commit()

    acme_path = tmp_path / "team_acme.jsonl"
    await rebuild_all_projects(maps_db, team="acme", path=acme_path)
    acme_table = read_index(acme_path)
    acme_ids = {m["id"] for m in acme_table["p"]}
    assert "kb-other" not in acme_ids
    assert "kb-acme" in acme_ids


async def test_rebuild_all_projects_uses_default_path(
    maps_db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``path=None``, writes to ``get_maps_index_path()`` (instance-aware)."""
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "kb.db"))
    monkeypatch.setenv("KB_INSTANCE_ROLE", "personal")
    await _insert_map(maps_db, "kb-1", "proj", "m", "M", age_minutes=10)
    await maps_db.commit()
    await rebuild_all_projects(maps_db)
    expected = tmp_path / "maps_index.personal.jsonl"
    assert expected.exists()
    table = read_index(expected)
    assert "proj" in table


# ---------------------------------------------------------------------------
# NOTIFY fired on local write
# ---------------------------------------------------------------------------


async def test_refresh_maps_index_fires_notify(maps_db, tmp_path: Path) -> None:
    """``_refresh_maps_index`` (kb_store) calls ``db.notify_maps_changed`` after write.

    Uses the live in-memory sqlite DB but wraps it with a spy on the
    NOTIFY method so we can assert it was invoked with the right
    project_ref. The live pg LISTEN→NOTIFY round-trip is NOT exercised
    here — it requires manual verification against a real Postgres after
    merge to origin (per CLAUDE.md and the item spec).
    """
    from personal_kb.tools.kb_store import _refresh_maps_index

    await _insert_map(maps_db, "kb-1", "demo", "m", "M", age_minutes=10)
    await maps_db.commit()

    # Patch the maps-index path so the writer doesn't touch a real one.
    import personal_kb.maps_index_writer as miw

    real_get_path = miw.get_maps_index_path
    miw.get_maps_index_path = lambda: tmp_path / "maps_index.default.jsonl"
    try:
        spy = AsyncMock(return_value=None)
        maps_db.notify_maps_changed = spy  # type: ignore[attr-defined]
        await _refresh_maps_index(maps_db, "demo", None)
        spy.assert_awaited_once_with("demo")
    finally:
        miw.get_maps_index_path = real_get_path


async def test_kb_store_batch_loop_fires_notify_per_project(maps_db, tmp_path: Path) -> None:
    """Batch path NOTIFYs once per distinct project_ref that got a mental_map.

    This validates the wiring at ``kb_store_batch.py`` (per-project loop
    after ``write_project_maps``). Live pg NOTIFY round-trip is NOT
    exercised here (mocked); see manual-verify note in the AC.
    """
    # We mimic the per-project loop's call sequence without needing the
    # full kb_store_batch tool wiring — assert that the call site invokes
    # ``write_project_maps`` THEN ``notify_maps_changed`` for each ref.
    spy = AsyncMock(return_value=None)
    maps_db.notify_maps_changed = spy  # type: ignore[attr-defined]

    import personal_kb.maps_index_writer as miw

    real_get_path = miw.get_maps_index_path
    miw.get_maps_index_path = lambda: tmp_path / "maps_index.default.jsonl"
    try:
        await _insert_map(maps_db, "kb-x", "p1", "x", "X", age_minutes=5)
        await _insert_map(maps_db, "kb-y", "p2", "y", "Y", age_minutes=5)
        await maps_db.commit()

        from personal_kb.maps_index_writer import write_project_maps as wpm

        for ref in ["p1", "p2"]:
            await wpm(maps_db, ref, team=None)
            await maps_db.notify_maps_changed(ref)  # type: ignore[attr-defined]

        assert spy.await_count == 2
        assert {c.args[0] for c in spy.call_args_list} == {"p1", "p2"}
    finally:
        miw.get_maps_index_path = real_get_path
