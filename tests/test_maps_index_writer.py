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

import pytest
from personal_kb_hook.index_reader import read_index

from personal_kb.db.connection import create_connection
from personal_kb.maps_index_writer import write_project_maps
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
    db, entry_id: str, project_ref: str, short: str, long: str, *, age_minutes: int
) -> None:
    created = (datetime.now(UTC) - timedelta(minutes=age_minutes)).isoformat()
    await db.execute(
        "INSERT INTO knowledge_entries "
        "(id, project_ref, short_title, long_title, knowledge_details, "
        " entry_type, created_at, updated_at, is_active) "
        "VALUES (?, ?, ?, ?, ?, 'mental_map', ?, ?, 1)",
        [entry_id, project_ref, short, long, "details", created, created],
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
