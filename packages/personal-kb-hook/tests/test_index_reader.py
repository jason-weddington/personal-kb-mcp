"""Tests for the on-disk JSONL maps index reader.

The writer-side tests (which exercise ``personal_kb.maps_index_writer``
against a live SQLite DB) stay in the main repo's test suite — the
standalone hook package has no DB dependencies.
"""

import json
from pathlib import Path

from personal_kb_hook.index_reader import read_index


def test_reader_missing_file_returns_empty(tmp_path: Path) -> None:
    """A missing JSONL file → empty mapping (no exception)."""
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


def test_reader_round_trips_well_formed_entry(tmp_path: Path) -> None:
    """A single well-formed JSONL entry parses back to the same shape."""
    path = tmp_path / "maps_index.jsonl"
    record = {
        "project_ref": "demo",
        "maps": [
            {"id": "kb-1", "short_title": "auth", "long_title": "Auth map"},
            {"id": "kb-2", "short_title": "ingest", "long_title": ""},
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    table = read_index(path)
    assert set(table.keys()) == {"demo"}
    assert [(m["id"], m["short_title"], m["long_title"]) for m in table["demo"]] == [
        ("kb-1", "auth", "Auth map"),
        ("kb-2", "ingest", ""),
    ]


def test_reader_uses_default_path_when_none(monkeypatch, tmp_path: Path) -> None:
    """``read_index(None)`` falls back to the env-derived default path."""
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    index_path = db_path.parent / "maps_index.jsonl"
    record = {
        "project_ref": "p",
        "maps": [{"id": "kb-X", "short_title": "x", "long_title": "X map"}],
    }
    index_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    table = read_index()
    assert "p" in table
    assert table["p"][0]["id"] == "kb-X"
