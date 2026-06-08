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
    """``read_index(None)`` falls back to the env-derived role-keyed file.

    With KB_INSTANCE_ROLE unset the default file is
    ``maps_index.default.jsonl``, which the ``maps_index.*.jsonl`` glob matches.
    """
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    index_path = db_path.parent / "maps_index.default.jsonl"
    record = {
        "project_ref": "p",
        "maps": [{"id": "kb-X", "short_title": "x", "long_title": "X map"}],
    }
    index_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    table = read_index()
    assert "p" in table
    assert table["p"][0]["id"] == "kb-X"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Helper: write a list of JSON dicts as JSONL."""
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_glob_merge_two_files_distinct_projects(monkeypatch, tmp_path: Path) -> None:
    """Two per-instance files in one dir → both project_refs resolved."""
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)

    _write_jsonl(
        db_path.parent / "maps_index.default.jsonl",
        [
            {
                "project_ref": "personal-only",
                "maps": [{"id": "kb-a", "short_title": "a", "long_title": "A"}],
            }
        ],
    )
    _write_jsonl(
        db_path.parent / "maps_index.team.jsonl",
        [
            {
                "project_ref": "team-only",
                "maps": [{"id": "kb-b", "short_title": "b", "long_title": "B"}],
            }
        ],
    )

    table = read_index()
    assert set(table.keys()) == {"personal-only", "team-only"}
    assert table["personal-only"][0]["id"] == "kb-a"
    assert table["team-only"][0]["id"] == "kb-b"


def test_glob_merge_collision_team_wins(monkeypatch, tmp_path: Path) -> None:
    """Same project_ref in both files → team's maps win (last-wins, sorted).

    ``maps_index.team.jsonl`` sorts after ``maps_index.default.jsonl``,
    so the team file's maps overwrite the default file's maps on
    collision.
    """
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)

    shared_ref = "shared"
    _write_jsonl(
        db_path.parent / "maps_index.default.jsonl",
        [
            {
                "project_ref": shared_ref,
                "maps": [{"id": "kb-DEFAULT", "short_title": "d", "long_title": "Default"}],
            }
        ],
    )
    _write_jsonl(
        db_path.parent / "maps_index.team.jsonl",
        [
            {
                "project_ref": shared_ref,
                "maps": [{"id": "kb-TEAM", "short_title": "t", "long_title": "Team"}],
            }
        ],
    )

    table = read_index()
    assert list(table.keys()) == [shared_ref]
    # Team file's maps win — kb-TEAM is the only entry.
    ids = [m["id"] for m in table[shared_ref]]
    assert ids == ["kb-TEAM"]


def test_glob_merge_legacy_suffixless_excluded(monkeypatch, tmp_path: Path) -> None:
    """Legacy ``maps_index.jsonl`` (no role suffix) is EXCLUDED by the glob.

    A frozen old-build ``maps_index.jsonl`` sorts AFTER
    ``maps_index.default.jsonl`` and would shadow it under last-wins; the
    role-suffix-required glob (``maps_index.*.jsonl``) keeps it out. With only
    the legacy file present, the result is empty.
    """
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)

    _write_jsonl(
        db_path.parent / "maps_index.jsonl",
        [
            {
                "project_ref": "legacy",
                "maps": [{"id": "kb-L", "short_title": "L", "long_title": "Legacy"}],
            }
        ],
    )

    table = read_index()
    assert "legacy" not in table
    assert table == {}


def test_glob_merge_explicit_path_unchanged(tmp_path: Path) -> None:
    """``read_index(explicit_path)`` reads ONLY that file (no globbing)."""
    other = tmp_path / "maps_index.team.jsonl"
    target = tmp_path / "maps_index.default.jsonl"

    # Write to both — call should only read 'target'.
    team_map = {"id": "kb-T", "short_title": "T", "long_title": ""}
    default_map = {"id": "kb-D", "short_title": "D", "long_title": ""}
    _write_jsonl(other, [{"project_ref": "team", "maps": [team_map]}])
    _write_jsonl(target, [{"project_ref": "default", "maps": [default_map]}])

    table = read_index(target)
    assert list(table.keys()) == ["default"]


def test_glob_merge_missing_dir_returns_empty(monkeypatch, tmp_path: Path) -> None:
    """A missing maps-index dir yields ``{}`` — never raises."""
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "does-not-exist" / "kb.db"))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    assert read_index() == {}
