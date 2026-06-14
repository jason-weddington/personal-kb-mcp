"""Tests for the per-session suppression scratch file.

Post-P1: ``should_emit`` / ``mark_emitted`` operate over
:class:`personal_kb_hook.index_reader.MapKey` (the typed
``(label, id)`` :class:`typing.NamedTuple`), and the on-disk
``surfaced_map_ids`` shape is a sorted list of two-element
``[label, id]`` lists. A pre-P1 bare-string element back-parses tolerantly
to ``MapKey(label='personal', id=<s>)`` so an upgraded session continues
suppressing without losing entries.
"""

import json
from pathlib import Path

from personal_kb_hook.index_reader import MapKey
from personal_kb_hook.suppression import (
    ScratchState,
    _read_scratch,
    _write_scratch,
    mark_emitted,
    should_emit,
)


def _mk(label: str, ident: str) -> MapKey:
    return MapKey(label=label, id=ident)


def test_first_surface_emits(tmp_path: Path) -> None:
    """No prior scratch + non-empty map_ids → emit."""
    scratch = tmp_path / "scratch.json"
    assert should_emit(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("personal", "kb-1"), _mk("personal", "kb-2")],
        source=None,
        scratch_path=scratch,
    )


def test_no_maps_is_silent(tmp_path: Path) -> None:
    """Empty map_ids → silent regardless of state."""
    scratch = tmp_path / "scratch.json"
    assert not should_emit(
        session_id="s1",
        scope="proj",
        map_ids=[],
        source=None,
        scratch_path=scratch,
    )


def test_duplicate_same_scope_same_ids_silent(tmp_path: Path) -> None:
    """After mark_emitted, the same scope+ids should not re-emit."""
    scratch = tmp_path / "scratch.json"
    map_ids = [_mk("personal", "kb-1"), _mk("personal", "kb-2")]
    mark_emitted(session_id="s1", scope="proj", map_ids=map_ids, scratch_path=scratch)
    assert not should_emit(
        session_id="s1",
        scope="proj",
        map_ids=map_ids,
        source=None,
        scratch_path=scratch,
    )


def test_subset_of_surfaced_silent(tmp_path: Path) -> None:
    """A strict subset of already-surfaced keys on the same scope is silent."""
    scratch = tmp_path / "scratch.json"
    mark_emitted(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("personal", "kb-1"), _mk("personal", "kb-2"), _mk("personal", "kb-3")],
        scratch_path=scratch,
    )
    assert not should_emit(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("personal", "kb-1")],
        source=None,
        scratch_path=scratch,
    )


def test_scope_drift_re_emits(tmp_path: Path) -> None:
    """A different scope on the same session re-emits."""
    scratch = tmp_path / "scratch.json"
    mark_emitted(
        session_id="s1", scope="proj-a", map_ids=[_mk("personal", "kb-1")], scratch_path=scratch
    )
    assert should_emit(
        session_id="s1",
        scope="proj-b",
        map_ids=[_mk("personal", "kb-9")],
        source=None,
        scratch_path=scratch,
    )


def test_new_ids_re_emits(tmp_path: Path) -> None:
    """Same scope but a new key (not a subset) re-emits."""
    scratch = tmp_path / "scratch.json"
    mark_emitted(
        session_id="s1", scope="proj", map_ids=[_mk("personal", "kb-1")], scratch_path=scratch
    )
    assert should_emit(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("personal", "kb-1"), _mk("personal", "kb-2")],
        source=None,
        scratch_path=scratch,
    )


def test_source_compact_bypasses_subset_check(tmp_path: Path) -> None:
    """source=compact re-emits even when the keys are already surfaced."""
    scratch = tmp_path / "scratch.json"
    mark_emitted(
        session_id="s1", scope="proj", map_ids=[_mk("personal", "kb-1")], scratch_path=scratch
    )
    assert should_emit(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("personal", "kb-1")],
        source="compact",
        scratch_path=scratch,
    )


def test_compact_then_mark_updates_scratch(tmp_path: Path) -> None:
    """After a compact bypass, mark_emitted should union new keys in."""
    scratch = tmp_path / "scratch.json"
    mark_emitted(
        session_id="s1", scope="proj", map_ids=[_mk("personal", "kb-1")], scratch_path=scratch
    )
    mark_emitted(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("personal", "kb-1"), _mk("personal", "kb-2")],
        scratch_path=scratch,
    )
    # Subsequent same scope+ids must now be silent (subset of surfaced)
    assert not should_emit(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("personal", "kb-1"), _mk("personal", "kb-2")],
        source=None,
        scratch_path=scratch,
    )


def test_missing_scratch_treated_as_fresh(tmp_path: Path) -> None:
    """A scratch path that does not exist behaves like fresh state."""
    scratch = tmp_path / "does-not-exist.json"
    assert should_emit(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("personal", "kb-1")],
        source=None,
        scratch_path=scratch,
    )


def test_corrupt_scratch_treated_as_fresh(tmp_path: Path) -> None:
    """A corrupt scratch file does not raise; behaves like fresh state."""
    scratch = tmp_path / "scratch.json"
    scratch.write_text("this is { not valid json", encoding="utf-8")
    assert should_emit(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("personal", "kb-1")],
        source=None,
        scratch_path=scratch,
    )


# ---------------------------------------------------------------------------
# MapKey-typed identity: distinct labels with the same id do NOT collide
# ---------------------------------------------------------------------------


def test_same_id_different_labels_do_not_collide(tmp_path: Path) -> None:
    """``MapKey(label='personal', id='kb-1')`` is NOT a subset of
    ``MapKey(label='team', id='kb-1')`` — same id under a different KB
    label is treated as a brand-new key, exactly what protects two KBs
    that share an id namespace from suppressing each other.
    """
    scratch = tmp_path / "scratch.json"
    mark_emitted(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("team", "kb-1")],
        scratch_path=scratch,
    )
    # personal/kb-1 is a NEW key even though the id matches an existing
    # team/kb-1 — should re-emit.
    assert should_emit(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("personal", "kb-1")],
        source=None,
        scratch_path=scratch,
    )


# ---------------------------------------------------------------------------
# On-disk shape: [label, id] pair list, write-then-read round-trip
# ---------------------------------------------------------------------------


def test_on_disk_shape_is_label_id_pair_list(tmp_path: Path) -> None:
    """The on-disk surfaced_map_ids value is a sorted list of [label, id] lists."""
    scratch = tmp_path / "scratch.json"
    mark_emitted(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("personal", "kb-2"), _mk("team", "kb-1")],
        scratch_path=scratch,
    )
    raw = json.loads(scratch.read_text(encoding="utf-8"))
    # Deterministic sort by (label, id)
    assert raw["surfaced_map_ids"] == [["personal", "kb-2"], ["team", "kb-1"]]


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    """_write_scratch followed by _read_scratch yields an equal set[MapKey]."""
    scratch = tmp_path / "scratch.json"
    keys = {
        _mk("personal", "kb-1"),
        _mk("team", "kb-1"),
        _mk("team", "kb-42"),
    }
    state_in = ScratchState(last_scope="proj-x", surfaced_map_ids=set(keys))
    _write_scratch(scratch, state_in)
    state_out = _read_scratch(scratch)
    assert state_out.last_scope == "proj-x"
    assert state_out.surfaced_map_ids == keys


# ---------------------------------------------------------------------------
# Legacy bare-string back-parse: pre-P1 scratch files still suppress correctly
# ---------------------------------------------------------------------------


def test_legacy_bare_string_ids_back_parse_to_personal_label(tmp_path: Path) -> None:
    """A pre-P1 scratch file with bare-string ids reads back as
    ``MapKey(label='personal', id=<s>)`` — matching the legacy roster
    label so an in-flight upgraded session continues to suppress.
    """
    scratch = tmp_path / "scratch.json"
    # Simulate a pre-P1 scratch file:
    legacy_payload = {
        "last_scope": "proj",
        "surfaced_map_ids": ["kb-1", "kb-2"],
    }
    scratch.write_text(json.dumps(legacy_payload), encoding="utf-8")
    # The same keys, post-migration, are now suppressed under the
    # 'personal' legacy label:
    assert not should_emit(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("personal", "kb-1"), _mk("personal", "kb-2")],
        source=None,
        scratch_path=scratch,
    )
    # A team-labelled key with the SAME id is treated as new — the legacy
    # entries were promoted to 'personal', not to every label.
    assert should_emit(
        session_id="s1",
        scope="proj",
        map_ids=[_mk("team", "kb-1")],
        source=None,
        scratch_path=scratch,
    )


def test_malformed_pair_elements_skipped(tmp_path: Path) -> None:
    """Malformed list elements (wrong length, non-str fields, empties) are skipped."""
    scratch = tmp_path / "scratch.json"
    payload = {
        "last_scope": "proj",
        "surfaced_map_ids": [
            ["personal", "kb-1"],  # OK
            ["", "kb-2"],  # empty label — skipped
            ["team", ""],  # empty id — skipped
            ["team", 42],  # non-str id — skipped
            [1, "kb-3"],  # non-str label — skipped
            ["one", "two", "three"],  # wrong length — skipped
            "kb-legacy",  # legacy bare string — kept as personal/kb-legacy
        ],
    }
    scratch.write_text(json.dumps(payload), encoding="utf-8")
    state = _read_scratch(scratch)
    assert state.surfaced_map_ids == {
        _mk("personal", "kb-1"),
        _mk("personal", "kb-legacy"),
    }
