"""Tests for the per-session suppression scratch file."""

from pathlib import Path

from personal_kb.hook.suppression import mark_emitted, should_emit


def test_first_surface_emits(tmp_path: Path) -> None:
    """No prior scratch + non-empty map_ids → emit."""
    scratch = tmp_path / "scratch.json"
    assert should_emit(
        session_id="s1",
        scope="proj",
        map_ids=["kb-1", "kb-2"],
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
    map_ids = ["kb-1", "kb-2"]
    mark_emitted(session_id="s1", scope="proj", map_ids=map_ids, scratch_path=scratch)
    assert not should_emit(
        session_id="s1",
        scope="proj",
        map_ids=map_ids,
        source=None,
        scratch_path=scratch,
    )


def test_subset_of_surfaced_silent(tmp_path: Path) -> None:
    """A strict subset of already-surfaced ids on the same scope is silent."""
    scratch = tmp_path / "scratch.json"
    mark_emitted(
        session_id="s1", scope="proj", map_ids=["kb-1", "kb-2", "kb-3"], scratch_path=scratch
    )
    assert not should_emit(
        session_id="s1",
        scope="proj",
        map_ids=["kb-1"],
        source=None,
        scratch_path=scratch,
    )


def test_scope_drift_re_emits(tmp_path: Path) -> None:
    """A different scope on the same session re-emits."""
    scratch = tmp_path / "scratch.json"
    mark_emitted(session_id="s1", scope="proj-a", map_ids=["kb-1"], scratch_path=scratch)
    assert should_emit(
        session_id="s1",
        scope="proj-b",
        map_ids=["kb-9"],
        source=None,
        scratch_path=scratch,
    )


def test_new_ids_re_emits(tmp_path: Path) -> None:
    """Same scope but a new id (not a subset) re-emits."""
    scratch = tmp_path / "scratch.json"
    mark_emitted(session_id="s1", scope="proj", map_ids=["kb-1"], scratch_path=scratch)
    assert should_emit(
        session_id="s1",
        scope="proj",
        map_ids=["kb-1", "kb-2"],
        source=None,
        scratch_path=scratch,
    )


def test_source_compact_bypasses_subset_check(tmp_path: Path) -> None:
    """source=compact re-emits even when the ids are already surfaced."""
    scratch = tmp_path / "scratch.json"
    mark_emitted(session_id="s1", scope="proj", map_ids=["kb-1"], scratch_path=scratch)
    assert should_emit(
        session_id="s1",
        scope="proj",
        map_ids=["kb-1"],
        source="compact",
        scratch_path=scratch,
    )


def test_compact_then_mark_updates_scratch(tmp_path: Path) -> None:
    """After a compact bypass, mark_emitted should union new ids in."""
    scratch = tmp_path / "scratch.json"
    mark_emitted(session_id="s1", scope="proj", map_ids=["kb-1"], scratch_path=scratch)
    mark_emitted(session_id="s1", scope="proj", map_ids=["kb-1", "kb-2"], scratch_path=scratch)
    # Subsequent same scope+ids must now be silent (subset of surfaced)
    assert not should_emit(
        session_id="s1",
        scope="proj",
        map_ids=["kb-1", "kb-2"],
        source=None,
        scratch_path=scratch,
    )


def test_missing_scratch_treated_as_fresh(tmp_path: Path) -> None:
    """A scratch path that does not exist behaves like fresh state."""
    scratch = tmp_path / "does-not-exist.json"
    assert should_emit(
        session_id="s1",
        scope="proj",
        map_ids=["kb-1"],
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
        map_ids=["kb-1"],
        source=None,
        scratch_path=scratch,
    )
