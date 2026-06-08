"""Tests for the .kb_project walk-up resolver."""

from pathlib import Path

from personal_kb_hook.resolver import resolve_project


def test_nested_subdir_resolves_repo_root(tmp_path: Path) -> None:
    """A deeply nested cwd resolves to the .kb_project at the repo root."""
    (tmp_path / ".kb_project").write_text("my-project\n", encoding="utf-8")
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert resolve_project(str(nested)) == "my-project"


def test_no_kb_project_anywhere_returns_none(tmp_path: Path) -> None:
    """No .kb_project on the walk → None."""
    nested = tmp_path / "x" / "y"
    nested.mkdir(parents=True)
    assert resolve_project(str(nested)) is None


def test_first_line_is_comment_returns_next_real_line(tmp_path: Path) -> None:
    """A # comment is skipped; the next non-blank line wins."""
    (tmp_path / ".kb_project").write_text(
        "# this is a comment\n\nactual-project\nignored\n", encoding="utf-8"
    )
    assert resolve_project(str(tmp_path)) == "actual-project"


def test_empty_file_returns_none(tmp_path: Path) -> None:
    """An empty .kb_project → None."""
    (tmp_path / ".kb_project").write_text("", encoding="utf-8")
    assert resolve_project(str(tmp_path)) is None


def test_comment_only_file_returns_none(tmp_path: Path) -> None:
    """A file with only comments → None."""
    (tmp_path / ".kb_project").write_text("# only\n# comments\n", encoding="utf-8")
    assert resolve_project(str(tmp_path)) is None


def test_cwd_none_returns_none() -> None:
    """A falsy cwd → None."""
    assert resolve_project(None) is None
    assert resolve_project("") is None


def test_strips_surrounding_whitespace(tmp_path: Path) -> None:
    """The project_ref is stripped."""
    (tmp_path / ".kb_project").write_text("   spaced-project   \n", encoding="utf-8")
    assert resolve_project(str(tmp_path)) == "spaced-project"


def test_kb_project_at_cwd_itself(tmp_path: Path) -> None:
    """The walk starts at cwd; a .kb_project there is found."""
    (tmp_path / ".kb_project").write_text("here\n", encoding="utf-8")
    assert resolve_project(str(tmp_path)) == "here"


def test_unreadable_kb_project_returns_none(tmp_path: Path) -> None:
    """An unreadable .kb_project does not raise."""
    marker = tmp_path / ".kb_project"
    marker.mkdir()  # directory at the marker path → read_text raises OSError
    assert resolve_project(str(tmp_path)) is None
