"""Drift guard: main-package path helpers vs standalone hook helpers.

The standalone ``personal-kb-hook`` package can no longer import
``personal_kb.config``. It therefore duplicates two on-disk path helpers
in ``personal_kb_hook.paths``:

* ``get_maps_index_path()`` — the JSONL maps index location.
* ``get_hook_scratch_path(session_id)`` — the per-session scratch file.

If those two implementations drift, the MCP-side writer and the CLI-side
reader would silently disagree on the file location and the hook would
silently stop seeing new maps. This test imports both and asserts they
produce identical paths for representative inputs, including a custom
``KB_DB_PATH``.
"""

from __future__ import annotations

import pytest
from personal_kb_hook import paths as hook_paths

from personal_kb import config as main_config


def test_default_maps_index_path_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no KB_DB_PATH set, both helpers agree on the default location."""
    monkeypatch.delenv("KB_DB_PATH", raising=False)
    assert main_config.get_maps_index_path() == hook_paths.get_maps_index_path()


def test_custom_kb_db_path_matches(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A custom KB_DB_PATH still yields identical maps_index_path on both sides."""
    custom_db = tmp_path / "weird" / "place" / "knowledge.db"
    monkeypatch.setenv("KB_DB_PATH", str(custom_db))
    assert main_config.get_maps_index_path() == hook_paths.get_maps_index_path()


def test_tilde_expansion_kb_db_path_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """A KB_DB_PATH containing ``~`` expands identically on both sides."""
    monkeypatch.setenv("KB_DB_PATH", "~/somewhere/kb.db")
    main_path = main_config.get_maps_index_path()
    hook_path = hook_paths.get_maps_index_path()
    assert main_path == hook_path
    # And tilde got expanded, not left literal.
    assert "~" not in str(main_path)


@pytest.mark.parametrize("session_id", ["abc", "session-with-hyphens", "1234-5678", ""])
def test_hook_scratch_path_matches(session_id: str) -> None:
    """The per-session scratch path is identical on both sides for many session_ids."""
    assert main_config.get_hook_scratch_path(session_id) == hook_paths.get_hook_scratch_path(
        session_id
    )
