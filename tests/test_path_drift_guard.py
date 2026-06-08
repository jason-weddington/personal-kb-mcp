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
``KB_DB_PATH`` and every supported ``KB_INSTANCE_ROLE`` value.
"""

from __future__ import annotations

import pytest
from personal_kb_hook import paths as hook_paths

from personal_kb import config as main_config


def test_default_maps_index_path_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no KB_DB_PATH/role set, both helpers agree (default role)."""
    monkeypatch.delenv("KB_DB_PATH", raising=False)
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    assert main_config.get_maps_index_path() == hook_paths.get_maps_index_path()
    # Unset role keys to the literal 'default'.
    assert main_config.get_maps_index_path().name == "maps_index.default.jsonl"


def test_custom_kb_db_path_matches(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A custom KB_DB_PATH still yields identical maps_index_path on both sides."""
    custom_db = tmp_path / "weird" / "place" / "knowledge.db"
    monkeypatch.setenv("KB_DB_PATH", str(custom_db))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    assert main_config.get_maps_index_path() == hook_paths.get_maps_index_path()
    assert main_config.get_maps_index_path().name == "maps_index.default.jsonl"


def test_tilde_expansion_kb_db_path_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """A KB_DB_PATH containing ``~`` expands identically on both sides."""
    monkeypatch.setenv("KB_DB_PATH", "~/somewhere/kb.db")
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    main_path = main_config.get_maps_index_path()
    hook_path = hook_paths.get_maps_index_path()
    assert main_path == hook_path
    # And tilde got expanded, not left literal.
    assert "~" not in str(main_path)
    assert main_path.name == "maps_index.default.jsonl"


# --- Per-instance role-keyed paths --------------------------------------------


@pytest.mark.parametrize(
    ("role_env", "expected_filename"),
    [
        (None, "maps_index.default.jsonl"),
        ("", "maps_index.default.jsonl"),
        ("personal", "maps_index.personal.jsonl"),
        ("team", "maps_index.team.jsonl"),
        # KB_INSTANCE_ROLE is case-insensitive — mixed case lowercases.
        ("Team", "maps_index.team.jsonl"),
        ("PERSONAL", "maps_index.personal.jsonl"),
    ],
)
def test_role_keyed_default_db_path(
    monkeypatch: pytest.MonkeyPatch, role_env: str | None, expected_filename: str
) -> None:
    """Role envar → ``maps_index.{role}.jsonl`` filename, both helpers agree."""
    monkeypatch.delenv("KB_DB_PATH", raising=False)
    if role_env is None:
        monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    else:
        monkeypatch.setenv("KB_INSTANCE_ROLE", role_env)

    main_path = main_config.get_maps_index_path()
    hook_path = hook_paths.get_maps_index_path()
    assert main_path == hook_path
    assert main_path.name == expected_filename


@pytest.mark.parametrize(
    ("role_env", "expected_filename"),
    [
        (None, "maps_index.default.jsonl"),
        ("personal", "maps_index.personal.jsonl"),
        ("team", "maps_index.team.jsonl"),
        ("Team", "maps_index.team.jsonl"),
    ],
)
def test_role_keyed_custom_db_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    role_env: str | None,
    expected_filename: str,
) -> None:
    """Role x custom KB_DB_PATH → both helpers produce the exact same path.

    Covers the critic-critical fix that the base directory comes from
    ``KB_DB_PATH``, NOT a hardcoded ``~/.local/share/personal_kb/`` —
    so a custom DB directory is honored on both sides.
    """
    custom_db = tmp_path / "custom" / "kb.db"
    monkeypatch.setenv("KB_DB_PATH", str(custom_db))
    if role_env is None:
        monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    else:
        monkeypatch.setenv("KB_INSTANCE_ROLE", role_env)

    main_path = main_config.get_maps_index_path()
    hook_path = hook_paths.get_maps_index_path()
    assert main_path == hook_path
    assert main_path == custom_db.parent / expected_filename


@pytest.mark.parametrize("session_id", ["abc", "session-with-hyphens", "1234-5678", ""])
def test_hook_scratch_path_matches(session_id: str) -> None:
    """The per-session scratch path is identical on both sides for many session_ids."""
    assert main_config.get_hook_scratch_path(session_id) == hook_paths.get_hook_scratch_path(
        session_id
    )
