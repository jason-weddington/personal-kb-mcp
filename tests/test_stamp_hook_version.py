"""Tests for ``scripts/stamp_hook_version.py``.

Locks in the contract that the standalone ``personal-kb-hook`` package's
``[project] version`` tracks the main repo's version, exactly. Called from
``release.sh`` after semantic-release bumps the root ``pyproject.toml``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "stamp_hook_version.py"


def _load_module():  # type: ignore[no-untyped-def]
    """Load ``scripts/stamp_hook_version.py`` as a module without packaging it."""
    spec = importlib.util.spec_from_file_location("stamp_hook_version", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["stamp_hook_version"] = module
    spec.loader.exec_module(module)
    return module


stamp_hook_version_mod = _load_module()


ROOT_PYPROJECT_TEMPLATE = """\
[project]
name = "personal-kb"
version = "{version}"
description = "test"

[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
"""

HOOK_PYPROJECT_TEMPLATE = """\
[project]
name = "personal-kb-hook"
version = "{version}"
description = "test hook"
dependencies = []
"""


def _write_pair(tmp_path: Path, root_version: str, hook_version: str) -> tuple[Path, Path]:
    root = tmp_path / "pyproject.toml"
    hook_dir = tmp_path / "packages" / "personal-kb-hook"
    hook_dir.mkdir(parents=True)
    hook = hook_dir / "pyproject.toml"
    root.write_text(ROOT_PYPROJECT_TEMPLATE.format(version=root_version))
    hook.write_text(HOOK_PYPROJECT_TEMPLATE.format(version=hook_version))
    return root, hook


def test_read_project_version_returns_top_level_version(tmp_path: Path) -> None:
    root, _ = _write_pair(tmp_path, "1.2.3", "0.0.1")
    assert stamp_hook_version_mod.read_project_version(root) == "1.2.3"


def test_stamp_updates_hook_to_new_version(tmp_path: Path) -> None:
    _, hook = _write_pair(tmp_path, "1.2.3", "0.0.1")
    changed = stamp_hook_version_mod.stamp_hook_version(hook, "1.2.3")
    assert changed is True
    assert 'version = "1.2.3"' in hook.read_text()
    assert 'version = "0.0.1"' not in hook.read_text()


def test_stamp_is_noop_when_already_matching(tmp_path: Path) -> None:
    _, hook = _write_pair(tmp_path, "1.2.3", "1.2.3")
    changed = stamp_hook_version_mod.stamp_hook_version(hook, "1.2.3")
    assert changed is False
    assert 'version = "1.2.3"' in hook.read_text()


def test_stamp_preserves_other_content(tmp_path: Path) -> None:
    _, hook = _write_pair(tmp_path, "1.2.3", "0.0.1")
    original = hook.read_text()
    stamp_hook_version_mod.stamp_hook_version(hook, "1.2.3")
    new = hook.read_text()
    # Only the version line should have changed.
    assert new.replace('version = "1.2.3"', 'version = "0.0.1"') == original


def test_stamp_only_replaces_first_version_line(tmp_path: Path) -> None:
    """Defensive: if a future maintainer adds another ``version = "..."`` line
    (e.g. inside a comment block or another table), the function must reject
    rather than silently rewrite the wrong one.
    """
    hook = tmp_path / "hook.toml"
    hook.write_text(
        '[project]\nversion = "0.0.1"\ndescription = "x"\n\n[other]\nversion = "9.9.9"\n'
    )
    with pytest.raises(ValueError, match="exactly one"):
        stamp_hook_version_mod.stamp_hook_version(hook, "1.2.3")


def test_stamp_rejects_missing_version_line(tmp_path: Path) -> None:
    hook = tmp_path / "hook.toml"
    hook.write_text('[project]\nname = "x"\n')
    with pytest.raises(ValueError, match="exactly one"):
        stamp_hook_version_mod.stamp_hook_version(hook, "1.2.3")


def test_read_rejects_pyproject_without_version(tmp_path: Path) -> None:
    root = tmp_path / "pyproject.toml"
    root.write_text('[project]\nname = "personal-kb"\n')
    with pytest.raises(ValueError, match="No top-level"):
        stamp_hook_version_mod.read_project_version(root)


def test_main_end_to_end_with_explicit_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, hook = _write_pair(tmp_path, "2.0.0", "0.1.0")
    svc = tmp_path / "svc.toml"
    svc.write_text(hook.read_text())
    rc = stamp_hook_version_mod.main(
        [
            "--root-pyproject",
            str(root),
            "--hook-pyproject",
            str(hook),
            "--service-pyproject",
            str(svc),
        ]
    )
    assert rc == 0
    assert 'version = "2.0.0"' in hook.read_text()
    assert 'version = "2.0.0"' in svc.read_text()
    out = capsys.readouterr().out
    assert "2.0.0" in out


def test_main_is_noop_when_already_matching(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, hook = _write_pair(tmp_path, "2.0.0", "2.0.0")
    svc = tmp_path / "svc.toml"
    svc.write_text(hook.read_text())
    rc = stamp_hook_version_mod.main(
        [
            "--root-pyproject",
            str(root),
            "--hook-pyproject",
            str(hook),
            "--service-pyproject",
            str(svc),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "no change" in out.lower()


def test_repo_root_and_hook_already_in_sync() -> None:
    """The committed repo should already have the hook version matching root.

    If this fails, either the helper was not run before the last release, or
    someone updated one pyproject without the other. Re-run ``release.sh`` or
    ``python3 scripts/stamp_hook_version.py`` to resync.
    """
    root_version = stamp_hook_version_mod.read_project_version(REPO_ROOT / "pyproject.toml")
    hook_version = stamp_hook_version_mod.read_project_version(
        REPO_ROOT / "packages" / "personal-kb-hook" / "pyproject.toml"
    )
    assert hook_version == root_version, (
        f"personal-kb-hook is at {hook_version} but main repo is at "
        f"{root_version}; run scripts/stamp_hook_version.py to resync."
    )


def test_pin_internal_deps_pins_project_deps_only(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "personal-kb-web-service"\nversion = "1.0.0"\n'
        'dependencies = [\n    "kb-core[postgres,ingest]",\n    "fastapi>=0.1",\n]\n'
        '[project.optional-dependencies]\nx = ["personal-kb-web-service[postgres]==0.1.0"]\n'
        '[dependency-groups]\ndev = ["kb-core", "personal-kb-hook"]\n'
    )
    assert stamp_hook_version_mod.pin_internal_deps(pyproject, "2.0.0") is True
    text = pyproject.read_text()
    assert 'name = "personal-kb-web-service"\n' in text
    assert '"kb-core[postgres,ingest]==2.0.0"' in text
    assert '"personal-kb-web-service[postgres]==2.0.0"' in text
    assert '"fastapi>=0.1"' in text
    assert 'dev = ["kb-core", "personal-kb-hook"]' in text
    assert stamp_hook_version_mod.pin_internal_deps(pyproject, "2.0.0") is False
