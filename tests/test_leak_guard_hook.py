"""The pre-push leak-guard hook runs the maintainer-local guard before origin pushes."""

import os
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / "scripts" / "run_leak_guard.sh"


def _guard(repo: Path, exit_code: int, *, executable: bool = True) -> Path:
    log = repo / "guard.log"
    script = repo / "release.guard.local.sh"
    script.write_text(f'#!/usr/bin/env bash\necho "args=$*" >> "{log}"\nexit {exit_code}\n')
    script.chmod(0o755 if executable else 0o644)
    return log


def _run(repo: Path, remote: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PRE_COMMIT_REMOTE_NAME=remote)
    return subprocess.run(  # noqa: S603
        ["/bin/bash", str(HOOK)], cwd=repo, env=env, capture_output=True, text=True, check=False
    )


def test_absent_guard_is_skipped(tmp_path: Path) -> None:
    assert _run(tmp_path, "origin").returncode == 0


def test_non_executable_guard_is_skipped(tmp_path: Path) -> None:
    log = _guard(tmp_path, 1, executable=False)
    assert _run(tmp_path, "origin").returncode == 0
    assert not log.exists()


def test_clean_guard_allows_the_push_in_tree_only_mode(tmp_path: Path) -> None:
    log = _guard(tmp_path, 0)
    assert _run(tmp_path, "origin").returncode == 0
    assert log.read_text() == "args=--tree-only\n"


@pytest.mark.parametrize("code", [1, 2, 7])
def test_any_guard_failure_blocks_the_push(tmp_path: Path, code: int) -> None:
    _guard(tmp_path, code)
    r = _run(tmp_path, "origin")
    assert r.returncode == code
    assert f"Push blocked by the leak guard (exit {code})" in r.stderr


def test_github_push_is_left_to_release_sh(tmp_path: Path) -> None:
    log = _guard(tmp_path, 1)
    assert _run(tmp_path, "github").returncode == 0
    assert not log.exists()


def test_hook_is_wired_as_a_pre_push_stage() -> None:
    config = (HOOK.parent.parent / ".pre-commit-config.yaml").read_text()
    assert "entry: scripts/run_leak_guard.sh" in config
    assert "release.guard.local.sh" in (HOOK.parent.parent / ".gitignore").read_text()
