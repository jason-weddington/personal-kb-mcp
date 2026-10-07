"""Tests for scripts/run_unless_docs_only.sh (fail-safe docs-only wrapper)."""

# ruff: noqa: S603, S607

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_unless_docs_only.sh"
ZEROS = "0" * 40


def _git(repo: Path, *args: str) -> str:
    env = {
        "PATH": "/usr/bin:/bin",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(repo),
    }
    out = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _commit(repo: Path, files: list[str]) -> str:
    for f in files:
        p = repo / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(p.read_text() + "x\n" if p.exists() else "x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    _git(tmp_path, "init", "-q", "-b", "main")
    base = _commit(tmp_path, ["seed.txt"])
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", base)
    return tmp_path, base


def _run(repo_dir: Path, frm: str | None, to: str | None):
    env = {"PATH": "/usr/bin:/bin"}
    if frm is not None:
        env["PRE_COMMIT_FROM_REF"] = frm
    if to is not None:
        env["PRE_COMMIT_TO_REF"] = to
    return subprocess.run(
        [str(SCRIPT), "sh", "-c", "echo RAN; exit 3"],
        cwd=repo_dir,
        env=env,
        capture_output=True,
        text=True,
    )


def test_docs_only_range_skips(repo):
    d, base = repo
    head = _commit(d, ["README.md", "proposals/x.md"])
    r = _run(d, base, head)
    assert r.returncode == 0
    assert "docs-only push" in r.stdout
    assert "RAN" not in r.stdout


def test_mixed_range_runs(repo):
    d, base = repo
    head = _commit(d, ["docs/a.md", "src/x.py"])
    r = _run(d, base, head)
    assert "RAN" in r.stdout
    assert r.returncode == 3


def test_empty_range_runs(repo):
    d, base = repo
    r = _run(d, base, base)
    assert "RAN" in r.stdout


def test_new_branch_zero_from_docs_only_skips(repo):
    d, _ = repo
    head = _commit(d, ["docs/a.md"])
    r = _run(d, ZEROS, head)
    assert r.returncode == 0
    assert "RAN" not in r.stdout


def test_unset_env_runs(repo):
    d, _ = repo
    r = _run(d, None, None)
    assert "RAN" in r.stdout
    assert r.returncode == 3


def test_nested_markdown_runs(repo):
    d, base = repo
    head = _commit(d, ["packages/kb-service/README.md"])
    r = _run(d, base, head)
    assert "RAN" in r.stdout
