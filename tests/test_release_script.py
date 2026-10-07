"""Run the real release.sh in a temp git repo with shims for uv, git push and the hook."""
# ruff: noqa: S603, S607

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NEW_VERSION = "1.2.3"
REAL_GIT = shutil.which("git") or "git"


def _write(path: Path, body: str, mode: int = 0o755) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    path.chmod(mode)


@pytest.fixture
def sandbox(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    shims = tmp_path / "shims"
    log = tmp_path / "calls.log"
    log.touch()
    repo.mkdir()
    shims.mkdir()
    shutil.copy(REPO / "release.sh", repo / "release.sh")

    # Stubs for the UI scripts and the version stamper (the real ones need node/uv).
    for name in ("build_static_ui.sh", "check_static_ui_fresh.sh"):
        _write(repo / "scripts" / name, "#!/usr/bin/env bash\nexit 0\n")
    _write(repo / "scripts" / "stamp_hook_version.py", "print('stamped')\n")
    (repo / "packages/kb-service/src/kb_service/static").mkdir(parents=True)
    (repo / "packages/kb-service/src/kb_service/static/.keep").write_text("")
    for pkg in ("personal-kb-hook", "kb-service", "kb-core"):
        _write(repo / "packages" / pkg / "pyproject.toml", "[project]\n", 0o644)
    _write(repo / "pyproject.toml", '[project]\nversion = "1.2.2"\n', 0o644)
    _write(repo / "uv.lock", "", 0o644)

    # uv shim: semantic-release bumps+commits+tags; build creates wheels.
    _write(
        shims / "uv",
        f"""\
        #!/usr/bin/env bash
        echo "uv $*" >> "{log}"
        case "$1" in
          run)
            if [ "$2" = "semantic-release" ]; then
              sed -i 's/^version = .*/version = "{NEW_VERSION}"/' pyproject.toml
              {REAL_GIT} commit -qam "chore(release): v{NEW_VERSION}"
              {REAL_GIT} tag -a "v{NEW_VERSION}" -m "v{NEW_VERSION}"
            fi ;;
          build)
            mkdir -p dist
            for n in personal_kb kb_core personal_kb_web_service personal_kb_hook; do
              touch "dist/$n-${{UV_SHIM_WHEEL_VERSION:-{NEW_VERSION}}}-py3-none-any.whl"
            done
            [ -n "${{UV_SHIM_DROP_WHEEL:-}}" ] && rm -f dist/kb_core-* ;;
        esac
        exit 0
        """,
    )
    # git shim: record pushes (never contact a remote), pass everything else through.
    _write(
        shims / "git",
        f"""\
        #!/usr/bin/env bash
        if [ "$1" = "push" ]; then
          echo "git $*" >> "{log}"
          exit 0
        fi
        exec {REAL_GIT} "$@"
        """,
    )
    for cmd in ("init -q -b main", "add -A", "-c user.name=t -c user.email=t@t commit -qm init"):
        subprocess.run([REAL_GIT, *cmd.split()], cwd=repo, check=True)
    (repo / ".gitignore").write_text("dist/\nrelease.local.sh\n")
    subprocess.run([REAL_GIT, "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        [REAL_GIT, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "ignore"],
        cwd=repo,
        check=True,
    )
    for name in ("origin", "github"):
        subprocess.run(
            [REAL_GIT, "remote", "add", name, f"/nonexistent/{name}.git"], cwd=repo, check=True
        )
    return {"repo": repo, "shims": shims, "log": log}


def _run(sb: dict[str, Path], *args: str, extra_env: dict[str, str] | None = None):
    env = {
        # Replaced PATH: shims first, then only the basics the script needs.
        "PATH": f"{sb['shims']}:/usr/bin:/bin",
        "HOME": str(sb["repo"]),
        "SKIP_UPGRADE_SMOKE": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        **(extra_env or {}),
    }
    return subprocess.run(
        ["bash", "release.sh", *args],
        cwd=sb["repo"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _tags(sb: dict[str, Path]) -> str:
    return subprocess.run(
        [REAL_GIT, "tag"], cwd=sb["repo"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _calls(sb: dict[str, Path]) -> list[str]:
    return sb["log"].read_text().splitlines()


def _hook(sb: dict[str, Path], exit_code: int) -> None:
    _write(
        sb["repo"] / "release.local.sh",
        f"""\
        #!/usr/bin/env bash
        echo "hook $1 $2 pwd=$PWD" >> "{sb["log"]}"
        ls "$2" | wc -l >> "{sb["log"]}"
        exit {exit_code}
        """,
    )


def test_hook_success_then_push(sandbox):
    _hook(sandbox, 0)
    r = _run(sandbox)
    assert r.returncode == 0, r.stderr
    calls = _calls(sandbox)
    hook_i = next(i for i, c in enumerate(calls) if c.startswith("hook "))
    push_is = [i for i, c in enumerate(calls) if c.startswith("git push")]
    assert len(push_is) == 2 and all(i > hook_i for i in push_is)
    assert calls[hook_i] == f"hook {NEW_VERSION} {sandbox['repo']}/dist pwd={sandbox['repo']}"
    assert f"v{NEW_VERSION}" in _tags(sandbox)


def test_hook_failure_aborts_without_push(sandbox):
    _hook(sandbox, 7)
    r = _run(sandbox)
    assert r.returncode != 0
    assert not any(c.startswith("git push") for c in _calls(sandbox))
    assert _tags(sandbox) == ""


def test_hook_published_but_deploy_incomplete_pushes_then_fails(sandbox):
    """Exit 10: artifacts shipped (immutable), so tags are pushed, then exit 3."""
    _hook(sandbox, 10)
    r = _run(sandbox)
    assert r.returncode == 3
    calls = _calls(sandbox)
    hook_i = next(i for i, c in enumerate(calls) if c.startswith("hook "))
    push_is = [i for i, c in enumerate(calls) if c.startswith("git push")]
    assert len(push_is) == 2 and all(i > hook_i for i in push_is)
    assert f"v{NEW_VERSION}" in _tags(sandbox)
    assert "INCOMPLETE deploy" in r.stderr


def test_no_github_remote_releases_to_origin_only(sandbox):
    subprocess.run([REAL_GIT, "remote", "remove", "github"], cwd=sandbox["repo"], check=True)
    _hook(sandbox, 0)
    r = _run(sandbox)
    assert r.returncode == 0, r.stderr
    pushes = [c for c in _calls(sandbox) if c.startswith("git push")]
    assert pushes == ["git push origin main --tags"]
    assert "git push github main --tags" in r.stderr


def test_missing_hook_aborts(sandbox):
    r = _run(sandbox)
    assert r.returncode != 0
    assert "release.local.sh" in r.stderr and "--no-publish" in r.stderr
    assert not any(c.startswith("git push") for c in _calls(sandbox))
    assert _tags(sandbox) == ""


def test_no_publish_pushes_with_notice(sandbox):
    r = _run(sandbox, "--no-publish")
    assert r.returncode == 0, r.stderr
    assert "NO ARTIFACTS PUBLISHED" in r.stderr
    assert len([c for c in _calls(sandbox) if c.startswith("git push")]) == 2
    assert f"v{NEW_VERSION}" in _tags(sandbox)


def test_missing_wheel_aborts_before_hook(sandbox):
    _hook(sandbox, 0)
    r = _run(sandbox, extra_env={"UV_SHIM_DROP_WHEEL": "1"})
    assert r.returncode != 0
    assert not any(c.startswith(("hook", "git push")) for c in _calls(sandbox))
    assert _tags(sandbox) == ""


def test_wrong_version_wheels_abort(sandbox):
    _hook(sandbox, 0)
    r = _run(sandbox, extra_env={"UV_SHIM_WHEEL_VERSION": "0.0.1"})
    assert r.returncode != 0
    assert not any(c.startswith(("hook", "git push")) for c in _calls(sandbox))
    assert _tags(sandbox) == ""
