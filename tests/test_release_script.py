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
    (repo / ".gitignore").write_text(
        "dist/\nrelease.local.sh\nrelease.post.local.sh\nrelease.guard.local.sh\n"
    )
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
    sb = {"repo": repo, "shims": shims, "log": log}
    # Every sandbox has a github remote, so it needs a passing leak guard by default.
    _guard(sb, 0)
    return sb


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


def _guard(sb: dict[str, Path], exit_code: int) -> None:
    _write(
        sb["repo"] / "release.guard.local.sh",
        f"""\
        #!/usr/bin/env bash
        echo "guard argc=$# pwd=$PWD" >> "{sb["log"]}"
        exit {exit_code}
        """,
    )


def _post_hook(sb: dict[str, Path], exit_code: int) -> None:
    _write(
        sb["repo"] / "release.post.local.sh",
        f"""\
        #!/usr/bin/env bash
        echo "post $1 argc=$# pwd=$PWD" >> "{sb["log"]}"
        exit {exit_code}
        """,
    )


def _assert_post_after_pushes(sb: dict[str, Path]) -> list[str]:
    calls = _calls(sb)
    post = [i for i, c in enumerate(calls) if c.startswith("post ")]
    pushes = [i for i, c in enumerate(calls) if c.startswith("git push")]
    assert len(post) == 1 and len(pushes) == 2 and all(i < post[0] for i in pushes)
    assert calls[post[0]] == f"post {NEW_VERSION} argc=1 pwd={sb['repo']}"
    return calls


def test_post_hook_runs_after_pushes(sandbox):
    _hook(sandbox, 0)
    _post_hook(sandbox, 0)
    r = _run(sandbox)
    assert r.returncode == 0, r.stderr
    _assert_post_after_pushes(sandbox)
    assert "release.post.local.sh failed" not in r.stderr


def test_post_hook_failure_exits_4_without_rollback(sandbox):
    _hook(sandbox, 0)
    _post_hook(sandbox, 7)
    r = _run(sandbox)
    assert r.returncode == 4
    _assert_post_after_pushes(sandbox)
    assert "failed (exit 7)" in r.stderr
    assert f"Rerun it by hand: ./release.post.local.sh {NEW_VERSION}" in r.stderr
    assert f"v{NEW_VERSION}" in _tags(sandbox)


def test_post_hook_runs_when_deploy_incomplete_and_exit_3_wins(sandbox):
    _hook(sandbox, 10)
    _post_hook(sandbox, 0)
    r = _run(sandbox)
    assert r.returncode == 3
    _assert_post_after_pushes(sandbox)


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
    head_before = subprocess.run(
        ["git", "-C", str(sandbox["repo"]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    r = _run(sandbox)
    assert r.returncode != 0
    assert not any(c.startswith("git push") for c in _calls(sandbox))
    assert _tags(sandbox) == ""
    head_after = subprocess.run(
        ["git", "-C", str(sandbox["repo"]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # The release commit is dropped too, so a re-run starts from the same main.
    assert head_after == head_before


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


# --- docs backstop ---------------------------------------------------------------

ABORT_MSG = (
    "feat commits since v1.0.0 but neither technical doc changed; "
    "update the docs or pass --no-docs-check"
)


def _commit(sb: dict[str, Path], subject: str, *files: str) -> None:
    repo = sb["repo"]
    for name in files:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.read_text() + "x\n" if path.exists() else "x\n")
    subprocess.run([REAL_GIT, "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        [REAL_GIT, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", subject],
        cwd=repo,
        check=True,
    )


def _tag_baseline(sb: dict[str, Path]) -> None:
    subprocess.run([REAL_GIT, "tag", "-a", "v1.0.0", "-m", "v1.0.0"], cwd=sb["repo"], check=True)


def test_feat_without_docs_aborts_before_bump(sandbox):
    _hook(sandbox, 0)
    _tag_baseline(sandbox)
    _commit(sandbox, "feat(search): new thing", "src/personal_kb/new.py")
    r = _run(sandbox)
    assert r.returncode != 0
    assert ABORT_MSG in r.stderr
    assert not any(c.startswith(("uv run", "hook", "git push")) for c in _calls(sandbox))
    assert _tags(sandbox) == "v1.0.0"


def test_feat_in_package_src_without_docs_aborts(sandbox):
    _hook(sandbox, 0)
    _tag_baseline(sandbox)
    _commit(sandbox, "feat: svc", "packages/kb-service/src/kb_service/new.py")
    assert ABORT_MSG in _run(sandbox).stderr


def test_feat_with_doc_change_in_range_passes(sandbox):
    _hook(sandbox, 0)
    _tag_baseline(sandbox)
    _commit(sandbox, "feat: new thing", "src/personal_kb/new.py")
    _commit(sandbox, "docs: describe it", "docs/intelligent-kb.md")
    r = _run(sandbox)
    assert r.returncode == 0, r.stderr
    assert "neither technical doc changed" not in r.stderr


def test_doc_change_before_last_tag_does_not_count(sandbox):
    _commit(sandbox, "docs: early", "how_it_works.md")
    _tag_baseline(sandbox)
    _commit(sandbox, "feat: later", "src/personal_kb/new.py")
    assert ABORT_MSG in _run(sandbox).stderr


def test_non_feat_and_feat_without_code_pass(sandbox):
    _hook(sandbox, 0)
    _tag_baseline(sandbox)
    _commit(sandbox, "fix: bug", "src/personal_kb/new.py")
    _commit(sandbox, "feat(ci): tooling only", "scripts/tool.sh")
    r = _run(sandbox)
    assert r.returncode == 0, r.stderr


def test_no_docs_check_overrides_loudly(sandbox):
    _hook(sandbox, 0)
    _tag_baseline(sandbox)
    _commit(sandbox, "feat: new thing", "src/personal_kb/new.py")
    r = _run(sandbox, "--no-docs-check")
    assert r.returncode == 0, r.stderr
    assert "--no-docs-check given" in r.stderr
    assert ABORT_MSG not in r.stderr


def test_guard_runs_before_publish_hook_and_pushes(sandbox):
    _hook(sandbox, 0)
    r = _run(sandbox)
    assert r.returncode == 0, r.stderr
    calls = _calls(sandbox)
    guard_i = next(i for i, c in enumerate(calls) if c.startswith("guard "))
    hook_i = next(i for i, c in enumerate(calls) if c.startswith("hook "))
    push_is = [i for i, c in enumerate(calls) if c.startswith("git push")]
    assert calls[guard_i] == f"guard argc=0 pwd={sandbox['repo']}"
    assert guard_i < hook_i and all(i > guard_i for i in push_is)


@pytest.mark.parametrize(
    ("code", "message"),
    [(1, "found private data"), (2, "could not run (exit 2)"), (7, "could not run (exit 7)")],
)
def test_guard_nonzero_aborts_before_publishing(sandbox, code, message):
    _hook(sandbox, 0)
    _guard(sandbox, code)
    r = _run(sandbox)
    assert r.returncode != 0
    assert message in r.stderr
    calls = _calls(sandbox)
    assert not any(c.startswith(("hook ", "git push")) for c in calls)
    assert _tags(sandbox) == ""


def test_missing_guard_aborts_unless_no_guard(sandbox):
    _hook(sandbox, 0)
    (sandbox["repo"] / "release.guard.local.sh").unlink()
    r = _run(sandbox)
    assert r.returncode != 0
    assert "release.guard.local.sh" in r.stderr and "--no-guard" in r.stderr
    assert not any(c.startswith(("hook ", "git push")) for c in _calls(sandbox))
    assert _tags(sandbox) == ""


def test_non_executable_guard_aborts(sandbox):
    _hook(sandbox, 0)
    (sandbox["repo"] / "release.guard.local.sh").chmod(0o644)
    r = _run(sandbox)
    assert r.returncode != 0
    assert "--no-guard" in r.stderr
    assert not any(c.startswith("git push") for c in _calls(sandbox))


def test_no_guard_flag_skips_loudly(sandbox):
    _hook(sandbox, 0)
    (sandbox["repo"] / "release.guard.local.sh").unlink()
    r = _run(sandbox, "--no-guard")
    assert r.returncode == 0, r.stderr
    assert "LEAK GUARD SKIPPED" in r.stderr
    assert len([c for c in _calls(sandbox) if c.startswith("git push")]) == 2


def test_no_github_remote_needs_no_guard(sandbox):
    subprocess.run([REAL_GIT, "remote", "remove", "github"], cwd=sandbox["repo"], check=True)
    (sandbox["repo"] / "release.guard.local.sh").unlink()
    _hook(sandbox, 0)
    r = _run(sandbox)
    assert r.returncode == 0, r.stderr
    assert not any(c.startswith("guard ") for c in _calls(sandbox))
    assert "LEAK GUARD SKIPPED" not in r.stderr
