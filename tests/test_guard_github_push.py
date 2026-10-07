"""The pre-push guard lets github receive release pushes only."""

import os
import subprocess
from pathlib import Path

GUARD = Path(__file__).resolve().parent.parent / "scripts" / "guard_github_push.sh"


def _run(remote: str, *, release: bool) -> int:
    env = {k: v for k, v in os.environ.items() if k not in ("KB_RELEASE_PUSH",)}
    env["PRE_COMMIT_REMOTE_NAME"] = remote
    if release:
        env["KB_RELEASE_PUSH"] = "1"
    return subprocess.run(  # noqa: S603
        ["/bin/bash", str(GUARD)], env=env, capture_output=True, check=False
    ).returncode


def test_github_push_outside_release_is_refused() -> None:
    assert _run("github", release=False) == 1


def test_github_push_from_release_is_allowed() -> None:
    assert _run("github", release=True) == 0


def test_origin_push_is_always_allowed() -> None:
    assert _run("origin", release=False) == 0


def test_release_script_marks_its_github_push() -> None:
    text = (GUARD.parent.parent / "release.sh").read_text()
    assert "KB_RELEASE_PUSH=1 git push github" in text
