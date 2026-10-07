"""Pinned vectors for the failure-cue normalizer (``kb_core.cues``)."""

from __future__ import annotations

import pytest

import kb_core
from kb_core import cues
from kb_core.cues import (
    CUE_NORMALIZER_VERSION,
    FailureCue,
    build_cue,
    cue_key,
    extract_target,
    host_class_from_cwd,
    normalize_error,
    resolve_cue_project,
    target_class,
)


def test_all_is_exact() -> None:
    assert set(cues.__all__) == {
        "CUE_NORMALIZER_VERSION",
        "FailureCue",
        "build_cue",
        "cue_key",
        "extract_target",
        "host_class_from_cwd",
        "normalize_error",
        "resolve_cue_project",
        "target_class",
    }


def test_not_reexported_from_package_root() -> None:
    assert not hasattr(kb_core, "build_cue")
    assert "cues" not in getattr(kb_core, "__all__", [])


def test_version_is_one() -> None:
    assert CUE_NORMALIZER_VERSION == 1


# --- AC2: extract_target ----------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "tool_input", "expected"),
    [
        ("Bash", {"command": "ls -la"}, "ls -la"),
        ("Read", {"file_path": "/a/b.py"}, "/a/b.py"),
        ("Edit", {"file_path": "/a/b.py"}, "/a/b.py"),
        ("Write", {"file_path": "/a/b.py"}, "/a/b.py"),
        ("MultiEdit", {"file_path": "/a/b.py"}, "/a/b.py"),
        ("NotebookEdit", {"notebook_path": "/a/n.ipynb"}, "/a/n.ipynb"),
        ("Glob", {"pattern": "**/*.py"}, "**/*.py"),
        ("Grep", {"pattern": "foo"}, "foo"),
        ("Bash", {"command": 5}, ""),
        ("Bash", {}, ""),
        ("mcp__x__y", {"command": "x"}, ""),
        ("Bash", None, ""),
        ("Read", "not-a-dict", ""),
    ],
)
def test_extract_target(tool: str, tool_input: object, expected: str) -> None:
    assert extract_target(tool, tool_input) == expected


# --- AC3: target_class ------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cd /x && git push origin main", "git push"),
        ("cd /x || git fetch origin", "git fetch"),
        ("FOO=1 uv run pytest -q", "uv run"),
        ("ls -la /tmp", "ls"),
        ("sudo -n -u dispatch git -C /x status", "git"),
        ("/usr/bin/python3 -m pytest", "python3"),
        ("", ""),
        ("cd /x", ""),
        ("FOO=1", ""),
        ("git", "git"),
        ("echo hi | grep h", "echo"),
    ],
)
def test_target_class_bash(command: str, expected: str) -> None:
    assert target_class("Bash", command) == expected


def test_target_class_file_tools() -> None:
    assert target_class("Read", "/a/b/c.PY") == "ext:py"
    assert target_class("Read", "/a/Makefile") == "ext:none"
    assert target_class("Read", extract_target("Read", {})) == ""
    assert target_class("NotebookEdit", "/a/n.ipynb") == "ext:ipynb"


def test_target_class_other_tools() -> None:
    assert target_class("mcp__agent-gtd__get_item", "") == ""
    assert target_class("Grep", "foo") == ""


# --- AC4: normalize_error ---------------------------------------------------


def test_normalize_exit_with_keyword() -> None:
    text = "Exit code 144\npkill: killing pid 3787822 failed: Operation not permitted"
    assert normalize_error(text) == (
        "exit 144 | pkill: killing pid <n> failed: operation not permitted",
        "keyword",
    )


def test_normalize_uuid() -> None:
    assert normalize_error("Item b2103f60-42cb-478d-87b5-297fd7a74041 not found") == (
        "item <uuid> not found",
        "keyword",
    )


def test_normalize_exit_only() -> None:
    assert normalize_error("Exit code 1") == ("exit 1", "exit_only")


def test_normalize_empty() -> None:
    assert normalize_error("") == ("", "empty")
    assert normalize_error("\x1b[0m") == ("", "empty")
    assert normalize_error("  \n \n") == ("", "empty")


def test_normalize_hex_last_line() -> None:
    assert normalize_error("deadbeefcafe1234") == ("<hex>", "last_line")


def test_normalize_traceback_paths_collapse() -> None:
    a = normalize_error('File "/home/a/b.py", line 3')
    b = normalize_error('File "/Users/z/q/r.py", line 9')
    assert a == b


def test_normalize_lint_paths_collapse() -> None:
    a = normalize_error("/x/y/z.py:12:4: E501 line too long")
    b = normalize_error("/p/q/w.py:7:80: E501 line too long")
    assert a == b


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            "Error: run 0f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b failed",
            "Error: run 11111111-2222-3333-4444-555555555555 failed",
        ),
        ("Error: pid 12345 failed", "Error: pid 999 failed"),
        ("Error at 2026-10-07T12:00:00Z failed", "Error at 2025-01-02T03:04:05.5+02:00 failed"),
        ("Error at 12:00:01", "Error at 23:59:59"),
        ("Error: cannot open /home/a/x.txt", "Error: cannot open /Users/b/c/y.txt"),
        ("Error: no such entry kb-00012", "Error: no such entry kb-99"),
        ("Error: toolu_abc123 failed", "Error: toolu_XYZ failed"),
    ],
)
def test_normalize_incidental_differences_collapse(a: str, b: str) -> None:
    assert normalize_error(a) == normalize_error(b)


def test_normalize_cap_200() -> None:
    normalized, _ = normalize_error("error " + "x" * 294)
    assert len(normalized) <= 200


def test_normalize_trunc_marker_and_ansi() -> None:
    normalized, rule = normalize_error("\x1b[31merror\x1b[0m ... [1234 characters truncated] ...")
    assert normalized == "error <trunc>"
    assert rule == "keyword"


def test_normalize_keeps_first_two_keyword_lines() -> None:
    text = "start\nError one\nnoise\nfatal two\nfailed three\nend"
    assert normalize_error(text) == ("error one | fatal two", "keyword")


def test_normalize_last_line_rule() -> None:
    assert normalize_error("first\nsecond line") == ("second line", "last_line")


def test_normalize_exit_line_keeps_code() -> None:
    assert normalize_error("Exit code 144\nsomething")[0] == "exit 144 | something"


# --- AC5: host_class, project, cue_key --------------------------------------


@pytest.mark.parametrize(
    ("cwd", "expected"),
    [
        ("/Users/j/x", "darwin"),
        ("/home/j/x", "linux"),
        ("/srv/x", "linux"),
        ("C:\\x", "windows"),
        (None, "unknown"),
        ("", "unknown"),
        ("rel/x", "unknown"),
    ],
)
def test_host_class(cwd: str | None, expected: str) -> None:
    assert host_class_from_cwd(cwd) == expected


def test_resolve_cue_project() -> None:
    assert resolve_cue_project(None, "/Users/jason/git/personal_kb") == (
        "personal-kb",
        "cwd_basename",
    )
    assert resolve_cue_project(None, "/home/jason/git/personal_kb") == (
        "personal-kb",
        "cwd_basename",
    )
    assert resolve_cue_project(
        None,
        "/home/dispatch/workspace/personal_kb-0f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b",
    ) == ("personal-kb", "cwd_basename")
    assert resolve_cue_project("personal-kb", "/anything") == ("personal-kb", "kb_project")
    assert resolve_cue_project("  ", "/home/j/Foo_Bar/") == ("foo-bar", "cwd_basename")
    assert resolve_cue_project(None, "C:\\Users\\j\\My_Repo") == ("my-repo", "cwd_basename")
    assert resolve_cue_project(None, None) == ("", "none")
    assert resolve_cue_project(None, "/") == ("", "none")


def test_cue_key_formula_and_stability() -> None:
    import hashlib

    expected = hashlib.sha256(b"1\x1fBash\x1fgit push\x1fexit 1\x1fp").hexdigest()[:16]
    assert cue_key("Bash", "git push", "exit 1", "p") == expected
    assert cue_key("Bash", "git push", "exit 1", "p") == cue_key("Bash", "git push", "exit 1", "p")
    assert len(expected) == 16


def test_cue_key_sensitivity() -> None:
    base = cue_key("Bash", "git push", "exit 1", "p")
    assert cue_key("Read", "git push", "exit 1", "p") != base
    assert cue_key("Bash", "git pull", "exit 1", "p") != base
    assert cue_key("Bash", "git push", "exit 2", "p") != base
    assert cue_key("Bash", "git push", "exit 1", "q") != base
    assert cue_key("Bash", "git push", "exit 1", "p", version=2) != base


def test_cue_key_ignores_host_class() -> None:
    a = build_cue("Bash", {"command": "git push"}, "Exit code 1", "proj", "/Users/j/x")
    b = build_cue("Bash", {"command": "git push"}, "Exit code 1", "proj", "/home/j/x")
    assert a.host_class == "darwin"
    assert b.host_class == "linux"
    assert a.cue_key == b.cue_key


# --- AC1: build_cue ---------------------------------------------------------


def test_build_cue_fields() -> None:
    cue = build_cue(
        "Bash",
        {"command": "cd /x && git push origin main"},
        "Exit code 1\nerror: failed to push some refs",
        None,
        "/home/j/personal_kb",
    )
    assert isinstance(cue, FailureCue)
    assert cue.tool == "Bash"
    assert cue.target == "cd /x && git push origin main"
    assert cue.target_class == "git push"
    assert cue.normalized_error == "exit 1 | error: failed to push some refs"
    assert cue.error_rule == "keyword"
    assert cue.project == "personal-kb"
    assert cue.project_source == "cwd_basename"
    assert cue.host_class == "linux"
    assert cue.normalizer_version == CUE_NORMALIZER_VERSION
    assert cue.cue_key == cue_key("Bash", "git push", cue.normalized_error, "personal-kb")


def test_build_cue_non_dict_input() -> None:
    cue = build_cue("Bash", ["not", "a", "dict"], "boom", "p", None)
    assert cue.target == ""
    assert cue.target_class == ""


def test_build_cue_truncation_equivalence() -> None:
    long_text = "x" * 3990 + "\nError: tail line beyond the cap " + "y" * 6000
    assert len(long_text) > 6000
    full = build_cue("Bash", {"command": "ls"}, long_text, "p", "/home/j")
    truncated = build_cue("Bash", {"command": "ls"}, long_text[:4000], "p", "/home/j")
    assert full == truncated
