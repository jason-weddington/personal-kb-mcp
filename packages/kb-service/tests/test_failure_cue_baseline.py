"""Hermetic tests for ``scripts/failure_cue_baseline.py`` (synthetic fixtures only)."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# scripts/ is not a package (no __init__.py, not installed) -- add it to
# sys.path directly, mirroring how the script is invoked standalone.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from failure_cue_baseline import (  # noqa: E402
    compute_report,
    main,
    parse_paths,
    render_markdown,
)

_PREFIX = "/home/dispatch/"


def _tool_use(
    tid: str, command: str = "pkill foo", name: str = "Bash"
) -> dict[str, Any]:
    return {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": tid,
                    "name": name,
                    "input": {"command": command},
                }
            ]
        },
    }


def _result(
    tid: str,
    content: Any,
    *,
    session: str,
    ts: str,
    cwd: str = "/home/j/proj",
    is_error: bool = True,
) -> dict[str, Any]:
    return {
        "type": "user",
        "sessionId": session,
        "timestamp": ts,
        "cwd": cwd,
        "toolUseResult": "Error: ignored",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "is_error": is_error,
                    "content": content,
                }
            ]
        },
    }


def _write(path: Path, lines: list[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines)
        + "\n"
    )
    return path


def _err(pid: int, path: str = "/home/j/x.py") -> str:
    return f"Exit code 1\nkill {pid} failed: Operation not permitted at {path}"


def _failure(tid: str, session: str, ts: str, pid: int = 1234) -> list[dict[str, Any]]:
    return [_tool_use(tid), _result(tid, _err(pid), session=session, ts=ts)]


def _parse(tmp_path: Path) -> Any:
    return parse_paths([str(tmp_path)], None, _PREFIX)


def test_a_repeat_rate_three_sessions(tmp_path: Path) -> None:
    _write(tmp_path / "a.jsonl", _failure("t1", "A", "2026-01-01T00:00:00Z"))
    _write(tmp_path / "b.jsonl", _failure("t2", "B", "2026-01-01T02:00:00Z"))
    _write(tmp_path / "c.jsonl", _failure("t3", "C", "2026-01-04T00:00:00Z"))
    parsed = _parse(tmp_path)
    report = compute_report(parsed.failures, 24, parsed)
    [cue] = report["cues"]
    assert cue["sessions"] == 3
    assert cue["repeat_sessions"] == 1
    assert cue["repeat_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert report["weeks"] == [{"week": "2026-W01", "repeat_sessions": 1}]
    assert report["totals"]["aggregate_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert report["totals"]["aggregate_rate_ge2"] == pytest.approx(1 / 3, abs=1e-4)
    assert report["totals"]["transcripts"] == 3


def test_b_pid_and_path_collapse(tmp_path: Path) -> None:
    _write(
        tmp_path / "a.jsonl",
        [
            _tool_use("t1"),
            _tool_use("t2"),
            _result(
                "t1", _err(1111, "/home/a/b.py"), session="A", ts="2026-01-01T00:00:00Z"
            ),
            _result(
                "t2", _err(2222, "/srv/c/d.py"), session="A", ts="2026-01-01T00:01:00Z"
            ),
        ],
    )
    parsed = _parse(tmp_path)
    report = compute_report(parsed.failures, 24, parsed)
    [cue] = report["cues"]
    assert cue["distinct_raw"] == 2
    assert len(cue["samples"]) == 2
    assert cue["sessions"] == 1


def test_c_exclusion_prefixes(tmp_path: Path) -> None:
    texts = [
        "<tool_use_error>InputValidationError</tool_use_error>",
        "Permission to use Bash has been denied",
        "The user doesn't want to proceed with this tool use.",
        "[Request interrupted by user for tool use]",
    ]
    lines: list[Any] = []
    for i, text in enumerate(texts):
        lines += [
            _tool_use(f"t{i}"),
            _result(f"t{i}", text, session="A", ts="2026-01-01T00:00:00Z"),
        ]
    _write(tmp_path / "a.jsonl", lines)
    parsed = _parse(tmp_path)
    assert parsed.excluded == 4
    assert len(parsed.failures) == 0


def test_d_duplicate_across_files(tmp_path: Path) -> None:
    lines = _failure("t1", "A", "2026-01-01T00:00:00Z")
    _write(tmp_path / "a.jsonl", lines)
    _write(tmp_path / "sub" / "b.jsonl", lines)
    parsed = _parse(tmp_path)
    report = compute_report(parsed.failures, 24, parsed)
    assert report["totals"]["failures"] == 1
    assert report["totals"]["duplicates_skipped"] == 1


def test_e_tie_breaks_on_session_id(tmp_path: Path) -> None:
    ts = "2026-01-01T00:00:00Z"
    # Same earliest ts; 'Alpha' (darwin) must win s0 over 'Zed' (linux).
    _write(
        tmp_path / "a.jsonl",
        [
            _tool_use("t1"),
            _result("t1", _err(1111), session="Zed", ts=ts, cwd="/home/j/p"),
        ],
    )
    _write(
        tmp_path / "b.jsonl",
        [
            _tool_use("t2"),
            _result("t2", _err(2222), session="Alpha", ts=ts, cwd="/Users/j/p"),
        ],
    )
    parsed = _parse(tmp_path)
    report = compute_report(parsed.failures, 0, parsed)
    [cue] = report["cues"]
    assert cue["host_class"] == "darwin"  # taken from s0's first failure
    assert cue["repeat_sessions"] == 1
    cuts = {r["key"]: r["repeat_sessions"] for r in report["by_host_class"]}
    assert cuts == {"darwin": 0, "linux": 1}  # the repeat is Zed, not Alpha


def test_f_malformed_and_unknown_counted(tmp_path: Path) -> None:
    _write(
        tmp_path / "a.jsonl",
        [
            "{not json",
            "[1, 2]",
            _result("ghost", "boom", session="A", ts="2026-01-01T00:00:00Z"),
            _tool_use("t1"),
            _result("t1", "boom", session="A", ts="not-a-ts"),
            _tool_use("t2"),
            _result(
                "t2", "boom", session="A", ts="2026-01-01T00:00:00", is_error=False
            ),
        ],
    )
    parsed = _parse(tmp_path)
    assert parsed.malformed == 3
    assert parsed.unknown_tool_use == 1
    assert parsed.failures == []


def test_g_mode_from_cwd_prefix(tmp_path: Path) -> None:
    _write(
        tmp_path / "a.jsonl",
        [
            _tool_use("t1"),
            _result(
                "t1",
                "boom",
                session="A",
                ts="2026-01-01T00:00:00Z",
                cwd="/home/dispatch/x",
            ),
            _tool_use("t2"),
            _result(
                "t2", "boom", session="B", ts="2026-01-01T00:00:00Z", cwd="/home/j/x"
            ),
        ],
    )
    parsed = _parse(tmp_path)
    modes = {f.session_id: f.mode for f in parsed.failures}
    assert modes == {"A": "headless", "B": "interactive"}
    report = compute_report(parsed.failures, 24, parsed)
    assert {r["key"] for r in report["by_mode"]} == {"headless", "interactive"}


def test_h_list_content_joined(tmp_path: Path) -> None:
    content = [
        {"type": "text", "text": "Exit code 2"},
        {"type": "image", "source": {}},
        {"type": "text", "text": "fatal: not a git repository"},
    ]
    _write(
        tmp_path / "a.jsonl",
        [
            _tool_use("t1", "git status"),
            _result("t1", content, session="A", ts="2026-01-01T00:00:00Z"),
        ],
    )
    [failure] = _parse(tmp_path).failures
    assert failure.raw == "Exit code 2\nfatal: not a git repository"
    assert failure.cue.normalized_error == "exit 2 | fatal: not a git repository"
    assert failure.cue.target_class == "git status"


def test_project_resolution(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    (repo / ".kb_project").write_text("# comment\n\nmy-proj\n")
    transcripts = tmp_path / "t"
    _write(
        transcripts / "a.jsonl",
        [
            _tool_use("t1"),
            _result(
                "t1",
                "boom",
                session="A",
                ts="2026-01-01T00:00:00Z",
                cwd=str(repo / "sub"),
            ),
            _tool_use("t2"),
            _result(
                "t2",
                "boom",
                session="A",
                ts="2026-01-01T00:00:00Z",
                cwd="/nope/Some_Repo",
            ),
            _tool_use("t3"),
            _result(
                "t3", "boom", session="A", ts="2026-01-01T00:00:00Z", cwd="/nope/mapped"
            ),
        ],
    )
    parsed = parse_paths([str(transcripts)], {"mapped": "from-map"}, _PREFIX)
    projects = [(f.cue.project, f.cue.project_source) for f in parsed.failures]
    assert projects == [
        ("my-proj", "kb_project"),
        ("some-repo", "cwd_basename"),
        ("from-map", "kb_project"),
    ]


def test_i_json_output_and_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "src"
    _write(src / "a.jsonl", _failure("t1", "A", "2026-01-01T00:00:00Z"))
    _write(src / "b.jsonl", _failure("t2", "B", "2026-01-05T00:00:00Z"))
    _write(src / "c.jsonl", _failure("t3", "C", "2025-06-01T00:00:00Z", pid=9))
    pmap = tmp_path / "map.json"
    pmap.write_text(json.dumps({"proj": "p"}))
    out = tmp_path / "out.json"
    rc = main(
        [
            str(src),
            "--json",
            str(out),
            "--since",
            "2026-01-01",
            "--project-map",
            str(pmap),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text())
    assert set(report) == {"totals", "cues", "weeks", "by_host_class", "by_mode"}
    assert {
        "normalizer_version",
        "harness",
        "transcripts",
        "failures",
        "excluded",
        "malformed",
        "unknown_tool_use",
        "duplicates_skipped",
        "distinct_cues",
        "cues_ge2_sessions",
        "aggregate_rate",
        "aggregate_rate_ge2",
        "singleton_cue_fraction",
        "top_cue_share",
        "by_project_source",
        "by_error_rule",
        "anomalies",
    } <= set(report["totals"])
    assert report["totals"]["failures"] == 2  # --since dropped the 2025 one
    assert report["totals"]["anomalies"] == {"empty_error": 0, "empty_bash_target": 0}
    [cue] = report["cues"]
    assert {
        "cue_key",
        "tool",
        "target_class",
        "project",
        "host_class",
        "normalized_error",
        "sessions",
        "repeat_sessions",
        "repeat_rate",
        "first_ts",
        "last_ts",
        "distinct_raw",
        "samples",
    } <= set(cue)
    assert cue["project"] == "p"
    stdout = capsys.readouterr().out
    order = [
        stdout.index(h)
        for h in (
            "## Totals",
            "## Top cues",
            "## Weeks",
            "## by_host_class",
            "## by_mode",
        )
    ]
    assert order == sorted(order)
    assert "WARN over-collapse" in stdout
    assert "WARN under-collapse" not in stdout


def test_warn_under_collapse() -> None:
    report = {
        "totals": {"singleton_cue_fraction": 0.95, "top_cue_share": 0.01},
        "cues": [],
        "weeks": [],
        "by_host_class": [],
        "by_mode": [],
    }
    text = render_markdown(report, 30)
    assert "WARN under-collapse" in text
    assert "WARN over-collapse" not in text


def test_empty_report_and_anomalies(tmp_path: Path) -> None:
    assert compute_report([], 24)["totals"]["aggregate_rate"] == 0.0
    _write(
        tmp_path / "a.jsonl",
        [
            _tool_use("t1", ""),
            _result("t1", "boom", session="A", ts="2026-01-01T00:00:00Z"),
            _tool_use("t2"),
            _result("t2", "\x1b[0m", session="A", ts="2026-01-01T00:00:00Z"),
        ],
    )
    report = compute_report(_parse(tmp_path).failures, 24)
    assert report["totals"]["anomalies"] == {"empty_error": 1, "empty_bash_target": 1}


def test_single_file_path_and_missing(tmp_path: Path) -> None:
    f = _write(tmp_path / "x.jsonl", _failure("t1", "A", "2026-01-01T00:00:00Z"))
    parsed = parse_paths([str(f), str(tmp_path / "missing")], None, _PREFIX)
    assert parsed.transcripts == 1
    assert len(parsed.failures) == 1
