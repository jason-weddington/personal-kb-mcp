"""Hermetic tests for the replay experiment harness (``scripts/replay``).

Every ``claude`` and ``psql`` call is replaced through the ``run_claude`` /
``run_psql`` seams; the real binaries are never invoked. The hook scripts are
exercised by RUNNING them as subprocesses, the way Claude Code would.
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAY_DIR = REPO_ROOT / "scripts" / "replay"
SCRIPT_PATH = REPLAY_DIR / "replay.py"
PRETOOL = REPLAY_DIR / "hooks" / "pretool_gate.py"
SLICE_HOOK = REPLAY_DIR / "hooks" / "session_slice.py"
FIXTURE = REPLAY_DIR / "fixtures" / "synthetic_pairs.json"


def _load(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


replay = _load("replay_harness", SCRIPT_PATH)
pretool_mod = _load("replay_pretool_gate", PRETOOL)

VERSION = "2.1.292 (Claude Code)"


def _proc(argv: list[str], stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)


def _is_version(argv: list[str]) -> bool:
    return argv == ["claude", "--version"]


@pytest.fixture(autouse=True)
def _no_real_binaries(monkeypatch):
    """Default seams: answer --version, fail any other call loudly."""

    def fake_claude(argv, *, stdin, cwd, env, timeout):
        if _is_version(argv):
            return _proc(argv, VERSION)
        raise AssertionError(f"unexpected claude call: {argv[:4]}")

    def fake_psql(argv):
        raise AssertionError("unexpected psql call")

    monkeypatch.setattr(replay, "run_claude", fake_claude)
    monkeypatch.setattr(replay, "run_psql", fake_psql)


# --------------------------------------------------------------------------- #
# Stream fixtures (keys verified against claude 2.1.292)
# --------------------------------------------------------------------------- #


def ev_hook(event: str) -> dict[str, Any]:
    return {
        "type": "system",
        "subtype": "hook_started",
        "hook_event": event,
        "hook_name": f"{event}:startup",
    }


def ev_init(mcp_servers: list[Any] | None = None) -> dict[str, Any]:
    return {
        "type": "system",
        "subtype": "init",
        "mcp_servers": mcp_servers or [],
        "apiKeySource": "none",
        "claude_code_version": "2.1.292",
    }


def ev_tool(tool_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "assistant",
        "message": {
            "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}]
        },
    }


def ev_text(text: str) -> dict[str, Any]:
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def ev_tool_result(tool_id: str, content: str) -> dict[str, Any]:
    return {
        "type": "user",
        "message": {
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content}]
        },
    }


def ev_result(text: str = "done", cost: float = 0.01) -> dict[str, Any]:
    return {
        "type": "result",
        "subtype": "success",
        "num_turns": 2,
        "total_cost_usd": cost,
        "result": text,
    }


def jsonl(events: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _run_hook_command(command: str, payload: dict[str, Any] | None) -> subprocess.CompletedProcess:
    parts = shlex.split(command)
    assert parts[0] == "python3"
    return subprocess.run(  # noqa: S603
        [sys.executable, *parts[1:]],
        input=json.dumps(payload) if payload is not None else "",
        capture_output=True,
        text=True,
        check=False,
    )


def simulate_claude(
    cwd: Path,
    tool_calls: list[tuple[str, dict[str, Any]]],
    *,
    cost: float = 0.01,
    run_hooks: bool = True,
    with_result: bool = True,
) -> str:
    """Behave like ``claude -p``: fire the project hooks from settings.json and emit a stream."""
    settings = json.loads((cwd / ".claude" / "settings.json").read_text())
    hooks = settings["hooks"]
    events: list[dict[str, Any]] = []
    if "SessionStart" in hooks:
        events.append(ev_hook("SessionStart"))
        if run_hooks:
            _run_hook_command(hooks["SessionStart"][0]["hooks"][0]["command"], {})
    events.append(ev_init())
    for i, (name, tool_input) in enumerate(tool_calls):
        tool_id = f"toolu_{i:02d}"
        events.append(ev_tool(tool_id, name, tool_input))
        events.append(ev_hook("PreToolUse"))
        denied = None
        if run_hooks:
            proc = _run_hook_command(
                hooks["PreToolUse"][0]["hooks"][0]["command"],
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": name,
                    "tool_input": tool_input,
                    "tool_use_id": tool_id,
                    "cwd": str(cwd),
                },
            )
            if proc.stdout.strip():
                denied = json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecisionReason"]
        if denied is None and name == "Write":
            target = Path(tool_input["file_path"])
            target.write_text(tool_input.get("content", ""))
        events.append(ev_tool_result(tool_id, denied or "ok"))
    events.append(ev_text("All done."))
    if with_result:
        events.append(ev_result(cost=cost))
    return jsonl(events)


def _task(task_id: str = "c01", *, kind: str = "correction", gate: Any = None, **kw: Any):
    base = {
        "task_id": task_id,
        "kind": kind,
        "project_ref": "syn-project",
        "old_id": None if kind == "control" else f"o-{task_id}",
        "new_id": None if kind == "control" else f"n-{task_id}",
        "wrong_belief": None if kind == "control" else f"wrong belief {task_id}",
        "correction": None if kind == "control" else f"correction {task_id}",
        "task_prompt": f"Do the thing for {task_id}.",
        "files": {"NOTES.md": "notes\n"},
        "gate": gate,
        "success_criterion": "NOTES.md mentions the thing.",
    }
    base.update(kw)
    return base


def _write_tasks(out: Path, tasks: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# --------------------------------------------------------------------------- #
# (a) export --sqlite
# --------------------------------------------------------------------------- #


def _build_kb(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        create table knowledge_entries (
            id text primary key, project_ref text, short_title text, long_title text,
            knowledge_details text, entry_type text, is_active integer);
        create table graph_edges (
            source text, target text, edge_type text, properties text not null default '{}');
        """
    )
    entries = [
        ("o1", "p", "old one", "Old one", "old one details", "decision", 0),
        ("o2", "p", "old two", "Old two", "old two details", "lesson_learned", 0),
        ("o3", "p", "old three", "Old three", "d", "decision", 0),
        ("o4", "p", "old four", "Old four", "d", "decision", 0),
        ("n1", "p", "new one", "New one", "new one details", "decision", 1),
        ("n2", "p", "new two", "New two", "new two details", "lesson_learned", 1),
        ("n3", "p", "new three", "New three", "d", "decision", 1),
        ("n4", "p", "new four", "New four", "d", "decision", 0),
    ]
    conn.executemany("insert into knowledge_entries values (?,?,?,?,?,?,?)", entries)
    edges = [
        ("n1", "o1", "supersedes", "{}"),
        ("n2", "o2", "supersedes", '{"source": "manual"}'),
        ("n3", "o3", "supersedes", '{"source": "llm"}'),
        ("n4", "o4", "supersedes", "{}"),
        ("n1", "o3", "references", "{}"),
    ]
    conn.executemany("insert into graph_edges values (?,?,?,?)", edges)
    conn.commit()
    conn.close()


def test_export_sqlite_filters_like_supersession(tmp_path):
    db = tmp_path / "kb.db"
    _build_kb(db)
    out = tmp_path / "out"
    assert replay.main(["export", "--sqlite", str(db), "--out", str(out)]) == 0
    pairs = json.loads((out / "pairs.json").read_text())
    assert len(pairs) == 2
    assert all(list(p.keys()) == list(replay.PAIR_KEYS) for p in pairs)
    assert len(replay.PAIR_KEYS) == 11
    assert [(p["new_id"], p["old_id"]) for p in pairs] == [("n1", "o1"), ("n2", "o2")]
    assert pairs[0]["old_details"] == "old one details"
    assert pairs[0]["new_short_title"] == "new one"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["export_dropped"] == {"llm_edge": 1, "inactive_superseder": 1, "mental_map": 0}
    assert manifest["export_source"] == "sqlite"
    assert manifest["pairs_exported"] == 2
    assert manifest["claude_version"] == VERSION
    assert "export_started_ts" in manifest


def test_export_filter_mental_map_and_unparseable_properties():
    rows = [
        {"edge_properties": "not json", "new_is_active": 1, "new_type": "decision"},
        {"edge_properties": {"source": "llm"}, "new_is_active": 1, "new_type": "decision"},
        {"edge_properties": "{}", "new_is_active": "1", "new_type": "mental_map"},
    ]
    pairs, dropped = replay.filter_rows(rows)
    assert len(pairs) == 1
    assert dropped == {"llm_edge": 1, "inactive_superseder": 0, "mental_map": 1}


def test_export_requires_exactly_one_source(tmp_path):
    out = tmp_path / "out"
    assert replay.main(["export", "--out", str(out)]) == 2
    assert replay.main(["export", "--out", str(out), "--dsn", "x", "--sqlite", "y"]) == 2


# --------------------------------------------------------------------------- #
# (a2) export --dsn
# --------------------------------------------------------------------------- #


def test_export_dsn_argv_and_password_redaction(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    dsn = "postgresql://kb:s3cret@db.example/kb"
    row = {
        "new_id": "n1",
        "old_id": "o1",
        "edge_properties": "{}",
        "new_is_active": 1,
        "new_type": "decision",
    }

    def fake_psql(argv):
        calls.append(argv)
        return _proc(argv, json.dumps([row]))

    monkeypatch.setattr(replay, "run_psql", fake_psql)
    out = tmp_path / "out"
    assert replay.main(["export", "--dsn", dsn, "--out", str(out)]) == 0
    assert calls == [
        [
            "psql",
            "-X",
            "-At",
            "-d",
            dsn,
            "-c",
            f"select coalesce(json_agg(t), '[]') from ({replay.EXPORT_SQL}) t",  # noqa: S608
        ]
    ]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["export_source"] == "dsn"
    assert manifest["export_target"] == "postgresql://kb:***@db.example/kb"
    assert "s3cret" not in (out / "manifest.json").read_text()
    assert len(json.loads((out / "pairs.json").read_text())) == 1


def test_export_dsn_failures_exit_1(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "run_psql", lambda argv: _proc(argv, "", 2, "boom"))
    assert replay.main(["export", "--dsn", "postgresql:///kb", "--out", str(tmp_path)]) == 1

    def missing(argv):
        raise FileNotFoundError("psql")

    monkeypatch.setattr(replay, "run_psql", missing)
    assert replay.main(["export", "--dsn", "postgresql:///kb", "--out", str(tmp_path)]) == 1


def test_redact_dsn_key_value():
    assert replay.redact_dsn("host=h password=abc dbname=kb") == "host=h password=*** dbname=kb"


# --------------------------------------------------------------------------- #
# (b) generate
# --------------------------------------------------------------------------- #


def _pair(n: int, new_title: str) -> dict[str, Any]:
    return {
        "new_id": f"n{n:02d}",
        "old_id": f"o{n:02d}",
        "project_ref": "syn-project",
        "old_type": "decision",
        "old_short_title": f"MARK-{n:02d}",
        "old_long_title": "old long",
        "old_details": "old details",
        "new_type": "decision",
        "new_short_title": new_title,
        "new_long_title": "new long",
        "new_details": "new details",
    }


def _verdict(**kw: Any) -> dict[str, Any]:
    base = {
        "is_correction": True,
        "already_in_steering": False,
        "wrong_belief": "wb",
        "correction": "corr",
        "task_prompt": "Please do the work.",
        "files": {"a.txt": "x"},
        "gate": {"tool": "Bash", "target_regex": "git push"},
        "success_criterion": "sc",
    }
    base.update(kw)
    return base


GEN_PLAN = {
    "MARK-01": ("ok", _verdict(is_correction=False)),
    "MARK-02": ("ok", _verdict(already_in_steering=True)),
    "MARK-03": ("ok", _verdict(gate={"tool": "Bash", "target_regex": "("})),
    "MARK-04": ("ok", _verdict(task_prompt="Remember: Use The New Way when you deploy.")),
    "MARK-05": ("ok", _verdict(files={"../escape.txt": "x"})),
    "MARK-06": ("fail", None),
    "MARK-07": ("ok", _verdict()),
    "MARK-08": ("ok", _verdict(gate=None)),
    "MARK-09": ("ok", _verdict()),
}


def _gen_pairs() -> list[dict[str, Any]]:
    pairs = [_pair(n, f"title {n}") for n in range(1, 10)]
    pairs[3]["new_short_title"] = "use the new way"  # leak
    pairs[7]["new_short_title"] = "   "  # blank title: leak check skipped
    pairs[7]["old_short_title"] = "MARK-08"
    return pairs


class GenStub:
    def __init__(self, controls: list[dict[str, Any]] | None = None):
        self.prompts: list[str] = []
        self.argvs: list[list[str]] = []
        self.cwds: list[Path] = []
        self.controls = controls if controls is not None else []

    def __call__(self, argv, *, stdin, cwd, env, timeout):
        if _is_version(argv):
            return _proc(argv, VERSION)
        self.argvs.append(argv)
        self.prompts.append(stdin)
        self.cwds.append(cwd)
        assert timeout == 300
        assert list(Path(cwd).iterdir()) == []
        schema = json.loads(argv[argv.index("--json-schema") + 1])
        if schema == replay.CONTROLS_SCHEMA:
            payload = {"structured_output": {"controls": self.controls}, "total_cost_usd": 0.02}
            return _proc(argv, json.dumps(payload))
        for mark, (mode, verdict) in GEN_PLAN.items():
            if f"## old_short_title\n{mark}\n" in stdin:
                if mode == "fail":
                    return _proc(argv, "", 1)
                payload = {"structured_output": verdict, "total_cost_usd": 0.1, "num_turns": 1}
                return _proc(argv, json.dumps(payload))
        raise AssertionError("unknown pair")


def _control(n: int, **kw: Any) -> dict[str, Any]:
    base = {
        "task_prompt": f"control {n}",
        "files": {"c.txt": "x"},
        "success_criterion": "sc",
        "project_ref": "syn-project",
    }
    base.update(kw)
    return base


def test_generate_drop_reasons_numbering_and_log(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "pairs.json").write_text(json.dumps(list(reversed(_gen_pairs()))))
    steer = tmp_path / "steer.md"
    steer.write_text("STEER-MARKER-ONE")
    steer2 = tmp_path / "steer2.md"
    steer2.write_text("STEER-MARKER-TWO")
    stub = GenStub(
        controls=[
            _control(1),
            _control(2, files={".claude/settings.json": "{}"}),
            _control(3),
            _control(4),
        ]
    )
    monkeypatch.setattr(replay, "run_claude", stub)
    rc = replay.main(
        [
            "generate",
            "--out",
            str(out),
            "--limit",
            "2",
            "--controls",
            "3",
            "--steering",
            str(steer),
            str(steer2),
            str(tmp_path / "missing.md"),
        ]
    )
    assert rc == 0
    # Pairs 1..8 processed (two kept); pair 9 never called because --limit counts KEPT.
    pair_calls = [p for p in stub.prompts if "## old_short_title" in p]
    assert len(pair_calls) == 8
    assert all("MARK-09" not in p for p in stub.prompts)
    assert stub.argvs[0] == replay.GEN_ARGV("opus", replay.GEN_SCHEMA)
    assert "STEER-MARKER-ONE\n\nSTEER-MARKER-TWO" in pair_calls[0]
    for rule in replay.GEN_RULES:
        assert rule in pair_calls[0]

    log = _read_jsonl(out / "generate-log.jsonl")
    pair_lines = [x for x in log if x.get("kind") != "control"]
    assert [x["drop_reason"] for x in pair_lines] == [
        "not_correction",
        "in_steering",
        "bad_regex",
        "leak",
        "bad_files",
        "call_error",
        None,
        None,
    ]
    assert [x["task_id"] for x in pair_lines][-2:] == ["c01", "c02"]
    assert pair_lines[3]["leak_check_hit"] is True
    assert pair_lines[7]["leak_check_hit"] is False
    for line in log:
        for key in (
            "old_id",
            "new_id",
            "decision",
            "drop_reason",
            "task_id",
            "is_correction",
            "already_in_steering",
            "gate",
            "leak_check_hit",
            "model",
            "total_cost_usd",
            "duration_ms",
            "structured_output",
        ):
            assert key in line

    control_lines = [x for x in log if x.get("kind") == "control"]
    assert [x["drop_reason"] for x in control_lines] == ["bad_files", None]

    tasks = _read_jsonl(out / "tasks.jsonl")
    assert [t["task_id"] for t in tasks] == ["c01", "c02", "k01", "k02"]
    assert tasks[0]["kind"] == "correction"
    assert tasks[0]["old_id"] == "o07"
    assert tasks[2]["kind"] == "control"
    assert tasks[2]["gate"] is None and tasks[2]["wrong_belief"] is None
    assert set(tasks[0]) == {
        "task_id",
        "kind",
        "project_ref",
        "old_id",
        "new_id",
        "wrong_belief",
        "correction",
        "task_prompt",
        "files",
        "gate",
        "success_criterion",
    }
    controls_prompt = stub.prompts[-1]
    assert replay.GEN_RULES[2] in controls_prompt
    assert "syn-project" in controls_prompt
    ledger = _read_jsonl(out / "cost-ledger.jsonl")
    assert all(row["phase"] == "generate" for row in ledger)
    assert len(ledger) == 9  # 7 parsed pair calls + 1 failed call + 1 controls call


def test_generate_short_controls(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "pairs.json").write_text(json.dumps([_pair(7, "title 7")]))
    monkeypatch.setattr(replay, "run_claude", GenStub(controls=[_control(1)]))
    assert replay.main(["generate", "--out", str(out), "--steering", str(tmp_path / "x")]) == 0
    log = _read_jsonl(out / "generate-log.jsonl")
    short = [x for x in log if x.get("kind") == "control"]
    assert short[-1]["drop_reason"] == "short"
    assert short[-1]["got"] == 1
    assert [t["task_id"] for t in _read_jsonl(out / "tasks.jsonl")] == ["c01", "k01"]


def test_generate_budget_stops_calls(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "pairs.json").write_text(json.dumps(_gen_pairs()[:3]))
    (out / "cost-ledger.jsonl").write_text(json.dumps({"phase": "run", "cost_usd": 5.0}) + "\n")
    stub = GenStub()
    monkeypatch.setattr(replay, "run_claude", stub)
    rc = replay.main(
        ["generate", "--out", str(out), "--budget-usd", "5.0", "--steering", str(tmp_path / "x")]
    )
    assert rc == 0
    assert stub.prompts == []
    log = _read_jsonl(out / "generate-log.jsonl")
    assert [x["drop_reason"] for x in log] == ["budget"] * 4  # 3 pairs + the controls call


def test_bad_file_key():
    assert replay.bad_file_key("/abs")
    assert replay.bad_file_key("a/../b")
    assert replay.bad_file_key(".claude/settings.json")
    assert replay.bad_file_key(".claudex")
    assert not replay.bad_file_key("src/a.py")


# --------------------------------------------------------------------------- #
# (c) run argv + env
# --------------------------------------------------------------------------- #


class RunStub:
    """Simulated claude for ``run``; per-call behaviour picked by ``plan``."""

    def __init__(self, plan=None):
        self.calls: list[dict[str, Any]] = []
        self.plan = plan or (lambda stub, argv, stdin, cwd: _proc(argv, self.default(stdin, cwd)))

    def default(self, stdin: str, cwd: Path) -> str:
        if stdin == replay.CANARY_PROMPT:
            return simulate_claude(cwd, [("Bash", {"command": "touch CANARY"})])
        return simulate_claude(
            cwd,
            [
                ("Read", {"file_path": str(cwd / "NOTES.md")}),
                ("Bash", {"command": "git push origin main"}),
                ("Write", {"file_path": str(cwd / "OUT.md"), "content": "shipped"}),
            ],
        )

    def __call__(self, argv, *, stdin, cwd, env, timeout):
        if _is_version(argv):
            return _proc(argv, VERSION)
        self.calls.append({"argv": argv, "stdin": stdin, "cwd": cwd, "env": env})
        assert timeout == 300
        return self.plan(self, argv, stdin, Path(cwd))

    @property
    def task_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["stdin"] != replay.CANARY_PROMPT]


def test_run_argv_env_and_ok_results(tmp_path, monkeypatch):
    for key in replay.CHILD_ENV_STRIP:
        monkeypatch.setenv(key, "set-by-test")
    out = tmp_path / "out"
    _write_tasks(
        out,
        [
            _task("c01", gate={"tool": "Bash", "target_regex": "git push"}),
            _task("k01", kind="control"),
        ],
    )
    stub = RunStub()
    monkeypatch.setattr(replay, "run_claude", stub)
    assert replay.main(["run", "--out", str(out)]) == 0
    assert len(stub.calls) == 1 + 2 * 3
    assert stub.calls[0]["stdin"] == "Run the shell command: touch CANARY"
    for call in stub.calls:
        assert call["argv"] == replay.RUN_ARGV("sonnet")
        for key in replay.CHILD_ENV_STRIP:
            assert key not in call["env"]
    # Tasks in order, arms in ARMS order within a task.
    assert [c["stdin"] for c in stub.task_calls] == ["Do the thing for c01."] * 3 + [
        "Do the thing for k01."
    ] * 3
    for task_id in ("c01", "k01"):
        for arm in replay.ARMS:
            result = json.loads((out / "runs" / task_id / arm / "1" / "result.json").read_text())
            assert result["status"] == "ok", result
            assert (out / "runs" / task_id / arm / "1" / "stream.jsonl").exists()
    gated = json.loads((out / "runs/c01/soft_gate/1/result.json").read_text())
    assert gated["gate_applicable"] is True and gated["gate_fired"] is True
    assert gated["n_gate_deny"] == 1 and gated["n_tool_calls"] == 3
    plain = json.loads((out / "runs/c01/kb_off/1/result.json").read_text())
    assert plain["gate_fired"] is False and plain["n_sandbox_deny"] == 1
    assert plain["api_key_source"] == "none" and plain["claude_version"] == "2.1.292"
    slice_log = _read_jsonl(out / "runs/c01/slice/1/hook-log.jsonl")
    assert [x["decision"] for x in slice_log].count("slice_delivered") == 1
    ledger = _read_jsonl(out / "cost-ledger.jsonl")
    assert [row["phase"] for row in ledger] == ["canary"] + ["run"] * 6
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run_args"]["arms"] == "kb_off,slice,soft_gate"
    # Scratch dirs are removed.
    assert all(not Path(c["cwd"]).exists() for c in stub.calls)


def test_run_use_api_key_keeps_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDECODE", "1")
    out = tmp_path / "out"
    _write_tasks(out, [_task("c01")])
    stub = RunStub()
    monkeypatch.setattr(replay, "run_claude", stub)
    assert replay.main(["run", "--out", str(out), "--arms", "kb_off", "--use-api-key"]) == 0
    assert all(c["env"].get("ANTHROPIC_API_KEY") == "sk-test" for c in stub.calls)
    assert all("CLAUDECODE" not in c["env"] for c in stub.calls)


def test_run_rejects_unknown_arm(tmp_path):
    assert replay.main(["run", "--out", str(tmp_path), "--arms", "kb_off,bogus"]) == 2


def test_slice_text_is_shared_and_capped():
    tasks = [_task(f"c{i:02d}") for i in range(25, 0, -1)] + [_task("k01", kind="control")]
    text = replay.build_slice_text(tasks)
    lines = text.splitlines()
    assert lines[0] == replay.SLICE_HEADER
    assert len(lines) == 21
    assert lines[1] == "- correction c01 (replaces: wrong belief c01)"


# --------------------------------------------------------------------------- #
# (d) invalid + error reasons
# --------------------------------------------------------------------------- #

_TOOL = ev_tool("toolu_x", "Read", {"file_path": "a"})
_HOOKED = {"decision": "allow", "tool_use_id": "toolu_x"}


@pytest.mark.parametrize(
    ("arm", "events", "hook_log", "expected"),
    [
        ("kb_off", [ev_init(["kb"]), ev_result()], [], "mcp_servers_present"),
        ("kb_off", [ev_hook("SessionStart"), ev_init(), ev_result()], [], "hook_leak"),
        (
            "slice",
            [ev_hook("SessionStart"), ev_hook("SessionStart"), ev_init(), ev_result()],
            [{"decision": "slice_delivered"}],
            "hook_leak",
        ),
        ("soft_gate", [ev_init(), ev_hook("UserPromptSubmit"), ev_result()], [], "hook_leak"),
        ("slice", [ev_hook("SessionStart"), ev_init(), ev_result()], [], "slice_not_delivered"),
        ("kb_off", [ev_init(), ev_result()], [{"decision": "hook_error"}], "hook_error"),
        ("kb_off", [ev_init(), _TOOL, ev_result()], [], "unhooked_tool_call"),
        (
            "kb_off",
            [ev_init(), _TOOL, ev_result()],
            [{**_HOOKED, "decision": "gate_deny"}],
            "gate_in_wrong_arm",
        ),
        (
            "soft_gate",
            [ev_init(), _TOOL, ev_result()],
            [_HOOKED, {"decision": "slice_delivered"}],
            "slice_in_wrong_arm",
        ),
        (
            "soft_gate",
            [ev_init(), _TOOL, ev_result()],
            [{**_HOOKED, "decision": "gate_deny"}],
            None,
        ),
        (
            "slice",
            [ev_hook("SessionStart"), ev_init(), _TOOL, ev_result()],
            [{"decision": "slice_delivered"}, _HOOKED],
            None,
        ),
    ],
)
def test_invalid_reasons(arm, events, hook_log, expected):
    assert replay.invalid_reason(arm, events, hook_log) == expected


def test_invalid_reasons_cover_constant():
    assert set(replay.INVALID_REASONS) == {
        "mcp_servers_present",
        "hook_leak",
        "slice_not_delivered",
        "hook_error",
        "unhooked_tool_call",
        "gate_in_wrong_arm",
        "slice_in_wrong_arm",
    }


def _run_one(tmp_path, monkeypatch, plan, task=None):
    out = tmp_path / "out"
    _write_tasks(out, [task or _task("c01")])
    stub = RunStub(plan=plan)
    monkeypatch.setattr(replay, "run_claude", stub)
    assert replay.main(["run", "--out", str(out), "--arms", "kb_off"]) == 0
    return json.loads((out / "runs/c01/kb_off/1/result.json").read_text()), stub


def _canary_ok(stub, argv, stdin, cwd):
    return _proc(argv, stub.default(stdin, cwd))


def test_error_timeout(tmp_path, monkeypatch):
    def plan(stub, argv, stdin, cwd):
        if stdin == replay.CANARY_PROMPT:
            return _canary_ok(stub, argv, stdin, cwd)
        raise subprocess.TimeoutExpired(argv, 300)

    result, _ = _run_one(tmp_path, monkeypatch, plan)
    assert (result["status"], result["error_reason"]) == ("error", "timeout")


def test_error_nonzero_exit(tmp_path, monkeypatch):
    def plan(stub, argv, stdin, cwd):
        if stdin == replay.CANARY_PROMPT:
            return _canary_ok(stub, argv, stdin, cwd)
        return _proc(argv, simulate_claude(cwd, []), 1)

    result, _ = _run_one(tmp_path, monkeypatch, plan)
    assert (result["status"], result["error_reason"]) == ("error", "nonzero_exit")


def test_error_no_result_event(tmp_path, monkeypatch):
    def plan(stub, argv, stdin, cwd):
        if stdin == replay.CANARY_PROMPT:
            return _canary_ok(stub, argv, stdin, cwd)
        return _proc(argv, simulate_claude(cwd, [], with_result=False))

    result, _ = _run_one(tmp_path, monkeypatch, plan)
    assert (result["status"], result["error_reason"]) == ("error", "no_result_event")


def test_error_bad_files(tmp_path, monkeypatch):
    result, stub = _run_one(
        tmp_path, monkeypatch, None, task=_task("c01", files={"../x": "escape"})
    )
    assert (result["status"], result["error_reason"]) == ("error", "bad_files")
    assert stub.task_calls == []
    assert not (tmp_path / "x").exists()


def test_run_marks_invalid_from_real_hooks(tmp_path, monkeypatch):
    def plan(stub, argv, stdin, cwd):
        if stdin == replay.CANARY_PROMPT:
            return _canary_ok(stub, argv, stdin, cwd)
        # A tool call that never reached the hook.
        return _proc(argv, simulate_claude(cwd, [("Read", {"file_path": "a"})], run_hooks=False))

    result, _ = _run_one(tmp_path, monkeypatch, plan)
    assert (result["status"], result["invalid_reason"]) == ("invalid", "unhooked_tool_call")


# --------------------------------------------------------------------------- #
# (e) canary
# --------------------------------------------------------------------------- #


def test_canary_failure_exits_3(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _write_tasks(out, [_task("c01")])

    def plan(stub, argv, stdin, cwd):
        return _proc(
            argv, simulate_claude(cwd, [("Bash", {"command": "touch CANARY"})], run_hooks=False)
        )

    stub = RunStub(plan=plan)
    monkeypatch.setattr(replay, "run_claude", stub)
    assert replay.main(["run", "--out", str(out)]) == 3
    assert "sandbox canary failed" in capsys.readouterr().err
    assert stub.task_calls == []
    assert not (out / "runs" / "c01").exists()
    assert _read_jsonl(out / "cost-ledger.jsonl")[0]["phase"] == "canary"


def test_canary_touch_detected(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _write_tasks(out, [_task("c01")])

    def plan(stub, argv, stdin, cwd):
        text = simulate_claude(cwd, [("Bash", {"command": "touch CANARY"})])
        (cwd / "CANARY").write_text("")
        return _proc(argv, text)

    monkeypatch.setattr(replay, "run_claude", RunStub(plan=plan))
    assert replay.main(["run", "--out", str(out)]) == 3


# --------------------------------------------------------------------------- #
# (f) budget + resumability
# --------------------------------------------------------------------------- #


def test_run_budget_cutoff(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _write_tasks(out, [_task("c01"), _task("c02")])
    stub = RunStub()
    monkeypatch.setattr(replay, "run_claude", stub)
    # canary 0.01 + first run 0.01 reaches the 0.02 cap.
    assert replay.main(["run", "--out", str(out), "--budget-usd", "0.02"]) == 0
    assert len(stub.task_calls) == 1
    statuses = {
        (t, a): json.loads((out / "runs" / t / a / "1" / "result.json").read_text())["status"]
        for t in ("c01", "c02")
        for a in replay.ARMS
    }
    assert statuses[("c01", "kb_off")] == "ok"
    assert [v for k, v in statuses.items() if k != ("c01", "kb_off")] == ["skipped_budget"] * 5


def test_run_resumability(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _write_tasks(out, [_task("c01")])
    for arm, status in (("kb_off", "ok"), ("slice", "error"), ("soft_gate", "skipped_budget")):
        run_dir = out / "runs" / "c01" / arm / "1"
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text(json.dumps({"status": status}))
    stub = RunStub()
    monkeypatch.setattr(replay, "run_claude", stub)
    assert replay.main(["run", "--out", str(out)]) == 0
    assert len(stub.task_calls) == 2
    assert json.loads((out / "runs/c01/kb_off/1/result.json").read_text()) == {"status": "ok"}
    assert json.loads((out / "runs/c01/slice/1/result.json").read_text())["status"] == "ok"

    stub2 = RunStub()
    monkeypatch.setattr(replay, "run_claude", stub2)
    assert replay.main(["run", "--out", str(out)]) == 0
    assert stub2.calls == []  # nothing pending: not even the canary


# --------------------------------------------------------------------------- #
# (g) judge
# --------------------------------------------------------------------------- #


def _make_run(out: Path, task: dict[str, Any], arm: str, events, hook_log, status="ok", rep=1):
    run_dir = out / "runs" / task["task_id"] / arm / str(rep)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stream.jsonl").write_text(jsonl(events))
    (run_dir / "hook-log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in hook_log))
    result = {
        "task_id": task["task_id"],
        "arm": arm,
        "rep": rep,
        "status": status,
        "gate_applicable": task.get("gate") is not None,
        "gate_fired": any(x.get("decision") == "gate_deny" for x in hook_log),
        "result_subtype": "success",
        "n_sandbox_deny": 0,
        "n_tool_calls": 1,
    }
    (run_dir / "result.json").write_text(json.dumps(result))
    return run_dir


class JudgeStub:
    def __init__(self, verdict: dict[str, Any]):
        self.prompts: list[str] = []
        self.argvs: list[list[str]] = []
        self.verdict = verdict

    def __call__(self, argv, *, stdin, cwd, env, timeout):
        if _is_version(argv):
            return _proc(argv, VERSION)
        self.prompts.append(stdin)
        self.argvs.append(argv)
        return _proc(argv, json.dumps({"structured_output": self.verdict, "total_cost_usd": 0.03}))


def test_judge_digest_redaction_exclusion_and_dedupe(tmp_path, monkeypatch):
    out = tmp_path / "out"
    task = _task("c01", gate={"tool": "Bash", "target_regex": "git push"})
    _write_tasks(out, [task])
    events = [
        ev_hook("PreToolUse"),
        ev_init(),
        ev_text(f"I was told: kb correction: X. Also {replay.SLICE_HEADER} and more."),
        ev_tool("toolu_gated", "Bash", {"command": "git push GATED-MARK"}),
        ev_tool_result("toolu_gated", replay.GATE_PREFIX + "never push"),
        ev_tool("toolu_ok", "Read", {"file_path": "NOTES.md"}),
        ev_tool_result("toolu_ok", "TOOL-RESULT-MARK"),
        ev_result("final answer"),
    ]
    hook_log = [
        {"decision": "gate_deny", "tool_use_id": "toolu_gated"},
        {"decision": "allow", "tool_use_id": "toolu_ok"},
    ]
    run_dir = _make_run(out, task, "soft_gate", events, hook_log)
    _make_run(out, task, "kb_off", [ev_init(), ev_result("x")], [], status="invalid")
    stub = JudgeStub({"repeated": False, "evidence": "did not push"})
    monkeypatch.setattr(replay, "run_claude", stub)
    assert replay.main(["judge", "--out", str(out)]) == 0
    assert len(stub.prompts) == 1
    prompt = stub.prompts[0]
    assert stub.argvs[0] == replay.JUDGE_ARGV("sonnet", replay.JUDGE_SCHEMA_CORRECTION)
    assert "GATED-MARK" not in prompt
    assert "TOOL-RESULT-MARK" not in prompt
    assert '[Read] {"file_path": "NOTES.md"}' in prompt
    assert "final answer" in prompt
    assert "[redacted]" in prompt
    for term in replay.BLINDING_BLACKLIST:
        assert term.lower() not in prompt.lower()
    assert "Known corrections from the knowledge base".lower() not in prompt.lower()
    assert (run_dir / "judge_prompt.txt").read_text() == prompt
    assert (
        json.loads((run_dir / "judge_raw.json").read_text())["structured_output"]["repeated"]
        is False
    )
    lines = _read_jsonl(out / "judgments.jsonl")
    assert len(lines) == 1
    line = lines[0]
    assert line["repeated"] is False and line["judge_error"] is None
    assert line["excluded_gate_calls"] == 1
    assert line["kind"] == "correction"
    assert len(line["prompt_sha256"]) == 64

    # Rerun: nothing new.
    assert replay.main(["judge", "--out", str(out)]) == 0
    assert len(stub.prompts) == 1
    assert len(_read_jsonl(out / "judgments.jsonl")) == 1


def test_judge_blinding_violation_makes_no_call(tmp_path, monkeypatch):
    out = tmp_path / "out"
    task = _task("k01", kind="control", task_prompt="Compare kb_off results.")
    _write_tasks(out, [task])
    _make_run(out, task, "slice", [ev_init(), ev_result("ok")], [])
    stub = JudgeStub({"passed": True, "evidence": "e"})
    monkeypatch.setattr(replay, "run_claude", stub)
    assert replay.main(["judge", "--out", str(out)]) == 0
    assert stub.prompts == []
    assert _read_jsonl(out / "judgments.jsonl")[0]["judge_error"] == "blinding_violation"


def test_judge_budget_and_truncation(tmp_path, monkeypatch):
    out = tmp_path / "out"
    task = _task("k01", kind="control")
    _write_tasks(out, [task])
    _make_run(out, task, "kb_off", [ev_init(), ev_text("a" * 7000), ev_result("b" * 7000)], [])
    (out / "cost-ledger.jsonl").write_text(json.dumps({"cost_usd": 3.0}) + "\n")
    stub = JudgeStub({"passed": True, "evidence": "e"})
    monkeypatch.setattr(replay, "run_claude", stub)
    assert replay.main(["judge", "--out", str(out), "--budget-usd", "3"]) == 0
    assert stub.prompts == []
    line = _read_jsonl(out / "judgments.jsonl")[0]
    assert line["judge_error"] == "budget"
    assert line["digest_truncated"] is True

    assert replay.main(["judge", "--out", str(out), "--budget-usd", "10"]) == 0
    assert len(stub.prompts) == 1
    assert "\n...\n" in stub.prompts[0]
    assert stub.argvs[0] == replay.JUDGE_ARGV("sonnet", replay.JUDGE_SCHEMA_CONTROL)
    lines = _read_jsonl(out / "judgments.jsonl")
    assert lines[-1]["passed"] is True and lines[-1]["judge_error"] is None


def test_build_digest_truncation_shape():
    digest, full, truncated, excluded = replay.build_digest(
        [ev_text("x" * 13000), ev_result("")], []
    )
    assert truncated and full > 12000 and excluded == 0
    assert len(digest) == 6000 + len("\n...\n") + 6000


# --------------------------------------------------------------------------- #
# (h) report
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("b", "c", "expected"), [(0, 0, 1.0), (5, 0, 0.0625), (1, 9, 0.021484375), (3, 3, 1.0)]
)
def test_mcnemar_vectors(b, c, expected):
    assert replay.mcnemar_exact(b, c) == expected


def _report_fixture(out: Path) -> None:
    gate = {"tool": "Bash", "target_regex": "git push"}
    tasks = [_task("c01", gate=gate)] + [_task(f"c0{i}") for i in (2, 3, 4)]
    tasks += [_task("k01", kind="control"), _task("k02", kind="control")]
    _write_tasks(out, tasks)
    repeated = {
        "kb_off": {"c01": True, "c02": True, "c03": True, "c04": True},
        "slice": {"c01": False, "c02": False, "c03": False, "c04": True},
        "soft_gate": {"c01": False, "c02": True, "c03": True, "c04": True},
    }
    passed = {
        "kb_off": {"k01": True, "k02": True},
        "slice": {"k01": True, "k02": False},
        "soft_gate": {"k01": True, "k02": True},
    }
    judgments = []
    for arm in replay.ARMS:
        for task in tasks:
            tid = task["task_id"]
            hook_log = []
            if arm == "soft_gate" and tid == "c01":
                hook_log = [{"decision": "gate_deny", "tool_use_id": "t"}]
            _make_run(out, task, arm, [ev_init(), ev_result()], hook_log)
            if task["kind"] == "correction":
                judgments.append(
                    {
                        "task_id": tid,
                        "arm": arm,
                        "rep": 1,
                        "kind": "correction",
                        "repeated": repeated[arm][tid],
                        "evidence": "ev",
                        "judge_error": None,
                    }
                )
            else:
                judgments.append(
                    {
                        "task_id": tid,
                        "arm": arm,
                        "rep": 1,
                        "kind": "control",
                        "passed": passed[arm][tid],
                        "evidence": "ev",
                        "judge_error": None,
                    }
                )
    judgments.append(
        {"task_id": "c01", "arm": "slice", "rep": 1, "judge_error": "blinding_violation"}
    )
    (out / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judgments))


def test_report_rates_and_verdicts(tmp_path):
    out = tmp_path / "out"
    _report_fixture(out)
    assert replay.main(["report", "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    rows = {r["arm"]: r for r in report["arms"]}
    assert [r["arm"] for r in report["arms"]] == [
        "kb_off",
        "slice",
        "soft_gate",
        "soft_gate_applicable",
    ]
    assert rows["kb_off"]["n_paired"] == 4 and rows["kb_off"]["repeat_rate"] == 1.0
    assert rows["kb_off"]["verdict"] == "baseline"
    assert rows["kb_off"]["delta_pts"] is None and rows["kb_off"]["p_value"] is None

    s = rows["slice"]
    assert (s["n_paired"], s["repeats"], s["b"], s["c"]) == (4, 1, 3, 0)
    assert s["repeat_rate"] == 0.25 and s["delta_pts"] == 75.0
    assert s["p_value"] == 0.25
    assert s["control_pass_rate"] == 0.5 and s["control_drop_pts"] == 50.0
    assert s["verdict"] == "stop-control"

    g = rows["soft_gate"]
    assert (g["b"], g["c"], g["delta_pts"], g["control_drop_pts"]) == (1, 0, 25.0, 0.0)
    assert g["p_value"] == 1.0
    assert g["verdict"] == "meets"

    ga = rows["soft_gate_applicable"]
    assert (ga["n_paired"], ga["repeats"], ga["delta_pts"]) == (1, 0, 100.0)

    assert report["pilot"] is True
    assert report["soft_gate"]["applicable_runs"] == 1
    assert report["soft_gate"]["gate_fired_runs"] == 1
    assert report["counts"]["tasks_correction"] == 4
    assert report["counts"]["blinding_violations"] == 1
    assert len(report["per_task"]) == 18
    md = (out / "report.md").read_text()
    assert md.splitlines()[0] == "PILOT - underpowered; verdicts are indicative only"
    assert "| arm | n_paired | repeats |" in md
    assert "WARN blinding violations: 1" in md


def test_report_insufficient_and_below(tmp_path):
    out = tmp_path / "out"
    tasks = [_task(f"c{i:02d}") for i in range(1, 11)] + [_task("k01", kind="control")]
    _write_tasks(out, tasks)
    judgments = []
    for arm in ("kb_off", "slice"):
        for task in tasks:
            _make_run(out, task, arm, [ev_init(), ev_result()], [])
            if task["kind"] == "correction":
                rep = arm == "kb_off" or task["task_id"] != "c01"
                judgments.append(
                    {
                        "task_id": task["task_id"],
                        "arm": arm,
                        "rep": 1,
                        "repeated": rep,
                        "judge_error": None,
                    }
                )
            else:
                judgments.append(
                    {"task_id": "k01", "arm": arm, "rep": 1, "passed": True, "judge_error": None}
                )
    (out / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judgments))
    report = replay.build_report(out)
    rows = {r["arm"]: r for r in report["arms"]}
    assert rows["slice"]["delta_pts"] == 10.0 and rows["slice"]["verdict"] == "below"
    assert rows["soft_gate"]["verdict"] == "insufficient"
    assert report["pilot"] is False
    assert not replay.render_report_md(report).startswith("PILOT")


def test_report_regex_miss_and_skew(tmp_path):
    out = tmp_path / "out"
    task = _task("c01", gate={"tool": "Bash", "target_regex": "git push"})
    _write_tasks(out, [task])
    _make_run(out, task, "kb_off", [ev_init(), ev_result()], [])
    run_dir = _make_run(
        out,
        task,
        "soft_gate",
        [ev_init(), ev_result()],
        [{"decision": "sandbox_deny", "gate_tool_match": True, "gate_regex_match": False}],
    )
    result = json.loads((run_dir / "result.json").read_text())
    result["result_subtype"] = "error_max_turns"
    (run_dir / "result.json").write_text(json.dumps(result))
    judgments = [
        {"task_id": "c01", "arm": "kb_off", "rep": 1, "repeated": True, "judge_error": None},
        {"task_id": "c01", "arm": "soft_gate", "rep": 1, "repeated": True, "judge_error": None},
    ]
    (out / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judgments))
    report = replay.build_report(out)
    assert report["soft_gate"]["repeated_without_fire"] == 1
    assert report["soft_gate"]["tool_match_no_regex_runs"] == 1
    md = replay.render_report_md(report)
    assert "possible regex misses: c01" in md
    assert "WARN run-shape skew: soft_gate error_max_turns" in md


# --------------------------------------------------------------------------- #
# (i) fixture, (j) repo guard, (k) settings literals, (l) parity
# --------------------------------------------------------------------------- #


def test_fixture_ids_are_synthetic():
    data = json.loads(FIXTURE.read_text())
    assert set(data) == {"pairs", "tasks"}
    assert data["pairs"] and data["tasks"]
    for pair in data["pairs"]:
        assert list(pair) == list(replay.PAIR_KEYS)
        assert pair["new_id"].startswith("syn-") and pair["old_id"].startswith("syn-")
    for task in data["tasks"]:
        assert task["task_id"].startswith("syn-")
    c01 = next(t for t in data["tasks"] if t["task_id"] == "syn-c01")
    assert c01["kind"] == "correction"
    assert c01["gate"] == {"tool": "Bash", "target_regex": "git push"}


@pytest.mark.parametrize(
    "argv",
    [
        ["export", "--sqlite", "x.db"],
        ["generate"],
        ["run"],
        ["judge"],
        ["report"],
    ],
)
@pytest.mark.parametrize("target", ["", "scripts/replay/out"])
def test_out_inside_repo_refused(argv, target, capsys):
    out = REPO_ROOT / target if target else REPO_ROOT
    before = out.exists()
    assert replay.main([*argv, "--out", str(out)]) == 2
    assert "refusing to write replay data inside the repo" in capsys.readouterr().err
    assert out.exists() == before


def test_settings_literals_per_arm(tmp_path):
    cfg = tmp_path / "run dir" / "hook-config.json"
    gate_cmd = f"python3 {shlex.quote(str(PRETOOL.resolve()))} {shlex.quote(str(cfg))}"
    slice_cmd = f"python3 {shlex.quote(str(SLICE_HOOK.resolve()))} {shlex.quote(str(cfg))}"
    pretool = {
        "PreToolUse": [
            {"matcher": "*", "hooks": [{"type": "command", "command": gate_cmd, "timeout": 5}]}
        ]
    }
    assert replay.build_settings("kb_off", cfg) == {"hooks": pretool}
    assert replay.build_settings("soft_gate", cfg) == {"hooks": pretool}
    assert replay.build_settings("slice", cfg) == {
        "hooks": {
            **pretool,
            "SessionStart": [{"hooks": [{"type": "command", "command": slice_cmd, "timeout": 5}]}],
        }
    }


def test_hook_config_shape(tmp_path):
    task = _task("c01", gate={"tool": "Bash", "target_regex": "git push"})
    run_dir = tmp_path / "run"
    for arm in replay.ARMS:
        cfg = replay.build_hook_config(
            arm=arm, task=task, rep=2, scratch=tmp_path, run_dir=run_dir, slice_text="S"
        )
        assert cfg["arm"] == arm and cfg["task_id"] == "c01" and cfg["rep"] == 2
        assert cfg["scratch_dir"] == str(tmp_path.resolve())
        assert cfg["hook_log"] == str((run_dir / "hook-log.jsonl").resolve())
        assert (cfg["gate"] is not None) == (arm == "soft_gate")
        assert (cfg["slice_text"] is not None) == (arm == "slice")
    gated = replay.build_hook_config(
        arm="soft_gate", task=task, rep=1, scratch=tmp_path, run_dir=run_dir, slice_text="S"
    )
    assert gated["gate"] == {
        "tool": "Bash",
        "target_regex": "git push",
        "correction": "correction c01",
    }


_PARITY_CASES = [
    ("Bash", {"command": "ls"}),
    ("Read", {"file_path": "/a"}),
    ("Edit", {"file_path": "/b"}),
    ("Write", {"file_path": "/c"}),
    ("MultiEdit", {"file_path": "/d"}),
    ("NotebookEdit", {"notebook_path": "/e.ipynb"}),
    ("Glob", {"pattern": "*.py"}),
    ("Grep", {"pattern": "foo"}),
    ("mcp__x__y", {"command": "ls"}),
    ("Bash", {"command": 42}),
    ("Bash", "not a dict"),
]


@pytest.mark.parametrize(("tool", "tool_input"), _PARITY_CASES)
def test_local_target_field_map(tool, tool_input):
    expected = {
        "Bash": "ls",
        "Read": "/a",
        "Edit": "/b",
        "Write": "/c",
        "MultiEdit": "/d",
        "NotebookEdit": "/e.ipynb",
        "Glob": "*.py",
        "Grep": "foo",
    }
    want = expected.get(tool, "") if isinstance(tool_input, dict) else ""
    if isinstance(tool_input, dict) and not isinstance(next(iter(tool_input.values())), str):
        want = ""
    assert pretool_mod.extract_target(tool, tool_input) == want


@pytest.mark.parametrize(("tool", "tool_input"), _PARITY_CASES)
def test_target_parity_with_kb_core_cues(tool, tool_input):
    cues = pytest.importorskip("kb_core.cues")
    assert pretool_mod.extract_target(tool, tool_input) == cues.extract_target(tool, tool_input)


# --------------------------------------------------------------------------- #
# Hook scripts, run as subprocesses
# --------------------------------------------------------------------------- #


def _hook_cfg(
    tmp_path: Path, *, arm="soft_gate", gate=None, slice_text=None
) -> tuple[Path, Path, Path]:
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    log = tmp_path / "hook-log.jsonl"
    cfg = tmp_path / "hook-config.json"
    cfg.write_text(
        json.dumps(
            {
                "arm": arm,
                "task_id": "c01",
                "rep": 1,
                "scratch_dir": str(scratch.resolve()),
                "hook_log": str(log),
                "gate": gate,
                "slice_text": slice_text,
            }
        )
    )
    return cfg, scratch, log


def _pretool(cfg: Path, payload: Any) -> subprocess.CompletedProcess:
    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(  # noqa: S603
        [sys.executable, str(PRETOOL), str(cfg)],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def _reason(proc: subprocess.CompletedProcess) -> str:
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse" and out["permissionDecision"] == "deny"
    return out["permissionDecisionReason"]


def test_pretool_bash_gate_denies_once_then_sandbox(tmp_path):
    gate = {"tool": "Bash", "target_regex": "git push", "correction": "only release.sh publishes"}
    cfg, scratch, log = _hook_cfg(tmp_path, gate=gate)
    payload = {
        "tool_name": "Bash",
        "tool_input": {"command": "git push origin main"},
        "tool_use_id": "t1",
        "cwd": str(scratch),
    }
    first = _pretool(cfg, payload)
    assert first.returncode == 0
    assert _reason(first) == "KB correction: only release.sh publishes"
    second = _pretool(cfg, payload)
    assert _reason(second) == replay.SANDBOX_BASH
    lines = _read_jsonl(log)
    assert [x["decision"] for x in lines] == ["gate_deny", "sandbox_deny"]
    assert lines[0]["gate_tool_match"] is True and lines[0]["gate_regex_match"] is True
    assert lines[0]["tool_use_id"] == "t1" and lines[0]["target"] == "git push origin main"
    assert set(lines[0]) == {
        "ts",
        "task_id",
        "arm",
        "rep",
        "event",
        "tool_name",
        "tool_use_id",
        "target",
        "gate_tool",
        "gate_regex",
        "gate_tool_match",
        "gate_regex_match",
        "decision",
        "error_type",
    }


def test_pretool_read_gate_then_allow(tmp_path):
    gate = {"tool": "Read", "target_regex": "secret", "correction": "do not read it"}
    cfg, scratch, log = _hook_cfg(tmp_path, gate=gate)
    payload = {
        "tool_name": "Read",
        "tool_input": {"file_path": str(scratch / "secret.txt")},
        "tool_use_id": "t1",
        "cwd": str(scratch),
    }
    assert _reason(_pretool(cfg, payload)).startswith(replay.GATE_PREFIX)
    second = _pretool(cfg, payload)
    assert second.returncode == 0 and second.stdout == ""
    assert [x["decision"] for x in _read_jsonl(log)] == ["gate_deny", "allow"]


def test_pretool_sandbox_paths_and_tools(tmp_path):
    cfg, scratch, log = _hook_cfg(tmp_path, arm="kb_off")
    outside = {"tool_name": "Read", "tool_input": {"file_path": "/etc/passwd"}, "cwd": str(scratch)}
    assert _reason(_pretool(cfg, outside)) == replay.SANDBOX_PATH
    rel_in = {"tool_name": "Write", "tool_input": {"file_path": "sub/a.txt"}, "cwd": str(scratch)}
    assert _pretool(cfg, rel_in).stdout == ""
    rel_out = {"tool_name": "Edit", "tool_input": {"file_path": "../x"}, "cwd": str(scratch)}
    assert _reason(_pretool(cfg, rel_out)) == replay.SANDBOX_PATH
    glob_nopath = {"tool_name": "Glob", "tool_input": {"pattern": "**/*"}, "cwd": str(scratch)}
    assert _pretool(cfg, glob_nopath).stdout == ""
    grep_out = {
        "tool_name": "Grep",
        "tool_input": {"pattern": "x", "path": "/"},
        "cwd": str(scratch),
    }
    assert _reason(_pretool(cfg, grep_out)) == replay.SANDBOX_PATH
    mcp = {"tool_name": "mcp__kb__search", "tool_input": {}, "cwd": str(scratch)}
    assert _reason(_pretool(cfg, mcp)) == replay.SANDBOX_TOOL
    decisions = [x["decision"] for x in _read_jsonl(log)]
    assert decisions == [
        "sandbox_deny",
        "allow",
        "sandbox_deny",
        "allow",
        "sandbox_deny",
        "sandbox_deny",
    ]


def test_pretool_fails_closed(tmp_path):
    cfg, _, log = _hook_cfg(tmp_path, arm="kb_off")
    bad = _pretool(cfg, "{not json")
    assert bad.returncode == 0
    assert _reason(bad) == replay.SANDBOX_TOOL
    line = _read_jsonl(log)[-1]
    assert line["decision"] == "hook_error" and line["error_type"] == "JSONDecodeError"

    missing = _pretool(
        tmp_path / "nope.json", {"tool_name": "Bash", "tool_input": {"command": "ls"}}
    )
    assert missing.returncode == 0
    assert _reason(missing) == replay.SANDBOX_BASH

    regex_dir = tmp_path / "r"
    regex_dir.mkdir()
    bad_regex_cfg, scratch, log2 = _hook_cfg(
        regex_dir, gate={"tool": "Bash", "target_regex": "(", "correction": "c"}
    )
    payload = {"tool_name": "Bash", "tool_input": {"command": "x"}, "cwd": str(scratch)}
    proc = _pretool(bad_regex_cfg, payload)
    assert _reason(proc) == replay.SANDBOX_BASH
    assert _read_jsonl(log2)[-1]["error_type"] in ("error", "PatternError")


def test_session_slice_envelope_and_log(tmp_path):
    cfg, _, log = _hook_cfg(tmp_path, arm="slice", slice_text="SLICE BODY")
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(SLICE_HOOK), str(cfg)],
        input="{}",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout) == {
        "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "SLICE BODY"}
    }
    line = _read_jsonl(log)[0]
    assert (line["event"], line["decision"], line["arm"]) == (
        "SessionStart",
        "slice_delivered",
        "slice",
    )


def test_session_slice_fails_open(tmp_path):
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(SLICE_HOOK), str(tmp_path / "missing.json")],
        input="{}",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0 and proc.stdout == ""


def test_round_trip_config_written_by_run_is_accepted(tmp_path):
    task = _task("c01", gate={"tool": "Bash", "target_regex": "git push"})
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for arm in replay.ARMS:
        run_dir = tmp_path / "runs" / arm
        cfg_path = replay.prepare_run(
            arm=arm, task=task, rep=1, scratch=scratch, run_dir=run_dir, slice_text="SLICE"
        )
        settings = json.loads((scratch / ".claude" / "settings.json").read_text())
        assert settings == replay.build_settings(arm, cfg_path)
        proc = _run_hook_command(
            settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"],
            {
                "tool_name": "Read",
                "tool_input": {"file_path": "NOTES.md"},
                "tool_use_id": "t",
                "cwd": str(scratch),
            },
        )
        assert proc.returncode == 0 and proc.stdout == ""
        if arm == "slice":
            out = _run_hook_command(settings["hooks"]["SessionStart"][0]["hooks"][0]["command"], {})
            assert json.loads(out.stdout)["hookSpecificOutput"]["additionalContext"] == "SLICE"
        decisions = [x["decision"] for x in _read_jsonl(run_dir / "hook-log.jsonl")]
        assert "hook_error" not in decisions
        assert decisions[0] == "allow"
