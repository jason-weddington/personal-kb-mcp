"""Turn digest: window, items, body, Stop flow and CLI wiring."""

from __future__ import annotations

import dataclasses
import io
import json
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import pytest

from personal_kb_hook import cli, prevention, turn_digest
from personal_kb_hook.paths import (
    get_event_drop_log_path,
    get_turn_digest_log_path,
    get_turn_state_path,
)
from personal_kb_hook.turn_digest import TurnWindow, build_digest, read_turn

if TYPE_CHECKING:
    from pathlib import Path

_REAL_SPAWN_SENDER = turn_digest.spawn_sender
_CN = "<" + "command-name>"
_CNE = "<" + "/command-name>"
_KEYS14 = {
    "event_id", "session_id", "harness", "mode", "engine", "host", "hook_version",
    "project", "turn_index", "ts", "user_prompt", "items", "final_message", "truncated",
}  # fmt: skip
_ROW_KEYS = {
    "ts", "op", "session_id", "turn_index", "event_id", "capture", "action", "error",
    "state", "boundary", "last_uuid_missing", "cut", "records_parsed",
    "unknown_block_types", "items_built", "items_sent", "bytes", "truncated",
    "user_prompt_null", "final_message_null", "elapsed_ms",
}  # fmt: skip


@pytest.fixture(autouse=True)
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.test")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-key")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("HEADLESS_BUILD_ENGINE", raising=False)
    monkeypatch.delenv("PERSONAL_KB_LISTENER", raising=False)
    (tmp_path / "home").mkdir()


def _msg(rtype: str, uuid: str, block: dict[str, Any]) -> dict[str, Any]:
    return {"type": rtype, "uuid": uuid, "message": {"content": [block]}}


def prompt(uuid: str, text: Any, **kw: Any) -> dict[str, Any]:
    rec = {
        "type": "user", "uuid": uuid, "isSidechain": False, "origin": {"kind": "human"},
        "message": {"role": "user", "content": text},
    }  # fmt: skip
    rec.update(kw)
    return rec


def atext(uuid: str, text: str, **kw: Any) -> dict[str, Any]:
    rec = {
        "type": "assistant",
        "uuid": uuid,
        "message": {"content": [{"type": "text", "text": text}]},
    }
    rec.update(kw)
    return rec


def tuse(uuid: str, tid: str, cmd: str) -> dict[str, Any]:
    block = {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": cmd}}
    return {"type": "assistant", "uuid": uuid, "message": {"content": [block]}}


def tres(uuid: str, tid: str, content: Any, err: bool = False) -> dict[str, Any]:
    block = {"type": "tool_result", "tool_use_id": tid, "content": content, "is_error": err}
    return {"type": "user", "uuid": uuid, "message": {"content": [block]}}


def fixture_f() -> list[dict[str, Any]]:
    return [
        prompt("u1", "first prompt"),
        atext("u2", "old turn"),
        prompt("u3", "use port 8080"),
        atext("u4", "port 8080 is free"),
        tuse("u5", "t1", "git push origin main"),
        tres("u6", "t1", "rejected", True),
        tuse("u7", "t2", "git push origin HEAD:main"),
        tres("u8", "t2", [{"type": "text", "text": "ok"}]),
        _msg("assistant", "u9", {"type": "thinking", "thinking": "x"}),
    ]  # fmt: skip


def write(path: Path, records: list[Any]) -> str:
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def call(tid: str, cmd: str, target_class: str, err: bool | None = None) -> dict[str, Any]:
    return {
        "kind": "tool_call", "tool_use_id": tid, "tool": "Bash", "target": cmd,
        "target_class": target_class,
    }  # fmt: skip


def res(tid: str, excerpt: str, err: bool = False) -> dict[str, Any]:
    return {"kind": "tool_result", "tool_use_id": tid, "is_error": err, "excerpt": excerpt}


ITEMS_A = [
    {"kind": "assistant_text", "text": "port 8080 is free"},
    call("t1", "git push origin main", "git push"),
    res("t1", "rejected", True),
    call("t2", "git push origin HEAD:main", "git push"),
    res("t2", "ok"),
]


@pytest.fixture
def tf(tmp_path: Path) -> Path:
    write(tmp_path / "t.jsonl", fixture_f())
    return tmp_path / "t.jsonl"


WIN_A = TurnWindow("use port 8080", ITEMS_A, "u9", False, "prompt", False, 7, {})


def test_window_a_b_c(tf: Path) -> None:
    assert read_turn(str(tf), None) == WIN_A
    w = read_turn(str(tf), "u6")
    assert w is not None
    assert (w.user_prompt, w.boundary, w.records_parsed, w.newest_uuid) == (
        None,
        "last_uuid",
        4,
        "u9",
    )
    assert w.items == ITEMS_A[3:]
    assert not w.last_uuid_missing
    assert read_turn(str(tf), "nope") == dataclasses.replace(WIN_A, last_uuid_missing=True)
    extra = [
        prompt("u10", "<" + "task-notification>x" + "<" + "/task-notification>",
               origin={"kind": "task-notification"}),
        tuse("u11", "t3", "ls"),
        tres("u12", "t3", "a"),
    ]  # fmt: skip
    with tf.open("a") as fh:
        fh.write("\n".join(json.dumps(r) for r in extra) + "\n")
    w = read_turn(str(tf), "u9")
    assert w is not None
    assert w.user_prompt is None
    assert w.boundary == "last_uuid"
    assert w.items == [call("t3", "ls", "ls"), res("t3", "a")]


def test_noise_is_not_a_boundary(tf: Path) -> None:
    echo = prompt(
        "e1",
        f"{_CN}/model{_CNE}\n   <" + "command-message>model<" + "/command-message>",
    )
    del echo["origin"]
    noise = [
        echo,
        {"type": "user", "uuid": "e2", "message": {"content": "<" + "local-command-stdout>x"}},
        prompt("e3", "meta text", isMeta=True),
        prompt("e4", [{"type": "text", "text": "[Request interrupted by user]"}]),
    ]
    with tf.open("a") as fh:
        fh.write("\n".join(json.dumps(r) for r in noise) + "\n")
    w = read_turn(str(tf), None)
    assert w is not None
    assert (w.user_prompt, w.boundary) == ("use port 8080", "prompt")


def test_prompt_shapes(tmp_path: Path) -> None:
    rec = prompt("p", "hi")
    del rec["origin"]
    w = read_turn(write(tmp_path / "a.jsonl", [atext("a", "x"), rec, atext("b", "y")]), None)
    assert w is not None
    assert (w.boundary, w.user_prompt) == ("prompt", "hi")
    multi = prompt(
        "p", [{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}]
    )
    assert turn_digest.human_prompt_text(multi) == "a\nb"
    side = [prompt("s", "side", isSidechain=True), atext("t", "gone", isSidechain=True)]
    w = read_turn(write(tmp_path / "b.jsonl", [prompt("p", "main"), *side]), None)
    assert w is not None
    assert (w.user_prompt, w.items) == ("main", [])


def test_excerpt_and_caps(tmp_path: Path) -> None:
    long = "A" * 1000 + "B" * 1000 + "C" * 1000
    assert turn_digest.result_excerpt(long) == "A" * 746 + "\n[...]\n" + "C" * 747
    assert len(turn_digest.result_excerpt(long)) == 1500
    assert turn_digest.result_excerpt("x" * 1500) == "x" * 1500
    recs = [prompt("p", "go"), tuse("a", "t", "echo " + "x" * 600), atext("b", "z" * 2500)]
    w = read_turn(write(tmp_path / "c.jsonl", recs), None)
    assert w is not None
    assert len(w.items[0]["target"]) == 500
    assert w.items[0]["target_class"] == "echo"
    assert len(w.items[1]["text"]) == 2000


def test_tail_cut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(turn_digest, "TURN_TAIL_BYTES", 2048)
    recs = [prompt("p", "go")] + [atext(f"a{i}", "x" * 100) for i in range(40)]
    w = read_turn(write(tmp_path / "d.jsonl", recs), None)
    assert w is not None
    assert (w.cut, w.boundary, w.user_prompt) == (True, "none", None)
    raw = build_digest({"session_id": "s"}, turn_index=0, project=None, window=w)
    assert raw is not None
    assert json.loads(raw)["truncated"] is True


def test_missing_file(tmp_path: Path) -> None:
    assert read_turn(str(tmp_path / "nope"), None) is None


def test_body_shape(tf: Path) -> None:
    raw = build_digest(
        {"session_id": "s1", "last_assistant_message": "Pushed to HEAD:main."},
        turn_index=2, project="personal-kb", window=read_turn(str(tf), None),
    )  # fmt: skip
    assert raw is not None
    body = json.loads(raw)
    assert set(body) == _KEYS14
    assert body["event_id"] == "s1:2"
    assert (body["harness"], body["mode"], body["engine"]) == ("claude-code", "interactive", None)
    assert body["project"] == "personal-kb"
    assert body["user_prompt"] == "use port 8080"
    assert body["items"] == ITEMS_A
    assert body["final_message"] == "Pushed to HEAD:main."
    assert body["truncated"] is False
    assert body["hook_version"] == prevention._hook_version()
    assert body["host"] == socket.gethostname()
    assert isinstance(body["ts"], str)


def test_headless_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADLESS_BUILD_ENGINE", "claude-code-sonnet")
    raw = build_digest({"session_id": "s"}, turn_index=0, project=None, window=None)
    assert raw is not None
    body = json.loads(raw)
    assert (body["mode"], body["engine"]) == ("headless", "claude-code-sonnet")
    assert (body["items"], body["user_prompt"], body["truncated"]) == ([], None, False)


@pytest.mark.parametrize("lam", [None, "", "   ", 5])
def test_final_message_none(lam: Any) -> None:
    payload: dict[str, Any] = {"session_id": "s"}
    if lam is not None:
        payload["last_assistant_message"] = lam
    raw = build_digest(payload, turn_index=0, project=None, window=None)
    assert raw is not None
    assert json.loads(raw)["final_message"] is None


def test_prompt_caps(tmp_path: Path) -> None:
    w = read_turn(write(tmp_path / "e.jsonl", [prompt("p", "p" * 5000)]), None)
    pl = {"session_id": "s", "last_assistant_message": "m" * 5000}
    raw = build_digest(pl, turn_index=0, project=None, window=w)
    assert raw is not None
    body = json.loads(raw)
    assert len(body["user_prompt"]) == 4000
    assert len(body["final_message"]) == 4000
    assert body["truncated"] is False


def test_fit_to_64k(tmp_path: Path) -> None:
    recs: list[Any] = [prompt("p", "go")]
    for i in range(60):
        recs += [tuse(f"a{i}", f"t{i}", f"ls {i}"), tres(f"b{i}", f"t{i}", "r" * 3000)]
    w = read_turn(write(tmp_path / "f.jsonl", recs), None)
    assert w is not None
    raw = build_digest({"session_id": "s"}, turn_index=0, project=None, window=w)
    assert raw is not None
    assert len(raw) <= 65536
    items = json.loads(raw)["items"]
    assert json.loads(raw)["truncated"] is True
    assert items
    assert items == w.items[-len(items) :]
    body = json.loads(raw)
    body["items"] = w.items[-len(items) - 1 :]
    assert len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()) > 65536


def test_items_cap(tmp_path: Path) -> None:
    recs = [prompt("p", "go")] + [atext(f"a{i}", f"t{i}") for i in range(250)]
    w = read_turn(write(tmp_path / "g.jsonl", recs), None)
    raw = build_digest({"session_id": "s"}, turn_index=0, project=None, window=w)
    assert raw is not None
    body = json.loads(raw)
    assert len(body["items"]) == 200
    assert body["items"][0]["text"] == "t50"
    assert body["truncated"] is True


def test_lone_surrogate(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"
    path.write_text(
        json.dumps(prompt("p", "go")) + "\n" + json.dumps(tres("r", "t", "bad \ud800 byte")) + "\n"
    )
    raw = build_digest(
        {"session_id": "s"}, turn_index=0, project=None, window=read_turn(str(path), None)
    )
    assert raw is not None
    assert json.loads(raw)["items"][0]["excerpt"] == "bad ? byte"


def test_pathological_session_id_fails_closed() -> None:
    assert (
        build_digest({"session_id": "x" * 70000}, turn_index=0, project=None, window=None) is None
    )


_BAD_PROMPTS: list[dict[str, Any]] = [
    prompt("1", "x", isSidechain=True),
    prompt("1", "x", isMeta=True),
    prompt("1", "x", isCompactSummary=True),
    prompt("1", "x", origin={"kind": "task-notification"}),
    prompt("1", "x", origin={"kind": "peer"}),
    {"type": "user", "message": "x"},
    prompt("1", [{"type": "tool_result", "tool_use_id": "t"}]),
    prompt("1", 5),
    prompt("1", "   "),
    prompt("1", "<" + "command-message>x"),
    prompt("1", "<" + "local-command-stdout>x"),
    prompt("1", "[Request interrupted by user]"),
    {"type": "assistant", "message": {"content": "x"}},
]


@pytest.mark.parametrize("rec", _BAD_PROMPTS)
def test_not_human_prompt(rec: dict[str, Any]) -> None:
    assert turn_digest.human_prompt_text(rec) is None


def test_non_dict_record_and_prefixes() -> None:
    assert turn_digest.human_prompt_text("x") is None
    for e in turn_digest.PROMPT_EXCLUDED_PREFIXES:
        assert "\\" not in e
        assert e[0] in "<["


def test_garbage_lines_ignored(tmp_path: Path) -> None:
    f = fixture_f()
    junk: list[Any] = [
        "", "not json", "[1,2]",
        _msg("assistant", "j1", {"type": "tool_use", "name": "Bash", "input": {}}),
        _msg("assistant", "j2", {"type": "tool_use", "id": "z", "name": "", "input": {}}),
        _msg("user", "j3", {"type": "tool_result", "content": "x"}),
        atext("j4", "  "),
        {**tuse("j5", "side", "ls"), "isSidechain": True},
    ]  # fmt: skip
    recs = f[:4] + junk + f[4:]
    w = read_turn(write(tmp_path / "i.jsonl", recs), None)
    assert w is not None
    assert w.items == ITEMS_A


def test_result_excerpt_variants() -> None:
    assert (
        turn_digest.result_excerpt([{"type": "image"}, {"type": "text", "text": "a"}, "junk"])
        == "a"
    )
    assert turn_digest.result_excerpt(None) == ""
    assert turn_digest.result_excerpt(5) == ""


def test_unknown_block_types(tmp_path: Path) -> None:
    rec = {"type": "assistant", "uuid": "a", "message": {"content": [
        {"type": "server_tool_use", "id": "s"}, "x", {"type": 5}, {"k": 1},
    ]}}  # fmt: skip
    w = read_turn(write(tmp_path / "j.jsonl", [prompt("p", "go"), rec]), None)
    assert w is not None
    assert w.unknown_block_types == {"server_tool_use": 1, "<non-dict>": 1, "<no-type>": 2}
    assert w.items == []


def test_message_without_content_list(tmp_path: Path) -> None:
    recs = [prompt("p", "go"), {"type": "assistant", "uuid": "a", "message": {"content": "str"}},
            {"type": "assistant", "uuid": "b", "message": "str"}]  # fmt: skip
    w = read_turn(write(tmp_path / "k.jsonl", recs), None)
    assert w is not None
    assert w.items == []


# ─── Stop flow ───────────────────────────────────────────────────────────────

SID = "s1"


def seed(mode: Any, project: str = "personal-kb") -> None:
    cache: dict[str, Any] = {
        "gate": {"enabled": False, "shadow": True, "max_denies": 2},
        "index": [],
        "project": project,
    }
    if mode is not None:
        cache["surprise_capture"] = mode
    prevention._write_cache(SID, cache)


def payload(tp: str | None, **kw: Any) -> dict[str, Any]:
    p: dict[str, Any] = {"hook_event_name": "Stop", "session_id": SID}
    if tp is not None:
        p["transcript_path"] = tp
    p.update(kw)
    return p


def rows() -> list[dict[str, Any]]:
    path = get_turn_digest_log_path(SID)
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def drops() -> list[dict[str, Any]]:
    path = get_event_drop_log_path()
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def state() -> Any:
    return json.loads(get_turn_state_path(SID).read_text())


def body_of(spawn: tuple[str, str]) -> dict[str, Any]:
    with open(spawn[0], encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def tmp_files() -> list[str]:
    from pathlib import Path

    return [p.name for p in Path(tempfile.gettempdir()).glob("kb-turn-*.json")]


@pytest.mark.parametrize("mode", ["off", None, "weird"])
def test_off_does_nothing(mode: Any, tf: Path, turn_spawns: list[tuple[str, str]]) -> None:
    seed(mode)
    turn_digest.stop(payload(str(tf)))
    assert turn_spawns == []
    assert tmp_files() == []
    assert drops() == []
    assert not get_turn_digest_log_path(SID).exists()
    assert state() == {"next_turn_index": 1, "last_uuid": None}


def test_no_cache(tf: Path, turn_spawns: list[tuple[str, str]]) -> None:
    turn_digest.stop(payload(str(tf)))
    assert turn_spawns == []
    assert state() == {"next_turn_index": 1, "last_uuid": None}


def test_no_session_id() -> None:
    turn_digest.stop({"hook_event_name": "Stop"})


def test_shadow_two_stops(tf: Path, turn_spawns: list[tuple[str, str]]) -> None:
    seed("shadow")
    turn_digest.stop(payload(str(tf)))
    assert len(turn_spawns) == 1
    assert body_of(turn_spawns[0])["event_id"] == "s1:0"
    assert state() == {"next_turn_index": 1, "last_uuid": "u9"}
    (row,) = rows()
    assert set(row) == _ROW_KEYS
    assert (row["op"], row["action"], row["event_id"], row["capture"]) == (
        "stop",
        "sent",
        "s1:0",
        "shadow",
    )
    assert (row["state"], row["boundary"], row["items_built"], row["items_sent"]) == (
        "missing",
        "prompt",
        5,
        5,
    )
    assert (row["truncated"], row["error"]) == (False, None)
    turn_digest.stop(payload(str(tf)))
    assert body_of(turn_spawns[1])["event_id"] == "s1:1"
    assert state()["next_turn_index"] == 2
    row2 = rows()[1]
    assert (row2["state"], row2["boundary"], row2["items_built"]) == ("ok", "last_uuid", 0)


def test_on_spawns(tf: Path, turn_spawns: list[tuple[str, str]]) -> None:
    seed("on")
    turn_digest.stop(payload(str(tf)))
    assert len(turn_spawns) == 1


def test_flip_after_off(tf: Path, turn_spawns: list[tuple[str, str]]) -> None:
    seed("off")
    for _ in range(3):
        turn_digest.stop(payload(str(tf)))
    seed("shadow")
    turn_digest.stop(payload(str(tf)))
    assert [body_of(s)["event_id"] for s in turn_spawns] == ["s1:3"]


def test_follow_on_window(tf: Path, turn_spawns: list[tuple[str, str]]) -> None:
    seed("shadow")
    turn_digest.stop(payload(str(tf)))
    extra = [tuse("u11", "t3", "ls"), tres("u12", "t3", "a")]
    with tf.open("a") as fh:
        fh.write("\n".join(json.dumps(r) for r in extra) + "\n")
    turn_digest.stop(payload(str(tf)))
    b = body_of(turn_spawns[1])
    assert b["user_prompt"] is None
    assert b["items"] == [call("t3", "ls", "ls"), res("t3", "a")]


@pytest.mark.parametrize("tp", [None, "/nonexistent/x.jsonl"])
def test_unreadable_transcript(tp: str | None, turn_spawns: list[tuple[str, str]]) -> None:
    seed("shadow")
    turn_digest.stop(payload(tp))
    assert len(turn_spawns) == 1
    assert body_of(turn_spawns[0])["items"] == []
    assert [(d["op"], d["reason"]) for d in drops()] == [("turn_digest", "transcript_unreadable")]
    assert rows()[0]["boundary"] is None


def test_no_url_key(
    tf: Path, monkeypatch: pytest.MonkeyPatch, turn_spawns: list[tuple[str, str]]
) -> None:
    monkeypatch.delenv("PERSONAL_KB_API_KEY")
    seed("shadow")
    turn_digest.stop(payload(str(tf)))
    assert turn_spawns == []
    assert [d["reason"] for d in drops()] == ["no_url_key"]
    assert rows()[0]["action"] == "no_url_key"
    assert state()["next_turn_index"] == 1


def test_spawn_failed(tf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any) -> None:
        raise OSError("no")

    monkeypatch.setattr(turn_digest, "spawn_sender", boom)
    seed("shadow")
    turn_digest.stop(payload(str(tf)))
    assert [d["reason"] for d in drops()] == ["spawn_failed"]
    assert tmp_files() == []
    assert rows()[0]["action"] == "spawn_failed"
    assert state()["next_turn_index"] == 1


def test_empty_project(tf: Path, turn_spawns: list[tuple[str, str]]) -> None:
    seed("shadow", project="")
    turn_digest.stop(payload(str(tf)))
    assert body_of(turn_spawns[0])["project"] is None


@pytest.mark.parametrize(
    "content", ["not json", '{"next_turn_index": true}', '{"next_turn_index": -3}']
)
def test_corrupt_state(content: str, tf: Path, turn_spawns: list[tuple[str, str]]) -> None:
    seed("shadow")
    p = get_turn_state_path(SID)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    turn_digest.stop(payload(str(tf)))
    assert body_of(turn_spawns[0])["event_id"] == "s1:0"
    assert rows()[0]["state"] == "corrupt"


def test_read_turn_raises(
    tf: Path, monkeypatch: pytest.MonkeyPatch, turn_spawns: list[tuple[str, str]]
) -> None:
    def boom(*a: Any) -> None:
        raise RuntimeError("x")

    monkeypatch.setattr(turn_digest, "read_turn", boom)
    seed("shadow")
    assert turn_digest.stop(payload(str(tf))) is None
    assert [(d["op"], d["reason"]) for d in drops()] == [("turn_digest", "error")]
    assert turn_spawns == []
    assert state()["next_turn_index"] == 1
    assert (rows()[0]["action"], rows()[0]["error"]) == ("error", "RuntimeError")


def test_tempfile_raises(
    tf: Path, monkeypatch: pytest.MonkeyPatch, turn_spawns: list[tuple[str, str]]
) -> None:
    def boom(*a: Any, **k: Any) -> None:
        raise OSError("x")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", boom)
    seed("shadow")
    turn_digest.stop(payload(str(tf)))
    assert [d["reason"] for d in drops()] == ["error"]
    assert turn_spawns == []


def test_too_large(
    tf: Path, monkeypatch: pytest.MonkeyPatch, turn_spawns: list[tuple[str, str]]
) -> None:
    monkeypatch.setattr(turn_digest, "TURN_DIGEST_MAX_BYTES", 100)
    seed("shadow")
    turn_digest.stop(payload(str(tf)))
    assert [d["reason"] for d in drops()] == ["too_large"]
    assert turn_spawns == []
    assert tmp_files() == []
    assert (rows()[0]["action"], rows()[0]["bytes"]) == ("too_large", None)


def test_state_write_failed(tmp_path: Path, tf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "f").write_text("file")
    monkeypatch.setattr(turn_digest, "get_turn_state_path", lambda sid: tmp_path / "f" / "s.json")
    seed("shadow")
    turn_digest.stop(payload(str(tf)))
    assert "state_write_failed" in [d["reason"] for d in drops()]


def test_real_spawn_sender(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class FakePopen:
        def __init__(self, argv: list[str], **kw: Any) -> None:
            calls.append((argv, kw))

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    _REAL_SPAWN_SENDER("/tmp/x.json", "s1")  # noqa: S108
    argv, kw = calls[0]
    assert argv == [sys.executable, "-m", "personal_kb_hook.turn_sender", "/tmp/x.json", "s1"]  # noqa: S108
    assert kw["start_new_session"] is True
    assert kw["stdin"] == kw["stdout"] == kw["stderr"] == subprocess.DEVNULL


def test_log_row_cap() -> None:
    p = get_turn_digest_log_path(SID)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x" * 262145)
    turn_digest.log_row(SID, {"a": 1})
    assert p.read_text() == '{"a": 1}\n'


# ─── CLI ─────────────────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, body: bytes = b"{}") -> None:
        self.status = 200
        self._b = body

    def read(self) -> bytes:
        return self._b

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *a: object) -> None:
        return None


class _Srv:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.urls: list[str] = []

    def __call__(self, req: Any, timeout: float = 30.0) -> _Resp:
        self.urls.append(req.full_url)
        if "/api/kb/prevention?" in req.full_url:
            body = {
                "project": "personal-kb",
                "gate": {"enabled": False, "shadow": True, "max_denies": 2},
                "index": [],
                "slice": [],
                "slice_text": "",
                "surprise_capture": self.mode,
            }
            return _Resp(json.dumps(body).encode())
        return _Resp()


def run(monkeypatch: pytest.MonkeyPatch, p: dict[str, Any]) -> str:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(p)))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    cli.main(["--format=claude-json"])
    return out.getvalue()


def cli_payloads(tmp_path: Path, tf: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / ".kb_project").write_text("personal-kb\n")
    base = {"session_id": SID, "cwd": str(repo)}
    return (
        {**base, "hook_event_name": "SessionStart", "source": "startup"},
        {**base, "hook_event_name": "Stop", "transcript_path": str(tf)},
    )


def test_cli_listener_off(
    tmp_path: Path, tf: Path, monkeypatch: pytest.MonkeyPatch, turn_spawns: list[tuple[str, str]]
) -> None:
    srv = _Srv("shadow")
    monkeypatch.setattr(urllib.request, "urlopen", srv)
    monkeypatch.setenv("PERSONAL_KB_LISTENER", "0")
    ss, stop = cli_payloads(tmp_path, tf)
    run(monkeypatch, ss)
    assert run(monkeypatch, stop) == ""
    assert len(turn_spawns) == 1


def test_cli_digest_after_refresh(
    tmp_path: Path, tf: Path, monkeypatch: pytest.MonkeyPatch, turn_spawns: list[tuple[str, str]]
) -> None:
    srv = _Srv("off")
    monkeypatch.setattr(urllib.request, "urlopen", srv)
    ss, stop = cli_payloads(tmp_path, tf)
    run(monkeypatch, ss)
    assert turn_spawns == []
    srv.mode = "shadow"
    run(monkeypatch, stop)
    assert [body_of(s)["event_id"] for s in turn_spawns] == ["s1:0"]


def test_cli_digest_precedes_listener(
    tmp_path: Path, tf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    srv = _Srv("shadow")
    monkeypatch.setattr(urllib.request, "urlopen", srv)
    monkeypatch.setenv("PERSONAL_KB_LISTENER", "1")
    order: list[str] = []
    monkeypatch.setattr(turn_digest, "spawn_sender", lambda *a: order.append("digest"))

    class FakePopen:
        def __init__(self, argv: list[str], **kw: Any) -> None:
            order.append("popen:" + " ".join(argv))

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    ss, stop = cli_payloads(tmp_path, tf)
    stop["last_assistant_message"] = "A" * 300
    run(monkeypatch, ss)
    run(monkeypatch, stop)
    assert order[0] == "digest", order
    assert len(order) > 1, order
    assert len([o for o in order if "listener_worker" in o]) == 1


def test_cli_headless(
    tmp_path: Path, tf: Path, monkeypatch: pytest.MonkeyPatch, turn_spawns: list[tuple[str, str]]
) -> None:
    srv = _Srv("shadow")
    monkeypatch.setattr(urllib.request, "urlopen", srv)
    monkeypatch.setenv("HEADLESS_BUILD_ENGINE", "claude-code-sonnet")
    monkeypatch.setenv("PERSONAL_KB_LISTENER", "1")
    monkeypatch.setenv("KB_LISTENER_HEADLESS", "FALSE")
    popens: list[Any] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: popens.append(a))
    _, stop = cli_payloads(tmp_path, tf)
    assert run(monkeypatch, stop) == ""
    assert len(turn_spawns) == 1
    b = body_of(turn_spawns[0])
    assert (b["mode"], b["engine"]) == ("headless", "claude-code-sonnet")
    assert popens == []


def test_unused_url_error() -> None:
    assert urllib.error.URLError("x")
