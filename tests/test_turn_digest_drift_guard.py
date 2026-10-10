"""Drift guard: the hook's turn-digest caps and body match kb_service's contract."""

from __future__ import annotations

import json
import typing
from typing import TYPE_CHECKING, Any

from kb_service import models as m
from kb_service import turn_digest as server_digest
from personal_kb_hook import prevention, turn_digest

if TYPE_CHECKING:
    from pathlib import Path

_CAPS = (
    "TURN_USER_PROMPT_MAX",
    "TURN_FINAL_MESSAGE_MAX",
    "TURN_TEXT_MAX",
    "TURN_TARGET_MAX",
    "TURN_EXCERPT_MAX",
    "TURN_ITEMS_MAX",
)


def _write(path: Path, records: list[dict[str, Any]]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return str(path)


def _prompt(text: str) -> dict[str, Any]:
    return {"type": "user", "uuid": "p", "origin": {"kind": "human"}, "message": {"content": text}}


def _use(i: int, cmd: str) -> dict[str, Any]:
    block = {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": cmd}}
    return {"type": "assistant", "uuid": f"a{i}", "message": {"content": [block]}}


def _result(i: int, text: str, is_error: bool = True) -> dict[str, Any]:
    block = {"type": "tool_result", "tool_use_id": f"t{i}", "content": text, "is_error": is_error}
    return {"type": "user", "uuid": f"r{i}", "message": {"content": [block]}}


def _text(text: str) -> dict[str, Any]:
    return {
        "type": "assistant",
        "uuid": "x",
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _build(path: Path, records: list[dict[str, Any]], final: str | None = "done") -> bytes:
    window = turn_digest.read_turn(_write(path, records), None)
    raw = turn_digest.build_digest(
        {"session_id": "s1", "last_assistant_message": final},
        turn_index=3,
        project="personal-kb",
        window=window,
    )
    assert raw is not None
    return raw


def test_caps_match_server() -> None:
    for name in _CAPS:
        assert getattr(turn_digest, name) == getattr(m, name), name
    assert turn_digest.TURN_DIGEST_MAX_BYTES == server_digest.TURN_DIGEST_MAX_BYTES


def test_modes_match_server() -> None:
    t: Any = m.SurpriseCaptureMode
    t = getattr(t, "__value__", t)
    assert typing.get_args(t) == prevention.SURPRISE_CAPTURE_MODES == ("off", "shadow", "on")
    assert set(prevention.SURPRISE_CAPTURE_MODES) >= turn_digest.SEND_MODES


def test_prevention_response_carries_mode() -> None:
    field = m.PreventionResponse.model_fields["surprise_capture"]
    assert field.default == "off"


def test_body_validates_with_exact_keys(tmp_path: Path) -> None:
    raw = _build(
        tmp_path / "a.jsonl",
        [_prompt("go"), _text("hi"), _use(1, "git push"), _result(1, "denied")],
    )
    v = m.TurnDigestRequest.model_validate_json(raw)
    assert v.event_id == f"{v.session_id}:{v.turn_index}"
    assert len(raw) <= server_digest.TURN_DIGEST_MAX_BYTES
    body = json.loads(raw)
    assert set(body) == set(m.TurnDigestRequest.model_fields)
    models = {
        "assistant_text": m.TurnAssistantTextItem,
        "tool_call": m.TurnToolCallItem,
        "tool_result": m.TurnToolResultItem,
    }
    assert {i["kind"] for i in body["items"]} == set(models)
    for item in body["items"]:
        assert set(item) == set(models[item["kind"]].model_fields)
    assert v.model_dump(mode="json")["items"] == body["items"]


def test_truncated_body_validates(tmp_path: Path) -> None:
    recs = [_prompt("go")]
    for i in range(60):
        recs += [_use(i, f"ls {i}"), _result(i, "r" * 3000)]
    raw = _build(tmp_path / "b.jsonl", recs)
    m.TurnDigestRequest.model_validate_json(raw)
    assert json.loads(raw)["truncated"] is True
    assert len(raw) <= 65536


def test_hook_caps_never_trip_server_max_length(tmp_path: Path) -> None:
    recs = [
        _prompt("p" * 5000),
        _text("t" * 2500),
        _use(1, "echo " + "x" * 700),
        _result(1, "e" * 3000),
    ]
    raw = _build(tmp_path / "c.jsonl", recs, final="f" * 5000)
    m.TurnDigestRequest.model_validate_json(raw)


def test_turn_state_gc_outlives_server_retention() -> None:
    assert prevention._TURN_STATE_GC_AGE_SECONDS > server_digest.TURN_EVENTS_RETENTION_DAYS * 86400
