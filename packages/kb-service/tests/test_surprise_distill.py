"""Pure tests for kb_service.surprise_distill (no DB, no network)."""

import json
from typing import Any

import pytest

import kb_service.surprise_distill as sd
from kb_service.prevention import Resolution
from kb_service.resolution_hint import validate_and_stamp_resolution
from kb_service.surprise import SurpriseCandidate
from kb_service.surprise_distill import (
    DISTILLER_INSTRUCTIONS,
    DISTILLER_SCHEMA_LINE,
    SHAPE_DESCRIPTIONS,
    SURPRISE_DISTILLER_SYSTEM,
    DistillVerdict,
    build_distill_prompt,
    build_knowledge_details,
    build_resolution,
    find_exact_match,
    known_sessions,
    merge_block_reason,
    merged_surprise_hint,
    normalize_text,
    not_durable_reason,
    parse_distill_response,
    shape1_cue,
    stored_capture,
    stored_observed_sessions,
    surprise_hint,
)

_OUTPUT1 = {
    "wrong_belief": "git push origin main",
    "corrected_fact": "git push origin HEAD:main",
    "evidence_excerpt": "rejected",
    "confidence": 1.0,
}


def _cand(
    *,
    cid: int = 1,
    shape: int = 1,
    session_id: str = "s1",
    project: str = "p",
    turn_event_ids: list[str] | None = None,
    output: dict[str, Any] | None = None,
) -> SurpriseCandidate:
    return SurpriseCandidate(
        id=cid,
        shape=shape,
        session_id=session_id,
        project=project,
        turn_event_ids=(
            [f"{session_id}:0"] if turn_event_ids is None else turn_event_ids
        ),
        detector_model="rule:shape1",
        detector_output=dict(_OUTPUT1 if output is None else output),
        status="pending",
        entry_id=None,
        created_at="2026-10-09T00:00:00+00:00",
    )


_VERDICT = DistillVerdict(
    short_title="Push to HEAD:main",
    long_title="Push the current branch with git push origin HEAD:main",
    corrected_fact="Push with git push origin HEAD:main",
    lesson="git push origin main is rejected as non-fast-forward here.",
)


def _valid(**kw: Any) -> str:
    obj: dict[str, Any] = {
        "durable": True,
        "short_title": "t",
        "long_title": "long t",
        "corrected_fact": "fact",
        "lesson": "lesson",
    }
    obj.update(kw)
    return json.dumps(obj)


# --- constants ---------------------------------------------------------------


def test_pinned_literals() -> None:
    assert SURPRISE_DISTILLER_SYSTEM == (
        "You turn corrected beliefs from coding-agent sessions into durable "
        "knowledge-base lessons. Reply with exactly one JSON object and nothing "
        "else."
    )
    assert DISTILLER_INSTRUCTIONS.startswith(
        "Decide whether this correction is a durable lesson for future sessions "
        "in this project. Set durable to false when"
    )
    assert DISTILLER_INSTRUCTIONS.endswith(
        "Never include secrets, tokens, passwords or credentials in any field."
    )
    assert "  " not in DISTILLER_INSTRUCTIONS
    assert DISTILLER_SCHEMA_LINE == (
        '{"durable": true|false, "why": str, "short_title": str, '
        '"long_title": str, "corrected_fact": str, "lesson": str}'
    )
    assert SHAPE_DESCRIPTIONS[2] == (
        "The human's next message corrected a claim or assumption from the "
        "assistant's previous turn."
    )
    assert SHAPE_DESCRIPTIONS[3] == (
        "In one turn the agent reached a corrected understanding or a root cause, "
        "backed by tool output, that contradicts what it, the code, a comment, a "
        "doc, a config or the environment had indicated before."
    )
    assert sd.SURPRISE_DISTILLER_VERSION == 3
    assert sd.SURPRISE_EVENT_IDS_CAP == 20
    assert sd.SURPRISE_HINT_LIST_CAP == 100


# --- AC-2 prompt -------------------------------------------------------------


def test_build_distill_prompt_shape1() -> None:
    prompt = build_distill_prompt(_cand())
    assert SHAPE_DESCRIPTIONS[1] in prompt
    assert "Project: p" in prompt
    assert "Wrong belief: git push origin main" in prompt
    assert "Corrected fact: git push origin HEAD:main" in prompt
    assert "Evidence: rejected" in prompt
    assert DISTILLER_INSTRUCTIONS in prompt
    assert '"durable"' in prompt
    assert prompt.rstrip().endswith(DISTILLER_SCHEMA_LINE)
    assert prompt.split("\n")[0] == SHAPE_DESCRIPTIONS[1]


def test_build_distill_prompt_shape2() -> None:
    prompt = build_distill_prompt(_cand(shape=2))
    assert SHAPE_DESCRIPTIONS[2] in prompt
    assert SHAPE_DESCRIPTIONS[1] not in prompt


def test_build_distill_prompt_missing_keys() -> None:
    prompt = build_distill_prompt(_cand(output={}))
    assert "Wrong belief: \n" in prompt


# --- AC-3 parser -------------------------------------------------------------


def test_parse_valid_in_fence() -> None:
    verdict, reject = parse_distill_response(
        "```json\n" + _valid(why="x", scope="global") + "\n```"
    )
    assert reject is None
    assert verdict == DistillVerdict("t", "long t", "fact", "lesson")


@pytest.mark.parametrize(
    ("raw", "reject"),
    [
        ('{"durable": false}', "not_durable"),
        ('{"short_title": "x"}', "not_durable"),
        (_valid(durable="true"), "not_durable"),
        ("not json", "unparseable"),
        (None, "llm_error"),
        (_valid(short_title=""), "invalid_fields"),
        (_valid(short_title="   "), "invalid_fields"),
        (_valid(lesson=5), "invalid_fields"),
    ],
)
def test_parse_rejects(raw: str | None, reject: str) -> None:
    assert parse_distill_response(raw) == (None, reject)


def test_parse_caps_and_strips() -> None:
    verdict, _ = parse_distill_response(
        _valid(corrected_fact="c" * 600, short_title="s" * 100, lesson="  l  ")
    )
    assert verdict is not None
    assert verdict.corrected_fact == "c" * 500
    assert verdict.short_title == "s" * 80
    assert verdict.lesson == "l"


def test_not_durable_reason() -> None:
    assert (
        not_durable_reason('{"durable": false, "why": "  transient network error "}')
        == "transient network error"
    )
    assert not_durable_reason(json.dumps({"why": "w" * 300})) == "w" * 200
    assert not_durable_reason('{"why": 5}') == ""
    assert not_durable_reason(None) == ""
    assert not_durable_reason("not json") == ""
    assert not_durable_reason("") == ""


# --- AC-4 builders -----------------------------------------------------------


def test_build_resolution_shape1() -> None:
    cand = _cand(turn_event_ids=["s1:0", "s1:1"])
    res = build_resolution(cand, _VERDICT)
    assert res == {
        "corrected_fact": _VERDICT.corrected_fact,
        "wrong_belief": "git push origin main",
        "evidence": "rejected",
        "provenance": {
            "capture": "autonomous",
            "grounding": "observed",
            "event_id": "s1:1",
        },
        "observed_sessions": 1,
        "scope": "project",
        "cue": {"tool": "Bash", "target_class": "git push"},
    }
    assert validate_and_stamp_resolution(
        {"resolution": res}, is_machine=True, entry_type="lesson_learned"
    ) == {"resolution": res}


def test_build_resolution_shape2_and_no_events() -> None:
    assert "cue" not in build_resolution(_cand(shape=2), _VERDICT)
    res = build_resolution(_cand(turn_event_ids=[]), _VERDICT)
    assert res["provenance"] == {"capture": "autonomous", "grounding": "asserted"}


def test_build_resolution_caps_evidence() -> None:
    res = build_resolution(
        _cand(output={**_OUTPUT1, "evidence_excerpt": "e" * 900}), _VERDICT
    )
    assert res["evidence"] == "e" * 500


def test_shape1_cue_fixed_point(monkeypatch: pytest.MonkeyPatch) -> None:
    assert shape1_cue("git push origin main") == {
        "tool": "Bash",
        "target_class": "git push",
    }
    assert shape1_cue("") is None

    def fake(tool: str, target: str) -> str:
        return {"weird cmd": "Weird", "Weird": "weird"}.get(target, "")

    monkeypatch.setattr(sd, "target_class", fake)
    assert shape1_cue("weird cmd") is None
    res = build_resolution(
        _cand(output={**_OUTPUT1, "wrong_belief": "weird cmd"}), _VERDICT
    )
    assert "cue" not in res


def test_shape1_cue_uses_cue_target_class() -> None:
    wb = "git show HEAD | tail -8; git push origin main"
    assert shape1_cue(wb, "git push") == {"tool": "Bash", "target_class": "git push"}
    assert shape1_cue(wb, "git push origin") is None
    assert shape1_cue(wb, None) == {"tool": "Bash", "target_class": "git show"}
    res = build_resolution(
        _cand(output={**_OUTPUT1, "wrong_belief": wb, "cue_target_class": "git push"}),
        _VERDICT,
    )
    assert res["cue"] == {"tool": "Bash", "target_class": "git push"}


def test_build_knowledge_details() -> None:
    cand = _cand(cid=7, turn_event_ids=["s1:0", "s1:1"])
    res = build_resolution(cand, _VERDICT)
    assert build_knowledge_details(cand, _VERDICT, res) == (
        "git push origin main is rejected as non-fast-forward here.\n\n"
        "Wrong belief: git push origin main\n"
        "Corrected fact: Push with git push origin HEAD:main\n"
        "Evidence: rejected\n\n"
        "Captured autonomously by surprise capture (shape 1, candidate 7) "
        "from session s1, turn events s1:0, s1:1."
    )


# --- AC-5 match helpers ------------------------------------------------------


def _r(entry_id: str, wb: str, tool: str = "", tc: str = "") -> Resolution:
    return Resolution(
        entry_id=entry_id,
        updated_at="2026-10-09",
        wrong_belief=wb,
        corrected_fact="f",
        evidence="",
        cue_tool=tool,
        cue_target_class=tc,
        capture="autonomous",
        grounding="observed",
        observed_sessions=1,
        observed_once=True,
    )


def test_normalize_text() -> None:
    assert normalize_text(" Git  Push\torigin MAIN ") == "git push origin main"


def test_find_exact_match() -> None:
    cue = {"tool": "Bash", "target_class": "git push"}
    a = _r("kb-1", "git push origin main", "Bash", "git push")
    b = _r("kb-2", "git push origin main", "Bash", "git push")
    assert find_exact_match([a, b], "git push origin main", cue) is a
    pull = {"tool": "Bash", "target_class": "git pull"}
    assert find_exact_match([a], "git push origin main", pull) is None
    c = _r("kb-3", "port is free")
    assert find_exact_match([c], "Port  IS free", None) is c
    assert find_exact_match([a], " Git  Push origin MAIN ", cue) is a
    empty = _r("kb-4", "")
    assert find_exact_match([empty], "  ", None) is None


def test_stored_capture_and_sessions() -> None:
    no_prov = {"resolution": {"corrected_fact": "f"}}
    assert stored_capture(no_prov) == "deliberate"
    assert merge_block_reason(no_prov, None) == "deliberate"
    assert stored_capture({}) is None
    assert merge_block_reason({}, None) == "no_resolution"
    auto = {"resolution": {"provenance": {"capture": "autonomous"}}}
    assert stored_capture(auto) == "autonomous"
    assert stored_observed_sessions({"resolution": {"observed_sessions": True}}) == 1
    assert stored_observed_sessions({"resolution": {"observed_sessions": 3}}) == 3
    assert stored_observed_sessions({"resolution": {"observed_sessions": 0}}) == 1
    assert stored_observed_sessions({}) == 1


def test_surprise_hint_parsing() -> None:
    assert surprise_hint({}) == ([], [], [])
    assert surprise_hint({"surprise_capture": "x"}) == ([], [], [])
    hint = {
        "surprise_capture": {
            "sessions": ["s1", 3],
            "candidate_ids": [1, True, "2"],
            "event_ids": ["a:0", 5],
        }
    }
    assert surprise_hint(hint) == (["s1"], [1], ["a:0"])
    assert surprise_hint({"surprise_capture": {"sessions": "s"}}) == ([], [], [])


def test_merged_surprise_hint() -> None:
    c = _cand(cid=7, session_id="s1", turn_event_ids=["s1:0", "s1:1"])
    assert merged_surprise_hint({}, c, new=True) == {
        "shape": 1,
        "sessions": ["s1"],
        "candidate_ids": [7],
        "event_ids": ["s1:1"],
    }
    c5 = _cand(cid=9, session_id="s5", turn_event_ids=["s5:0"])
    stored = {
        "surprise_capture": {
            "sessions": ["old"],
            "candidate_ids": [1],
            "event_ids": [f"e{i}" for i in range(20)],
        }
    }
    assert merged_surprise_hint(stored, c5) == {
        "sessions": ["old", "s5"],
        "candidate_ids": [1, 9],
        "event_ids": [f"e{i}" for i in range(1, 20)] + ["s5:0"],
    }
    many = {
        "surprise_capture": {
            "sessions": [f"x{i}" for i in range(100)],
            "candidate_ids": list(range(100)),
            "event_ids": ["e"],
        }
    }
    merged = merged_surprise_hint(many, c5)
    assert len(merged["sessions"]) == 100
    assert merged["sessions"][0] == "x1"
    assert merged["sessions"][-1] == "s5"
    assert len(merged["candidate_ids"]) == 100
    no_events = _cand(cid=10, turn_event_ids=[])
    assert merged_surprise_hint(many, no_events)["event_ids"] == ["e"]


def test_merge_keeps_first_shape() -> None:
    first = merged_surprise_hint({}, _cand(cid=1, shape=2, session_id="s1"), new=True)
    assert first["shape"] == 2
    again = merged_surprise_hint(
        {"surprise_capture": first}, _cand(cid=2, shape=3, session_id="s2")
    )
    assert again["shape"] == 2
    legacy = merged_surprise_hint(
        {"surprise_capture": {"sessions": ["a"]}}, _cand(cid=3, shape=1)
    )
    assert "shape" not in legacy


def test_known_sessions() -> None:
    hints = {
        "surprise_capture": {"sessions": ["a"]},
        "resolution": {
            "corrected_fact": "f",
            "provenance": {"capture": "autonomous", "event_id": "x:y:2"},
        },
    }
    assert known_sessions(hints) == {"a", "x:y"}
    hints2 = {
        "resolution": {
            "corrected_fact": "f",
            "provenance": {"capture": "autonomous", "event_id": "nocolon"},
        }
    }
    assert known_sessions(hints2) == set()


def test_merge_block_reason() -> None:
    prov = {"capture": "autonomous", "grounding": "observed", "event_id": "s:0"}
    glob = {
        "resolution": {"corrected_fact": "f", "provenance": prov, "scope": "global"}
    }
    assert merge_block_reason(glob, None) == "global"
    pull = {
        "resolution": {
            "corrected_fact": "f",
            "provenance": prov,
            "cue": {"tool": "Bash", "target_class": "git pull"},
        }
    }
    push = {"tool": "Bash", "target_class": "git push"}
    assert merge_block_reason(pull, push) == "cue_mismatch"
    bare = {"resolution": {"corrected_fact": "f", "provenance": prov}}
    assert merge_block_reason(bare, None) is None
    assert merge_block_reason(bare, push) == "cue_mismatch"
    deliberate = {
        "resolution": {"corrected_fact": "f", "provenance": {"capture": "deliberate"}}
    }
    assert merge_block_reason(deliberate, None) == "deliberate"
