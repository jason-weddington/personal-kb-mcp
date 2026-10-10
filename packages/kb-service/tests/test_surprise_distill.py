"""Pure tests for kb_service.surprise_distill (no DB, no network)."""

import json
from typing import Any

import pytest

import kb_service.surprise_distill as sd
from kb_service.prevention import Resolution
from kb_service.resolution_hint import validate_and_stamp_resolution
from kb_service.surprise import SurpriseCandidate
from kb_service.surprise_distill import (
    CRITIC_INSTRUCTIONS,
    CRITIC_SCHEMA_LINE,
    DISTILLER_INSTRUCTIONS,
    DISTILLER_SCHEMA_LINE,
    SHAPE_DESCRIPTIONS,
    SURPRISE_CRITIC_SYSTEM,
    SURPRISE_DISTILLER_SYSTEM,
    CriticVerdict,
    DistillVerdict,
    build_critic_prompt,
    build_distill_prompt,
    build_knowledge_details,
    build_resolution,
    find_exact_match,
    known_sessions,
    merge_block_reason,
    merged_surprise_hint,
    normalize_text,
    not_durable_reason,
    parse_critic_response,
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
    assert sd.SURPRISE_DISTILLER_VERSION == 5
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
        "cue": {
            "tool": "Bash",
            "target_class": "git push",
            "args_prefix": "origin main",
        },
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
    assert shape1_cue("git push origin main", "git push origin HEAD:main") == {
        "tool": "Bash",
        "target_class": "git push",
        "args_prefix": "origin main",
    }
    assert shape1_cue("", "git push origin main") is None

    def fake(tool: str, target: str) -> str:
        return {"weird cmd": "Weird", "Weird": "weird"}.get(target, "")

    monkeypatch.setattr(sd, "target_class", fake)
    assert shape1_cue("weird cmd", "weird other") is None
    res = build_resolution(
        _cand(output={**_OUTPUT1, "wrong_belief": "weird cmd"}), _VERDICT
    )
    assert "cue" not in res


def test_shape1_cue_uses_cue_target_class() -> None:
    wb = "git show HEAD | tail -8; git push origin main"
    ok = "git show HEAD~1 && git push origin HEAD:main"
    push = {"tool": "Bash", "target_class": "git push", "args_prefix": "origin main"}
    assert shape1_cue(wb, ok, "git push") == push
    assert shape1_cue(wb, ok, "git push origin") is None
    assert shape1_cue(wb, ok, None) == {
        "tool": "Bash",
        "target_class": "git show",
        "args_prefix": "HEAD",
    }
    res = build_resolution(
        _cand(
            output={
                **_OUTPUT1,
                "wrong_belief": wb,
                "corrected_fact": ok,
                "cue_target_class": "git push",
            }
        ),
        _VERDICT,
    )
    assert res["cue"] == push


@pytest.mark.parametrize(
    ("failed", "success", "expected"),
    [
        (
            "git push origin main",
            "git push origin HEAD:refs/for/main",
            {"tool": "Bash", "target_class": "git push", "args_prefix": "origin main"},
        ),
        (
            "cd /x && uv run --frozen ruff check . && uv run --frozen ruff format .",
            "uv run --frozen ruff check pkg/a.py",
            {"tool": "Bash", "target_class": "uv run", "args_prefix": "ruff check ."},
        ),
        (
            "git checkout origin/feat-x",
            "git checkout -b feat-x FETCH_HEAD",
            {
                "tool": "Bash",
                "target_class": "git checkout",
                "args_prefix": "origin/feat-x",
            },
        ),
        ("make test", "make test", None),
        ("npm run build", "npm run build -- --mode prod", None),
        # F empty: no args to narrow on.
        ("git push", "git push origin main", None),
        # The success command has no segment of the cue class.
        ("git push origin main", "ls", None),
        # Identical first three args: the prefix cannot tell them apart.
        ("uv run a b c d", "uv run a b c e", None),
    ],
)
def test_shape1_cue_pinned(
    failed: str, success: str, expected: dict[str, str] | None
) -> None:
    cue = shape1_cue(failed, success)
    assert cue == expected
    if cue is not None:
        res = {"corrected_fact": "f", "cue": cue}
        stamped = validate_and_stamp_resolution(
            {"resolution": res}, is_machine=True, entry_type="lesson_learned"
        )
        assert stamped is not None
        assert stamped["resolution"]["cue"] == cue


def test_distiller_instructions_durability_test() -> None:
    assert (
        "Also set durable to false when the failure was caused by this session's"
        " own in-progress changes or by a transient state of this checkout or"
        " environment (for example lint or test errors present only at that"
        " moment, a branch not yet fetched, a file not yet created, a service"
        " that was briefly down), or when the successful command only narrowed"
        " the scope of the same check for this task. A durable lesson states a"
        " fact about the project, its tools or its environment that will still be"
        " true for a fresh session tomorrow."
    ) in DISTILLER_INSTRUCTIONS
    examples_end = DISTILLER_INSTRUCTIONS.index("leave the other fields empty.")
    assert DISTILLER_INSTRUCTIONS.index("Also set durable to false") > examples_end
    assert sd.SURPRISE_DISTILLER_VERSION == 5


_SCOPE_RULE = (
    "State the corrected fact at exactly the scope the evidence supports: if"
    " the user corrected one narrow point, record that narrow point, never a"
    " broader rule. Use only facts present in the evidence; add no details,"
    " motives or events that are not there. If the correction describes a"
    " temporary state, a work-in-progress, or something the user says will"
    " change, set durable to false. If the correction is a priority or"
    " preference for the current task rather than a fact about the project,"
    " its tools or its environment, set durable to false."
)


def test_distiller_instructions_scope_rule() -> None:
    assert _SCOPE_RULE in DISTILLER_INSTRUCTIONS
    durability_end = DISTILLER_INSTRUCTIONS.index(
        "would plausibly hold the same wrong belief."
    )
    assert DISTILLER_INSTRUCTIONS.index(_SCOPE_RULE) > durability_end
    assert sd.SURPRISE_DISTILLER_VERSION == 5
    assert _SCOPE_RULE in build_distill_prompt(_cand())


# --- critic -------------------------------------------------------------------


def _critic_reply(**kw: Any) -> str:
    obj: dict[str, Any] = {
        "supported": True,
        "scope_ok": True,
        "durable": True,
        "misleading": False,
        "reason": "matches the evidence",
    }
    obj.update(kw)
    return json.dumps(obj)


def test_critic_pinned() -> None:
    assert sd.SURPRISE_CRITIC_VERSION == 1
    assert SURPRISE_CRITIC_SYSTEM.endswith(
        "Reply with exactly one JSON object and nothing else."
    )
    assert CRITIC_SCHEMA_LINE == (
        '{"supported": true|false, "scope_ok": true|false, "durable": true|false,'
        ' "misleading": true|false, "reason": str}'
    )
    assert "  " not in CRITIC_INSTRUCTIONS


def test_build_critic_prompt_shape2_has_evidence_and_draft() -> None:
    cand = _cand(
        shape=2,
        output={
            "wrong_belief": "DRO must be enabled",
            "corrected_fact": "DRO doesn't matter for this shot",
            "evidence_excerpt": "DRO doesn't matter here",
            "confidence": 0.9,
        },
    )
    prompt = build_critic_prompt(
        cand,
        _VERDICT,
        user_correction="no, DRO doesn't matter here",
        tool_results=["ignored for shape 2"],
    )
    for text in (
        SHAPE_DESCRIPTIONS[2],
        "Wrong belief: DRO must be enabled",
        "Corrected fact: DRO doesn't matter for this shot",
        "Evidence excerpt: DRO doesn't matter here",
        "The human's correction: no, DRO doesn't matter here",
        f"Short title: {_VERDICT.short_title}",
        f"Corrected fact: {_VERDICT.corrected_fact}",
        f"Lesson: {_VERDICT.lesson}",
        CRITIC_INSTRUCTIONS,
        CRITIC_SCHEMA_LINE,
    ):
        assert text in prompt
    assert "ignored for shape 2" not in prompt
    assert prompt.index("Evidence:") < prompt.index("Drafted lesson:")


def test_build_critic_prompt_shape3_tool_results() -> None:
    cand = _cand(shape=3)
    prompt = build_critic_prompt(
        cand, _VERDICT, user_correction="ignored", tool_results=["boom 1", " ", "ok"]
    )
    assert "Tool results from the turn:" in prompt
    assert "[tool_result] boom 1" in prompt
    assert "[tool_result] ok" in prompt
    assert "ignored" not in prompt
    assert "[tool_result] \n" not in prompt


def test_build_critic_prompt_without_digest() -> None:
    prompt = build_critic_prompt(_cand(shape=3), _VERDICT)
    assert "Tool results" not in prompt
    assert "Drafted lesson:" in prompt


@pytest.mark.parametrize(
    ("raw", "accepted"),
    [
        (_critic_reply(), True),
        (_critic_reply(supported=False), False),
        (_critic_reply(scope_ok=False), False),
        (_critic_reply(durable=False), False),
        (_critic_reply(misleading=True), False),
    ],
)
def test_parse_critic_response(raw: str, accepted: bool) -> None:
    verdict, reject = parse_critic_response(raw)
    assert reject is None
    assert isinstance(verdict, CriticVerdict)
    assert verdict.accepted is accepted
    assert verdict.reason == "matches the evidence"


@pytest.mark.parametrize(
    ("raw", "reject"),
    [
        (None, "llm_error"),
        ("not json", "unparseable"),
        (_critic_reply(supported="yes"), "unparseable"),
        (json.dumps({"supported": True, "reason": "x"}), "unparseable"),
    ],
)
def test_parse_critic_response_rejects(raw: str | None, reject: str) -> None:
    assert parse_critic_response(raw) == (None, reject)


def test_parse_critic_response_reason_optional_and_capped() -> None:
    verdict, _ = parse_critic_response(_critic_reply(reason="x" * 500))
    assert verdict is not None and len(verdict.reason) == sd.CRITIC_REASON_MAX
    verdict, _ = parse_critic_response(_critic_reply(reason=3))
    assert verdict is not None and verdict.reason == ""


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


def test_merge_records_and_keeps_first_mode() -> None:
    c1 = _cand(cid=1, session_id="s1")
    assert "mode" not in merged_surprise_hint({}, c1, new=True)
    assert "mode" not in merged_surprise_hint({}, c1, new=True, mode="batch")
    first = merged_surprise_hint({}, c1, new=True, mode="headless")
    assert first["mode"] == "headless"
    again = merged_surprise_hint(
        {"surprise_capture": first},
        _cand(cid=2, session_id="s2"),
        mode="interactive",
    )
    assert again["mode"] == "headless"
    legacy = merged_surprise_hint(
        {"surprise_capture": {"sessions": ["a"]}},
        _cand(cid=3, session_id="s3"),
        mode="interactive",
    )
    assert "mode" not in legacy
    bad = merged_surprise_hint(
        {"surprise_capture": {"mode": "weird"}}, _cand(cid=4, session_id="s4")
    )
    assert "mode" not in bad


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
