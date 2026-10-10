"""Pure-function tests for kb_service.surprise (shapes 1-3, parser, grounding)."""

from typing import Any

from kb_core.cues import target_class

from kb_service.surprise import (
    DETECTOR_SCHEMA_LINE,
    DetectorVerdict,
    TurnDigest,
    build_shape2_prompt,
    build_shape3_prompt,
    detect_shape1,
    evidence_grounded,
    parse_detector_response,
    render_items,
    shape2_skip_reason,
    shape3_skip_reason,
)


def _digest(
    turn: int,
    items: list[dict[str, Any]] | None = None,
    *,
    user_prompt: str | None = "do it",
    final_message: str | None = None,
) -> TurnDigest:
    return TurnDigest(
        event_id=f"s:{turn}",
        session_id="s",
        project="p",
        turn_index=turn,
        user_prompt=user_prompt,
        items=items or [],
        final_message=final_message,
        truncated=False,
        ts="2026-10-09T00:00:00+00:00",
    )


_n = 0


def _bash(
    target: str,
    *,
    error: bool,
    excerpt: str = "",
    cls: str | None = None,
    tool: str = "Bash",
) -> list[dict[str, Any]]:
    global _n
    _n += 1
    tid = f"t{_n}"
    call: dict[str, Any] = {
        "kind": "tool_call",
        "tool_use_id": tid,
        "tool": tool,
        "target": target,
        "target_class": target_class(tool, target) if cls is None else cls,
    }
    result = {
        "kind": "tool_result",
        "tool_use_id": tid,
        "is_error": error,
        "excerpt": excerpt,
    }
    return [call, result]


# --- shape 1 -------------------------------------------------------------------


def test_shape1_cross_turn_pair() -> None:
    d0 = _digest(0, _bash("git push origin main", error=True, excerpt="rejected"))
    d1 = _digest(1, _bash("git push origin HEAD:main", error=False))
    res = detect_shape1([d0, d1], d1.event_id)
    assert len(res.candidates) == 1
    cand = res.candidates[0]
    assert cand.shape == 1
    assert cand.detector_model == "rule:shape1"
    assert cand.detector_output == {
        "wrong_belief": "git push origin main",
        "corrected_fact": "git push origin HEAD:main",
        "evidence_excerpt": "rejected",
        "confidence": 1.0,
    }
    assert cand.turn_event_ids == [d0.event_id, d1.event_id]
    assert target_class("Bash", cand.detector_output["wrong_belief"]) == "git push"
    assert set(res.stats) == {
        "bash_calls",
        "failures",
        "calls_without_result",
        "dropped_ignored_class",
        "pairs",
        "dropped_identical",
    }
    assert res.stats["bash_calls"] == 1
    assert res.stats["pairs"] == 1


def test_shape1_identical_command_dropped() -> None:
    d0 = _digest(0, _bash("git push origin main", error=True))
    d1 = _digest(1, _bash("git push origin main ", error=False))
    res = detect_shape1([d0, d1], d1.event_id)
    assert res.candidates == []
    assert res.stats["dropped_identical"] == 1


def test_shape1_failure_without_success() -> None:
    d0 = _digest(0, _bash("git push origin main", error=True))
    assert detect_shape1([d0], d0.event_id).candidates == []


def test_shape1_pair_completed_earlier_not_reemitted() -> None:
    d0 = _digest(0, _bash("git push origin main", error=True))
    d1 = _digest(1, _bash("git push origin HEAD:main", error=False))
    d2 = _digest(2, [{"kind": "assistant_text", "text": "done"}])
    assert detect_shape1([d0, d1, d2], d2.event_id).candidates == []


def test_shape1_non_bash_ignored() -> None:
    d0 = _digest(
        0,
        _bash("/a.py", error=True, tool="Read")
        + _bash("/b.py", error=False, tool="Read"),
    )
    assert detect_shape1([d0], d0.event_id).candidates == []


def test_shape1_later_failure_wins() -> None:
    d0 = _digest(
        0,
        _bash("git push origin main", error=True)
        + _bash("git push -f origin main", error=True)
        + _bash("git push origin HEAD:main", error=False),
    )
    res = detect_shape1([d0], d0.event_id)
    assert len(res.candidates) == 1
    assert (
        res.candidates[0].detector_output["wrong_belief"] == "git push -f origin main"
    )


def test_shape1_empty_target_class_falls_back() -> None:
    d0 = _digest(0, _bash("git push x", error=True, cls=""))
    d1 = _digest(1, _bash("git push origin HEAD:main", error=False, cls="git push"))
    res = detect_shape1([d0, d1], d1.event_id)
    assert len(res.candidates) == 1


def test_shape1_ignored_class() -> None:
    d0 = _digest(
        0, _bash("cat /nope", error=True) + _bash("cat /etc/hosts", error=False)
    )
    res = detect_shape1([d0], d0.event_id)
    assert res.candidates == []
    assert res.stats["dropped_ignored_class"] == 2


def test_shape1_same_digest_pair() -> None:
    d0 = _digest(
        0,
        _bash("git push origin main", error=True)
        + _bash("git push origin HEAD:main", error=False),
    )
    res = detect_shape1([d0], d0.event_id)
    assert len(res.candidates) == 1
    assert res.candidates[0].turn_event_ids == [d0.event_id]


def test_shape1_other_class_success_does_not_break_pair() -> None:
    d0 = _digest(
        0,
        _bash("git push origin main", error=True)
        + _bash("make smoke", error=False)
        + _bash("git push origin HEAD:main", error=False),
    )
    res = detect_shape1([d0], d0.event_id)
    assert len(res.candidates) == 1
    out = res.candidates[0].detector_output
    assert target_class("Bash", out["wrong_belief"]) == "git push"


def test_shape1_call_without_result() -> None:
    items = [
        {
            "kind": "tool_call",
            "tool_use_id": "orphan",
            "tool": "Bash",
            "target": "git push origin main",
            "target_class": "git push",
        },
        *_bash("git push origin HEAD:main", error=False),
    ]
    d0 = _digest(0, items)
    res = detect_shape1([d0], d0.event_id)
    assert res.candidates == []
    assert res.stats["calls_without_result"] == 1


def test_shape1_missing_is_error_is_success() -> None:
    fail = _bash("git push origin main", error=True)
    ok = _bash("git push origin HEAD:main", error=False)
    del ok[1]["is_error"]
    d0 = _digest(0, fail + ok)
    res = detect_shape1([d0], d0.event_id)
    assert len(res.candidates) == 1
    assert res.stats["failures"] == 1


# --- shape 2 -------------------------------------------------------------------


def test_shape2_skip_reasons() -> None:
    prev = _digest(2, final_message="port 8080 is free")
    assert shape2_skip_reason(prev, _digest(3, user_prompt=None)) == "no_user_prompt"
    assert shape2_skip_reason(None, _digest(0)) == "no_prev"
    assert shape2_skip_reason(None, _digest(3)) == "turn_gap"
    assert shape2_skip_reason(_digest(1, final_message="x"), _digest(3)) == "turn_gap"
    blank = _digest(2, [{"kind": "assistant_text", "text": "   "}], final_message=None)
    assert shape2_skip_reason(blank, _digest(3)) == "no_prev_text"
    assert shape2_skip_reason(prev, _digest(3)) is None
    only_text = _digest(2, [{"kind": "assistant_text", "text": "claim"}])
    assert shape2_skip_reason(only_text, _digest(3)) is None


def test_shape2_prompt() -> None:
    prev = _digest(
        0,
        [{"kind": "assistant_text", "text": "I think so"}],
        final_message="port 8080 is free",
    )
    cur = _digest(1, user_prompt="no, 8080 is taken")
    prompt = build_shape2_prompt(prev, cur)
    assert prompt.index("I think so") < prompt.index("port 8080 is free")
    assert prompt.index("port 8080 is free") < prompt.index("no, 8080 is taken")
    assert '"corrected_fact"' in prompt
    assert prompt.rstrip().endswith(DETECTOR_SCHEMA_LINE)


# --- shape 3 -------------------------------------------------------------------


def test_shape3_skip_reasons() -> None:
    after = _digest(
        0,
        [*_bash("ls", error=False), {"kind": "assistant_text", "text": "claim"}],
    )
    assert shape3_skip_reason(after) == "no_claim_before_result"
    blank = _digest(
        0, [{"kind": "assistant_text", "text": "  "}, *_bash("ls", error=False)]
    )
    assert shape3_skip_reason(blank) == "no_claim_before_result"
    ok = _digest(0, [{"kind": "assistant_text", "text": "x"}, *_bash("ls", error=True)])
    assert shape3_skip_reason(ok) is None


def test_shape3_prompt_renders_items_in_order() -> None:
    items = [
        {"kind": "assistant_text", "text": "config is in /etc/foo.conf"},
        {"kind": "tool_call", "tool_use_id": "a", "tool": "Bash", "target": "cat x"},
        {"kind": "tool_result", "tool_use_id": "a", "is_error": True, "excerpt": "no"},
        {"kind": "tool_result", "tool_use_id": "b", "excerpt": "yes"},
        {"kind": "mystery"},
    ]
    rendered = render_items(items)
    assert rendered.splitlines() == [
        "[assistant] config is in /etc/foo.conf",
        "[tool_call Bash] cat x",
        "[tool_result error=true] no",
        "[tool_result error=false] yes",
    ]
    prompt = build_shape3_prompt(_digest(0, items))
    positions = [prompt.index(line) for line in rendered.splitlines()]
    assert positions == sorted(positions)
    assert prompt.rstrip().endswith(DETECTOR_SCHEMA_LINE)


# --- parser and grounding ------------------------------------------------------

_GOOD = (
    '{"surprise": true, "wrong_belief": "a", "corrected_fact": "b",'
    ' "evidence_excerpt": "c", "confidence": 0.8}'
)


def test_parse_fenced_verdict() -> None:
    verdict, reject = parse_detector_response(f"```json\n{_GOOD}\n```")
    assert reject is None
    assert verdict == DetectorVerdict("a", "b", "c", 0.8)
    assert verdict.to_output() == {
        "wrong_belief": "a",
        "corrected_fact": "b",
        "evidence_excerpt": "c",
        "confidence": 0.8,
    }


def test_parse_rejects() -> None:
    assert parse_detector_response('{"surprise": false}') == (None, "no_surprise")
    assert parse_detector_response("not json") == (None, "unparseable")
    assert parse_detector_response(None) == (None, "llm_error")
    bad = [
        '{"surprise": true, "wrong_belief": "a", "evidence_excerpt": "c",'
        ' "confidence": 0.8}',
        _GOOD.replace("0.8", '"high"'),
        _GOOD.replace("0.8", "true"),
        _GOOD.replace("0.8", "1.5"),
        _GOOD.replace("0.8", "NaN"),
    ]
    for raw in bad:
        assert parse_detector_response(raw) == (None, "invalid_fields"), raw


def test_parse_truncates_and_strips() -> None:
    raw = _GOOD.replace('"a"', '"' + "w" * 600 + '"').replace('"b"', '" b "')
    verdict, _ = parse_detector_response(raw)
    assert verdict is not None
    assert len(verdict.wrong_belief) == 500
    assert verdict.corrected_fact == "b"
    int_conf, _ = parse_detector_response(_GOOD.replace("0.8", "1"))
    assert int_conf is not None
    assert isinstance(int_conf.confidence, float)


def test_evidence_grounded() -> None:
    assert evidence_grounded("taken by caddy", ["no, 8080 is free"]) is False
    assert evidence_grounded("Taken  BY\ncaddy", ["8080 is taken by caddy"]) is True
    assert evidence_grounded("   ", ["anything"]) is False
