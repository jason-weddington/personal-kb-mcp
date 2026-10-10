"""Harness tool-name normalization (canonical_tool, tool_map, normalize_turn_tools)."""

from typing import Any

import pytest

from kb_service.harness_tools import (
    HARNESS_TOOL_MAPS,
    canonical_tool,
    normalize_turn_tools,
    tool_map,
)
from kb_service.models import TurnDigestRequest, TurnToolCallItem

TALOS_MAP = {
    "bash": "Bash",
    "edit_file": "Edit",
    "read_file": "Read",
    "list_files": "LS",
    "run_checks": "run_checks",
    "finish": "finish",
}


@pytest.mark.parametrize(
    ("harness", "tool", "expected"),
    [
        ("talos", "bash", "Bash"),
        ("talos", "edit_file", "Edit"),
        ("talos", "write_file", "write_file"),
        ("talos", "read_file", "Read"),
        ("talos", "list_files", "LS"),
        ("talos", "run_checks", "run_checks"),
        ("talos", "finish", "finish"),
        ("talos", "web_fetch", "web_fetch"),
        ("talos", "Bash", "Bash"),
        ("claude-code", "bash", "bash"),
        (None, "bash", "bash"),
        ("Talos", "bash", "bash"),
    ],
)
def test_canonical_tool(harness: str | None, tool: str, expected: str) -> None:
    assert canonical_tool(harness, tool) == expected


def test_tool_map_returns_copy() -> None:
    assert HARNESS_TOOL_MAPS == {"talos": TALOS_MAP}
    m = tool_map("talos")
    assert m == TALOS_MAP
    m["x"] = "y"
    m.pop("bash")
    assert HARNESS_TOOL_MAPS["talos"] == TALOS_MAP
    for h in ("claude-code", None, "unknown"):
        assert tool_map(h) == {}


def test_maps_are_injective() -> None:
    for m in HARNESS_TOOL_MAPS.values():
        assert len(set(m.values())) == len(m)


def _body(harness: str, calls: list[tuple[str, str, str]]) -> TurnDigestRequest:
    items: list[dict[str, Any]] = []
    for n, (tool, target, cls) in enumerate(calls):
        items.append(
            {
                "kind": "tool_call",
                "tool_use_id": f"t{n}",
                "tool": tool,
                "target": target,
                "target_class": cls,
            }
        )
        items.append({"kind": "tool_result", "tool_use_id": f"t{n}", "is_error": False})
    return TurnDigestRequest.model_validate(
        {
            "event_id": "s:0",
            "session_id": "s",
            "turn_index": 0,
            "harness": harness,
            "items": [{"kind": "assistant_text", "text": "x"}, *items],
        }
    )


@pytest.mark.parametrize(
    ("tool", "target", "cls", "out_tool", "out_cls", "unmapped"),
    [
        ("bash", "git push github main", "", "Bash", "git push", []),
        ("bash", "cd repo && uv run pytest -q", "", "Bash", "uv run", []),
        ("edit_file", "src/a.py", "", "Edit", "ext:py", []),
        ("edit_file", "README.md", "", "Edit", "ext:md", []),
        ("read_file", "Makefile", "", "Read", "ext:none", []),
        ("list_files", "src", "", "LS", "", []),
        ("finish", "", "", "finish", "", []),
        ("run_checks", "", "", "run_checks", "", []),
        ("bash", "git push github main", "WRONG", "Bash", "git push", []),
        ("web_fetch", "https://x", "zzz", "web_fetch", "", ["web_fetch"]),
        ("Bash", "git push x", "", "Bash", "git push", ["Bash"]),
    ],
)
def test_normalize_table(
    tool: str,
    target: str,
    cls: str,
    out_tool: str,
    out_cls: str,
    unmapped: list[str],
) -> None:
    body = _body("talos", [(tool, target, cls)])
    out, un = normalize_turn_tools(body)
    call = out.items[1]
    assert isinstance(call, TurnToolCallItem)
    assert (call.tool, call.target_class) == (out_tool, out_cls)
    assert call.target == target
    assert call.tool_use_id == "t0"
    assert un == unmapped
    assert out.items[0] is body.items[0]
    assert out.items[2] is body.items[2]
    first = body.items[1]
    assert isinstance(first, TurnToolCallItem)
    assert (first.tool, first.target_class) == (tool, cls)


def test_unmapped_distinct_sorted() -> None:
    body = _body(
        "talos", [("web_fetch", "a", ""), ("Grep", "b", ""), ("web_fetch", "c", "")]
    )
    assert normalize_turn_tools(body)[1] == ["Grep", "web_fetch"]


def test_other_harness_untouched() -> None:
    body = _body("claude-code", [("bash", "git push x", "x")])
    out, un = normalize_turn_tools(body)
    assert out is body
    assert un == []
    call = out.items[1]
    assert isinstance(call, TurnToolCallItem)
    assert (call.tool, call.target_class) == ("bash", "x")
