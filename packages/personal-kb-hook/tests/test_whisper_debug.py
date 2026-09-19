"""Tests for the whisper-debug header's ``event=`` field (GTD 022837d4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from personal_kb_hook import whisper_debug

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def debug_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin HOME so the debug log lands under a throwaway dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".cache" / "personal_kb").mkdir(parents=True, exist_ok=True)
    return tmp_path / ".cache" / "personal_kb"


@pytest.mark.parametrize("event", ["SessionStart", "UserPromptSubmit", "Stop", "PostToolUse"])
def test_header_contains_event_for_each_supported_event(debug_env: Path, event: str) -> None:
    session_id = f"sess-{event}"
    whisper_debug.append_run_block(
        session_id,
        event_name=event,
        cwd_project="photoqueue",
        operating=["mcp:agent-gtd"],
        text="hello",
        per_kb_lines=[],
        outcome_lines=["  => no whisper"],
    )
    content = (debug_env / f"whisper-debug-{session_id}.log").read_text(encoding="utf-8")
    assert f"event={event}" in content
    # Existing fields + order preserved so the fleet's greps keep working.
    assert "listener run | event=" in content
    assert "cwd_project=photoqueue" in content
    assert "operating=[mcp:agent-gtd]" in content
    # event= comes before cwd_project= (added, not restructured).
    assert content.index("event=") < content.index("cwd_project=")


@pytest.mark.parametrize("bad_event", [None, 123, [], {}, ""])
def test_header_renders_event_placeholder_when_missing_or_non_string(
    debug_env: Path, bad_event: object
) -> None:
    session_id = "sess-bad-event"
    whisper_debug.append_run_block(
        session_id,
        event_name=bad_event,
        cwd_project="photoqueue",
        operating=[],
        text="hello",
        per_kb_lines=[],
        outcome_lines=["  => no whisper"],
    )
    content = (debug_env / f"whisper-debug-{session_id}.log").read_text(encoding="utf-8")
    assert "event=?" in content


def test_header_defaults_event_placeholder_when_omitted(debug_env: Path) -> None:
    """``event_name`` is an optional kwarg; omitting it must not raise."""
    session_id = "sess-omitted"
    whisper_debug.append_run_block(
        session_id,
        cwd_project=None,
        operating=[],
        text=None,
        per_kb_lines=[],
        outcome_lines=["  => no whisper"],
    )
    content = (debug_env / f"whisper-debug-{session_id}.log").read_text(encoding="utf-8")
    assert "event=?" in content
