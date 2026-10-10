"""Drift guard: hook failure-context gate-log rows validate as kb_service rows."""

from __future__ import annotations

import json
import urllib.request
from typing import TYPE_CHECKING, Any

from kb_service.models import GateDecisionRow
from personal_kb_hook import prevention
from personal_kb_hook.paths import get_gate_log_path, get_prevention_cache_path

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _cue() -> dict[str, Any]:
    return {
        "resolution_id": "kb-00001",
        "updated_at": "2026-10-07T00:00:00",
        "tool": "Bash",
        "target_class": "git push",
        "args_prefix": "",
        "wrong_belief": "push to github",
        "corrected_fact": "Push to origin; github is release-only",
        "evidence": "",
        "provenance_label": "deliberate/observed",
        "observed_once": False,
    }


def _payload(tool_use_id: str) -> dict[str, Any]:
    return {
        "hook_event_name": "PostToolUseFailure",
        "session_id": "s1",
        "tool_name": "Bash",
        "tool_use_id": tool_use_id,
        "tool_input": {"command": "git commit -qam x && git push github main"},
        "error": "Exit code 1",
    }


def test_failure_context_rows_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KB_FAILURE_CONTEXT", "1")
    monkeypatch.delenv("HEADLESS_BUILD_ENGINE", raising=False)

    def _no_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("failure_context must not touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    cache_path = get_prevention_cache_path("s1")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "project": "personal-kb",
                "gate": {"enabled": True, "shadow": False, "max_denies": 2},
                "index": [_cue()],
                "denied_resolution_ids": [],
                "deny_count": 0,
                "pending_retry": None,
            }
        ),
        encoding="utf-8",
    )

    assert prevention.failure_context(_payload("t1")) is not None
    assert prevention.failure_context(_payload("t2")) is None

    def _boom(*args: Any) -> Any:
        raise RuntimeError("x")

    monkeypatch.setattr(prevention, "_find_match", _boom)
    assert prevention.failure_context(_payload("t3")) is None

    lines = get_gate_log_path("s1").read_text(encoding="utf-8").splitlines()
    rows = [GateDecisionRow.model_validate(json.loads(line)) for line in lines]
    assert [r.decision for r in rows] == [
        "failure_context",
        "failure_context_repeat",
        "failure_context_error",
    ]
