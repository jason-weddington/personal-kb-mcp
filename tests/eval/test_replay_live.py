"""Opt-in live smoke for the replay harness (hits the real ``claude`` CLI).

Excluded from the default gate by ``-m "not eval"``. Runs the synthetic
fixture tasks through the ``kb_off`` and ``slice`` arms and checks the canary,
the run statuses and slice delivery.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "replay" / "replay.py"
FIXTURE = REPO_ROOT / "scripts" / "replay" / "fixtures" / "synthetic_pairs.json"

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not installed"),
]


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("replay_harness_live", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_harness_live"] = module
    spec.loader.exec_module(module)
    return module


def test_replay_live_smoke(tmp_path):
    replay = _load()
    fixture = json.loads(FIXTURE.read_text())
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in fixture["tasks"]))
    rc = replay.main(
        [
            "run",
            "--out",
            str(tmp_path),
            "--arms",
            "kb_off,slice",
            "--reps",
            "1",
            "--budget-usd",
            "1.0",
        ]
    )
    assert rc == 0, "sandbox canary failed or run refused"
    canary_log = (tmp_path / "runs" / "_canary" / "hook-log.jsonl").read_text()
    assert '"sandbox_deny"' in canary_log
    task_id = fixture["tasks"][0]["task_id"]
    for arm in ("kb_off", "slice"):
        result = json.loads((tmp_path / "runs" / task_id / arm / "1" / "result.json").read_text())
        assert result["status"] == "ok", result
    slice_log = (tmp_path / "runs" / task_id / "slice" / "1" / "hook-log.jsonl").read_text()
    decisions = [json.loads(x)["decision"] for x in slice_log.splitlines() if x.strip()]
    assert decisions.count("slice_delivered") == 1
