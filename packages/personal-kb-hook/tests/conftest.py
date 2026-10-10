"""Suite-wide fixtures for the standalone hook tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_tool_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the SessionStart tool inventory OFF unless a test opts in."""
    monkeypatch.delenv("KB_TOOL_DIRS", raising=False)
    monkeypatch.setenv("KB_TOOL_INVENTORY", "0")
    monkeypatch.delenv("KB_FAILURE_CONTEXT", raising=False)


@pytest.fixture(autouse=True)
def turn_spawns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record turn-digest sender spawns instead of starting a real process."""
    d = tmp_path / "tmpdir"
    d.mkdir(exist_ok=True)
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    spawns: list[tuple[str, str]] = []

    def recorder(body_path: str, session_id: str) -> None:
        spawns.append((body_path, session_id))

    monkeypatch.setattr("personal_kb_hook.turn_digest.spawn_sender", recorder)
    return spawns
