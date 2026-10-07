"""Suite-wide fixtures for the standalone hook tests."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_tool_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the SessionStart tool inventory OFF unless a test opts in."""
    monkeypatch.delenv("KB_TOOL_DIRS", raising=False)
    monkeypatch.setenv("KB_TOOL_INVENTORY", "0")
