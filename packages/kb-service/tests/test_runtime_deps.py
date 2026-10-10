"""kb-service must declare the MCP stack as runtime deps (kb-01738 shape).

In-workspace tests pass even when a dep is missing from [project], because
uv sync installs every workspace member; the wheel would then crash at import.
"""

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _deps() -> list[str]:
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    deps: list[str] = data["project"]["dependencies"]
    return deps


def test_fastmcp_is_a_runtime_dependency() -> None:
    assert any(d.startswith("fastmcp") for d in _deps())


def test_mcp_is_a_runtime_dependency() -> None:
    assert any(d.startswith("mcp>=1.26") for d in _deps())
