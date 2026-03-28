"""Validate that explorer.js has no syntax errors.

Runs `node --check` on the static JS file. Catches escaping bugs,
unclosed strings, and other syntax issues that previously required
manual browser debugging.
"""

import subprocess
from pathlib import Path

import pytest

_JS_PATH = Path(__file__).resolve().parent.parent.parent / (
    "src/personal_kb/explorer/static/explorer.js"
)


def test_explorer_js_syntax():
    """explorer.js passes Node.js syntax validation."""
    assert _JS_PATH.exists(), f"JS file not found: {_JS_PATH}"
    result = subprocess.run(  # noqa: S603
        ["node", "--check", str(_JS_PATH)],  # noqa: S607
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"JS syntax error:\n{result.stderr}")
