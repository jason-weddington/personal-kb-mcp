"""Hermetic tests for the ``pg_url`` fixture's skip-vs-fail decision logic.

None of these tests touch a real Postgres or even invoke the ``pg_url``
fixture itself — they call the pure ``_pg_gate_decision`` /
``_is_truthy_kb_flag`` helpers extracted in ``conftest.py`` directly, so
they run unconditionally as part of the hermetic default suite (no
``KB_TEST_DATABASE_URL`` needed).

Covers the four flag/DSN combinations from this item's acceptance
criteria:

(a) flag unset + DSN unset  => skip
(b) flag set   + DSN unset  => fail
(c) flag set   + DSN set    => proceed
(d) flag unset + DSN set    => proceed
"""

from __future__ import annotations

import pytest

from conftest import _is_truthy_kb_flag, _pg_gate_decision

_DSN = "postgresql:///postgres"


@pytest.mark.parametrize("require_raw", [None, "", "false", "FALSE", "0", "no"])
@pytest.mark.parametrize("dsn_raw", [None, "", "   "])
def test_flag_unset_dsn_unset_skips(require_raw: str | None, dsn_raw: str | None) -> None:
    """(a) No DSN and the suite was never required -> skip, not fail."""
    action, message = _pg_gate_decision(require_raw, dsn_raw)
    assert action == "skip"
    assert message == "KB_TEST_DATABASE_URL not set — Postgres integration tests skipped"


@pytest.mark.parametrize("require_raw", ["true", "TRUE", "True", "1"])
@pytest.mark.parametrize("dsn_raw", [None, "", "   "])
def test_flag_set_dsn_unset_fails(require_raw: str, dsn_raw: str | None) -> None:
    """(b) Suite required but no DSN -> fail, naming both env vars."""
    action, message = _pg_gate_decision(require_raw, dsn_raw)
    assert action == "fail"
    assert message is not None
    assert "KB_REQUIRE_POSTGRES_TESTS" in message
    assert "KB_TEST_DATABASE_URL" in message


@pytest.mark.parametrize("require_raw", ["true", "TRUE", "1", "false", "", None])
def test_dsn_set_always_proceeds(require_raw: str | None) -> None:
    """(c) + (d) A DSN present always proceeds, regardless of the flag."""
    action, message = _pg_gate_decision(require_raw, _DSN)
    assert action == "proceed"
    assert message is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        ("", False),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("2", False),
        ("true", True),
        ("TRUE", True),
        ("True", True),
        ("1", True),
        ("  true  ", True),
        ("  1  ", True),
    ],
)
def test_is_truthy_kb_flag(raw: str | None, expected: bool) -> None:
    assert _is_truthy_kb_flag(raw) is expected
