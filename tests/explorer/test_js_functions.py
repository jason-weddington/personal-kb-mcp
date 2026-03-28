"""Unit tests for pure JS functions in explorer.js via Node.js."""

import subprocess
from pathlib import Path

import pytest

_JS_PATH = Path(__file__).resolve().parent.parent.parent / (
    "src/personal_kb/explorer/static/explorer.js"
)


def _run_js(script: str) -> str:
    """Run a JS snippet in Node.js and return stdout."""
    # Load the pure functions from explorer.js by extracting them,
    # then run the test script.
    preamble = """
    // Stub globals that explorer.js references at parse time
    const GRAPH_DATA = {nodes:[], edges:[], stats:{node_count:0, edge_count:0}};
    const NODE_COLORS = {};
    // Deep proxy that absorbs any property access / method call
    function makeStub() {
        return new Proxy(function(){}, {
            get(t, p) {
                if (p === 'then') return undefined; // prevent Promise detection
                if (p === Symbol.toPrimitive) return () => '';
                if (p === 'length') return 0;
                if (p === 'forEach') return () => {};
                if (p === 'map') return () => [];
                return makeStub();
            },
            apply() { return makeStub(); },
            set() { return true; },
        });
    }
    const document = makeStub();
    const location = {protocol: 'http:'};
    const setTimeout = () => {};
    const crypto = {randomUUID: () => 'test-uuid'};
    const ForceGraph = () => () => makeStub();
    const marked = undefined;
    // Stub fetch to prevent network calls during eval
    const fetch = () => Promise.resolve({json: () => Promise.resolve([])});
    """
    full = preamble + "\n" + _JS_PATH.read_text() + "\n" + script
    result = subprocess.run(  # noqa: S603
        ["node", "-e", full],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        pytest.fail(f"Node.js error:\n{result.stderr}")
    return result.stdout.strip()


class TestEscapeHtml:
    def test_basic_escaping(self):
        out = _run_js(r"""console.log(escapeHtml('<script>alert("xss")</script>'))""")
        assert "&lt;script&gt;" in out
        assert "&quot;" in out
        assert "<script>" not in out

    def test_ampersand(self):
        out = _run_js(r"""console.log(escapeHtml('a & b'))""")
        assert out == "a &amp; b"

    def test_null_returns_empty(self):
        out = _run_js(r"""console.log(escapeHtml(null))""")
        assert out == ""

    def test_undefined_returns_empty(self):
        out = _run_js(r"""console.log(escapeHtml(undefined))""")
        assert out == ""

    def test_plain_text_unchanged(self):
        out = _run_js(r"""console.log(escapeHtml('hello world'))""")
        assert out == "hello world"


class TestEscapeAttr:
    def test_single_quotes(self):
        out = _run_js(r"""console.log(escapeAttr("it's"))""")
        assert "\\'" in out

    def test_double_quotes(self):
        out = _run_js(r"""console.log(escapeAttr('say "hi"'))""")
        assert "&quot;" in out


class TestFormatRelativeTime:
    def test_just_now(self):
        out = _run_js("""console.log(formatRelativeTime(new Date().toISOString()))""")
        assert out == "just now"

    def test_minutes_ago(self):
        out = _run_js("""
        const d = new Date(Date.now() - 5 * 60 * 1000);
        console.log(formatRelativeTime(d.toISOString()));
        """)
        assert out == "5m ago"

    def test_hours_ago(self):
        out = _run_js("""
        const d = new Date(Date.now() - 3 * 3600 * 1000);
        console.log(formatRelativeTime(d.toISOString()));
        """)
        assert out == "3h ago"

    def test_yesterday(self):
        out = _run_js("""
        const d = new Date(Date.now() - 30 * 3600 * 1000);
        console.log(formatRelativeTime(d.toISOString()));
        """)
        assert out == "yesterday"

    def test_with_timezone_offset(self):
        """Timestamps with +00:00 should not produce NaN."""
        out = _run_js("""
        const d = new Date(Date.now() - 120 * 1000);
        const iso = d.toISOString().replace('Z', '+00:00');
        console.log(formatRelativeTime(iso));
        """)
        assert "NaN" not in out
        assert out == "2m ago"

    def test_bare_iso_no_timezone(self):
        """Timestamps without timezone get Z appended."""
        out = _run_js("""
        const d = new Date(Date.now() - 60 * 1000);
        const iso = d.toISOString().replace('Z', '');
        console.log(formatRelativeTime(iso));
        """)
        assert "NaN" not in out

    def test_older_date_shows_month_day(self):
        out = _run_js("""
        const d = new Date(Date.now() - 10 * 86400 * 1000);
        const result = formatRelativeTime(d.toISOString());
        console.log(result);
        """)
        # Should be like "Mar 17" — not "just now" or "NaN"
        assert "NaN" not in out
        assert len(out) > 0


class TestTruncateLabel:
    def test_short_label_unchanged(self):
        out = _run_js("""console.log(truncateLabel('hello', 1.0))""")
        assert out == "hello"

    def test_long_label_at_low_zoom(self):
        long_text = "a very long label that exceeds the limit"
        out = _run_js(f"console.log(truncateLabel('{long_text}', 1.0))")
        assert out.endswith("\u2026")
        assert len(out) <= 21  # LABEL_TRUNCATE_LEN + ellipsis

    def test_long_label_at_high_zoom_unchanged(self):
        long_text = "a very long label that exceeds the limit"
        out = _run_js(f"console.log(truncateLabel('{long_text}', 3.0))")
        assert out == long_text
