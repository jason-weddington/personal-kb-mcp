"""Tests for FTS query escaping in SQLite backend."""

from personal_kb.db.sqlite_backend import _escape_fts_query


class TestEscapeFtsQuery:
    def test_normal_tokens(self):
        assert _escape_fts_query("hello world") == '"hello" "world"'

    def test_empty_string(self):
        assert _escape_fts_query("") == ""

    def test_whitespace_only(self):
        assert _escape_fts_query("   ") == ""

    def test_strips_quotes_from_tokens(self):
        assert _escape_fts_query('"hello" "world"') == '"hello" "world"'

    def test_strips_embedded_quotes(self):
        assert _escape_fts_query('he"llo wo"rld') == '"hello" "world"'

    def test_token_that_is_only_quotes(self):
        """A token consisting entirely of quotes should be dropped."""
        result = _escape_fts_query('"" hello')
        assert result == '"hello"'

    def test_single_token(self):
        assert _escape_fts_query("sqlite") == '"sqlite"'

    def test_mixed_quotes_and_normal(self):
        assert _escape_fts_query('normal "quoted" mix"ed') == '"normal" "quoted" "mixed"'
