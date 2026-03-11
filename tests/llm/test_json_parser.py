"""Tests for the shared LLM JSON parser."""

from personal_kb.llm.json_parser import parse_json_array, parse_json_object


class TestParseJsonObject:
    def test_bare_object(self):
        assert parse_json_object('{"key": "value"}') == {"key": "value"}

    def test_fenced_object(self):
        raw = '```json\n{"key": "value"}\n```'
        assert parse_json_object(raw) == {"key": "value"}

    def test_fenced_no_lang(self):
        raw = '```\n{"key": 1}\n```'
        assert parse_json_object(raw) == {"key": 1}

    def test_object_with_surrounding_text(self):
        raw = 'Here is the result:\n{"tool": "search", "args": {"q": "test"}}\nDone.'
        result = parse_json_object(raw)
        assert result == {"tool": "search", "args": {"q": "test"}}

    def test_nested_braces(self):
        raw = '{"outer": {"inner": "value"}}'
        result = parse_json_object(raw)
        assert result == {"outer": {"inner": "value"}}

    def test_no_object(self):
        assert parse_json_object("no json here") is None

    def test_empty_string(self):
        assert parse_json_object("") is None

    def test_malformed_json(self):
        assert parse_json_object("{not: valid json}") is None

    def test_array_returns_none(self):
        assert parse_json_object("[1, 2, 3]") is None

    def test_fenced_with_commentary(self):
        raw = 'I\'ll help with that.\n```json\n{"a": 1}\n```\nLet me know!'
        assert parse_json_object(raw) == {"a": 1}


class TestParseJsonArray:
    def test_bare_array(self):
        assert parse_json_array("[1, 2, 3]") == [1, 2, 3]

    def test_fenced_array(self):
        raw = '```json\n[{"name": "a"}, {"name": "b"}]\n```'
        result = parse_json_array(raw)
        assert result == [{"name": "a"}, {"name": "b"}]

    def test_array_with_surrounding_text(self):
        raw = 'Results:\n[{"x": 1}]\nEnd.'
        result = parse_json_array(raw)
        assert result == [{"x": 1}]

    def test_no_array(self):
        assert parse_json_array("no json here") is None

    def test_empty_string(self):
        assert parse_json_array("") is None

    def test_object_returns_none(self):
        assert parse_json_array('{"key": "value"}') is None

    def test_nested_arrays(self):
        raw = "[[1, 2], [3, 4]]"
        assert parse_json_array(raw) == [[1, 2], [3, 4]]

    def test_empty_array(self):
        assert parse_json_array("[]") == []
