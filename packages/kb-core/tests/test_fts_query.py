"""Pinned tokenizer, stopword and coverage-threshold tests for kb_core.db.fts_query."""

from __future__ import annotations

from kb_core.db.fts_query import FTS_STOPWORDS, fts_min_coverage, fts_query_tokens

_PG_ENGLISH_STOP = (
    "i me my myself we our ours ourselves you your yours yourself yourselves he him his "
    "himself she her hers herself it its itself they them their theirs themselves what "
    "which who whom this that these those am is are was were be been being have has had "
    "having do does did doing a an the and but if or because as until while of at by for "
    "with about against between into through during before after above below to from up "
    "down in out on off over under again further then once here there when where why how "
    "all any both each few more most other some such no nor not only own same so than too "
    "very s t can will just don should now"
)


def test_empty_and_stopword_only_queries() -> None:
    assert fts_query_tokens("") == []
    assert fts_query_tokens("   ") == []
    assert fts_query_tokens("the and of") == []
    assert fts_query_tokens("don't it's") == []


def test_dedupe_and_lowercase() -> None:
    assert fts_query_tokens("Whisper whisper WHISPER") == ["whisper"]


def test_double_quotes_stripped() -> None:
    assert fts_query_tokens('he"llo wo"rld') == ["hello", "world"]
    assert fts_query_tokens('"" hello') == ["hello"]
    assert fts_query_tokens('"hello" "world"') == ["hello", "world"]


def test_apostrophes_are_word_boundaries() -> None:
    assert fts_query_tokens("user's cooldown\u2019s") == ["user", "cooldown"]


def test_multi_part_tokens_kept_whole() -> None:
    assert fts_query_tokens("fan-out kb_get") == ["fan-out", "kb_get"]


def test_stopwords_dropped() -> None:
    assert fts_query_tokens("how do I deploy production") == ["deploy", "production"]


def test_syntax_characters() -> None:
    assert fts_query_tokens("cooldown NEAR * ^ : ( ) {hour}") == ["cooldown", "near", "{hour}"]


def test_token_cap() -> None:
    assert fts_query_tokens(" ".join(f"tok{i}" for i in range(40))) == [
        f"tok{i}" for i in range(32)
    ]


def test_stopwords_match_postgres_english() -> None:
    assert frozenset(_PG_ENGLISH_STOP.split()) == FTS_STOPWORDS
    assert len(FTS_STOPWORDS) == 127


def test_min_coverage() -> None:
    assert [fts_min_coverage(n) for n in (1, 2, 3, 4, 5, 6, 7, 8, 32)] == [
        1,
        2,
        2,
        2,
        3,
        4,
        5,
        6,
        30,
    ]
