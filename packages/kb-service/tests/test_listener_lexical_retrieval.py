"""Hermetic unit tests for the lexical project-name / map-title candidate
path, and its merge with detail-match retrieval (GTD be964e94).

Companion to ``tests/test_listener_retrieval.py`` (which drives the
detail-match path, :func:`_retrieve_candidate_maps`, alone — left entirely
untouched by this item) and ``tests/test_listener_decisions.py`` (telemetry
sink). This file drives, directly (no TestClient, no HTTP layer):

  * :func:`_normalize_lexical` / :func:`_lexical_word_match` — the
    normalization + whole-word-bounded substring primitives
  * :func:`_retrieve_lexical_candidates` — the lexical signal alone: substring
    matching with hyphen/underscore/space variants, the
    ``MIN_LEXICAL_TOKEN_LEN`` guard, the ``cwd_project`` exclusion, and the
    title-stopword guard
  * :func:`_retrieve_candidates` — the MERGE of the lexical and detail
    signals: precedence tiers, evidence tightening, provenance
    (``signal_source``), and the post-merge ``_MAP_CANDIDATE_CAP``

All tests are hermetic — no live Postgres, Ollama, or network. ``FakeKbDb``
(``tests/conftest.py``) fakes the kb-core ``Database`` handle. The lexical
path's active-maps scan is a flat, param-free SQL query
(``_ACTIVE_MAPS_SQL``); its rows are seeded via
``kb.db.rows_for[_ACTIVE_MAPS_ROWS_FOR_KEY]`` — a substring of that SQL
('project_ref, short_title') that does not also appear in the detail path's
``_owning_active_maps`` query, so the two fakes never collide.
"""

from typing import Any

from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.models.search import SearchResult

from kb_service.routes.listener_routes import (
    MIN_LEXICAL_TOKEN_LEN,
    _CandidateEvidence,
    _lexical_word_match,
    _meets_second_slot_bar,
    _normalize_lexical,
    _retrieve_candidates,
    _retrieve_lexical_candidates,
)
from tests.conftest import FakeKnowledgeBase

# Substring of _ACTIVE_MAPS_SQL unique to it (does not also appear in
# _owning_active_maps' SQL, which also mentions 'knowledge_entries' and
# 'mental_map' but never this exact column list).
_ACTIVE_MAPS_ROWS_FOR_KEY = "project_ref, short_title"


# ─── helpers (mirrors tests/test_listener_retrieval.py) ─────────────────────


def _detail_entry(entry_id: str, **kwargs: Any) -> KnowledgeEntry:
    defaults: dict[str, Any] = {
        "id": entry_id,
        "short_title": f"Detail {entry_id}",
        "long_title": f"Detail {entry_id} (long)",
        "knowledge_details": "Some chunky detail with paths and identifiers.",
        "entry_type": EntryType.FACTUAL_REFERENCE,
    }
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


def _map_entry(entry_id: str, **kwargs: Any) -> KnowledgeEntry:
    defaults: dict[str, Any] = {
        "id": entry_id,
        "short_title": f"Map {entry_id}",
        "long_title": f"Map {entry_id} (long)",
        "knowledge_details": "Orientation map body.",
        "entry_type": EntryType.MENTAL_MAP,
        "is_active": True,
    }
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


def _detail_results(*entries: KnowledgeEntry) -> list[SearchResult]:
    return [
        SearchResult(
            entry=e,
            score=1.0 / (60 + i + 1),
            effective_confidence=0.9,
            staleness_warning=None,
            match_source="fts",
        )
        for i, e in enumerate(entries)
    ]


def _make_kb(entries: list[KnowledgeEntry]) -> FakeKnowledgeBase:
    return FakeKnowledgeBase(results=_detail_results(*entries), filtered_count=0)


# ─── _normalize_lexical / _lexical_word_match ───────────────────────────────


def test_normalize_lexical_collapses_hyphen_underscore_whitespace() -> None:
    assert _normalize_lexical("Camera-Profiles_Data") == "camera profiles data"
    assert _normalize_lexical("camera   profiles") == "camera profiles"
    assert _normalize_lexical("  camera_profiles  ") == "camera profiles"


def test_lexical_word_match_hyphenated_path_token() -> None:
    """'camera-profiles-data' (one hyphen-joined token) must match the needle
    'camera-profiles' -- the load-bearing AC example."""
    haystack = _normalize_lexical(
        "rsync the camera-profiles-data raw-pairs from a7r6 to dispatch-host-a"
    )
    needle = _normalize_lexical("camera-profiles")
    assert _lexical_word_match(haystack, needle) is True


def test_lexical_word_match_underscore_and_space_variants() -> None:
    needle = _normalize_lexical("camera-profiles")
    assert _lexical_word_match(_normalize_lexical("camera_profiles ready"), needle)
    assert _lexical_word_match(_normalize_lexical("the camera profiles here"), needle)


def test_lexical_word_match_rejects_partial_word_contamination() -> None:
    """'megacamera profiles' must NOT match needle 'camera profiles' -- the
    'camera' inside 'megacamera' is not bounded by a separator on its left."""
    haystack = _normalize_lexical("megacamera profiles-lens")
    needle = _normalize_lexical("camera profiles")
    assert _lexical_word_match(haystack, needle) is False


def test_lexical_word_match_empty_needle_never_matches() -> None:
    assert _lexical_word_match("anything here", "") is False


# ─── _retrieve_lexical_candidates: signal alone ─────────────────────────────


async def test_project_ref_matches_hyphen_underscore_space_variants() -> None:
    for variant_text in (
        "rsync the camera-profiles-data raw-pairs from a7r6 to dispatch-host-a",
        "check camera_profiles config before running the job",
        "look at the camera profiles archive from last night",
    ):
        kb = _make_kb([])
        kb.db.rows_for[_ACTIVE_MAPS_ROWS_FOR_KEY] = [
            ("kb-20000", "camera-profiles", "camera-profiles fitter map")
        ]
        hits = await _retrieve_lexical_candidates(kb.db, variant_text, None)
        assert hits == {"kb-20000": True}, variant_text


async def test_short_project_ref_and_title_tokens_not_eligible() -> None:
    """MIN_LEXICAL_TOKEN_LEN guard: a ref/title token shorter than the pinned
    literal cannot drive a match, even when it appears verbatim in the text."""
    kb = _make_kb([])
    kb.db.rows_for[_ACTIVE_MAPS_ROWS_FOR_KEY] = [
        ("kb-20000", "r7", "R7 Host Notes"),
    ]
    assert len("r7") < MIN_LEXICAL_TOKEN_LEN
    hits = await _retrieve_lexical_candidates(
        kb.db, "copying raw files to r7 host notes tonight", None
    )
    assert hits == {}


async def test_cwd_project_excluded_before_matching() -> None:
    """A message naming the cwd project itself yields no lexical candidate
    for that project (excluded early, independent of Rule A)."""
    kb = _make_kb([])
    kb.db.rows_for[_ACTIVE_MAPS_ROWS_FOR_KEY] = [
        ("kb-20000", "camera-profiles", "camera-profiles fitter map")
    ]
    hits = await _retrieve_lexical_candidates(
        kb.db, "rsync the camera-profiles-data raw-pairs from a7r6", "camera-profiles"
    )
    assert hits == {}


async def test_stopword_title_token_does_not_match_alone() -> None:
    """A generic/stopword title word appearing in the text must NOT, on its
    own, produce a lexical candidate."""
    kb = _make_kb([])
    kb.db.rows_for[_ACTIVE_MAPS_ROWS_FOR_KEY] = [
        ("kb-20000", "internal-tools", "Internal Tools Database Overview")
    ]
    # "database" and "overview" are pinned stopwords; the ref "internal-tools"
    # and the non-stopword title token "internal" are never mentioned.
    hits = await _retrieve_lexical_candidates(
        kb.db, "check the database backup logs and overview report", None
    )
    assert hits == {}


async def test_single_title_token_is_not_enough_to_match() -> None:
    """One non-stopword title token is not evidence — it was a cost regression.

    This test previously asserted the OPPOSITE (a lone 'internal' surfaced the
    map) and so enshrined the defect: probed against real map titles, 4 of 4
    ordinary sentences produced a candidate, and each false candidate turns a
    zero-candidate short-circuit into three concurrent Sonnet calls. A
    title-only match now needs MIN_LEXICAL_TITLE_TOKEN_MATCHES distinct
    non-stopword tokens.
    """
    kb = _make_kb([])
    kb.db.rows_for[_ACTIVE_MAPS_ROWS_FOR_KEY] = [
        ("kb-20000", "internal-tools", "Internal Tools Database Overview")
    ]
    hits = await _retrieve_lexical_candidates(
        kb.db, "checking the internal build pipeline today", None
    )
    assert hits == {}


async def test_two_title_tokens_match_when_ref_absent() -> None:
    """Two distinct non-stopword title tokens DO surface the map.

    Note both tokens must clear MIN_LEXICAL_TOKEN_LEN and avoid the stopword
    set, which is why this uses 'ingestion'/'pipeline' rather than a title
    whose only eligible token is one word.
    """
    kb = _make_kb([])
    kb.db.rows_for[_ACTIVE_MAPS_ROWS_FOR_KEY] = [
        ("kb-20000", "internal-tools", "Ingestion Pipeline Internals")
    ]
    hits = await _retrieve_lexical_candidates(
        kb.db, "the ingestion pipeline needs a rewrite", None
    )
    assert hits == {"kb-20000": False}


async def test_no_active_maps_yields_no_candidates() -> None:
    kb = _make_kb([])
    kb.db.rows_for[_ACTIVE_MAPS_ROWS_FOR_KEY] = []
    hits = await _retrieve_lexical_candidates(kb.db, "camera-profiles rsync", None)
    assert hits == {}


# ─── _retrieve_candidates: merge precedence + provenance ───────────────────


async def test_merge_precedence_tier_order() -> None:
    """4 maps, one of each provenance -- confirms the AC3 tier ordering:
    both-signals > lexical project_ref > lexical title-token > detail-only.
    """
    detail_one = _detail_entry("kb-10001")
    kb = _make_kb([detail_one])
    # kb-20000 (BOTH signals) and kb-20003 (detail-only) both owned by the
    # single detail hit -> tie at best_rank=1/hit_count=1 in the detail
    # signal alone; the merge tier is what actually separates them.
    kb.db.rows_for["graph_edges"] = [
        ("kb-10001", "kb-20000"),
        ("kb-10001", "kb-20003"),
    ]
    kb.entries["kb-20000"] = _map_entry(
        "kb-20000", project_ref="alpha-cluster", short_title="Alpha Cluster Map"
    )
    kb.entries["kb-20001"] = _map_entry(
        "kb-20001", project_ref="bravo-cluster", short_title="Bravo Cluster Map"
    )
    kb.entries["kb-20002"] = _map_entry(
        "kb-20002", project_ref="echo-project", short_title="Echo Corridor Archive"
    )
    kb.entries["kb-20003"] = _map_entry(
        "kb-20003", project_ref="foxtrot-archive", short_title="Foxtrot Archive Log"
    )
    kb.db.rows_for[_ACTIVE_MAPS_ROWS_FOR_KEY] = [
        ("kb-20000", "alpha-cluster", "Alpha Cluster Map"),
        ("kb-20001", "bravo-cluster", "Bravo Cluster Map"),
        ("kb-20002", "echo-project", "Echo Corridor Archive"),
        ("kb-20003", "foxtrot-archive", "Foxtrot Archive Log"),
    ]

    text = "alpha-cluster bravo-cluster echo corridor archive tasks"
    candidates, used_fallback, evidence, signal_source = await _retrieve_candidates(
        kb, text, None
    )

    assert used_fallback is False
    assert [c.id for c in candidates] == [
        "kb-20000",  # both signals
        "kb-20001",  # lexical project_ref only
        "kb-20002",  # lexical title only (two tokens: "corridor" + "archive")
        "kb-20003",  # detail only
    ]
    assert signal_source["kb-20000"] == "lexical"
    assert signal_source["kb-20001"] == "lexical"
    assert signal_source["kb-20002"] == "lexical"
    assert signal_source["kb-20003"] == "detail"
    # Lexical-only candidates get synthesized evidence that clears the
    # second-slot bar automatically.
    assert evidence["kb-20001"] == _CandidateEvidence(hit_count=1, best_rank=1)
    assert _meets_second_slot_bar(evidence["kb-20001"]) is True


async def test_merge_both_signal_map_gets_best_rank_tightened_to_one() -> None:
    """A map found by BOTH signals keeps its detail hit_count but its
    best_rank is tightened to min(existing, 1) -- it ranks (and evidences)
    at least as well as a rank-1 detail hit."""
    detail_one = _detail_entry("kb-10001")  # resolves to the decoy, rank 1
    detail_two = _detail_entry("kb-10002")  # resolves to the golden map, rank 2
    kb = _make_kb([detail_one, detail_two])
    kb.db.rows_for["graph_edges"] = [
        ("kb-10001", "kb-20099"),
        ("kb-10002", "kb-20000"),
    ]
    kb.entries["kb-20099"] = _map_entry(
        "kb-20099", project_ref="decoy-project", short_title="Decoy Project Map"
    )
    kb.entries["kb-20000"] = _map_entry(
        "kb-20000", project_ref="alpha-cluster", short_title="Alpha Cluster Map"
    )
    # Only the golden map is in the active-maps roster used for lexical
    # matching in this fixture -- irrelevant to the decoy's own ranking,
    # which stays entirely detail-driven.
    kb.db.rows_for[_ACTIVE_MAPS_ROWS_FOR_KEY] = [
        ("kb-20000", "alpha-cluster", "Alpha Cluster Map"),
    ]

    candidates, used_fallback, evidence, signal_source = await _retrieve_candidates(
        kb, "alpha-cluster deployment notes", None
    )

    assert used_fallback is False
    assert [c.id for c in candidates] == ["kb-20000", "kb-20099"]
    assert evidence["kb-20000"].best_rank == 1  # tightened from 2
    assert evidence["kb-20000"].hit_count == 1  # detail hit_count preserved
    assert signal_source["kb-20000"] == "lexical"
    assert signal_source["kb-20099"] == "detail"


async def test_signal_source_fallback_when_detail_signal_uses_its_own_fallback() -> (
    None
):
    """When detail-matching itself falls back to a direct mental_map search
    (GTD bf40d4f1), a candidate surfaced ONLY via that fallback is attributed
    'fallback', not 'detail'."""
    fallback_map = _map_entry(
        "kb-40001", project_ref="zulu-project", short_title="Zulu Project Map"
    )
    # Seeded as the search results directly (entry_type=MENTAL_MAP), so the
    # primary detail-match leg filters it out entirely (zero detail hits) and
    # the route falls back to the direct mental_map search, which (per
    # FakeKnowledgeBase.search) reuses these same seeded results.
    kb = FakeKnowledgeBase(results=_detail_results(fallback_map), filtered_count=0)

    candidates, used_fallback, _evidence, signal_source = await _retrieve_candidates(
        kb, "totally unrelated text mentioning neither ref nor title", None
    )

    assert used_fallback is True
    assert [c.id for c in candidates] == ["kb-40001"]
    assert signal_source["kb-40001"] == "fallback"


async def test_merge_cap_still_applies_after_merge() -> None:
    """The pinned _MAP_CANDIDATE_CAP (5) still bounds the MERGED list."""
    from kb_service.routes.listener_routes import _MAP_CANDIDATE_CAP

    kb = _make_kb([])
    rows = []
    for i in range(6):
        map_id = f"kb-3000{i}"
        ref = f"proj-alpha-{i}"
        rows.append((map_id, ref, f"Proj Alpha {i} Map"))
        kb.entries[map_id] = _map_entry(map_id, project_ref=ref, short_title=ref)
    kb.db.rows_for[_ACTIVE_MAPS_ROWS_FOR_KEY] = rows

    text = " ".join(f"proj-alpha-{i}" for i in range(6))
    candidates, _used_fallback, _evidence, _signal_source = await _retrieve_candidates(
        kb, text, None
    )

    assert len(candidates) == _MAP_CANDIDATE_CAP == 5
    # All 6 tie at tier=lexical-ref, best_rank=1 -> tie-break by id ascending.
    assert [c.id for c in candidates] == [f"kb-3000{i}" for i in range(5)]


async def test_no_lexical_or_detail_signal_yields_no_candidates() -> None:
    kb = _make_kb([])
    kb.db.rows_for[_ACTIVE_MAPS_ROWS_FOR_KEY] = [
        ("kb-20000", "alpha-cluster", "Alpha Cluster Map"),
    ]
    candidates, used_fallback, _evidence, signal_source = await _retrieve_candidates(
        kb, "nothing relevant mentioned here at all", None
    )
    assert candidates == []
    assert used_fallback is True  # detail-signal fell back (zero hits) too
    assert signal_source == {}
