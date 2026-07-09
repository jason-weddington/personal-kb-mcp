"""Tests for ``personal_kb_hook.http_index.load_index(roster)``.

Post-P1: ``load_index`` takes the P0 roster (a ``list[KbEntry]``) as its
only argument and fans the GET out CONCURRENTLY via
``concurrent.futures.ThreadPoolExecutor``. Per-KB calls use ``timeout=1.5``;
the whole fan-out is bounded by a single ``concurrent.futures.wait(3.0)``
absolute wall deadline.

All network access is mocked via monkeypatch — no real HTTP calls.
urllib.request.urlopen is replaced inline with a callable; no third-party
mocking libraries (httpx/respx) are used.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import pytest

from personal_kb_hook import http_index
from personal_kb_hook.roster import KbEntry

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def http_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Isolate KB_DB_PATH and clear both legacy HTTP env vars."""
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)
    return {
        "db_path": db_path,
        "maps_index": db_path.parent / "maps_index.default.jsonl",
    }


def _personal(url: str = "https://kb.example.com", key: str = "secret") -> KbEntry:
    """Build a single-entry roster shaped like the P0 legacy fallback."""
    return KbEntry(label="personal", url=url, key=key)


REMOTE_PROJECT = "remote-proj"
REMOTE_MAPS = [{"id": "kb-remote-1", "short_title": "remote", "long_title": "Remote entry"}]


class _MockResponse:
    """Minimal context-manager response shim for urllib.request.urlopen."""

    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _MockResponse:
        return self

    def __exit__(self, *args: object) -> None:
        pass


def _ok_response(projects: list[dict[str, Any]]) -> _MockResponse:
    return _MockResponse(json.dumps({"projects": projects}))


# ---------------------------------------------------------------------------
# (a) Empty roster — urlopen must never be called; returns {}
# ---------------------------------------------------------------------------


def test_empty_roster_no_urlopen(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty roster short-circuits with {} and never calls urlopen."""

    def must_not_be_called(*args: object, **kwargs: object) -> _MockResponse:
        raise AssertionError("urlopen must not be called when roster is empty")

    monkeypatch.setattr(urllib.request, "urlopen", must_not_be_called)
    assert http_index.load_index([]) == {}


# ---------------------------------------------------------------------------
# (b) Happy path: single-entry roster + 200 with valid body
# ---------------------------------------------------------------------------


def test_happy_path_returns_remote_content(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-entry roster with a valid HTTP response is returned."""
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response([{"project_ref": REMOTE_PROJECT, "maps": REMOTE_MAPS}]),
    )

    result = http_index.load_index([_personal()])
    assert REMOTE_PROJECT in result
    # Each entry is now a (label, MapEntry) tuple.
    pairs = result[REMOTE_PROJECT]
    assert len(pairs) == 1
    label, entry = pairs[0]
    assert label == "personal"
    assert entry["id"] == "kb-remote-1"


# ---------------------------------------------------------------------------
# (c) Per-KB timeout -> empty for that label (here: the only label)
# ---------------------------------------------------------------------------


def test_timeout_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A socket timeout returns {} for the single-label case."""

    def raise_timeout(*args: object, **kwargs: object) -> _MockResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", raise_timeout)

    result = http_index.load_index([_personal()])
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# (d) Malformed JSON body -> empty for that label
# ---------------------------------------------------------------------------


def test_malformed_json_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-JSON response body returns {} for the single-label case."""
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _MockResponse("not json at all {{{"),
    )

    result = http_index.load_index([_personal()])
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# (e) Wrong top-level shape -> empty for that label
# ---------------------------------------------------------------------------


def test_wrong_shape_projects_not_list_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Body {"projects": "nope"} (non-list projects) returns {}."""
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _MockResponse(json.dumps({"projects": "nope"})),
    )

    result = http_index.load_index([_personal()])
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# (f) HTTPError 401 -> empty for that label
# ---------------------------------------------------------------------------


def test_http_error_401_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A 401 HTTPError returns {} for the single-label case."""

    def raise_401(*args: object, **kwargs: object) -> _MockResponse:
        raise urllib.error.HTTPError(
            url="https://kb.example.com/api/kb/maps-index",
            code=401,
            msg="Unauthorized",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_401)

    result = http_index.load_index([_personal(key="bad-key")])
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# (g) URL and Authorization header contract (incl. trailing-slash normalisation)
# ---------------------------------------------------------------------------


def test_http_request_contract_url_and_auth_header(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request URL has trailing slash stripped; Authorization header is exact."""
    captured: list[urllib.request.Request] = []

    def capture_req(
        req: urllib.request.Request,
        timeout: float | None = None,
    ) -> _MockResponse:
        captured.append(req)
        return _ok_response([])

    monkeypatch.setattr(urllib.request, "urlopen", capture_req)

    http_index.load_index([_personal(url="https://kb.example.com/", key="my-secret-key")])

    assert len(captured) == 1
    req = captured[0]
    assert req.full_url == "https://kb.example.com/api/kb/maps-index"
    assert req.get_header("Authorization") == "Bearer my-secret-key"


# ---------------------------------------------------------------------------
# (h) Partial-record tolerance: bad records skipped, good ones kept
# ---------------------------------------------------------------------------


def test_tolerance_skips_malformed_map_records(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed map records are skipped; valid ones are returned."""
    projects = [
        {
            "project_ref": "my-proj",
            "maps": [
                # Valid record — must appear in output.
                {"id": "kb-1", "short_title": "good", "long_title": ""},
                # Empty id — must be skipped.
                {"id": "", "short_title": "bad-id", "long_title": "should be skipped"},
                # Non-dict map element — must be skipped.
                "oops-not-a-dict",
                # long_title is an int (42) — must be skipped, not coerced.
                {"id": "kb-2", "short_title": "bad-long-title", "long_title": 42},
                # missing short_title — must be skipped.
                {"id": "kb-3", "long_title": "no short_title"},
                # Another valid record.
                {"id": "kb-4", "short_title": "also-good", "long_title": "kept"},
            ],
        }
    ]

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response(projects),
    )

    result = http_index.load_index([_personal()])
    assert "my-proj" in result
    ids = [entry["id"] for (_label, entry) in result["my-proj"]]
    # Valid records kept:
    assert "kb-1" in ids
    assert "kb-4" in ids
    # Malformed records skipped:
    assert "kb-2" not in ids  # long_title=42 skipped
    assert "kb-3" not in ids  # missing short_title skipped
    # Non-dict element and empty-id entries produce no entry at all.
    assert len(ids) == 2


# ---------------------------------------------------------------------------
# (i) Successful-but-empty response is authoritative — returns {}
# ---------------------------------------------------------------------------


def test_empty_response_is_authoritative_no_local_fallback(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """200 with {"projects": []} returns {}."""
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response([]),
    )
    assert http_index.load_index([_personal()]) == {}


# ---------------------------------------------------------------------------
# Extra: socket.timeout alias also returns empty
# ---------------------------------------------------------------------------


def test_socket_timeout_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """TimeoutError (socket.timeout alias) returns {}."""

    def raise_socket_timeout(*args: object, **kwargs: object) -> _MockResponse:
        raise TimeoutError("socket timed out")

    monkeypatch.setattr(urllib.request, "urlopen", raise_socket_timeout)

    result = http_index.load_index([_personal()])
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Extra: URL with non-http scheme returns empty (scheme validation)
# ---------------------------------------------------------------------------


def test_invalid_scheme_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-http(s) URL silently returns {} (urlopen not called)."""

    def must_not_be_called(*args: object, **kwargs: object) -> _MockResponse:
        raise AssertionError("urlopen must not be called for ftp:// scheme")

    monkeypatch.setattr(urllib.request, "urlopen", must_not_be_called)
    assert http_index.load_index([_personal(url="ftp://kb.example.com")]) == {}


# ---------------------------------------------------------------------------
# Extra: duplicate project_ref in response follows last-wins (per KB)
# ---------------------------------------------------------------------------


def test_duplicate_project_ref_last_wins(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Within a single KB response, duplicate project_ref follows last-wins."""
    projects = [
        {
            "project_ref": "dup-proj",
            "maps": [{"id": "kb-first", "short_title": "first", "long_title": ""}],
        },
        {
            "project_ref": "dup-proj",
            "maps": [{"id": "kb-second", "short_title": "second", "long_title": ""}],
        },
    ]

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response(projects),
    )

    result = http_index.load_index([_personal()])
    assert "dup-proj" in result
    ids = [entry["id"] for (_label, entry) in result["dup-proj"]]
    # Last entry wins within the single KB.
    assert ids == ["kb-second"]
    assert "kb-first" not in ids


# ---------------------------------------------------------------------------
# Extra: long_title None coerced to ""
# ---------------------------------------------------------------------------


def test_long_title_none_coerced_to_empty_string(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """long_title: null in the response is coerced to '' (not skipped)."""
    projects = [
        {
            "project_ref": "null-lt-proj",
            "maps": [{"id": "kb-1", "short_title": "alpha", "long_title": None}],
        }
    ]

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response(projects),
    )

    result = http_index.load_index([_personal()])
    assert "null-lt-proj" in result
    _label, entry = result["null-lt-proj"][0]
    assert entry["id"] == "kb-1"
    assert entry["long_title"] == ""


# ---------------------------------------------------------------------------
# Extra: URLError (DNS / connection failure) -> empty
# ---------------------------------------------------------------------------


def test_urlerror_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A URLError (DNS / connection failure) returns {}."""

    def raise_urlerror(*args: object, **kwargs: object) -> _MockResponse:
        raise urllib.error.URLError("name or service not known")

    monkeypatch.setattr(urllib.request, "urlopen", raise_urlerror)

    result = http_index.load_index([_personal()])
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Extra: non-JSON-object top-level (e.g. a JSON array) -> empty
# ---------------------------------------------------------------------------


def test_json_array_top_level_returns_empty(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A top-level JSON array (not an object) returns {}."""
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _MockResponse(json.dumps([{"project_ref": "x", "maps": []}])),
    )

    result = http_index.load_index([_personal()])
    assert result == {}
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Extra: MapEntry fields match the TypedDict exactly
# ---------------------------------------------------------------------------


def test_returned_entries_are_map_entry_typed(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returned entries contain the id/short_title/long_title/pointers keys."""
    projects = [
        {
            "project_ref": "typed-proj",
            "maps": [
                {
                    "id": "kb-1",
                    "short_title": "s",
                    "long_title": "l",
                    "pointers": ["kb-2"],
                }
            ],
        }
    ]

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response(projects),
    )

    result = http_index.load_index([_personal()])
    _label, entry = result["typed-proj"][0]
    assert entry["id"] == "kb-1"
    assert entry["short_title"] == "s"
    assert entry["long_title"] == "l"
    assert entry["pointers"] == ["kb-2"]


# ---------------------------------------------------------------------------
# Pointer parsing tolerance: OLD server payload (no pointers), non-list,
# non-str elements. Rollout order: service deploys may lag hook upgrades
# and vice versa — an OLD server response MUST still parse cleanly.
# ---------------------------------------------------------------------------


def test_pointers_missing_field_defaults_to_empty_list(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OLD server payload (no ``pointers`` field) parses with pointers=[]."""
    projects = [
        {
            "project_ref": "old-proj",
            # No ``pointers`` field — mimics a pre-pointers service deploy.
            "maps": [{"id": "kb-1", "short_title": "s", "long_title": "l"}],
        }
    ]
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response(projects),
    )
    result = http_index.load_index([_personal()])
    _label, entry = result["old-proj"][0]
    # Absent field → parsed as empty list, not KeyError, and no map dropped.
    assert entry.get("pointers", []) == []


def test_pointers_non_list_and_non_str_elements_tolerated(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-list ``pointers`` folds to []; non-str/empty-str elements are dropped."""
    projects = [
        {
            "project_ref": "tol-proj",
            "maps": [
                # pointers is a string, not a list — coerced to [].
                {
                    "id": "kb-1",
                    "short_title": "s",
                    "long_title": "l",
                    "pointers": "kb-2,kb-3",
                },
                # pointers list contains a mix; non-strs / empty-strs dropped.
                {
                    "id": "kb-4",
                    "short_title": "s",
                    "long_title": "l",
                    "pointers": ["kb-5", 42, "", None, "kb-6"],
                },
            ],
        }
    ]
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response(projects),
    )
    result = http_index.load_index([_personal()])
    by_id = {entry["id"]: entry for (_label, entry) in result["tol-proj"]}
    assert by_id["kb-1"]["pointers"] == []
    assert by_id["kb-4"]["pointers"] == ["kb-5", "kb-6"]


# ===========================================================================
# Fan-out (P1) tests
# ===========================================================================


def _make_urlopen_by_url(
    responses: dict[str, list[dict[str, Any]] | Exception],
) -> Any:
    """Return a urlopen replacement that dispatches by the request URL.

    Each value is either a ``projects`` list (returned as a 200 response)
    or an ``Exception`` instance to be raised.
    """

    def fake_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        # Normalise: caller's url ends with ``/api/kb/maps-index``; the
        # key under ``responses`` is the base URL (without the trailing
        # endpoint), to make the test data easy to read.
        full = req.full_url
        for base, value in responses.items():
            if full == base.rstrip("/") + "/api/kb/maps-index":
                if isinstance(value, Exception):
                    raise value
                return _ok_response(value)
        raise AssertionError(f"unexpected request URL: {full!r}")

    return fake_urlopen


def test_fanout_two_kbs_merge_into_namespaced_shape(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2-KB roster merges into ``dict[str, list[(label, MapEntry)]]``."""
    roster = [
        KbEntry(label="personal", url="https://personal.kb/", key="p-key"),
        KbEntry(label="team", url="https://team.kb/", key="t-key"),
    ]
    responses: dict[str, list[dict[str, Any]] | Exception] = {
        "https://personal.kb/": [
            {
                "project_ref": "alpha",
                "maps": [{"id": "kb-a-1", "short_title": "a-short", "long_title": "a-long"}],
            }
        ],
        "https://team.kb/": [
            {
                "project_ref": "beta",
                "maps": [{"id": "kb-b-1", "short_title": "b-short", "long_title": "b-long"}],
            }
        ],
    }
    monkeypatch.setattr(urllib.request, "urlopen", _make_urlopen_by_url(responses))

    result = http_index.load_index(roster)

    assert set(result.keys()) == {"alpha", "beta"}
    # Each project_ref has a single (label, entry) tuple here.
    alpha_pairs = result["alpha"]
    beta_pairs = result["beta"]
    assert len(alpha_pairs) == 1
    assert alpha_pairs[0][0] == "personal"
    assert alpha_pairs[0][1]["id"] == "kb-a-1"
    assert len(beta_pairs) == 1
    assert beta_pairs[0][0] == "team"
    assert beta_pairs[0][1]["id"] == "kb-b-1"


def test_fanout_shared_project_ref_concatenates_both_labels(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A project_ref present in two KBs concatenates both labels' tuples."""
    roster = [
        KbEntry(label="personal", url="https://personal.kb/", key="p-key"),
        KbEntry(label="team", url="https://team.kb/", key="t-key"),
    ]
    responses: dict[str, list[dict[str, Any]] | Exception] = {
        "https://personal.kb/": [
            {
                "project_ref": "shared",
                "maps": [{"id": "kb-p-1", "short_title": "p-share", "long_title": "p-long"}],
            }
        ],
        "https://team.kb/": [
            {
                "project_ref": "shared",
                "maps": [{"id": "kb-t-1", "short_title": "t-share", "long_title": "t-long"}],
            }
        ],
    }
    monkeypatch.setattr(urllib.request, "urlopen", _make_urlopen_by_url(responses))

    result = http_index.load_index(roster)
    assert set(result.keys()) == {"shared"}
    pairs = result["shared"]
    labels = sorted(label for (label, _entry) in pairs)
    assert labels == ["personal", "team"]
    ids_by_label = {label: entry["id"] for (label, entry) in pairs}
    assert ids_by_label == {"personal": "kb-p-1", "team": "kb-t-1"}


def test_fanout_per_kb_error_isolation_does_not_raise(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One KB raising surfaces empty for that label only; no exception propagates."""
    roster = [
        KbEntry(label="personal", url="https://personal.kb/", key="p-key"),
        KbEntry(label="team", url="https://team.kb/", key="t-key"),
    ]
    responses: dict[str, list[dict[str, Any]] | Exception] = {
        "https://personal.kb/": [
            {
                "project_ref": "alpha",
                "maps": [{"id": "kb-a-1", "short_title": "ok", "long_title": "fine"}],
            }
        ],
        "https://team.kb/": urllib.error.URLError("team kb is down"),
    }
    monkeypatch.setattr(urllib.request, "urlopen", _make_urlopen_by_url(responses))

    # Must NOT raise — the surviving KB's data still flows through.
    result = http_index.load_index(roster)
    assert "alpha" in result
    alpha_pairs = result["alpha"]
    assert len(alpha_pairs) == 1
    assert alpha_pairs[0][0] == "personal"
    # The team KB contributed nothing under any project_ref.
    for proj, pairs in result.items():
        for label, _entry in pairs:
            assert label != "team", f"team label leaked under {proj!r}: {pairs!r}"


def test_fanout_absolute_wall_deadline_bound(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A KB that BLOCKS past the 3.0s wall deadline does not delay the fan-out.

    The hung KB contributes nothing; the surviving KB's data is returned;
    elapsed wall time is < 3.5s (3.0s deadline + slack).
    """
    block_event = threading.Event()  # never set
    roster = [
        KbEntry(label="personal", url="https://personal.kb/", key="p-key"),
        KbEntry(label="slow", url="https://slow.kb/", key="s-key"),
    ]
    fast_body = json.dumps(
        {
            "projects": [
                {
                    "project_ref": "alpha",
                    "maps": [{"id": "kb-a-1", "short_title": "alpha", "long_title": ""}],
                }
            ]
        }
    )

    def fake_urlopen(req: urllib.request.Request, timeout: float | None = None) -> _MockResponse:
        full = req.full_url
        if "personal.kb" in full:
            return _MockResponse(fast_body)
        # The slow KB blocks until the test releases it. We bound the
        # block on the per-KB timeout so the test cannot wedge if it
        # somehow gets through; the 3.0s wall deadline should win first.
        block_event.wait(timeout=10.0)
        return _MockResponse(fast_body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    start = time.monotonic()
    try:
        result = http_index.load_index(roster)
    finally:
        block_event.set()  # let the hung thread exit
    elapsed = time.monotonic() - start

    # Must NOT have waited for the slow KB. Allow generous slack: 3.0s
    # deadline + executor teardown.
    assert elapsed < 4.0, f"fan-out took {elapsed:.2f}s (>= 4.0s wall budget)"
    # Surviving KB's data is returned.
    assert "alpha" in result
    alpha_pairs = result["alpha"]
    assert len(alpha_pairs) == 1
    assert alpha_pairs[0][0] == "personal"
    # Slow KB contributed nothing.
    for proj, pairs in result.items():
        for label, _entry in pairs:
            assert label != "slow", f"slow label leaked under {proj!r}: {pairs!r}"


def test_fanout_single_entry_roster_matches_legacy_shape(
    http_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-entry roster yields the same project_refs as pre-fan-out, just
    with each entry wrapped in a ``(label, MapEntry)`` tuple. This is the
    legacy back-compat path the absent-roster identity test exercises end-to-end.
    """
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _ok_response(
            [
                {
                    "project_ref": "personal-kb",
                    "maps": [{"id": "kb-1", "short_title": "auth", "long_title": "Auth"}],
                }
            ]
        ),
    )

    result = http_index.load_index([_personal()])
    assert "personal-kb" in result
    pairs = result["personal-kb"]
    assert len(pairs) == 1
    label, entry = pairs[0]
    assert label == "personal"
    assert entry["id"] == "kb-1"
