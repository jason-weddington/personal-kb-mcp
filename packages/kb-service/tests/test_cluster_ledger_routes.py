"""Hermetic tests for the cluster/decline ledger HTTP surface.

Covers ``POST /api/kb/cluster-ledger/match`` (batch candidate matching,
server-owned Jaccard via kb-core) and ``POST /api/kb/cluster-ledger/decline``
(201 with the ledger row id), the 404/409 admission gate shared with
map-loop-input, the 401 auth gate (HTTPBearer ``auto_error=False`` — kb-01745)
and the reopen-eligibility verdict in BOTH directions (the design's only
escape from a permanent decline — asserting only one direction would let the
mechanism silently invert).

Everything delegates to the ``FakeKnowledgeBase`` ledger methods added in
``conftest.py``: canned ``ClusterLedgerMatchResult``/``ClusterLedgerRow``
returns, no ``self.db`` access, no live Postgres, no network.
"""

import pytest
from fastapi.testclient import TestClient
from kb_core.cluster_ledger import (
    ClusterLedgerMatchResult,
    ClusterLedgerRow,
    ClusterLedgerVerdict,
)

from kb_service.auth import get_current_user
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase, fake_user, make_map_eligibility_verdict

PROJ = "proj"
MATCH_URL = "/api/kb/cluster-ledger/match"
DECLINE_URL = "/api/kb/cluster-ledger/decline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wire(fake_kb: FakeKnowledgeBase, *, verdicts: list | None = None) -> None:
    """Point get_current_user at a plain non-admin principal and make PROJ
    map-eligible (the nightly loop runs against eligible projects only).
    """
    app.dependency_overrides[get_current_user] = fake_user
    fake_kb.map_eligibility_verdicts = (
        [
            make_map_eligibility_verdict(
                PROJ, mappable=50, ingested=0, top_prefix_count=5, top_prefix="Run"
            )
        ]
        if verdicts is None
        else verdicts
    )


def _ineligible() -> list:
    """PROJ present but computed ineligible (mappable=0)."""
    return [
        make_map_eligibility_verdict(
            PROJ, mappable=0, ingested=0, top_prefix_count=0, top_prefix=""
        )
    ]


def _row(
    key: str = "aaaaaaaaaaaaaaaa",
    *,
    members: tuple[str, ...] = ("kb-00001", "kb-00002", "kb-00003"),
    status: str = "proposed",
    declined_member_count: int | None = None,
) -> ClusterLedgerRow:
    """A canned ledger row for the fake's match/decline returns."""
    return ClusterLedgerRow(
        cluster_key=key,
        project_ref=PROJ,
        member_entry_ids=members,
        last_label="VPN and DNS",
        status=status,  # type: ignore[arg-type]
        sightings=1,
        first_seen_at="2026-03-01T12:00:00+00:00",
        last_seen_at="2026-03-01T12:00:00+00:00",
        declined_member_count=declined_member_count,
        declined_reason=None,
        declined_by=None,
        declined_at=None,
    )


def _verdict(
    members: list[str],
    jaccard: float,
    *,
    matched: ClusterLedgerRow | None = None,
    suppressed: bool = False,
    reopened: bool = False,
) -> ClusterLedgerVerdict:
    return ClusterLedgerVerdict(
        member_entry_ids=tuple(sorted(set(members))),
        jaccard=jaccard,
        matched=matched,
        suppressed=suppressed,
        reopened=reopened,
        near_miss=False,
        marginal_match=False,
        oversized=False,
    )


def _match_result(verdicts: list[ClusterLedgerVerdict]) -> ClusterLedgerMatchResult:
    """A canned whole-batch match result; thresholds are kb-core's pinned values."""
    return ClusterLedgerMatchResult(
        project_ref=PROJ,
        jaccard_threshold=0.5,
        near_miss_floor=0.3,
        marginal_match_ceiling=0.7,
        reopen_growth_factor=2,
        rows_considered=1,
        verdicts=tuple(verdicts),
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_match_and_decline_require_auth_401(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    """No Authorization header -> 401, NOT 403 (kb-01745: HTTPBearer 0.136)."""
    monkeypatch.delenv("KB_AUTH_MODE", raising=False)
    assert (
        client.post(MATCH_URL, json={"project_ref": PROJ, "candidates": []}).status_code
        == 401
    )
    assert (
        client.post(
            DECLINE_URL,
            json={"project_ref": PROJ, "member_entry_ids": ["kb-00001"], "reason": "r"},
        ).status_code
        == 401
    )
    # The gate fires BEFORE the admission gate or the ledger is touched.
    assert fake_kb.map_eligibility_verdicts == []


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


def test_match_empty_candidates_returns_empty_results_200(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """An empty batch is a 200 with ``{"results": []}``, not an error."""
    _wire(fake_kb)
    fake_kb.cluster_ledger_match_result = _match_result([])
    response = client.post(MATCH_URL, json={"project_ref": PROJ, "candidates": []})
    assert response.status_code == 200
    assert response.json() == {"results": []}
    # The batch never reaches the facade (kb-core raises on an empty batch).
    assert fake_kb.match_clusters_calls == []


def test_match_above_threshold_returns_matched_with_ledger_id_and_jaccard(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """A candidate matching a stored cluster above threshold: matched=true,
    non-null ledger_id, an unrounded jaccard."""
    _wire(fake_kb)
    row = _row("feedfacefeedface")
    fake_kb.cluster_ledger_match_result = _match_result(
        [
            _verdict(
                ["kb-00001", "kb-00002", "kb-00003"], 0.5333333333333333, matched=row
            )
        ]
    )
    response = client.post(
        MATCH_URL,
        json={
            "project_ref": PROJ,
            "candidates": [{"member_entry_ids": ["kb-00001", "kb-00002", "kb-00003"]}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == [
        {
            "index": 0,
            "matched": True,
            "ledger_id": "feedfacefeedface",
            "status": "proposed",
            "jaccard": 0.5333333333333333,
            "reopen_eligible": False,
        }
    ]
    # The unrounded float crossed the wire verbatim.
    assert body["results"][0]["jaccard"] == 0.5333333333333333
    assert fake_kb.match_clusters_calls == [
        (PROJ, [["kb-00001", "kb-00002", "kb-00003"]])
    ]


def test_match_below_threshold_returns_unmatched_with_facade_jaccard(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """Below threshold: matched=false, ledger_id=null, and the jaccard is
    EXACTLY what the facade returned — the near-miss band value is echoed,
    never nulled or re-derived."""
    _wire(fake_kb)
    fake_kb.cluster_ledger_match_result = _match_result(
        [_verdict(["kb-00001", "kb-00002"], 0.375)]
    )
    response = client.post(
        MATCH_URL,
        json={
            "project_ref": PROJ,
            "candidates": [{"member_entry_ids": ["kb-00001", "kb-00002"]}],
        },
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["matched"] is False
    assert result["ledger_id"] is None
    assert result["jaccard"] == 0.375


def test_match_three_candidates_request_order_and_index(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """A 3-candidate batch where only the MIDDLE one matches: results come
    back in request order with correct zero-based index values."""
    _wire(fake_kb)
    fake_kb.cluster_ledger_match_result = _match_result(
        [
            _verdict(["kb-00001", "kb-00002"], 0.0),
            _verdict(["kb-00003", "kb-00004"], 0.75, matched=_row("badc0ffeebadc0ff")),
            _verdict(["kb-00005", "kb-00006"], 0.0),
        ]
    )
    response = client.post(
        MATCH_URL,
        json={
            "project_ref": PROJ,
            "candidates": [
                {"member_entry_ids": ["kb-00001", "kb-00002"]},
                {"member_entry_ids": ["kb-00003", "kb-00004"]},
                {"member_entry_ids": ["kb-00005", "kb-00006"]},
            ],
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["index"] for r in results] == [0, 1, 2]
    assert [r["matched"] for r in results] == [False, True, False]
    assert results[1]["ledger_id"] == "badc0ffeebadc0ff"
    # Candidates cross to the facade in REQUEST order, one call, whole batch.
    assert fake_kb.match_clusters_calls == [
        (
            PROJ,
            [
                ["kb-00001", "kb-00002"],
                ["kb-00003", "kb-00004"],
                ["kb-00005", "kb-00006"],
            ],
        )
    ]


def test_match_reopen_eligible_true_when_member_set_doubled(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """A declined cluster whose member set HAS doubled: reopen_eligible=true
    — the design's ONLY escape from a permanent decline."""
    _wire(fake_kb)
    declined = _row(
        "0123456789abcdef",
        members=("kb-00001", "kb-00002", "kb-00003", "kb-00004", "kb-00005"),
        status="declined",
        declined_member_count=5,
    )
    doubled = [f"kb-{i:05d}" for i in range(1, 11)]
    fake_kb.cluster_ledger_match_result = _match_result(
        [_verdict(doubled, 0.5, matched=declined, reopened=True)]
    )
    response = client.post(
        MATCH_URL,
        json={"project_ref": PROJ, "candidates": [{"member_entry_ids": doubled}]},
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["reopen_eligible"] is True
    assert result["matched"] is True
    assert result["ledger_id"] == "0123456789abcdef"
    assert result["status"] == "declined"


def test_match_reopen_eligible_false_when_member_set_not_doubled(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """A declined cluster whose member set has NOT doubled stays suppressed:
    reopen_eligible=false. A decline is permanent without the doubling."""
    _wire(fake_kb)
    declined = _row(
        "0123456789abcdef",
        members=("kb-00001", "kb-00002", "kb-00003", "kb-00004", "kb-00005"),
        status="declined",
        declined_member_count=5,
    )
    grown = [f"kb-{i:05d}" for i in range(1, 8)]  # 7 members: grown, NOT doubled
    fake_kb.cluster_ledger_match_result = _match_result(
        [_verdict(grown, 0.6, matched=declined, suppressed=True, reopened=False)]
    )
    response = client.post(
        MATCH_URL,
        json={"project_ref": PROJ, "candidates": [{"member_entry_ids": grown}]},
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["reopen_eligible"] is False
    assert result["matched"] is True
    assert result["status"] == "declined"


def test_match_unknown_project_ref_404(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """A ref with no eligibility verdict: 404, ledger untouched."""
    _wire(fake_kb, verdicts=[])
    response = client.post(
        MATCH_URL,
        json={
            "project_ref": "nope",
            "candidates": [{"member_entry_ids": ["kb-00001"]}],
        },
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "project_ref not found"}
    assert fake_kb.match_clusters_calls == []


def test_match_ineligible_project_ref_409(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """A ref with an ineligible effective verdict: 409, ledger untouched."""
    _wire(fake_kb, verdicts=_ineligible())
    response = client.post(
        MATCH_URL,
        json={"project_ref": PROJ, "candidates": [{"member_entry_ids": ["kb-00001"]}]},
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "project_ref not map-eligible"}
    assert fake_kb.match_clusters_calls == []


# ---------------------------------------------------------------------------
# Decline
# ---------------------------------------------------------------------------


def test_decline_success_returns_201_with_ledger_id(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """A successful decline is 201 with the (possibly new) row's ledger_id."""
    _wire(fake_kb)
    fake_kb.record_cluster_result = _row("feedfacefeedface")
    fake_kb.decline_cluster_result = _row(
        "feedfacefeedface",
        status="declined",
        declined_member_count=3,
    )
    response = client.post(
        DECLINE_URL,
        json={
            "project_ref": PROJ,
            "member_entry_ids": ["kb-00001", "kb-00002", "kb-00003"],
            "reason": "duplicate of an existing map",
        },
    )
    assert response.status_code == 201
    assert response.json() == {"ledger_id": "feedfacefeedface"}
    assert fake_kb.record_cluster_calls == [
        (PROJ, ["kb-00001", "kb-00002", "kb-00003"], "duplicate of an existing map")
    ]
    assert fake_kb.decline_cluster_calls == [
        {
            "cluster_key": "feedfacefeedface",
            "reason": "duplicate of an existing map",
            "declined_by": "tester@example.com",
        }
    ]


def test_decline_blank_reason_422(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """A whitespace-only reason is 422 — the reason IS the audit trail."""
    _wire(fake_kb)
    for reason in ("", "   ", "\t\n "):
        response = client.post(
            DECLINE_URL,
            json={
                "project_ref": PROJ,
                "member_entry_ids": ["kb-00001"],
                "reason": reason,
            },
        )
        assert response.status_code == 422, reason
    assert fake_kb.record_cluster_calls == []
    assert fake_kb.decline_cluster_calls == []


def test_decline_unknown_project_ref_404_writes_no_ledger_row(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """An unknown ref: 404 and NO ledger write of any kind."""
    _wire(fake_kb, verdicts=[])
    response = client.post(
        DECLINE_URL,
        json={"project_ref": "nope", "member_entry_ids": ["kb-00001"], "reason": "r"},
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "project_ref not found"}
    assert fake_kb.record_cluster_calls == []
    assert fake_kb.decline_cluster_calls == []


def test_decline_ineligible_project_ref_409_writes_no_ledger_row(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """An ineligible ref: 409 and NO ledger write of any kind."""
    _wire(fake_kb, verdicts=_ineligible())
    response = client.post(
        DECLINE_URL,
        json={"project_ref": PROJ, "member_entry_ids": ["kb-00001"], "reason": "r"},
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "project_ref not map-eligible"}
    assert fake_kb.record_cluster_calls == []
    assert fake_kb.decline_cluster_calls == []


def test_decline_whitespace_reason_is_stripped_before_facade(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    """A reason with surrounding whitespace is accepted, stored stripped."""
    _wire(fake_kb)
    response = client.post(
        DECLINE_URL,
        json={
            "project_ref": PROJ,
            "member_entry_ids": ["kb-00001"],
            "reason": "  dup  ",
        },
    )
    assert response.status_code == 201
    assert fake_kb.record_cluster_calls == [(PROJ, ["kb-00001"], "dup")]
    assert fake_kb.decline_cluster_calls[0]["reason"] == "dup"
