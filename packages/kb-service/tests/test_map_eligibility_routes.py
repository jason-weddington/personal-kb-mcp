"""Hermetic tests for the map-eligibility review + override HTTP surface.

Covers ``GET /api/kb/map-eligibility`` (every verdict, full evidence,
unrounded, in kb-core's own order) and the two POST override endpoints
(set / clear), plus the ``inspect.signature`` parity test against the real
``kb_core.KnowledgeBase`` facade — the tripwire that fails loudly if the
pinned ``uv.lock`` rev predates the owner item's merge.
"""

import dataclasses
import inspect
import logging

import pytest
from fastapi.testclient import TestClient

from kb_service.auth import get_current_user
from kb_service.main import app
from kb_service.models import (
    MapEligibilityOverrideSetRequest,
    MapEligibilityVerdictModel,
)
from tests.conftest import (
    FakeKnowledgeBase,
    fake_admin_user,
    fake_user,
    make_map_eligibility_verdict,
)

MARKER = "map-eligibility-override"
LOGGER = "kb_service.routes.map_eligibility_routes"


def _admin() -> None:
    """Point get_current_user at fake_admin_user (call-site helper)."""
    app.dependency_overrides[get_current_user] = fake_admin_user


def _marker_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return every captured record whose message carries the route marker."""
    return [r for r in caplog.records if MARKER in r.getMessage()]


# ---------------------------------------------------------------------------
# 401 / 403 gates — all three endpoints
# ---------------------------------------------------------------------------


def test_map_eligibility_requires_auth_401(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No Authorization header -> 401, NOT 403 (kb-01745: HTTPBearer 0.136)."""
    monkeypatch.delenv("KB_AUTH_MODE", raising=False)
    assert client.get("/api/kb/map-eligibility").status_code == 401
    assert client.post("/api/kb/map-eligibility/override", json={}).status_code == 401
    assert (
        client.post("/api/kb/map-eligibility/override/clear", json={}).status_code
        == 401
    )


def test_map_eligibility_non_admin_403(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-admin authenticated user gets 403 from require_admin on all three."""
    monkeypatch.delenv("KB_AUTH_MODE", raising=False)
    app.dependency_overrides[get_current_user] = fake_user
    assert client.get("/api/kb/map-eligibility").status_code == 403
    assert (
        client.post(
            "/api/kb/map-eligibility/override",
            json={"project_ref": "x", "eligible": True, "reason": "r"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/kb/map-eligibility/override/clear", json={"project_ref": "x"}
        ).status_code
        == 403
    )


# ---------------------------------------------------------------------------
# Read endpoint — exact five-verdict body, unsorted order, both polarities
# ---------------------------------------------------------------------------


def _five_verdicts() -> list:
    """The deliberately UNSORTED five-verdict fixture (ascending would be
    agent-gtd-dev-typo, cleanr, dispatch-performance-log, harness-design,
    ml-papers) covering every evidence flag in both polarities."""
    from kb_core.map_eligibility import MapEligibilityOverride

    a = make_map_eligibility_verdict(
        "dispatch-performance-log",
        mappable=624,
        ingested=0,
        top_prefix_count=611,
        top_prefix="Run",
    )
    b = make_map_eligibility_verdict(
        "cleanr",
        mappable=194,
        ingested=122,
        top_prefix_count=24,
        top_prefix="kb",
        maps=3,
    )
    c = make_map_eligibility_verdict(
        "agent-gtd-dev-typo",
        mappable=0,
        ingested=0,
        top_prefix_count=0,
        top_prefix="",
        override=MapEligibilityOverride(
            "agent-gtd-dev-typo",
            True,
            "alias typo, force on to test the surface",
            None,
            "2026-09-19T13:00:00+00:00",
        ),
        orphaned=True,
    )
    d = make_map_eligibility_verdict(
        "harness-design",
        mappable=71,
        ingested=0,
        top_prefix_count=3,
        top_prefix="talos",
        override=MapEligibilityOverride(
            "harness-design",
            False,
            "dated session journal, no topical partition",
            "jason@example.com",
            "2026-09-19T12:00:00+00:00",
        ),
    )
    e = make_map_eligibility_verdict(
        "ml-papers",
        mappable=41,
        ingested=41,
        top_prefix_count=2,
        top_prefix="Fact",
    )
    return [a, b, c, d, e]


def test_read_returns_every_verdict_unsorted_exact_body(client: TestClient) -> None:
    """GET returns all five verdicts in the fake's (unsorted) order, with the
    exact evidence dicts, bit-exact unrounded floats and a non-zero ``maps``."""
    kb: FakeKnowledgeBase = app.state.kb
    kb.map_eligibility_verdicts = _five_verdicts()
    _admin()
    resp = client.get("/api/kb/map-eligibility")
    assert resp.status_code == 200
    body = resp.json()
    assert [p["evidence"]["project_ref"] for p in body["projects"]] == [
        "dispatch-performance-log",
        "cleanr",
        "agent-gtd-dev-typo",
        "harness-design",
        "ml-papers",
    ]
    a, b, c, d, e = body["projects"]
    assert a == {
        "evidence": {
            "project_ref": "dispatch-performance-log",
            "mappable": 624,
            "ingested": 0,
            "hand_authored": 624,
            "maps": 0,
            "top_prefix": "Run",
            "top_prefix_share": 0.9791666666666666,
            "is_ingest_corpus": False,
            "is_too_thin": False,
            "is_journal": True,
            "computed_eligible": False,
        },
        "override": None,
        "effective_eligible": False,
        "decided_by": "computed",
        "orphaned": False,
    }
    assert b["evidence"] == {
        "project_ref": "cleanr",
        "mappable": 194,
        "ingested": 122,
        "hand_authored": 72,
        "maps": 3,
        "top_prefix": "kb",
        "top_prefix_share": 0.12371134020618557,
        "is_ingest_corpus": False,
        "is_too_thin": False,
        "is_journal": False,
        "computed_eligible": True,
    }
    assert b["override"] is None
    assert b["effective_eligible"] is True
    assert b["decided_by"] == "computed"
    assert c["evidence"]["top_prefix_share"] == 0.0
    assert c["override"] == {
        "project_ref": "agent-gtd-dev-typo",
        "eligible": True,
        "reason": "alias typo, force on to test the surface",
        "set_by": None,
        "set_at": "2026-09-19T13:00:00+00:00",
    }
    assert c["effective_eligible"] is True
    assert c["decided_by"] == "override"
    assert c["orphaned"] is True
    assert d["evidence"]["top_prefix_share"] == 0.04225352112676056
    assert d["override"] == {
        "project_ref": "harness-design",
        "eligible": False,
        "reason": "dated session journal, no topical partition",
        "set_by": "jason@example.com",
        "set_at": "2026-09-19T12:00:00+00:00",
    }
    assert d["effective_eligible"] is False
    assert d["decided_by"] == "override"
    assert e["evidence"] == {
        "project_ref": "ml-papers",
        "mappable": 41,
        "ingested": 41,
        "hand_authored": 0,
        "maps": 0,
        "top_prefix": "Fact",
        "top_prefix_share": 0.04878048780487805,
        "is_ingest_corpus": True,
        "is_too_thin": True,
        "is_journal": False,
        "computed_eligible": False,
    }
    assert e["override"] is None


def test_read_fixture_covers_every_flag_in_both_polarities(client: TestClient) -> None:
    """The five-verdict fixture exercises every flag in BOTH states — a
    polarity that silently flips to constant cannot pass this."""
    kb: FakeKnowledgeBase = app.state.kb
    kb.map_eligibility_verdicts = _five_verdicts()
    _admin()
    body = client.get("/api/kb/map-eligibility").json()["projects"]
    for flag in ("is_ingest_corpus", "is_too_thin", "is_journal", "computed_eligible"):
        assert {p["evidence"][flag] for p in body} == {True, False}, flag
    assert {p["orphaned"] for p in body} == {True, False}
    assert {p["decided_by"] for p in body} == {"computed", "override"}


def test_read_empty_returns_empty_projects(client: TestClient) -> None:
    """An empty verdict list yields exactly {\"projects\": []}."""
    _admin()
    resp = client.get("/api/kb/map-eligibility")
    assert resp.status_code == 200
    assert resp.json() == {"projects": []}


def test_routes_mounted(client: TestClient) -> None:
    """All three paths are registered on the app (mirrors test_app.py:72)."""
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert {
        "/api/kb/map-eligibility",
        "/api/kb/map-eligibility/override",
        "/api/kb/map-eligibility/override/clear",
    } <= paths


def test_openapi_declares_no_parameters(client: TestClient) -> None:
    """The read endpoint takes NO query/path/header parameters (AC2a) and
    neither POST path does either."""
    _admin()
    schema = client.get("/openapi.json").json()
    assert schema["paths"]["/api/kb/map-eligibility"]["get"].get("parameters", []) == []
    assert (
        schema["paths"]["/api/kb/map-eligibility/override"]["post"].get(
            "parameters", []
        )
        == []
    )
    assert (
        schema["paths"]["/api/kb/map-eligibility/override/clear"]["post"].get(
            "parameters", []
        )
        == []
    )


def test_read_emits_no_log(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The read path emits ZERO marker log records (AC8)."""
    _admin()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        client.get("/api/kb/map-eligibility")
    assert _marker_records(caplog) == []


# ---------------------------------------------------------------------------
# Field parity against the kb-core verdict dataclass (LEAD-ADDED 1)
# ---------------------------------------------------------------------------


def test_verdict_response_model_field_parity_with_kb_core() -> None:
    """FastAPI silently DROPS any field a response_model does not declare, so
    the Pydantic model's field set must equal kb-core's dataclass field set
    in BOTH directions — otherwise a new kb-core evidence field would
    vanish from the wire with no error anywhere."""
    from kb_core.map_eligibility import MapEligibilityVerdict

    pydantic_fields = set(MapEligibilityVerdictModel.model_fields)
    dataclass_fields = {f.name for f in dataclasses.fields(MapEligibilityVerdict)}
    assert pydantic_fields == dataclass_fields


# ---------------------------------------------------------------------------
# Set endpoint
# ---------------------------------------------------------------------------


def test_set_override_records_call_and_returns_verdict(client: TestClient) -> None:
    """A set records the STRIPPED reason and the attributed set_by, and
    returns the post-write resolved verdict — the route's recompute is
    observable, so a route that never called the setter cannot pass (the
    fake's map_eligibility() list is set independently)."""
    kb: FakeKnowledgeBase = app.state.kb
    kb.map_eligibility_verdicts = [_five_verdicts()[3]]  # verdict D
    _admin()
    resp = client.post(
        "/api/kb/map-eligibility/override",
        json={
            "project_ref": "harness-design",
            "eligible": False,
            "reason": "  dated session journal, no topical partition  ",
        },
    )
    assert resp.status_code == 200
    assert kb.set_map_eligibility_override_calls[-1] == (
        "harness-design",
        {
            "eligible": False,
            "reason": "dated session journal, no topical partition",
            "set_by": "admin@example.com",
        },
    )
    body = resp.json()
    assert body["changed"] is True
    assert body["verdict"]["evidence"]["project_ref"] == "harness-design"
    assert body["verdict"]["decided_by"] == "override"
    assert body["verdict"]["override"] is not None
    assert body["verdict"]["override"]["eligible"] is False


def test_set_override_response_verdict_carries_non_null_override(
    client: TestClient,
) -> None:
    """LEAD-ADDED 4: a successful set's verdict MUST carry a non-null
    override block with decided_by == \"override\" — a row was just upserted."""
    kb: FakeKnowledgeBase = app.state.kb
    kb.map_eligibility_verdicts = _five_verdicts()
    _admin()
    resp = client.post(
        "/api/kb/map-eligibility/override",
        json={"project_ref": "harness-design", "eligible": False, "reason": "journal"},
    )
    assert resp.status_code == 200
    verdict = resp.json()["verdict"]
    assert verdict is not None
    assert verdict["override"] is not None
    assert verdict["decided_by"] == "override"


def test_set_override_blank_reason_is_stripped(client: TestClient) -> None:
    """min_length=1 does NOT reject whitespace-only values — only the
    validator does; the route stores the stripped form."""
    kb: FakeKnowledgeBase = app.state.kb
    _admin()
    resp = client.post(
        "/api/kb/map-eligibility/override",
        json={"project_ref": "harness-design", "eligible": True, "reason": "  x  "},
    )
    assert resp.status_code == 200
    assert kb.set_map_eligibility_override_calls[-1][1]["reason"] == "x"


# ---------------------------------------------------------------------------
# Clear endpoint
# ---------------------------------------------------------------------------


def test_clear_override_changed_true(client: TestClient) -> None:
    """A clear with an existing row returns 200 changed=true, never 404."""
    kb: FakeKnowledgeBase = app.state.kb
    kb.map_eligibility_verdicts = [_five_verdicts()[3]]
    kb.clear_map_eligibility_override_result = True
    _admin()
    resp = client.post(
        "/api/kb/map-eligibility/override/clear", json={"project_ref": "harness-design"}
    )
    assert resp.status_code == 200
    assert kb.clear_map_eligibility_override_calls[-1] == "harness-design"
    body = resp.json()
    assert body["changed"] is True
    assert body["verdict"]["evidence"]["project_ref"] == "harness-design"


def test_clear_override_noop_changed_false_not_404(client: TestClient) -> None:
    """A no-op clear is a 200 with changed=false — NOT a 404."""
    kb: FakeKnowledgeBase = app.state.kb
    kb.clear_map_eligibility_override_result = False
    _admin()
    resp = client.post(
        "/api/kb/map-eligibility/override/clear", json={"project_ref": "harness-design"}
    )
    assert resp.status_code == 200
    assert resp.json()["changed"] is False


def test_clear_override_ref_absent_returns_null_verdict_zero_warnings(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Clearing a ref absent from the post-write list is the legitimate
    orphan-cleanup path: 200, verdict null, ZERO warning records."""
    kb: FakeKnowledgeBase = app.state.kb
    kb.map_eligibility_verdicts = []
    _admin()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        resp = client.post(
            "/api/kb/map-eligibility/override/clear",
            json={"project_ref": "agent-gtd-dev-typo"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"changed": True, "verdict": None}
    warnings = [r for r in _marker_records(caplog) if r.levelno == logging.WARNING]
    assert warnings == []


def test_clear_strips_and_records_project_ref(client: TestClient) -> None:
    """The clear request's validator strips and normalises the ref."""
    kb: FakeKnowledgeBase = app.state.kb
    _admin()
    resp = client.post(
        "/api/kb/map-eligibility/override/clear", json={"project_ref": " x "}
    )
    assert resp.status_code == 200
    assert kb.clear_map_eligibility_override_calls[-1] == "x"


# ---------------------------------------------------------------------------
# The set-absent defensive branch (unreachable against the real engine)
# ---------------------------------------------------------------------------


def test_set_ref_absent_returns_null_verdict_with_warning(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """DEFENSIVE-BRANCH EXERCISE — UNREACHABLE against the real engine.

    kb-core's ``resolve_eligibility`` returns the UNION of counts-refs and
    override-refs, so a just-upserted ref is ALWAYS present in the
    post-write list (as a synthesized ``orphaned`` row when it has no
    mappable entries). This path is reachable here ONLY because the fake's
    ``map_eligibility()`` list is independent of the write. It is NOT
    ordinary behaviour — the MCP consumer must NOT be told to handle it.
    """
    kb: FakeKnowledgeBase = app.state.kb
    kb.map_eligibility_verdicts = []
    _admin()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        resp = client.post(
            "/api/kb/map-eligibility/override",
            json={"project_ref": "harness-design", "eligible": False, "reason": "r"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"changed": True, "verdict": None}
    warnings = [r for r in _marker_records(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "invariant breach" in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# Log assertions (first caplog use in this repo — mechanism pinned here)
# ---------------------------------------------------------------------------


def test_set_and_clear_each_emit_one_info_record(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Exactly one marker INFO record per write, carrying the branch and the
    inputs that drove it; the read path stays silent (tested separately)."""
    kb: FakeKnowledgeBase = app.state.kb
    kb.map_eligibility_verdicts = _five_verdicts()
    _admin()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        client.post(
            "/api/kb/map-eligibility/override",
            json={
                "project_ref": "harness-design",
                "eligible": False,
                "reason": "dated session journal, no topical partition",
            },
        )
        client.post(
            "/api/kb/map-eligibility/override/clear",
            json={"project_ref": "harness-design"},
        )
    infos = [r for r in _marker_records(caplog) if r.levelno == logging.INFO]
    assert len(infos) == 2
    set_msg = infos[0].getMessage()
    assert "op=set" in set_msg
    assert "project_ref='harness-design'" in set_msg
    assert "set_by='admin@example.com'" in set_msg
    assert "machine_principal=False" in set_msg
    assert "decided_by=override" in set_msg
    assert "'dated session journal, no topical partition'" in set_msg
    clear_msg = infos[1].getMessage()
    assert "op=clear" in clear_msg
    assert "set_by=None" in clear_msg


def test_machine_principal_true_logged(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The machine-principal flag is read per write and logged (log-only —
    no wire field, no column). Patching the module-global get_setting takes
    effect because is_machine_principal resolves it at call time."""
    import kb_service.attribution as attribution_module

    async def _fake_get_setting(key: str) -> str | None:
        if key == attribution_module.MACHINE_PRINCIPAL_EMAIL_KEY:
            return "admin@example.com"
        return None

    monkeypatch.setattr(attribution_module, "get_setting", _fake_get_setting)
    kb: FakeKnowledgeBase = app.state.kb
    kb.map_eligibility_verdicts = _five_verdicts()
    _admin()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        resp = client.post(
            "/api/kb/map-eligibility/override",
            json={"project_ref": "harness-design", "eligible": False, "reason": "r"},
        )
    assert resp.status_code == 200
    infos = [r for r in _marker_records(caplog) if r.levelno == logging.INFO]
    assert len(infos) == 1
    assert "machine_principal=True" in infos[0].getMessage()


# ---------------------------------------------------------------------------
# 422 matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/kb/map-eligibility/override", {"eligible": True, "reason": "r"}),
        ("/api/kb/map-eligibility/override", {"project_ref": "x", "eligible": True}),
        (
            "/api/kb/map-eligibility/override",
            {"project_ref": "x", "eligible": True, "reason": ""},
        ),
        (
            "/api/kb/map-eligibility/override",
            {"project_ref": "x", "eligible": True, "reason": "   "},
        ),
        (
            "/api/kb/map-eligibility/override",
            {"project_ref": "   ", "eligible": True, "reason": "r"},
        ),
        (
            "/api/kb/map-eligibility/override",
            {"project_ref": "x", "eligible": "maybe", "reason": "r"},
        ),
        (
            "/api/kb/map-eligibility/override",
            {"project_ref": "x", "eligible": True, "reason": "r" * 2001},
        ),
        ("/api/kb/map-eligibility/override/clear", {"project_ref": "  "}),
        ("/api/kb/map-eligibility/override/clear", {}),
    ],
)
def test_invalid_bodies_are_422(client: TestClient, path: str, body: dict) -> None:
    """Every malformed request body is a 422, never a 500 or a silent 200."""
    _admin()
    resp = client.post(path, json=body)
    assert resp.status_code == 422


def test_set_request_validator_is_registered() -> None:
    """Sanity: the field validators are actually registered on the models
    (decorator order is load-bearing — inverted, Pydantic v2 drops it)."""
    assert "project_ref" in MapEligibilityOverrideSetRequest.model_fields
    model = MapEligibilityOverrideSetRequest(
        project_ref=" x ", eligible=True, reason="  r  "
    )
    assert model.project_ref == "x"
    assert model.reason == "r"


# ---------------------------------------------------------------------------
# Facade signature parity against the REAL kb-core (AC12 — the tripwire)
# ---------------------------------------------------------------------------


def test_kb_core_facade_signature_parity() -> None:
    """Kind-and-default parity against the REAL ``kb_core.KnowledgeBase``.

    A name-and-order check alone would miss a dropped keyword-only ``*`` or
    a lost ``set_by`` default, either of which breaks the route's real call
    site in production while every fake-based test stays green.
    """
    from kb_core import KnowledgeBase

    sig = inspect.signature(KnowledgeBase.set_map_eligibility_override)
    assert [(p.name, p.kind) for p in sig.parameters.values()] == [
        ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("project_ref", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("eligible", inspect.Parameter.KEYWORD_ONLY),
        ("reason", inspect.Parameter.KEYWORD_ONLY),
        ("set_by", inspect.Parameter.KEYWORD_ONLY),
    ]
    assert sig.parameters["set_by"].default is None
    assert sig.parameters["eligible"].default is inspect.Parameter.empty
    assert sig.parameters["reason"].default is inspect.Parameter.empty

    clear_sig = inspect.signature(KnowledgeBase.clear_map_eligibility_override)
    assert [(p.name, p.kind) for p in clear_sig.parameters.values()] == [
        ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("project_ref", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]

    read_sig = inspect.signature(KnowledgeBase.map_eligibility)
    assert [(p.name, p.kind) for p in read_sig.parameters.values()] == [
        ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]
