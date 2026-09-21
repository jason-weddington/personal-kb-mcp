"""Hermetic tests for POST /api/kb/map-op (map_op_routes.py) — somnus's write path.

The contract of record is docs/somnus-functional-spec.md, "The write path".

Everything is driven through the FakeKnowledgeBase and a stateful fake
app_config pool: no live Postgres, no Ollama, no network.

The per-night cap counts are monkeypatched at the route module's import seam
(``map_op_routes.count_maps_created_since``) rather than run against the fake kb
DB, because the kb-core function needs a real cursor with ``fetchone``.

The hermetic FakeKbDb only implements ``fetchall``.

Making it faithful here would break the "fakes return canned data" convention.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.map_lint import count_map_pointers, map_body_budget
from kb_core.models.entry import EntryType, KnowledgeEntry

import kb_service.attribution as attribution_module
import kb_service.routes.map_op_routes as map_op_routes
from kb_service.auth import get_current_user
from kb_service.main import app
from kb_service.models import User
from tests.conftest import (
    FakeKnowledgeBase,
    StatefulFakeDbPool,
    fake_admin_user,
    fake_user,
)

MARKER = "map-op-route"
LOGGER = "kb_service.routes.map_op_routes"
URL = "/api/kb/map-op"
PROJ = "proj"
OTHER_PROJ = "other-proj"
MACHINE_EMAIL = "somnus@example.com"

MAP_ID = "kb-00001"
TARGET_ID = "kb-00002"
GAP = "the split tunnel gap"

STORED_BODY = "Lives in proj.\n\nDetail entries:\n- kb-00001 one\n"
ADDED_BODY = "Lives in proj.\n\nDetail entries:\n- kb-00001 one\n- kb-00002 two\n"
GAP_STORED_BODY = (
    f"Lives in proj.\n\nDetail entries:\n- kb-00001 one\n\nNot yet documented: {GAP}.\n"
)
GAP_STRUCK_BODY = (
    "Lives in proj.\n\nDetail entries:\n- kb-00001 one\n\n"
    "Not yet documented: nothing yet.\n"
)
LINT_BAD_BODY = (
    "Lives in proj.\n\nDetail entries:\n- kb-00001 one\n\n"
    "See src/foo.py for kb-00002 two.\n"
)


def fake_machine_user() -> User:
    """A non-admin user designated as the machine principal (somnus)."""

    return User(
        id="00000000-0000-0000-0000-000000000003",
        email=MACHINE_EMAIL,
        hashed_password="x",
        is_admin=False,
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
    )


def _seed_app_config(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, str]
) -> StatefulFakeDbPool:
    """Serve *config* as the app_config table via a fresh stateful pool.

    is_machine_principal reads ``machine_principal_email`` through
    ``attribution.get_setting`` -> ``attribution.get_db``.

    Re-pointing that one binding (the seam the ``client`` fixture patches) keeps
    every other DB touch hermetic.
    """

    pool = StatefulFakeDbPool(dict(config))

    async def _get_db() -> StatefulFakeDbPool:
        return pool

    monkeypatch.setattr(attribution_module, "get_db", _get_db)
    return pool


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    user: Any = None,
    *,
    machine_configured: bool = True,
) -> StatefulFakeDbPool:
    """Point get_current_user at *user* and seed the machine-principal config."""

    app.dependency_overrides[get_current_user] = user or fake_machine_user
    config = {"machine_principal_email": MACHINE_EMAIL} if machine_configured else {}
    return _seed_app_config(monkeypatch, config)


def _caps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    per_project: int = 0,
    per_kb: int = 0,
) -> list[dict[str, Any]]:
    """Replace the kb-core cap counter with a recording fake."""

    calls: list[dict[str, Any]] = []

    async def _fake(
        db: Any,
        *,
        contributor: str,
        since: datetime,
        project_ref: str | None = None,
    ) -> int:
        calls.append(
            {
                "db": db,
                "contributor": contributor,
                "since": since,
                "project_ref": project_ref,
            }
        )
        return per_project if project_ref is not None else per_kb

    monkeypatch.setattr(map_op_routes, "count_maps_created_since", _fake)
    return calls


def _entry(
    entry_id: str,
    *,
    entry_type: EntryType = EntryType.FACTUAL_REFERENCE,
    project_ref: str | None = PROJ,
    is_active: bool = True,
    knowledge_details: str = "Some details.",
) -> KnowledgeEntry:
    """Build one KnowledgeEntry for the fake kb's get() map."""

    entry = KnowledgeEntry(
        id=entry_id,
        short_title="T",
        long_title="T long",
        knowledge_details=knowledge_details,
        entry_type=entry_type,
        project_ref=project_ref,
    )
    entry.is_active = is_active
    return entry


def _wire_map(
    fake_kb: FakeKnowledgeBase,
    *,
    body: str = STORED_BODY,
    entry_id: str = MAP_ID,
    version: int = 1,
    entry_type: EntryType = EntryType.MENTAL_MAP,
) -> None:
    """Seed one stored map (or stand-in entry) the update ops will read."""

    fake_kb.entries[entry_id] = _entry(
        entry_id,
        entry_type=entry_type,
        knowledge_details=body,
    )
    fake_kb.entries[entry_id].version = version


def _seed_targets(fake_kb: FakeKnowledgeBase) -> None:
    """Seed the pointed-at detail entry in PROJ plus a cross-project one."""

    fake_kb.entries[TARGET_ID] = _entry(TARGET_ID)
    fake_kb.entries["kb-00003"] = _entry("kb-00003", project_ref=OTHER_PROJ)


def _create(body: str = ADDED_BODY, project_ref: str = PROJ) -> dict[str, Any]:
    return {
        "op": "create_map",
        "project_ref": project_ref,
        "short_title": "Area map",
        "long_title": "One subject area of proj",
        "body": body,
    }


def _add(
    body: str = ADDED_BODY,
    added_entry_id: str = TARGET_ID,
    map_id: str = MAP_ID,
    base_version: int | None = None,
) -> dict[str, Any]:
    req = {
        "op": "add_pointer",
        "map_id": map_id,
        "added_entry_id": added_entry_id,
        "body": body,
    }
    if base_version is not None:
        req["base_version"] = base_version
    return req


def _strike(
    body: str = GAP_STRUCK_BODY,
    map_id: str = MAP_ID,
    closing_entry_id: str = MAP_ID,
    base_version: int | None = None,
) -> dict[str, Any]:
    req = {
        "op": "strike_gap",
        "map_id": map_id,
        "gap_text": f"{GAP}.",
        "closing_entry_id": closing_entry_id,
        "body": body,
    }
    if base_version is not None:
        req["base_version"] = base_version
    return req


def _propose(
    body: str = GAP_STORED_BODY,
    map_id: str = MAP_ID,
    base_version: int | None = None,
) -> dict[str, Any]:
    req = {
        "op": "propose_gap",
        "map_id": map_id,
        "gap_text": f"{GAP}.",
        "body": body,
    }
    if base_version is not None:
        req["base_version"] = base_version
    return req


def _marker_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return every captured record whose message carries the route marker."""

    return [r for r in caplog.records if MARKER in r.getMessage()]


def _update_kwargs(fake_kb: FakeKnowledgeBase) -> dict[str, Any]:
    """The kwargs of the single kb.update call (fails if there was none)."""

    assert len(fake_kb.update_calls) == 1
    return fake_kb.update_calls[0][1]


# ---------------------------------------------------------------------------
# Auth: 401, then the machine-principal 403 gate
# ---------------------------------------------------------------------------


def test_no_auth_header_401(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No Authorization header -> 401, never 403 (auto_error=False bearer)."""

    monkeypatch.delenv("KB_AUTH_MODE", raising=False)
    assert client.post(URL, json=_create()).status_code == 401


def test_non_machine_principal_403(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A plain human caller is 403 even with a machine principal configured."""

    _wire(monkeypatch, fake_user)
    resp = client.post(URL, json=_add())
    assert resp.status_code == 403
    assert fake_kb.update_calls == []
    assert fake_kb.store_calls == []


def test_admin_403_when_no_machine_principal_configured(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """No config row: nobody is the machine principal, admins included."""

    _wire(monkeypatch, fake_admin_user, machine_configured=False)
    resp = client.post(URL, json=_create())
    assert resp.status_code == 403
    assert fake_kb.store_calls == []


# ---------------------------------------------------------------------------
# Happy paths — one per op
# ---------------------------------------------------------------------------


def test_create_map_201(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """create_map stores a mental_map and returns the uniform envelope."""

    _wire(monkeypatch)
    calls = _caps(monkeypatch)
    resp = client.post(URL, json=_create())
    assert resp.status_code == 201
    pointer_count = count_map_pointers(ADDED_BODY)
    assert resp.json() == {
        "map_id": "kb-00001",
        "version": 1,
        "pointer_count": pointer_count,
        "budget": map_body_budget(pointer_count),
    }
    assert len(fake_kb.store_calls) == 1
    kwargs = fake_kb.store_calls[0]
    assert kwargs["entry_type"] is EntryType.MENTAL_MAP
    assert kwargs["knowledge_details"] == ADDED_BODY
    assert kwargs["project_ref"] == PROJ
    assert kwargs["contributor"] == MACHINE_EMAIL
    assert fake_kb.update_calls == []
    # Both cap queries ran, scoped then unscoped, since UTC midnight.
    assert len(calls) == 2
    assert calls[0]["project_ref"] == PROJ
    assert calls[0]["contributor"] == MACHINE_EMAIL
    assert calls[1]["project_ref"] is None
    midnight = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    assert calls[0]["since"] == midnight
    assert calls[0]["since"].tzinfo is not None


def test_add_pointer_200(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """add_pointer appends exactly one pointer and updates via the facade."""

    _wire(monkeypatch)
    calls = _caps(monkeypatch)
    _wire_map(fake_kb)
    _seed_targets(fake_kb)
    resp = client.post(URL, json=_add(base_version=1))
    assert resp.status_code == 200
    pointer_count = count_map_pointers(ADDED_BODY)
    assert resp.json() == {
        "map_id": MAP_ID,
        "version": 1,
        "pointer_count": pointer_count,
        "budget": map_body_budget(pointer_count),
    }
    entry_id, kwargs = fake_kb.update_calls[0]
    assert entry_id == MAP_ID
    assert kwargs["knowledge_details"] == ADDED_BODY
    assert kwargs["updated_by"] == MACHINE_EMAIL
    assert "add_pointer" in kwargs["change_reason"]
    assert TARGET_ID in kwargs["change_reason"]
    assert fake_kb.store_calls == []
    # The caps are a create_map-only check.
    assert calls == []


def test_strike_gap_200(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """strike_gap removes the gap text without moving any pointer."""

    _wire(monkeypatch)
    _wire_map(fake_kb, body=GAP_STORED_BODY)
    resp = client.post(URL, json=_strike(closing_entry_id=MAP_ID))
    assert resp.status_code == 200
    assert resp.json()["map_id"] == MAP_ID
    assert resp.json()["pointer_count"] == 1
    kwargs = _update_kwargs(fake_kb)
    assert kwargs["knowledge_details"] == GAP_STRUCK_BODY
    assert "strike_gap" in kwargs["change_reason"]


def test_propose_gap_200(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """propose_gap adds the gap text without moving any pointer."""

    _wire(monkeypatch)
    _wire_map(fake_kb, body=STORED_BODY)
    resp = client.post(URL, json=_propose())
    assert resp.status_code == 200
    assert resp.json()["pointer_count"] == 1
    kwargs = _update_kwargs(fake_kb)
    assert kwargs["knowledge_details"] == GAP_STORED_BODY
    assert "propose_gap" in kwargs["change_reason"]


# ---------------------------------------------------------------------------
# 422: the closed vocabulary (no_change) and the lint
# ---------------------------------------------------------------------------


def test_no_change_is_not_a_valid_op(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """no_change never reaches HTTP: Pydantic rejects it with 422."""

    _wire(monkeypatch)
    resp = client.post(URL, json={"op": "no_change", "cluster_id": "c1"})
    assert resp.status_code == 422


def test_create_map_without_any_pointer_422(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """The cardinal rule: a map must point at something."""

    _wire(monkeypatch)
    resp = client.post(URL, json=_create(body="Lives in proj.\n\nNo pointers here.\n"))
    assert resp.status_code == 422
    assert "at least one outbound pointer" in resp.json()["detail"]
    assert fake_kb.store_calls == []


def test_create_map_lint_findings_422(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A lint-failing body is 422 with the findings rendered in the detail."""

    _wire(monkeypatch)
    resp = client.post(URL, json=_create(body=LINT_BAD_BODY))
    assert resp.status_code == 422
    assert "Map lint findings reject this mental_map write" in resp.json()["detail"]
    assert "dotted_identifier" in resp.json()["detail"]
    assert fake_kb.store_calls == []


def test_add_pointer_lint_findings_422(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """The lint applies to the submitted body on update ops too."""

    _wire(monkeypatch)
    _wire_map(fake_kb)
    _seed_targets(fake_kb)
    resp = client.post(URL, json=_add(body=LINT_BAD_BODY))
    assert resp.status_code == 422
    assert "Map lint findings reject this mental_map write" in resp.json()["detail"]
    assert fake_kb.update_calls == []


def test_strike_gap_lint_findings_422(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A gap op with a lint-failing submitted body is 422."""

    _wire(monkeypatch)
    _wire_map(fake_kb, body=GAP_STORED_BODY)
    bad = GAP_STRUCK_BODY.replace("Lives in proj.", "Lives in proj, see src/foo.py.")
    resp = client.post(URL, json=_strike(body=bad, closing_entry_id=MAP_ID))
    assert resp.status_code == 422
    assert "Map lint findings" in resp.json()["detail"]
    assert fake_kb.update_calls == []


def test_propose_gap_lint_findings_422(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A propose_gap with a lint-failing submitted body is 422."""

    _wire(monkeypatch)
    _wire_map(fake_kb, body=STORED_BODY)
    bad = GAP_STORED_BODY.replace("Lives in proj.", "Lives in proj, see src/foo.py.")
    resp = client.post(URL, json=_propose(body=bad))
    assert resp.status_code == 422
    assert fake_kb.update_calls == []


# ---------------------------------------------------------------------------
# add_pointer 409s: the additive-only and exactly-one invariants
# ---------------------------------------------------------------------------


def test_add_pointer_dropping_a_pointer_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """new must be a superset of old — additive-only, no revert."""

    _wire(monkeypatch)
    _wire_map(fake_kb)
    body = "Lives in proj.\n\nDetail entries:\n- kb-00002 two\n"
    resp = client.post(URL, json=_add(body=body))
    assert resp.status_code == 409
    assert "additive-only" in resp.json()["detail"]
    assert "kb-00001" in resp.json()["detail"]
    assert fake_kb.update_calls == []


def test_add_pointer_nothing_added_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A body identical in refs to the stored one adds nothing -> 409."""

    _wire(monkeypatch)
    _wire_map(fake_kb)
    resp = client.post(URL, json=_add(body=STORED_BODY))
    assert resp.status_code == 409
    assert "no pointer added" in resp.json()["detail"]


def test_add_pointer_more_than_one_added_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """Two new pointers in one call -> 409 with a distinct detail."""

    _wire(monkeypatch)
    _wire_map(fake_kb)
    _seed_targets(fake_kb)
    body = (
        "Lives in proj.\n\nDetail entries:\n- kb-00001 one\n- kb-00002 two\n"
        "- kb-00003 three\n"
    )
    resp = client.post(URL, json=_add(body=body, added_entry_id=TARGET_ID))
    assert resp.status_code == 409
    assert "more than one pointer added" in resp.json()["detail"]
    assert "kb-00003" in resp.json()["detail"]


def test_add_pointer_added_id_not_the_claimed_one_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """Exactly one pointer added, but not the id the request claimed -> 409."""

    _wire(monkeypatch)
    _wire_map(fake_kb)
    fake_kb.entries["kb-00003"] = _entry("kb-00003")
    resp = client.post(
        URL,
        json=_add(
            body="Lives in proj.\n\nDetail entries:\n- kb-00001 one\n- kb-00003 three\n"
        ),
    )
    assert resp.status_code == 409
    assert "kb-00003" in resp.json()["detail"]
    assert "kb-00002" in resp.json()["detail"]
    assert "not the claimed added_entry_id" in resp.json()["detail"]


def test_add_pointer_unknown_entry_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """The added id must resolve to an entry, else 404."""

    _wire(monkeypatch)
    _wire_map(fake_kb)
    body = "Lives in proj.\n\nDetail entries:\n- kb-00001 one\n- kb-99999 gone\n"
    resp = client.post(URL, json=_add(body=body, added_entry_id="kb-99999"))
    assert resp.status_code == 404
    assert "kb-99999" in resp.json()["detail"]
    assert fake_kb.update_calls == []


def test_add_pointer_cross_project_entry_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """The added entry must live in the map's own project_ref, else 409."""

    _wire(monkeypatch)
    _wire_map(fake_kb)
    fake_kb.entries["kb-00003"] = _entry("kb-00003", project_ref=OTHER_PROJ)
    resp = client.post(
        URL,
        json=_add(
            body=(
                "Lives in proj.\n\nDetail entries:\n- kb-00001 one\n- kb-00003 three\n"
            ),
            added_entry_id="kb-00003",
        ),
    )
    assert resp.status_code == 409
    assert "different project" in resp.json()["detail"]
    assert fake_kb.update_calls == []


# ---------------------------------------------------------------------------
# The gap ops: pointer-set equality plus the plain substring check
# ---------------------------------------------------------------------------


def test_strike_gap_moving_a_pointer_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A gap op must never move pointers — new must equal old."""

    _wire(monkeypatch)
    _wire_map(fake_kb, body=GAP_STORED_BODY)
    resp = client.post(URL, json=_strike(body=ADDED_BODY, closing_entry_id=MAP_ID))
    assert resp.status_code == 409
    assert "must not move pointers" in resp.json()["detail"]
    assert fake_kb.update_calls == []


def test_strike_gap_closing_entry_not_pointed_at_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """closing_entry_id must already be a pointer of the stored map."""

    _wire(monkeypatch)
    _wire_map(fake_kb, body=GAP_STORED_BODY)
    fake_kb.entries["kb-00009"] = _entry("kb-00009")
    resp = client.post(URL, json=_strike(closing_entry_id="kb-00009"))
    assert resp.status_code == 409
    assert "closing_entry_id" in resp.json()["detail"]
    assert "kb-00009" in resp.json()["detail"]


def test_strike_gap_absent_from_stored_body_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """The claimed gap is not in the stored body at all -> 409."""

    _wire(monkeypatch)
    _wire_map(fake_kb, body=STORED_BODY)
    resp = client.post(URL, json=_strike(closing_entry_id=MAP_ID))
    assert resp.status_code == 409
    assert "not present in the stored map body" in resp.json()["detail"]
    assert fake_kb.update_calls == []


def test_strike_gap_still_present_in_submitted_body_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A renderer that silently dropped the strike is caught -> 409."""

    _wire(monkeypatch)
    _wire_map(fake_kb, body=GAP_STORED_BODY)
    resp = client.post(URL, json=_strike(body=GAP_STORED_BODY, closing_entry_id=MAP_ID))
    assert resp.status_code == 409
    assert "still present in the submitted map body" in resp.json()["detail"]
    assert fake_kb.update_calls == []


def test_propose_gap_moving_a_pointer_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """propose_gap must never move pointers either."""

    _wire(monkeypatch)
    _wire_map(fake_kb, body=STORED_BODY)
    _seed_targets(fake_kb)
    resp = client.post(
        URL,
        json=_propose(
            body=(
                "Lives in proj.\n\nDetail entries:\n- kb-00001 one\n- kb-00002 two\n\n"
                f"Not yet documented: {GAP}.\n"
            )
        ),
    )
    assert resp.status_code == 409
    assert "must not move pointers" in resp.json()["detail"]
    assert fake_kb.update_calls == []


def test_propose_gap_already_present_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """The gap is already recorded in the stored body -> 409."""

    _wire(monkeypatch)
    _wire_map(fake_kb, body=GAP_STORED_BODY)
    resp = client.post(URL, json=_propose(body=GAP_STORED_BODY))
    assert resp.status_code == 409
    assert "already present in the stored map body" in resp.json()["detail"]


def test_propose_gap_absent_from_submitted_body_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A renderer that silently dropped the proposed gap is caught -> 409."""

    _wire(monkeypatch)
    _wire_map(fake_kb, body=STORED_BODY)
    resp = client.post(URL, json=_propose(body=STORED_BODY))
    assert resp.status_code == 409
    assert "absent from the submitted map body" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# base_version: the cheap guard for a manual run racing the timer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "req",
    [_add(base_version=99), _strike(base_version=99), _propose(base_version=99)],
    ids=["add_pointer", "strike_gap", "propose_gap"],
)
def test_stale_base_version_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
    req: dict[str, Any],
) -> None:
    """A supplied base_version that disagrees with the stored row is 409."""

    _wire(monkeypatch)
    _wire_map(fake_kb, body=GAP_STORED_BODY, version=4)
    fake_kb.entries[TARGET_ID] = _entry(TARGET_ID)
    resp = client.post(URL, json=req)
    assert resp.status_code == 409
    assert "base_version" in resp.json()["detail"]
    assert fake_kb.update_calls == []


@pytest.mark.parametrize(
    ("stored_body", "req"),
    [
        (GAP_STORED_BODY, _add(base_version=4)),
        (GAP_STORED_BODY, _strike(base_version=4)),
        (GAP_STRUCK_BODY, _propose(base_version=4)),
    ],
    ids=["add_pointer", "strike_gap", "propose_gap"],
)
def test_matching_base_version_succeeds(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
    stored_body: str,
    req: dict[str, Any],
) -> None:
    """A matching base_version is accepted (no check runs when omitted)."""

    _wire(monkeypatch)
    _wire_map(fake_kb, body=stored_body, version=4)
    fake_kb.entries[TARGET_ID] = _entry(TARGET_ID)
    assert client.post(URL, json=req).status_code == 200


# ---------------------------------------------------------------------------
# 404: unknown map_id, and a map_id that is not a mental_map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "req",
    [
        _add(map_id="kb-99999"),
        _strike(map_id="kb-99999"),
        _propose(map_id="kb-99999"),
    ],
    ids=["add_pointer", "strike_gap", "propose_gap"],
)
def test_unknown_map_id_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
    req: dict[str, Any],
) -> None:
    """An id that resolves to nothing is 404, never a silent update."""

    _wire(monkeypatch)
    resp = client.post(URL, json=req)
    assert resp.status_code == 404
    assert "kb-99999" in resp.json()["detail"]
    assert fake_kb.update_calls == []


def test_non_map_entry_type_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A map_id that resolves to a factual_reference entry is 404."""

    _wire(monkeypatch)
    _wire_map(fake_kb, entry_type=EntryType.FACTUAL_REFERENCE)
    resp = client.post(URL, json=_add())
    assert resp.status_code == 404
    assert "not a mental_map" in resp.json()["detail"]
    # Critically: the non-map entry was not silently updated.
    assert fake_kb.update_calls == []


# ---------------------------------------------------------------------------
# The per-night structural caps (create_map only)
# ---------------------------------------------------------------------------


def test_project_cap_reached_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """One new map per project_ref per night; the second is 409."""

    _wire(monkeypatch)
    _caps(monkeypatch, per_project=1)
    resp = client.post(URL, json=_create())
    assert resp.status_code == 409
    assert "per_project_ref" in resp.json()["detail"]
    assert "per_kb" not in resp.json()["detail"]
    assert fake_kb.store_calls == []


def test_kb_cap_reached_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """Three new maps per KB per night; the fourth is 409."""

    _wire(monkeypatch)
    _caps(monkeypatch, per_project=0, per_kb=3)
    resp = client.post(URL, json=_create())
    assert resp.status_code == 409
    assert "per_kb" in resp.json()["detail"]
    assert "per_project_ref" not in resp.json()["detail"]
    assert fake_kb.store_calls == []


# ---------------------------------------------------------------------------
# The operator-readable trail
# ---------------------------------------------------------------------------


def test_one_structured_log_line_per_handled_op(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A successful op leaves one INFO line: op, project_ref, map_id, outcome."""

    _wire(monkeypatch)
    _caps(monkeypatch)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        resp = client.post(URL, json=_create())
    assert resp.status_code == 201
    records = _marker_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    msg = records[0].getMessage()
    assert "op=create_map" in msg
    assert f"project_ref={PROJ!r}" in msg
    assert "outcome=created" in msg


def test_rejected_op_logs_outcome_too(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A 409 rejection leaves one line with the machine-readable outcome."""

    _wire(monkeypatch)
    _wire_map(fake_kb)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        resp = client.post(URL, json=_add(body=STORED_BODY))
    assert resp.status_code == 409
    records = _marker_records(caplog)
    assert len(records) == 1
    assert "outcome=no_pointer_added" in records[0].getMessage()
    assert f"map_id={MAP_ID!r}" in records[0].getMessage()
