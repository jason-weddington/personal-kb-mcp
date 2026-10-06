"""Hermetic tests for DELETE /api/kb/maps/{map_id} (map_delete_routes.py).

This is the loop's only irreversible operation, so the tests hold it to the
ruling (Jason, 2026-09-21) rather than to convenience: a ``mental_map`` holds no
facts, so a bad one is DELETED with its edges, and the audit event — not a human
reviewer — is what makes the deletion reconstructable.

Everything runs against the FakeKnowledgeBase and a stateful fake app_config
pool: no live Postgres, no Ollama, no network.

The inbound-referrer fixture mirrors a verified real case: map ``kb-01707``
"Map: agent-gtd Backend System" carries the ids of 13 other maps in its body,
the documented hub pattern, so deleting a component map genuinely dangles a
text reference in the most-read map of that project.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.map_delete import DeletedMapRecord
from kb_core.models.entry import EntryType, KnowledgeEntry

import kb_service.attribution as attribution_module
from kb_service.auth import get_current_user
from kb_service.main import app
from kb_service.models import User
from tests.conftest import (
    FakeKnowledgeBase,
    StatefulFakeDbPool,
    fake_admin_user,
    fake_user,
)

MARKER = "map-delete-route"
LOGGER = "kb_service.routes.map_delete_routes"
MACHINE_EMAIL = "somnus@example.com"

PROJ = "agent-gtd"
MAP_ID = "kb-01703"
HUB_ID = "kb-01707"
UNKNOWN_ID = "kb-99999"

REASON = "cluster was a split of a real subject area; map double-points it"

MAP_BODY = "Lives in agent-gtd.\n\nDetail entries:\n- kb-03001 one\n- kb-03002 two\n"
SHORT_TITLE = "Map: agent-gtd Queue Workers"
LONG_TITLE = "How agent-gtd's queue workers fit together"

NOT_FOUND_ERROR = (
    f"map-delete: no entry with id {UNKNOWN_ID!r} — refusing to silently no-op"
)
WRONG_TYPE_ERROR = (
    "map-delete: 'kb-00001' is a factual_reference, not a mental_map"
    " — hard delete is for maps; deactivate a knowledge entry instead"
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

    ``is_machine_principal`` reads ``machine_principal_email`` through
    ``attribution.get_setting`` -> ``attribution.get_db``, so re-pointing that
    one binding keeps every other DB touch hermetic.
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


def _record(**overrides: Any) -> DeletedMapRecord:
    """Build the DeletedMapRecord the fake kb returns, with per-test overrides."""

    fields: dict[str, Any] = {
        "entry_id": MAP_ID,
        "project_ref": PROJ,
        "short_title": SHORT_TITLE,
        "long_title": LONG_TITLE,
        "knowledge_details": MAP_BODY,
        "pointer_ids": ["kb-03001", "kb-03002"],
        "outbound_edges_deleted": 2,
        "inbound_edges_deleted": 1,
        "inbound_referrer_ids": [],
    }
    fields.update(overrides)
    return DeletedMapRecord(**fields)


def _entry(entry_id: str) -> KnowledgeEntry:
    """Build one KnowledgeEntry, for tests that also seed kb.get."""

    return KnowledgeEntry(
        id=entry_id,
        short_title="T",
        long_title="T long",
        knowledge_details="Some details.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref=PROJ,
    )


def _delete(map_id: str = MAP_ID, *, reason: str | None = REASON) -> dict[str, Any]:
    """The request body for one deletion; None reason omits the body entirely."""

    return {} if reason is None else {"reason": reason}


def _do_delete(
    client: TestClient,
    map_id: str,
    *,
    reason: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    query_reason: str | None = None,
) -> Any:
    """Issue one DELETE, with the reason in the body (or as a query parameter).

    ``client.delete`` in this starlette version takes no ``json`` kwarg, so the
    request goes through ``client.request`` instead — same wire shape.
    """

    if query_reason is not None:
        return client.request(
            "DELETE", f"/api/kb/maps/{map_id}", params={"reason": query_reason}
        )
    return client.request(
        "DELETE", f"/api/kb/maps/{map_id}", json=body if body is not None else reason
    )


def _audit_calls(fake_kb: FakeKnowledgeBase) -> list[tuple[str, Any]]:
    """The (sql, params) pairs of every audit_events INSERT the route issued."""

    return [(sql, params) for sql, params in fake_kb.db.calls if "audit_events" in sql]


def _marker_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return every captured record whose message carries the route marker."""

    return [r for r in caplog.records if MARKER in r.getMessage()]


# ---------------------------------------------------------------------------
# Auth: 401, then the machine-principal 403 gate
# ---------------------------------------------------------------------------


def test_no_auth_header_401(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No Authorization header -> 401, never 403 (auto_error=False bearer)."""

    monkeypatch.delenv("KB_AUTH_MODE", raising=False)
    resp = _do_delete(client, MAP_ID, reason=_delete())
    assert resp.status_code == 401


def test_non_machine_principal_403(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A plain human caller is 403 even with a machine principal configured.

    A human deleting a map is a different workflow with different confirmation
    needs, and is explicitly out of scope for this endpoint.
    """

    _wire(monkeypatch, fake_user)
    resp = _do_delete(client, MAP_ID, reason=_delete())
    assert resp.status_code == 403
    assert fake_kb.delete_map_calls == []
    assert _audit_calls(fake_kb) == []


def test_admin_403_when_no_machine_principal_configured(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """No config row: nobody is the machine principal, admins included."""

    _wire(monkeypatch, fake_admin_user, machine_configured=False)
    resp = _do_delete(client, MAP_ID, reason=_delete())
    assert resp.status_code == 403
    assert fake_kb.delete_map_calls == []


# ---------------------------------------------------------------------------
# 200: the full DeletedMapRecord, both with and without inbound referrers
# ---------------------------------------------------------------------------


def test_delete_map_200_returns_the_full_record(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """The whole record comes back, because deletion is irreversible."""

    _wire(monkeypatch)
    fake_kb.delete_map_record = _record()
    resp = _do_delete(client, MAP_ID, reason=_delete())
    assert resp.status_code == 200
    assert resp.json() == {
        "map_id": MAP_ID,
        "project_ref": PROJ,
        "short_title": SHORT_TITLE,
        "long_title": LONG_TITLE,
        "knowledge_details": MAP_BODY,
        "pointer_ids": ["kb-03001", "kb-03002"],
        "outbound_edges_deleted": 2,
        "inbound_edges_deleted": 1,
        "inbound_referrer_ids": [],
    }
    assert fake_kb.delete_map_calls == [MAP_ID]


def test_inbound_referrer_is_reported_not_blocking(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A referrer does not block the delete — it is reported for body repair.

    Real case: ``kb-01707`` "Map: agent-gtd Backend System" carries the ids of
    13 component maps in its body, the documented hub pattern, so deleting this
    component map dangles a text reference in the project's most-read map. The
    referrer list is how the caller finds and repairs that body.
    """

    _wire(monkeypatch)
    fake_kb.delete_map_record = _record(inbound_referrer_ids=[HUB_ID])
    resp = _do_delete(client, MAP_ID, reason=_delete())
    assert resp.status_code == 200
    assert resp.json()["inbound_referrer_ids"] == [HUB_ID]
    assert fake_kb.delete_map_calls == [MAP_ID]


def test_reason_accepted_as_query_parameter(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """The reason may arrive as a query parameter instead of a body."""

    _wire(monkeypatch)
    fake_kb.delete_map_record = _record()
    resp = _do_delete(client, MAP_ID, query_reason=REASON)
    assert resp.status_code == 200
    assert fake_kb.delete_map_calls == [MAP_ID]


# ---------------------------------------------------------------------------
# 404 vs 409: a stale id is not the same bug as deleting real knowledge
# ---------------------------------------------------------------------------


def test_unknown_map_id_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """An id matching no row is 404 — the loop holds a stale id."""

    _wire(monkeypatch)
    fake_kb._delete_map_raises = ValueError(NOT_FOUND_ERROR)
    resp = _do_delete(client, UNKNOWN_ID, reason=_delete())
    assert resp.status_code == 404
    assert UNKNOWN_ID in resp.json()["detail"]
    assert fake_kb.delete_map_calls == [UNKNOWN_ID]
    assert _audit_calls(fake_kb) == []


def test_non_map_entry_type_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """A real knowledge entry is 409 — pointed at content, never deleted here.

    This must not be conflated with the 404: it means the caller is pointed at
    real knowledge, a much more serious bug than a stale id.
    """

    _wire(monkeypatch)
    fake_kb._delete_map_raises = ValueError(WRONG_TYPE_ERROR)
    resp = _do_delete(client, "kb-00001", reason=_delete())
    assert resp.status_code == 409
    assert "factual_reference" in resp.json()["detail"]
    assert fake_kb.delete_map_calls == ["kb-00001"]
    assert _audit_calls(fake_kb) == []


# ---------------------------------------------------------------------------
# 422: the reason is required and cannot be blank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [_delete(reason=None), {"reason": ""}, {"reason": "   \t  "}],
    ids=["no_reason", "empty_reason", "whitespace_reason"],
)
def test_blank_reason_422(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
    body: dict[str, Any],
) -> None:
    """A deletion with no recorded reason is untraceable rot-removal."""

    _wire(monkeypatch)
    resp = _do_delete(client, MAP_ID, body=body)
    assert resp.status_code == 422
    assert fake_kb.delete_map_calls == []


# ---------------------------------------------------------------------------
# The audit event: the instrumentation that replaces the human reviewer
# ---------------------------------------------------------------------------


def test_audit_event_written_with_every_required_field(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """One audit_events row carrying the whole deleted map plus the reason.

    kb-core's primitive deliberately writes no audit event — it leaves the
    caller to name who deleted what and why — so this row is the only durable
    record that survives the map, and a deletion that cannot be reconstructed
    from it is the one outcome this endpoint must never produce.
    """

    _wire(monkeypatch)
    fake_kb.delete_map_record = _record(inbound_referrer_ids=[HUB_ID])
    resp = _do_delete(client, MAP_ID, reason=_delete())
    assert resp.status_code == 200
    calls = _audit_calls(fake_kb)
    assert len(calls) == 1
    sql, params = calls[0]
    assert "INSERT INTO audit_events" in sql
    event_type, entry_id, contributor, detail, _created_at = params
    assert event_type == "map_deleted"
    assert entry_id == MAP_ID
    assert contributor == MACHINE_EMAIL

    # PARSED, not substring-matched. The payload's stated purpose is to
    # reconstruct a map nobody can recover any other way, so the test has to
    # prove it is readable back — a substring assertion passes just as well on
    # a payload whose fields have run together, which is the failure the JSON
    # encoding exists to prevent (a map body carries newlines, semicolons and
    # equals signs, and `reason` is caller-supplied text).
    payload = json.loads(detail)
    assert payload["map_id"] == MAP_ID
    assert payload["project_ref"] == PROJ
    assert payload["short_title"] == SHORT_TITLE
    assert payload["knowledge_details"] == MAP_BODY
    assert payload["pointer_ids"] == ["kb-03001", "kb-03002"]
    assert payload["inbound_referrer_ids"] == [HUB_ID]
    assert payload["reason"] == REASON
    assert payload["deleted_by"] == MACHINE_EMAIL
    assert fake_kb.db.committed >= 1


# ---------------------------------------------------------------------------
# The operator-readable trail
# ---------------------------------------------------------------------------


def test_one_structured_log_line_per_deletion(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    fake_kb: FakeKnowledgeBase,
) -> None:
    """One INFO line: map_id, project_ref, both edge counts, referrer count."""

    _wire(monkeypatch)
    fake_kb.delete_map_record = _record(inbound_referrer_ids=[HUB_ID])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        resp = _do_delete(client, MAP_ID, reason=_delete())
    assert resp.status_code == 200
    records = _marker_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    msg = records[0].getMessage()
    assert "op=delete_map" in msg
    assert f"map_id={MAP_ID!r}" in msg
    assert f"project_ref={PROJ!r}" in msg
    assert "outbound_edges_deleted=2" in msg
    assert "inbound_edges_deleted=1" in msg
    assert "inbound_referrer_count=1" in msg
    assert "outcome=deleted" in msg
