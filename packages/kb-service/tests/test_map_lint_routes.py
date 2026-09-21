"""Hermetic tests for the map-lint surfaces: dry-run endpoint + write gate.

Three surfaces, one lint (``kb_core.map_lint`` — the single source of truth):

* the kb-core entry point (structured findings),
* the MCP channel's advisory adapter (``personal_kb.tools.map_lint``),
* this service's dry-run endpoint (``POST /api/kb/map-lint``).

The agreement table feeds the SAME bodies through all three and asserts the
same verdict — that is the test that makes "single source of truth" a
property rather than a claim.

Also covered:

* the machine-principal write gate on /store (create), /store (update) and
  /store_batch — 422 for the machine principal, success + stored entry for
  everyone else, and INERT (including for admins) when no
  ``machine_principal_email`` config row is set,
* the dry-run endpoint's failure STATUS (asserted as a status code, not just
  a body field — the somnus gate is ``curl -sf`` and keys on the exit code,
  so a 200 carrying ``valid: false`` would read as a green gate),
* the endpoint is authed like every other kb route but NOT admin-only.

Hermetic: no live Postgres, no Ollama, no network. The MCP adapter is loaded
from the sibling ``personal_kb`` checkout when the repos sit side by side (as
they do in this workspace); without a sibling checkout the adapter column
skips, but the kb-core vs endpoint agreement still runs everywhere.
"""

import sys
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.map_lint import MapLintCode, lint_map_body, map_body_budget
from kb_core.models.entry import EntryType

import kb_service.attribution as attribution_module
from kb_service.auth import get_current_user
from kb_service.main import app
from kb_service.models import User
from tests.conftest import (
    FakeKnowledgeBase,
    StatefulFakeDbPool,
    fake_admin_user,
    fake_user,
    make_entry,
)

# ─── the machine principal fixture user (non-admin BY DESIGN) ───────────────

MACHINE_EMAIL = "somnus@example.com"


def fake_machine_user() -> User:
    """A non-admin user designated as the machine principal (somnus)."""
    return User(
        id="00000000-0000-0000-0000-000000000003",
        email=MACHINE_EMAIL,
        hashed_password="x",
        is_admin=False,
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
    )


def _override_user(user_fn: Any) -> None:
    """Point get_current_user at *user_fn* (the suite's call-site helper)."""
    app.dependency_overrides[get_current_user] = user_fn


def _seed_app_config(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, str]
) -> StatefulFakeDbPool:
    """Serve *config* as the app_config table via a fresh stateful pool.

    The write gate reads ``machine_principal_email`` through
    ``attribution.get_setting`` -> ``attribution.get_db``; re-pointing that
    one binding (the same seam the ``client`` fixture patches) keeps every
    other DB touch hermetic.
    """
    pool = StatefulFakeDbPool(dict(config))

    async def _get_db() -> StatefulFakeDbPool:
        return pool

    monkeypatch.setattr(attribution_module, "get_db", _get_db)
    return pool


# ─── the MCP adapter (sibling checkout, best effort) ─────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ADAPTER_SRC = _REPO_ROOT.parent / "personal_kb" / "src"


def _load_mcp_adapter() -> ModuleType | None:
    """Import personal_kb.tools.map_lint from the sibling checkout, or None."""
    if not (_ADAPTER_SRC / "personal_kb" / "tools" / "map_lint.py").is_file():
        return None
    sys.path.insert(0, str(_ADAPTER_SRC))
    try:
        return import_module("personal_kb.tools.map_lint")
    finally:
        sys.path.remove(str(_ADAPTER_SRC))


ADAPTER = _load_mcp_adapter()


# ─── agreement bodies: clean, each purity rule, over/at budget ──────────────


def _calibration_body(total_chars: int, pointer_count: int) -> str:
    """Synthetic body with exact length and distinct kb- count (filler else)."""
    ids_text = " ".join(f"kb-{1000 + i:05d}" for i in range(pointer_count))
    body = ids_text + "x" * (total_chars - len(ids_text))
    assert len(body) == total_chars
    return body


AGREEMENT_BODIES = [
    # clean
    "orients the auth subsystem; follow kb-00421 and kb-01670 to sources",
    # one per purity rule
    "see https://example.com/x",
    "config at ~/.config/kb/settings.toml",
    "set KB_DB_PATH to override",
    "call store.create_entry(short_title)",
    'the flag is "--no-push"',
    "the explorer runs on port 8767",
    # over budget (zero pointers -> budget 900)
    "y" * (map_body_budget(0) + 1),
    # exactly at the budget boundary (passes)
    "y" * map_body_budget(0),
    # exactly at the budget boundary with pointers (passes)
    _calibration_body(map_body_budget(2), 2),
]


@pytest.mark.parametrize("body", AGREEMENT_BODIES)
def test_three_surfaces_agree(client: TestClient, body: str) -> None:
    """kb-core, the MCP adapter and the dry-run endpoint give one verdict."""
    _override_user(fake_user)

    kb_core_clean = lint_map_body(body) == []

    response = client.post("/api/kb/map-lint", json={"body": body})
    if kb_core_clean:
        assert response.status_code == 200
        assert response.json() == {"valid": True, "findings": []}
    else:
        # The status IS the verdict for curl -sf: assert the actual code,
        # not just the body, so the gate can't silently go green.
        assert response.status_code == 422
        payload = response.json()
        assert payload["valid"] is False
        assert {f["code"] for f in payload["findings"]} == {
            f.code.value for f in lint_map_body(body)
        }

    if ADAPTER is None:
        # No sibling personal_kb checkout: kb-core vs endpoint agreement still
        # ran above; the adapter column is covered wherever the repos sit
        # side by side (as they do in this workspace).
        pytest.skip("personal_kb sibling checkout not present")
    adapter_clean = ADAPTER.lint_map_body(body) == []  # type: ignore[union-attr]
    assert adapter_clean == kb_core_clean


def test_adapter_renders_kb_core_messages_with_the_advisory_prefix() -> None:
    """The MCP adapter is a pure renderer over kb-core's findings."""
    if ADAPTER is None:
        pytest.skip("personal_kb sibling checkout not present")
    body = 'port 8767, see https://example.com/x, KB_DB_PATH, store.f(), "x"'
    strings = ADAPTER.lint_map_body(body)  # type: ignore[union-attr]
    findings = lint_map_body(body)
    assert strings == [f"Map lint (advisory): {f.message}" for f in findings]


# ─── dry-run endpoint: auth shape + no writes ────────────────────────────────


def test_map_lint_requires_auth_401(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No Authorization header -> 401, NOT 403 (kb-01745: HTTPBearer 0.136)."""
    monkeypatch.delenv("KB_AUTH_MODE", raising=False)
    assert client.post("/api/kb/map-lint", json={"body": "x"}).status_code == 401


def test_map_lint_is_callable_by_a_non_admin_machine_principal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate endpoint itself must NOT require admin (somnus is non-admin)."""
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    assert (
        client.post("/api/kb/map-lint", json={"body": "orients kb-00421"}).status_code
        == 200
    )
    assert (
        client.post(
            "/api/kb/map-lint", json={"body": "see https://x.example/y"}
        ).status_code
        == 422
    )


def test_dry_run_writes_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    """A dry-run call never reaches the KB write surface."""
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    client.post("/api/kb/map-lint", json={"body": "orients kb-00421"})
    client.post("/api/kb/map-lint", json={"body": "port 8767"})
    assert fake_kb.store_calls == []
    assert fake_kb.update_calls == []
    assert fake_kb.store_batch_calls == []


def test_failure_status_is_one_curl_dash_sf_treats_as_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure verdict must live in the STATUS, not the body.

    curl -f exits non-zero only on non-2xx: a 200 with ``valid: false``
    would make the somnus run_checks gate read green while rejecting the
    body. Assert the concrete status so a refactor to "200 + flag" fails
    here instead of silently defeating the gate.
    """
    _override_user(fake_user)
    _seed_app_config(monkeypatch, {})
    response = client.post("/api/kb/map-lint", json={"body": "port 8767"})
    assert response.status_code == 422
    assert response.status_code // 100 != 2  # what curl -f actually keys on


# ─── machine-principal write gate: /store create ────────────────────────────

OFFENDING_MAP_BODY = (
    "orients kb-00421; set KB_DB_PATH=/tmp/kb.db on port 8767, "
    'pass "--no-push", and see https://example.com/x'
)
CLEAN_MAP_BODY = "orients kb-00421; follow it to the sources and their edges"


def _map_store_payload(body: str) -> dict[str, Any]:
    return {
        "short_title": "Map: test",
        "long_title": "A test map",
        "knowledge_details": body,
        "entry_type": "mental_map",
        "hints": {"related_entities": [{"id": "kb-00421"}]},
    }


def test_offending_map_422_for_machine_principal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    response = client.post("/api/kb/store", json=_map_store_payload(OFFENDING_MAP_BODY))
    assert response.status_code == 422
    assert "machine principal" in response.json()["detail"]
    assert "env_var" in response.json()["detail"]
    assert fake_kb.store_calls == []  # nothing stored


def test_clean_map_succeeds_for_machine_principal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    """The gate rejects findings, not the principal: a clean map stores."""
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    response = client.post("/api/kb/store", json=_map_store_payload(CLEAN_MAP_BODY))
    assert response.status_code == 200
    assert len(fake_kb.store_calls) == 1


def test_offending_map_succeeds_for_a_non_machine_user(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    """Same config row, different writer: advisory only, the write stores."""
    _override_user(fake_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    response = client.post("/api/kb/store", json=_map_store_payload(OFFENDING_MAP_BODY))
    assert response.status_code == 200
    assert len(fake_kb.store_calls) == 1


def test_offending_map_succeeds_for_admin_when_config_absent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    """No machine_principal_email row: is_machine_principal is False for
    EVERYONE INCLUDING ADMINS — the hard gate is inert and no write is
    rejected."""
    _override_user(fake_admin_user)
    pool = _seed_app_config(monkeypatch, {})
    assert "machine_principal_email" not in pool._app_config
    response = client.post("/api/kb/store", json=_map_store_payload(OFFENDING_MAP_BODY))
    assert response.status_code == 200
    assert len(fake_kb.store_calls) == 1


def test_offending_map_succeeds_for_machine_user_when_config_absent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    """The absent-config case for the machine user itself: the email alone
    does not make anyone the machine principal without the config row."""
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {})
    response = client.post("/api/kb/store", json=_map_store_payload(OFFENDING_MAP_BODY))
    assert response.status_code == 200
    assert len(fake_kb.store_calls) == 1


# ─── machine-principal write gate: /store update ─────────────────────────────


def _existing_map(fake_kb: FakeKnowledgeBase, body: str = CLEAN_MAP_BODY) -> str:
    """Seed a stored mental_map entry and return its id.

    ``model_copy`` does NOT re-validate, so the enum is passed as the enum —
    the update path compares ``existing.entry_type`` against
    ``EntryType.MENTAL_MAP`` by identity/ equality and a bare string would
    silently skip every gate.
    """
    entry = make_entry("kb-09999")
    entry = entry.model_copy(
        update={
            "entry_type": EntryType.MENTAL_MAP,
            "knowledge_details": body,
            "hints": {"related_entities": [{"id": "kb-00421"}]},
        }
    )
    fake_kb.entries[entry.id] = entry
    return entry.id


def test_offending_update_422_for_machine_principal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    entry_id = _existing_map(fake_kb)
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    response = client.post(
        "/api/kb/store",
        json={
            "update_entry_id": entry_id,
            "knowledge_details": OFFENDING_MAP_BODY,
            "change_reason": "nightly gloss pass",
        },
    )
    assert response.status_code == 422
    assert fake_kb.update_calls == []


def test_offending_update_succeeds_for_non_machine_user(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    entry_id = _existing_map(fake_kb)
    _override_user(fake_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    response = client.post(
        "/api/kb/store",
        json={
            "update_entry_id": entry_id,
            "knowledge_details": OFFENDING_MAP_BODY,
            "change_reason": "human gloss pass",
        },
    )
    assert response.status_code == 200
    assert len(fake_kb.update_calls) == 1


def test_metadata_only_update_of_stored_offending_body_422_for_machine(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    """The update gate uses the EFFECTIVE post-update body, like the orphan
    check it follows: a machine-principal metadata-only update of a stored
    body that fails the lint is rejected too, so the principal can never
    leave a map in a failing state."""
    entry_id = _existing_map(fake_kb, body=OFFENDING_MAP_BODY)
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    response = client.post(
        "/api/kb/store",
        json={
            "update_entry_id": entry_id,
            "tags": ["nightly"],
            "change_reason": "metadata only",
        },
    )
    assert response.status_code == 422
    assert fake_kb.update_calls == []


def test_metadata_only_update_of_stored_offending_body_succeeds_for_human(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    entry_id = _existing_map(fake_kb, body=OFFENDING_MAP_BODY)
    _override_user(fake_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    response = client.post(
        "/api/kb/store",
        json={
            "update_entry_id": entry_id,
            "tags": ["human pass"],
            "change_reason": "metadata only",
        },
    )
    assert response.status_code == 200
    assert len(fake_kb.update_calls) == 1


# ─── machine-principal write gate: /store_batch ──────────────────────────────


def _batch_payload(body: str) -> dict[str, Any]:
    return {
        "entries": [
            {
                "short_title": "Map: batch test",
                "long_title": "A batch test map",
                "knowledge_details": body,
                "entry_type": "mental_map",
                "hints": {"related_entities": [{"id": "kb-00421"}]},
            }
        ]
    }


def test_offending_batch_entry_422_for_machine_principal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    response = client.post(
        "/api/kb/store_batch", json=_batch_payload(OFFENDING_MAP_BODY)
    )
    assert response.status_code == 422
    assert "entry 0" in response.json()["detail"]
    assert "machine principal" in response.json()["detail"]
    assert fake_kb.store_batch_calls == []


def test_offending_batch_entry_succeeds_for_non_machine_user(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    _override_user(fake_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    response = client.post(
        "/api/kb/store_batch", json=_batch_payload(OFFENDING_MAP_BODY)
    )
    assert response.status_code == 200
    assert len(fake_kb.store_batch_calls) == 1


def test_non_map_write_never_linted_even_for_machine_principal(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_kb: FakeKnowledgeBase
) -> None:
    """The gate is scoped to mental_map: a factual_reference full of paths
    and numerals is legitimate and stores regardless of writer."""
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    response = client.post(
        "/api/kb/store",
        json={
            "short_title": "Runbook",
            "long_title": "A runbook entry",
            "knowledge_details": "set KB_DB_PATH=/tmp/kb.db on port 8767",
            "entry_type": "factual_reference",
        },
    )
    assert response.status_code == 200
    assert len(fake_kb.store_calls) == 1


def test_every_purity_code_rejects_a_machine_principal_map_write(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each rule's code appears in the 422 detail — the structured findings
    surface to the agent so it can fix without parsing prose."""
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    response = client.post("/api/kb/store", json=_map_store_payload(OFFENDING_MAP_BODY))
    assert response.status_code == 422
    detail = response.json()["detail"]
    for code in (
        "url",
        "path",
        "env_var",
        "dotted_identifier",
        "quoted_literal",
        "config_numeral",
    ):
        assert code in detail


def test_over_budget_machine_principal_map_write_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-pointer budget is enforced on the write path too — a body over
    its compositional budget rejects with the over_budget code."""
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    body = _calibration_body(map_body_budget(1) + 1, 1)
    response = client.post("/api/kb/store", json=_map_store_payload(body))
    assert response.status_code == 422
    assert MapLintCode.OVER_BUDGET.value in response.json()["detail"]


def test_raw_text_body_is_accepted_as_the_map_body(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate's real shape: raw prose, not a JSON object.

    somnus writes a composed body to a file and gates it with
    ``curl --data-binary @<path>``. On 2026-09-21 that returned 422 six times
    in one run — the file held the body verbatim while this endpoint demanded
    ``{"body": ...}`` — so every composed map was withheld with no map ever
    written. Reproduces the wire shape rather than the unit call.
    """
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    body = (
        "This map orients an agent in one subject area without holding facts.\n\n"
        "Detail entries (kb_get for specifics): kb-00513, kb-03186.\n\n"
        "Not yet documented: the publish bookkeeping path.\n"
    )
    resp = client.post(
        "/api/kb/map-lint", content=body, headers={"Content-Type": "text/plain"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"valid": True, "findings": []}


def test_raw_text_body_that_fails_the_lint_is_still_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dirty raw body is 422 for LINT reasons, not parse reasons.

    The two 422s are indistinguishable to ``curl -sf``, so the raw path must
    still reject a dirty body — accepting raw bytes must not have widened the
    gate, only changed how the body arrives.
    """
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    resp = client.post(
        "/api/kb/map-lint",
        content="Lives in /srv/talos and see kb-00513.",
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 422
    assert [f["code"] for f in resp.json()["findings"]] == ["path"]


def test_json_shape_missing_body_field_is_422_not_linted_as_prose(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed JSON request must not fall back to linting its own text.

    Falling back would let ``{"bdy": "..."}`` lint the JSON source as a map
    body and very likely PASS, which is the fail-open direction: a typo'd
    field name would read as a green gate.
    """
    _override_user(fake_machine_user)
    _seed_app_config(monkeypatch, {"machine_principal_email": MACHINE_EMAIL})
    resp = client.post("/api/kb/map-lint", json={"bdy": "kb-00513 typo'd field"})
    assert resp.status_code == 422
    assert "string 'body' field" in resp.json()["detail"]
