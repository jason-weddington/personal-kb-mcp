"""The startup supersession reconcile is best-effort and runs exactly once.

``main.lifespan`` calls ``kb.reconcile_supersession()`` right after
``_open_kb``: a failure is logged as a WARNING and never blocks startup, and
a report is summarized on one INFO line plus one WARNING per drifted row.
"""

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient
from kb_core.supersession import SupersessionReconcileReport

import kb_service.main as main_module
from kb_service.main import app
from tests.conftest import FakeKnowledgeBase


class _RaisingReconcileKb(FakeKnowledgeBase):
    async def reconcile_supersession(self) -> SupersessionReconcileReport:
        self.reconcile_calls += 1
        raise RuntimeError("db unavailable")


class _DriftReconcileKb(FakeKnowledgeBase):
    async def reconcile_supersession(self) -> SupersessionReconcileReport:
        self.reconcile_calls += 1
        return SupersessionReconcileReport(
            edges_added=1,
            set_count=1,
            cleared_count=0,
            changed=(("kb-00002", None, "kb-00003"),),
            edges_added_ids=(("kb-00003", "kb-00002"),),
        )


def _patch_lifespan(monkeypatch: pytest.MonkeyPatch, kb: FakeKnowledgeBase) -> None:
    async def _noop() -> None:
        return None

    async def _open_kb() -> Any:
        return kb

    monkeypatch.setattr(main_module, "init_db", _noop)
    monkeypatch.setattr(main_module, "close_db", _noop)
    monkeypatch.setattr(main_module, "_open_kb", _open_kb)


def test_reconcile_failure_never_blocks_startup(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    kb = _RaisingReconcileKb(results=[], filtered_count=0)
    _patch_lifespan(monkeypatch, kb)
    with caplog.at_level(logging.INFO, logger="kb_service.main"), TestClient(app):
        pass
    assert kb.reconcile_calls == 1
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("supersession-reconcile failed: db unavailable" in m for m in warnings)


def test_reconcile_report_logged_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    kb = _DriftReconcileKb(results=[], filtered_count=0)
    _patch_lifespan(monkeypatch, kb)
    with caplog.at_level(logging.INFO, logger="kb_service.main"), TestClient(app):
        pass
    assert kb.reconcile_calls == 1
    messages = [r.getMessage() for r in caplog.records]
    assert "supersession-reconcile edges_added=1 set=1 cleared=0" in messages
    assert (
        "supersession-reconcile drift target=kb-00002 old=None new='kb-00003'"
        in messages
    )


def test_client_fixture_runs_reconcile_once(
    client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    assert fake_kb.reconcile_calls == 1
