"""``kb_service.main._log_embedding_config`` startup observability."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

import kb_service.main as main

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")


async def _run(model_row: str | None, caplog: pytest.LogCaptureFixture) -> str:
    kb = await main._open_kb()
    try:
        if model_row is not None:
            await kb.db.execute(
                "INSERT INTO deployment_config (key, value, set_at)"
                " VALUES ('embedding_model', ?, 'then')",
                (model_row,),
            )
            await kb.db.commit()
        with caplog.at_level(logging.INFO, logger="kb_service.main"):
            await main._log_embedding_config(kb)
    finally:
        await kb.close()
    return caplog.text


async def test_mismatch_warns(
    env: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("KB_EMBEDDING_MODEL", "fake-model")
    text = await _run("qwen3-embedding:0.6b", caplog)
    assert "embedding provider=ollama model=fake-model dim=1024" in text
    assert (
        "embedding model mismatch: vectors were built with qwen3-embedding:0.6b" in text
    )


async def test_equal_no_warning(
    env: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("KB_EMBEDDING_MODEL", "fake-model")
    text = await _run("fake-model", caplog)
    assert "embedding provider=ollama" in text
    assert "mismatch" not in text


async def test_no_row_no_warning(env: None, caplog: pytest.LogCaptureFixture) -> None:
    text = await _run(None, caplog)
    assert "embedding provider=ollama" in text
    assert "mismatch" not in text
