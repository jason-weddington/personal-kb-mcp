"""Tests for deployment config check."""

import pytest

from personal_kb.db.connection import _check_deployment_config


@pytest.mark.asyncio
async def test_deployment_config_seeds_on_first_startup(db):
    """First startup seeds embedding_model and embedding_dim."""
    await _check_deployment_config(db, embedding_dim=1024)

    cursor = await db.execute("SELECT key, value FROM deployment_config ORDER BY key")
    rows = await cursor.fetchall()
    config = {row["key"]: row["value"] for row in rows}
    assert "embedding_model" in config
    assert config["embedding_dim"] == "1024"


@pytest.mark.asyncio
async def test_deployment_config_idempotent(db):
    """Running config check twice doesn't fail or duplicate rows."""
    await _check_deployment_config(db, embedding_dim=1024)
    await _check_deployment_config(db, embedding_dim=1024)

    cursor = await db.execute("SELECT COUNT(*) FROM deployment_config")
    row = await cursor.fetchone()
    assert row[0] == 2  # embedding_model + embedding_dim


@pytest.mark.asyncio
async def test_deployment_config_dim_mismatch_raises(db):
    """Mismatched embedding_dim raises RuntimeError."""
    await _check_deployment_config(db, embedding_dim=1024)

    with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
        await _check_deployment_config(db, embedding_dim=512)


@pytest.mark.asyncio
async def test_deployment_config_model_mismatch_warns(db, caplog):
    """Mismatched embedding_model logs a warning but doesn't raise."""
    await _check_deployment_config(db, embedding_dim=1024)

    # Change model env var for second check
    import os

    original = os.environ.get("KB_EMBEDDING_MODEL")
    os.environ["KB_EMBEDDING_MODEL"] = "different-model:1b"
    try:
        with caplog.at_level("WARNING"):
            await _check_deployment_config(db, embedding_dim=1024)
        assert "model mismatch" in caplog.text.lower()
    finally:
        if original:
            os.environ["KB_EMBEDDING_MODEL"] = original
        else:
            os.environ.pop("KB_EMBEDDING_MODEL", None)
