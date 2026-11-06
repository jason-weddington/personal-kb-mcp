"""Drift guard: the ingest dedup similarity floor is one number in every channel.

``kb_core.config._DEFAULT_INGEST_DEDUP_SIMILARITY_FLOOR``,
``kb_core.ingest.dedup_agent._DEFAULT_DEDUP_SIMILARITY_FLOOR`` and the
``KB_INGEST_DEDUP_SIMILARITY_FLOOR`` default in both env channels
(``personal_kb.config`` and ``kb_service.config``) must agree. If one side
changes, this test fails until they agree again.

The floor also stays strictly below ``NEAR_DUPLICATE_FLOOR_DEFAULT`` (0.88,
the store-time near-duplicate floor). The two compare different text pairs —
a chunk against whole entries here, an entry against an entry at store time —
so the inequality is an invariant about the configured constants, not a claim
about how any individual chunk is treated.
"""

from __future__ import annotations

import pytest
from kb_core import config as kb_config
from kb_core.ingest import dedup_agent
from kb_service import config as svc_config

import personal_kb.config as pk_config

ENV_VAR = "KB_INGEST_DEDUP_SIMILARITY_FLOOR"
DEFAULT = 0.85


def test_defaults_agree() -> None:
    """kb_core's config constant and the dedup agent default are the same number."""
    assert kb_config._DEFAULT_INGEST_DEDUP_SIMILARITY_FLOOR == DEFAULT
    assert dedup_agent._DEFAULT_DEDUP_SIMILARITY_FLOOR == DEFAULT


def test_both_channels_default_to_the_same_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the env var unset, both env getters and both builders yield 0.85."""
    monkeypatch.delenv(ENV_VAR, raising=False)

    assert pk_config.get_ingest_dedup_similarity_floor() == DEFAULT
    assert svc_config.get_ingest_dedup_similarity_floor() == DEFAULT
    assert pk_config.build_ingest_config().dedup_similarity_floor == DEFAULT
    assert svc_config.build_ingest_config().dedup_similarity_floor == DEFAULT


def test_both_channels_follow_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """An env override reaches both getters and both builders."""
    monkeypatch.setenv(ENV_VAR, "0.9")

    assert pk_config.get_ingest_dedup_similarity_floor() == 0.9
    assert svc_config.get_ingest_dedup_similarity_floor() == 0.9
    assert pk_config.build_ingest_config().dedup_similarity_floor == 0.9
    assert svc_config.build_ingest_config().dedup_similarity_floor == 0.9


@pytest.mark.parametrize("module", [pk_config, svc_config])
def test_malformed_value_raises(module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric value raises ValueError naming the env var."""
    monkeypatch.setenv(ENV_VAR, "abc")

    with pytest.raises(ValueError, match=ENV_VAR):
        module.get_ingest_dedup_similarity_floor()  # type: ignore[attr-defined]


def test_floor_sits_below_the_near_duplicate_floor() -> None:
    """The dedup similarity floor stays strictly below the store-time floor."""
    assert DEFAULT < svc_config.NEAR_DUPLICATE_FLOOR_DEFAULT
