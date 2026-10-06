"""Drift guard: main package and standalone hook agree on local-mode defaults."""

from __future__ import annotations

from personal_kb_hook import defaults

from personal_kb import config


def test_local_default_url_and_key_agree() -> None:
    assert config.LOCAL_KB_URL == defaults.LOCAL_KB_URL
    assert config.LOCAL_KB_API_KEY == defaults.LOCAL_KB_API_KEY


def test_unset_env_resolves_to_defaults_in_both(monkeypatch) -> None:
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)
    assert defaults.resolve_url_key() == (
        config.get_personal_kb_url(),
        config.get_personal_kb_api_key(),
    )


def test_remote_without_key_fails_closed_in_both(monkeypatch) -> None:
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)
    assert defaults.resolve_url_key() is None
    assert config.get_personal_kb_api_key() is None
