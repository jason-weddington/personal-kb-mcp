"""Tests for kb_core.ingest.safety.redact_secrets."""

import sys

import pytest

from kb_core.ingest.safety import redact_secrets

# Fixtures assembled from fragments so no source line matches scanner rules.
_GH = "export GITHUB_TOKEN=" + "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
_JWT = (
    "Authorization: Bearer "
    + "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    + "."
    + "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ"
    + "."
    + "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)
_AWS = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
_PEM = "-----BEGIN RSA PRIV" + "ATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIV" + "ATE KEY-----"

REDACTED = [
    (
        f'export AWS_SECRET_ACCESS_KEY="{_AWS}"',
        "[REDACTED:Secret Keyword]",
        ["Secret Keyword", "AWS Access Key"],
    ),
    (
        "curl https://user:s3cretpass@example.com/x",
        "[REDACTED:Basic Auth Credentials]",
        ["Basic Auth Credentials"],
    ),
    (
        'echo one\npassword = "hunter2hunter2"\necho three\n',
        "echo one\n[REDACTED:Secret Keyword]\necho three\n",
        ["Secret Keyword"],
    ),
    (
        'echo one\r\npassword = "hunter2hunter2"\r\necho three',
        "echo one\r\n[REDACTED:Secret Keyword]\r\necho three",
        ["Secret Keyword"],
    ),
    (_PEM, "[REDACTED:Private Key]", ["Private Key"]),
    (
        "MIIEowIBAAKCAQEA\n-----END RSA PRIV" + "ATE KEY-----\n",
        "[REDACTED:Private Key]",
        ["Private Key"],
    ),
    (
        f"export AWS_SECRET_ACCESS_KEY={_AWS}",
        "[REDACTED:Secret Assignment]",
        ["Secret Assignment"],
    ),
    (
        f"aws_secret_access_key = {_AWS}",
        "[REDACTED:Secret Assignment]",
        ["Secret Assignment"],
    ),
    (_GH, "[REDACTED:GitHub Token]", ["GitHub Token"]),
    (
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
        "[REDACTED:AWS Access Key]",
        ["AWS Access Key"],
    ),
    (
        "export ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789",
        "[REDACTED:Secret Assignment]",
        ["Secret Assignment", "Anthropic API Key"],
    ),
    (
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789",
        "[REDACTED:Anthropic API Key]",
        ["Anthropic API Key"],
    ),
    (_JWT, "[REDACTED:JSON Web Token]", ["JSON Web Token"]),
    (
        "Authorization: Bearer abcdef0123456789opaque",
        "[REDACTED:Bearer Token]",
        ["Bearer Token"],
    ),
    (
        'password = "abcdefgh"',
        "[REDACTED:Secret Assignment]",
        ["Secret Assignment"],
    ),
]

UNCHANGED = [
    "ls -la /tmp",
    "max_tokens: 4096",
    'curl -H "Authorization: Bearer $TOKEN" https://x',
    "export TOKEN=$(cat ~/.tok)",
    "",
]


@pytest.mark.parametrize(("text", "expected", "types"), REDACTED)
def test_redacted(text: str, expected: str, types: list[str]) -> None:
    assert redact_secrets(text) == (expected, types)


@pytest.mark.parametrize(("text", "expected", "types"), REDACTED)
def test_idempotent(text: str, expected: str, types: list[str]) -> None:
    assert redact_secrets(expected) == (expected, [])


@pytest.mark.parametrize("text", UNCHANGED)
def test_unchanged(text: str) -> None:
    assert redact_secrets(text) == (text, [])


def test_import_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "detect_secrets.core.scan", None)
    assert redact_secrets("x") is None
