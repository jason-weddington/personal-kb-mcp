"""Safety pipeline — deny-list, secret detection, and PII redaction."""

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Patterns that should never be ingested
_DENY_PATTERNS: list[str] = [
    # Private keys and certificates
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.crt",
    "*.cer",
    # SSH keys
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    "id_dsa",
    "id_ecdsa",
    # Environment / secrets
    ".env",
    ".env.*",
    "*.env",
    # VPN / WireGuard
    "wg*.conf",
    # Password / credential files
    "*.keychain",
    "*.keychain-db",
    "credentials.json",
    "token.json",
    # Binary / image / archive (not useful text)
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.gz",
    "*.bz2",
    "*.xz",
    "*.7z",
    "*.rar",
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",
    "*.bin",
    "*.o",
    "*.a",
    "*.class",
    "*.jar",
    "*.pyc",
    "*.pyo",
    "*.wasm",
    # Images
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.bmp",
    "*.ico",
    "*.svg",
    "*.webp",
    # Audio / video
    "*.mp3",
    "*.mp4",
    "*.wav",
    "*.avi",
    "*.mov",
    # Database files
    "*.sqlite",
    "*.sqlite3",
    "*.db",
]


def check_deny_list(path: Path) -> str | None:
    """Check if a file matches the deny-list.

    Returns the matching pattern if denied, None if allowed.
    """
    name = path.name
    for pattern in _DENY_PATTERNS:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name.lower(), pattern):
            return pattern
    return None


def detect_secrets_in_content(content: str) -> list[str] | None:
    """Scan content for secrets using detect-secrets.

    Returns a list of secret types found, or None if the library
    is not installed.
    """
    try:
        from detect_secrets.core.scan import scan_line
        from detect_secrets.settings import transient_settings
    except ImportError:
        logger.debug("detect-secrets not installed — skipping secret detection")
        return None

    secret_types: list[str] = []
    seen: set[str] = set()

    with transient_settings(
        {
            "plugins_used": [
                {"name": "HexHighEntropyString"},
                {"name": "Base64HighEntropyString"},
                {"name": "KeywordDetector"},
                {"name": "PrivateKeyDetector"},
            ]
        }
    ):
        for i, line in enumerate(content.splitlines(), 1):
            for secret in scan_line(filename="<content>", line=line, line_number=i):
                stype = secret.type
                if stype not in seen:
                    seen.add(stype)
                    secret_types.append(stype)

    return secret_types


def redact_pii(content: str) -> tuple[str, list[str]]:
    """Redact PII from content using scrubadub.

    Returns (cleaned_content, list_of_pii_types). If scrubadub
    is not installed, returns the original content unchanged.
    """
    try:
        import scrubadub
    except ImportError:
        logger.debug("scrubadub not installed — skipping PII redaction")
        return content, []

    scrubber = scrubadub.Scrubber()
    cleaned = scrubber.clean(content)

    # Detect what types were redacted by looking for {{TYPE}} markers
    pii_types: list[str] = []
    seen: set[str] = set()
    import re

    for match in re.finditer(r"\{\{(.+?)\}\}", cleaned):
        pii_type = match.group(1).split("-")[0].strip()
        if pii_type not in seen:
            seen.add(pii_type)
            pii_types.append(pii_type)

    return cleaned, pii_types


@dataclass
class SafetyResult:
    """Result from the safety pipeline."""

    action: str  # "allow", "skip", "flag"
    content: str
    reason: str | None = None
    redactions: list[str] = field(default_factory=list)


def run_safety_pipeline(path: Path, content: str) -> SafetyResult:
    """Run all safety checks on a file.

    Returns a SafetyResult indicating whether to proceed.
    """
    # 1. Deny-list check
    denied = check_deny_list(path)
    if denied:
        return SafetyResult(
            action="skip",
            content=content,
            reason=f"Matches deny-list pattern: {denied}",
        )

    # 2. Secret detection
    secrets = detect_secrets_in_content(content)
    if secrets:
        return SafetyResult(
            action="flag",
            content=content,
            reason=f"Secrets detected: {', '.join(secrets)}",
        )

    # 3. PII redaction
    cleaned, pii_types = redact_pii(content)

    return SafetyResult(
        action="allow",
        content=cleaned,
        redactions=pii_types,
    )
