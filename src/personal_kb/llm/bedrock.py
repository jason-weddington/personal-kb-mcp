"""Re-export shim — real code moved to ``kb_core.llm.bedrock``.

Shim: real code moved to kb_core (kb-core extraction wave 3a).
Channel-rewiring wave removes this.
"""

from kb_core.llm.bedrock import (
    _CONVENTION_PROFILE,
    _SONNET_MODEL,
    BedrockLLMClient,
    _configure_bearer_auth,
    _find_aws_profile,
    _resolve_profile_credentials,
)

__all__ = [
    "_CONVENTION_PROFILE",
    "_SONNET_MODEL",
    "BedrockLLMClient",
    "_configure_bearer_auth",
    "_find_aws_profile",
    "_resolve_profile_credentials",
]
