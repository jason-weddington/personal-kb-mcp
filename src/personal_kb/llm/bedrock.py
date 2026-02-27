"""AWS Bedrock LLM client with graceful degradation."""

from __future__ import annotations

import logging
import os
from typing import Any

from personal_kb.config import get_bedrock_model, get_bedrock_region

logger = logging.getLogger(__name__)

_BEARER_TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"  # noqa: S105 (env var name, not a secret)


def _has_bearer_token() -> bool:
    """Check if a Bedrock bearer token is configured."""
    return bool(os.environ.get(_BEARER_TOKEN_ENV))


def _has_aws_credentials() -> bool:
    """Check if traditional AWS credentials are configured."""
    return bool(os.environ.get("AWS_ACCESS_KEY_ID"))


def _configure_bearer_auth(config: Any) -> None:
    """Add bearer token auth scheme to a Bedrock Config object.

    Monkey-patches the generated Config to support httpBearerAuth, which the
    service model declares but the codegen doesn't wire up yet.  Uses the
    existing smithy_http APIKeyAuthScheme plumbing with Authorization: Bearer.
    """
    from smithy_core.auth import AuthOption
    from smithy_core.shapes import ShapeID
    from smithy_core.traits import APIKeyLocation
    from smithy_http.aio.auth.apikey import APIKeyAuthScheme
    from smithy_http.aio.identity.apikey import (
        APIKeyIdentity,
        APIKeyIdentityProperties,
        APIKeyIdentityResolver,
    )

    bearer_scheme_id = ShapeID("smithy.api#httpBearerAuth")

    # --- Auth scheme: signs requests with Authorization: Bearer <token> ---
    class BearerAuthScheme(APIKeyAuthScheme):
        scheme_id = bearer_scheme_id

        def __init__(self) -> None:
            super().__init__(
                name="Authorization",
                location=APIKeyLocation.HEADER,
                scheme="Bearer",
            )

        def identity_properties(self, *, context: Any) -> APIKeyIdentityProperties:
            return {"api_key": os.environ.get(_BEARER_TOKEN_ENV, "")}

        def identity_resolver(self, *, context: Any) -> APIKeyIdentityResolver:
            return _EnvBearerTokenResolver()

    # --- Identity resolver: reads token from env var ---
    class _EnvBearerTokenResolver(APIKeyIdentityResolver):
        async def get_identity(self, *, properties: APIKeyIdentityProperties) -> APIKeyIdentity:
            token = properties.get("api_key") or os.environ.get(_BEARER_TOKEN_ENV)
            if not token:
                from smithy_core.exceptions import SmithyIdentityError

                raise SmithyIdentityError(f"{_BEARER_TOKEN_ENV} not set")
            return APIKeyIdentity(api_key=token)

    # --- Inject into Config ---
    config.auth_schemes[bearer_scheme_id] = BearerAuthScheme()

    # Patch the resolver to prefer bearer auth when token is available
    original_resolve = config.auth_scheme_resolver.resolve_auth_scheme

    def patched_resolve(auth_parameters: Any) -> list[Any]:
        options: list[Any] = original_resolve(auth_parameters)
        if _has_bearer_token():
            # Prepend bearer option so it's tried first
            bearer_option = AuthOption(
                scheme_id=bearer_scheme_id,
                identity_properties={},  # type: ignore[arg-type]
                signer_properties={},  # type: ignore[arg-type]
            )
            options.insert(0, bearer_option)
        return options

    config.auth_scheme_resolver.resolve_auth_scheme = patched_resolve


class BedrockLLMClient:
    """Generates text via the AWS Bedrock Converse API."""

    def __init__(self) -> None:
        """Initialize with lazy client creation."""
        self._client: Any = None
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Check availability. Only caches success — retries on failure."""
        if self._available is True:
            return True
        try:
            client = self._get_client()
            if client is None:
                return False
            if not _has_bearer_token() and not _has_aws_credentials():
                logger.warning(
                    "Neither %s nor AWS_ACCESS_KEY_ID set — Bedrock LLM disabled",
                    _BEARER_TOKEN_ENV,
                )
                return False
            return True
        except Exception:
            return False

    async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
        """Generate text from a prompt. Returns None if unavailable."""
        try:
            client = self._get_client()
            if client is None:
                return None

            from aws_sdk_bedrock_runtime.models import (
                ContentBlockText,
                ConverseInput,
                InferenceConfiguration,
                Message,
                SystemContentBlockText,
            )

            converse_input = ConverseInput(
                model_id=get_bedrock_model(),
                messages=[
                    Message(
                        role="user",
                        content=[ContentBlockText(value=prompt)],
                    ),
                ],
                inference_config=InferenceConfiguration(max_tokens=4096),
            )
            if system is not None:
                converse_input.system = [SystemContentBlockText(value=system)]

            response = await client.converse(converse_input)
            result: str = response.output.value.content[0].value
            self._available = True
            return result
        except Exception:
            logger.warning("Bedrock generation failed", exc_info=True)
            self._available = None
            return None

    def _get_client(self) -> Any:
        """Lazily create the BedrockRuntimeClient. Returns None if SDK missing."""
        if self._client is None:
            try:
                from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
                from aws_sdk_bedrock_runtime.config import Config

                config = Config(region=get_bedrock_region())

                # Wire up auth based on what credentials are available
                if _has_bearer_token():
                    _configure_bearer_auth(config)
                    logger.info("Bedrock: using bearer token auth")
                elif _has_aws_credentials():
                    from smithy_aws_core.identity import EnvironmentCredentialsResolver

                    config.aws_credentials_identity_resolver = EnvironmentCredentialsResolver()  # type: ignore[no-untyped-call]
                    logger.info("Bedrock: using SigV4 auth")

                self._client = BedrockRuntimeClient(config)
            except ImportError:
                logger.warning(
                    "aws-sdk-bedrock-runtime package not installed — Bedrock LLM disabled"
                )
                return None
        return self._client

    async def close(self) -> None:
        """No-op — SDK client doesn't need explicit cleanup."""
