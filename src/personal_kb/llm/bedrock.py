"""AWS Bedrock LLM client with graceful degradation."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from personal_kb.config import get_aws_profile, get_bedrock_model, get_bedrock_region

if TYPE_CHECKING:
    from personal_kb.llm.provider import Message

logger = logging.getLogger(__name__)

_BEARER_TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"  # noqa: S105 (env var name, not a secret)


def _has_bearer_token() -> bool:
    """Check if a Bedrock bearer token is configured."""
    return bool(os.environ.get(_BEARER_TOKEN_ENV))


def _has_aws_credentials() -> bool:
    """Check if traditional AWS credentials are configured."""
    return bool(os.environ.get("AWS_ACCESS_KEY_ID"))


def _resolve_profile_credentials(profile_name: str) -> dict[str, str] | None:
    """Resolve AWS credentials from a named profile via boto3.

    Returns a dict with access_key, secret_key, and optionally token,
    or None if the profile doesn't exist or credentials can't be resolved.
    """
    try:
        import boto3
    except ImportError:
        logger.debug("boto3 not installed — cannot resolve profile '%s'", profile_name)
        return None

    try:
        session = boto3.Session(profile_name=profile_name)
        creds = session.get_credentials()
        if creds is None:
            return None
        frozen = creds.get_frozen_credentials()
        result: dict[str, str] = {
            "access_key": frozen.access_key,
            "secret_key": frozen.secret_key,
        }
        if frozen.token:
            result["token"] = frozen.token
        return result
    except Exception:
        logger.debug("Failed to resolve credentials for profile '%s'", profile_name, exc_info=True)
        return None


def _find_aws_profile() -> str | None:
    """Find the AWS profile to use for Bedrock credentials.

    Resolution order:
    1. KB_AWS_PROFILE env var (explicit override)
    2. 'personal_kb_bedrock' if it exists in available profiles
    3. None
    """
    from personal_kb.config import _CONVENTION_PROFILE

    explicit = get_aws_profile()
    if explicit:
        return explicit

    # Check if convention profile exists
    try:
        import boto3

        if _CONVENTION_PROFILE in boto3.Session().available_profiles:
            logger.info("Bedrock: found convention profile '%s'", _CONVENTION_PROFILE)
            return _CONVENTION_PROFILE
    except ImportError:
        pass
    except Exception:
        logger.debug("Failed to check available profiles", exc_info=True)

    return None


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


_SONNET_MODEL = "us.anthropic.claude-sonnet-4-6"

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds — exponential: 1s, 2s, 4s


class BedrockLLMClient:
    """Generates text via the AWS Bedrock Converse API."""

    def __init__(self, *, model_override: str | None = None) -> None:
        """Initialize with lazy client creation."""
        self._client: Any = None
        self._available: bool | None = None
        self._model_override = model_override
        self._auth_method: str | None = None

    async def is_available(self) -> bool:
        """Check availability. Only caches success — retries on failure."""
        if self._available is True:
            return True
        try:
            client = self._get_client()
            if client is None:
                return False
            if self._auth_method is None:
                logger.warning(
                    "No AWS credentials found (checked KB_AWS_PROFILE, "
                    "personal_kb_bedrock profile, %s, AWS_ACCESS_KEY_ID) — Bedrock LLM disabled",
                    _BEARER_TOKEN_ENV,
                )
                return False
            return True
        except Exception:
            return False

    def _model(self) -> str:
        return self._model_override or get_bedrock_model()

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
                model_id=self._model(),
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

            return await self._converse_with_retry(client, converse_input)
        except Exception:
            logger.warning("Bedrock generation failed", exc_info=True)
            self._available = None
            return None

    async def generate_chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> str | None:
        """Generate text from a conversation history."""
        try:
            client = self._get_client()
            if client is None:
                return None

            from aws_sdk_bedrock_runtime.models import (
                ContentBlockText,
                ConverseInput,
                InferenceConfiguration,
                SystemContentBlockText,
            )
            from aws_sdk_bedrock_runtime.models import Message as BRMessage

            br_messages = [
                BRMessage(
                    role=m["role"],
                    content=[ContentBlockText(value=m["content"])],
                )
                for m in messages
            ]
            converse_input = ConverseInput(
                model_id=self._model(),
                messages=br_messages,
                inference_config=InferenceConfiguration(max_tokens=4096),
            )
            if system is not None:
                converse_input.system = [SystemContentBlockText(value=system)]

            return await self._converse_with_retry(client, converse_input)
        except Exception:
            logger.warning("Bedrock chat generation failed", exc_info=True)
            self._available = None
            return None

    async def _converse_with_retry(self, client: Any, converse_input: Any) -> str:
        """Call client.converse with exponential-backoff retries."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await client.converse(converse_input)
                result: str = response.output.value.content[0].value
                self._available = True
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "Bedrock converse attempt %d/%d failed, retrying in %.1fs: %s",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _get_client(self) -> Any:
        """Lazily create the BedrockRuntimeClient. Returns None if SDK missing."""
        if self._client is None:
            try:
                from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
                from aws_sdk_bedrock_runtime.config import Config

                config = Config(region=get_bedrock_region())

                # Wire up auth — priority: profile > bearer token > env vars
                profile = _find_aws_profile()
                if profile:
                    creds = _resolve_profile_credentials(profile)
                    if creds:
                        from smithy_aws_core.identity.static import StaticCredentialsResolver

                        config.aws_access_key_id = creds["access_key"]
                        config.aws_secret_access_key = creds["secret_key"]
                        if "token" in creds:
                            config.aws_session_token = creds["token"]
                        config.aws_credentials_identity_resolver = StaticCredentialsResolver()  # type: ignore[no-untyped-call]
                        self._auth_method = f"profile:{profile}"
                        logger.info("Bedrock: using SigV4 auth (profile '%s')", profile)
                if self._auth_method is None and _has_bearer_token():
                    _configure_bearer_auth(config)
                    self._auth_method = "bearer"
                    logger.info("Bedrock: using bearer token auth")
                elif self._auth_method is None and _has_aws_credentials():
                    from smithy_aws_core.identity import EnvironmentCredentialsResolver

                    config.aws_credentials_identity_resolver = EnvironmentCredentialsResolver()  # type: ignore[no-untyped-call]
                    self._auth_method = "env"
                    logger.info("Bedrock: using SigV4 auth (env vars)")

                self._client = BedrockRuntimeClient(config)
            except ImportError:
                logger.warning(
                    "aws-sdk-bedrock-runtime package not installed — Bedrock LLM disabled"
                )
                return None
        return self._client

    async def close(self) -> None:
        """No-op — SDK client doesn't need explicit cleanup."""
