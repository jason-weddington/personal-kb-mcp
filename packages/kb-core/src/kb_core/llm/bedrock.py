"""AWS Bedrock LLM client with graceful degradation.

Channel-agnostic: takes an :class:`~kb_core.config.BedrockProviderConfig`
at construction time. No ``os.environ`` reads inside this module — the
channel snapshots ``AWS_BEARER_TOKEN_BEDROCK`` and the presence of
``AWS_ACCESS_KEY_ID`` into the config. The smithy ``Environment-
CredentialsResolver`` and boto3's default credential chain may still read
AWS env vars at use time; those reads happen inside smithy_aws_core and
boto3, not in kb_core.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from kb_core.llm.provider import LLMCallRecord

if TYPE_CHECKING:
    from kb_core.config import BedrockProviderConfig
    from kb_core.llm.provider import LLMCallOutcome, LLMCallRecorder, LLMPurpose, Message

logger = logging.getLogger(__name__)


_CONVENTION_PROFILE = "personal_kb_bedrock"
"""Auto-detected AWS profile name when ``config.profile`` is unset."""


def _credentialed_session(profile_name: str | None) -> Any | None:
    """Return a boto3 Session whose credentials resolve now, or None.

    ``profile_name=None`` means boto3's default credential chain: environment
    variables, the shared config (including SSO and ``aws login`` sessions),
    then container and instance roles. Returns None when boto3 is missing,
    the profile doesn't exist, or no credentials resolve.
    """
    try:
        import boto3
    except ImportError:
        logger.debug("boto3 not installed — cannot resolve AWS credentials")
        return None

    try:
        session = boto3.Session(profile_name=profile_name)
        if session.get_credentials() is None:
            return None
        return session
    except Exception:
        logger.debug("Failed to resolve AWS credentials (profile %r)", profile_name, exc_info=True)
        return None


def _frozen_credentials(session: Any) -> Any | None:
    """Return the session's current credentials, refreshing them if they are near expiry."""
    creds = session.get_credentials()
    return None if creds is None else creds.get_frozen_credentials()


def _session_credentials_resolver(session: Any) -> Any:
    """Return a smithy credentials resolver that asks *session* on every request.

    boto3 refreshes temporary credentials (SSO, ``aws login``, assumed and
    instance roles) as they near expiry, so a long-running service keeps
    signing with live keys. A snapshot taken when the client was created
    would start failing the moment the session behind it expired.
    """
    from smithy_aws_core.identity.components import AWSCredentialsIdentity
    from smithy_core.exceptions import SmithyIdentityError

    # The session is held in this closure, not on the instance: the generated
    # client deep-copies its config on every call, and a deep-copied session
    # would lose the credentials it has already refreshed.
    class _SessionCredentialsResolver:
        async def get_identity(self, *, properties: Any) -> AWSCredentialsIdentity:
            # A refresh can make a network call; keep it off the event loop.
            frozen = await asyncio.to_thread(_frozen_credentials, session)
            if frozen is None:
                raise SmithyIdentityError("AWS credentials no longer resolve")
            return AWSCredentialsIdentity(
                access_key_id=frozen.access_key,
                secret_access_key=frozen.secret_key,
                session_token=frozen.token,
            )

    return _SessionCredentialsResolver()


def _find_aws_profile(explicit_profile: str | None) -> str | None:
    """Find the AWS profile to use for Bedrock credentials.

    Resolution order:
    1. ``explicit_profile`` (from :class:`BedrockProviderConfig`)
    2. :data:`_CONVENTION_PROFILE` if it exists in available profiles
    3. ``None``
    """
    if explicit_profile:
        return explicit_profile

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


def _configure_bearer_auth(config: Any, bearer_token: str) -> None:
    """Add bearer token auth scheme to a Bedrock Config object.

    Monkey-patches the generated Config to support ``httpBearerAuth``,
    which the service model declares but the codegen doesn't wire up
    yet. Uses the existing smithy_http APIKeyAuthScheme plumbing with
    ``Authorization: Bearer``. The bearer token is captured here from
    the caller (the channel's snapshot of ``AWS_BEARER_TOKEN_BEDROCK``);
    we do not read env inside this module.
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
            return {"api_key": bearer_token}

        def identity_resolver(self, *, context: Any) -> APIKeyIdentityResolver:
            return _StaticBearerTokenResolver()

    # --- Identity resolver: returns the captured token ---
    class _StaticBearerTokenResolver(APIKeyIdentityResolver):
        async def get_identity(self, *, properties: APIKeyIdentityProperties) -> APIKeyIdentity:
            token = properties.get("api_key") or bearer_token
            if not token:
                from smithy_core.exceptions import SmithyIdentityError

                raise SmithyIdentityError("Bedrock bearer token not configured")
            return APIKeyIdentity(api_key=token)

    # --- Inject into Config ---
    config.auth_schemes[bearer_scheme_id] = BearerAuthScheme()

    # Patch the resolver to prefer bearer auth when token is available
    original_resolve = config.auth_scheme_resolver.resolve_auth_scheme

    def patched_resolve(auth_parameters: Any) -> list[Any]:
        options: list[Any] = original_resolve(auth_parameters)
        # Prepend bearer option so it's tried first
        bearer_option = AuthOption(
            scheme_id=bearer_scheme_id,
            identity_properties={},  # type: ignore[arg-type]
            signer_properties={},  # type: ignore[arg-type]
        )
        options.insert(0, bearer_option)
        return options

    config.auth_scheme_resolver.resolve_auth_scheme = patched_resolve


_SONNET_MODEL = "us.anthropic.claude-sonnet-5-5"
"""Default Bedrock model identifier for the human-facing synthesis role."""

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds — exponential: 1s, 2s, 4s

_CACHE_POINT_MODEL_MARKER = "anthropic.claude"
"""A model id containing this gets a system cache point; any other model id gets none."""


def _int_or_none(value: Any) -> int | None:
    """Keep ``value`` only when it is a real int (not a bool)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _cache_write_1h_tokens(usage: Any) -> int | None:
    """Sum the 1-hour cache-write tokens in ``usage.cache_details``; None when it is not a list."""
    details = getattr(usage, "cache_details", None)
    if not isinstance(details, list):
        return None
    total = 0
    for detail in details:
        if getattr(detail, "ttl", None) != "1h":
            continue
        tokens = _int_or_none(getattr(detail, "input_tokens", None))
        if tokens is not None:
            total += tokens
    return total


def _wire_sha256(wire: list[dict[str, Any]]) -> str:
    """Short stable hash of the system blocks as they go out on the wire."""
    payload = json.dumps(wire, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _to_document(fields: dict[str, Any] | None) -> Any:
    """Wrap request fields in a smithy ``Document``; None stays None."""
    if not fields:
        return None
    from smithy_core.documents import Document

    return Document(fields)


def _response_text(response: Any) -> str:
    """Join the str ``value`` items of a Converse response's output content."""
    items = response.output.value.content
    return "".join(v for v in (getattr(i, "value", None) for i in items) if isinstance(v, str))


class BedrockLLMClient:
    """Generates text via the AWS Bedrock Converse API."""

    def __init__(
        self,
        config: BedrockProviderConfig,
        *,
        recorder: LLMCallRecorder | None = None,
        system_cache_1h_purposes: frozenset[LLMPurpose] = frozenset(),
    ) -> None:
        """Initialize with an explicit provider config and an optional call recorder.

        ``recorder`` receives one :class:`LLMCallRecord` per Converse attempt.
        ``system_cache_1h_purposes`` lists the purposes whose system cache
        point gets the 1-hour TTL instead of the 5-minute default.
        """
        self._config = config
        self._recorder = recorder
        self._system_cache_1h_purposes = system_cache_1h_purposes
        self._client: Any = None
        self._available: bool | None = None
        self._auth_method: str | None = None

    def _additional_fields_dict(self) -> dict[str, Any] | None:
        """Return the ``additionalModelRequestFields`` as a plain dict, or None.

        Mirrors the Anthropic client: ``thinking`` goes out as
        ``thinking = {"type": mode}`` and ``effort`` as ``output_config.effort``.
        Neither configured sends no field, which keeps the model's defaults.
        """
        fields: dict[str, Any] = {}
        if self._config.thinking is not None:
            fields["thinking"] = {"type": self._config.thinking}
        if self._config.effort is not None:
            fields["output_config"] = {"effort": self._config.effort}
        return fields or None

    def _additional_fields(self) -> Any:
        """Return the Converse ``additionalModelRequestFields`` document, or None."""
        return _to_document(self._additional_fields_dict())

    def _system_blocks(
        self, system: str, purpose: LLMPurpose
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Build the system blocks as (SDK blocks, wire dicts).

        A Claude model id gets a cache point after the system text, the
        Converse form of the Anthropic client's system breakpoint, with the
        1-hour TTL when ``purpose`` is in ``system_cache_1h_purposes``. Any
        other model id gets the bare text block, as before caching existed.
        """
        from aws_sdk_bedrock_runtime.models import SystemContentBlockText

        if _CACHE_POINT_MODEL_MARKER not in self._config.model:
            return [SystemContentBlockText(value=system)], [{"text": system}]

        from aws_sdk_bedrock_runtime.models import (
            CachePointBlock,
            CachePointType,
            CacheTTL,
            SystemContentBlockCachePoint,
        )

        wire_point: dict[str, Any] = {"type": "default"}
        ttl: str | None = None
        if purpose in self._system_cache_1h_purposes:
            ttl = CacheTTL.ONE_HOUR.value
            wire_point["ttl"] = ttl
        blocks: list[Any] = [
            SystemContentBlockText(value=system),
            SystemContentBlockCachePoint(
                value=CachePointBlock(type=CachePointType.DEFAULT.value, ttl=ttl)
            ),
        ]
        return blocks, [{"text": system}, {"cachePoint": wire_point}]

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
                    "No AWS credentials found (checked profile, "
                    "personal_kb_bedrock profile, bearer token, env credentials, "
                    "default credential chain)"
                    " — Bedrock LLM disabled",
                )
                return False
            return True
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        *,
        purpose: LLMPurpose,
        system: str | None = None,
    ) -> str | None:
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
            )

            fields = self._additional_fields_dict()
            converse_input = ConverseInput(
                model_id=self._config.model,
                messages=[
                    Message(
                        role="user",
                        content=[ContentBlockText(value=prompt)],
                    ),
                ],
                inference_config=InferenceConfiguration(max_tokens=4096),
                additional_model_request_fields=_to_document(fields),
            )
            system_sha256: str | None = None
            if system is not None:
                blocks, wire = self._system_blocks(system, purpose)
                converse_input.system = blocks
                system_sha256 = _wire_sha256(wire)

            return await self._converse_with_retry(
                client,
                converse_input,
                purpose=purpose,
                system_sha256=system_sha256,
                fields=fields,
            )
        except Exception:
            logger.warning(
                "Bedrock generation failed purpose=%s model=%s",
                purpose,
                self._config.model,
                exc_info=True,
            )
            self._available = None
            return None

    async def generate_chat(
        self,
        messages: list[Message],
        *,
        purpose: LLMPurpose,
        system: str | None = None,
    ) -> str | None:
        """Generate text from a conversation history.

        A Claude model id gets the same system cache point as ``generate``.
        There is no rolling message breakpoint.
        """
        try:
            client = self._get_client()
            if client is None:
                return None

            from aws_sdk_bedrock_runtime.models import (
                ContentBlockText,
                ConverseInput,
                InferenceConfiguration,
            )
            from aws_sdk_bedrock_runtime.models import Message as BRMessage

            br_messages = [
                BRMessage(
                    role=m["role"],
                    content=[ContentBlockText(value=m["content"])],
                )
                for m in messages
            ]
            fields = self._additional_fields_dict()
            converse_input = ConverseInput(
                model_id=self._config.model,
                messages=br_messages,
                inference_config=InferenceConfiguration(max_tokens=4096),
                additional_model_request_fields=_to_document(fields),
            )
            system_sha256: str | None = None
            if system is not None:
                blocks, wire = self._system_blocks(system, purpose)
                converse_input.system = blocks
                system_sha256 = _wire_sha256(wire)

            return await self._converse_with_retry(
                client,
                converse_input,
                purpose=purpose,
                system_sha256=system_sha256,
                fields=fields,
            )
        except Exception:
            logger.warning(
                "Bedrock chat generation failed purpose=%s model=%s",
                purpose,
                self._config.model,
                exc_info=True,
            )
            self._available = None
            return None

    def _record(
        self,
        *,
        purpose: LLMPurpose,
        start: float,
        outcome: LLMCallOutcome,
        response: Any,
        error: BaseException | None,
        system_sha256: str | None,
        fields: dict[str, Any] | None,
    ) -> None:
        """Hand one attempt's record to the recorder; a recorder failure is only logged."""
        recorder = self._recorder
        if recorder is None:
            return
        try:
            latency_ms = int((time.monotonic() - start) * 1000)
            tokens: list[int | None] = [None] * 5
            stop_reason: str | None = None
            if error is None:
                raw_stop = getattr(response, "stop_reason", None)
                stop_reason = raw_stop if isinstance(raw_stop, str) else None
                usage = getattr(response, "usage", None)
                tokens = [
                    _int_or_none(getattr(usage, "input_tokens", None)),
                    _int_or_none(getattr(usage, "output_tokens", None)),
                    _int_or_none(getattr(usage, "cache_write_input_tokens", None)),
                    _cache_write_1h_tokens(usage),
                    _int_or_none(getattr(usage, "cache_read_input_tokens", None)),
                ]
            request_fields = fields or {}
            recorder(
                LLMCallRecord(
                    ts=datetime.now(UTC).isoformat(timespec="seconds"),
                    purpose=purpose,
                    model=self._config.model,
                    outcome=outcome,
                    error_type=type(error).__name__ if error is not None else None,
                    response_id=None,
                    stop_reason=stop_reason,
                    input_tokens=tokens[0],
                    output_tokens=tokens[1],
                    cache_creation_input_tokens=tokens[2],
                    cache_creation_1h_input_tokens=tokens[3],
                    cache_read_input_tokens=tokens[4],
                    diagnostics_state="off_config",
                    previous_message_id=None,
                    cache_miss_reason=None,
                    cache_missed_input_tokens=None,
                    diagnostics_raw=None,
                    system_sha256=system_sha256,
                    latency_ms=latency_ms,
                    thinking=request_fields.get("thinking", {}).get("type"),
                    effort=request_fields.get("output_config", {}).get("effort"),
                )
            )
        except Exception:
            logger.warning("llm-call recorder failed purpose=%s", purpose, exc_info=True)

    async def _converse_with_retry(
        self,
        client: Any,
        converse_input: Any,
        *,
        purpose: LLMPurpose,
        system_sha256: str | None,
        fields: dict[str, Any] | None,
    ) -> str:
        """Call client.converse with a per-attempt timeout and exponential-backoff retries.

        Each attempt is bounded by ``config.timeout`` and, with a recorder
        set, writes exactly one record: ``ok``, ``no_text`` (converse returned
        but carried no text) or ``error`` (converse or the timeout raised).
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            start = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    client.converse(converse_input), timeout=self._config.timeout
                )
            except (Exception, asyncio.CancelledError) as exc:
                self._record(
                    purpose=purpose,
                    start=start,
                    outcome="error",
                    response=None,
                    error=exc,
                    system_sha256=system_sha256,
                    fields=fields,
                )
                if not isinstance(exc, Exception):
                    raise
                last_exc = exc
            else:
                try:
                    result = _response_text(response)
                except Exception as exc:
                    result = ""
                    last_exc = exc
                else:
                    if not result:
                        # Reasoning-only (or empty) output: treat as a failed call.
                        last_exc = ValueError("Bedrock response contained no text content")
                self._record(
                    purpose=purpose,
                    start=start,
                    outcome="ok" if result else "no_text",
                    response=response,
                    error=None,
                    system_sha256=system_sha256,
                    fields=fields,
                )
                if result:
                    self._available = True
                    return result
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "bedrock converse failed attempt=%d/%d purpose=%s model=%s "
                    "error_type=%s retry_in=%.1fs: %s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    purpose,
                    self._config.model,
                    type(last_exc).__name__,
                    delay,
                    last_exc,
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _get_client(self) -> Any:
        """Lazily create the BedrockRuntimeClient. Returns None if SDK missing."""
        if self._client is None:
            try:
                from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
                from aws_sdk_bedrock_runtime.config import Config

                config = Config(region=self._config.region)

                # Wire up auth — priority: profile > bearer token > env vars >
                # default chain. A named profile that doesn't resolve never
                # falls through to the default chain, which may be another account.
                profile = _find_aws_profile(self._config.profile)
                if profile:
                    session = _credentialed_session(profile)
                    if session is not None:
                        config.aws_credentials_identity_resolver = _session_credentials_resolver(
                            session
                        )
                        self._auth_method = f"profile:{profile}"
                        logger.info("Bedrock: using SigV4 auth (profile '%s')", profile)
                    else:
                        logger.warning("Bedrock: AWS profile '%s' has no credentials", profile)
                if self._auth_method is None and self._config.bearer_token:
                    _configure_bearer_auth(config, self._config.bearer_token)
                    self._auth_method = "bearer"
                    logger.info("Bedrock: using bearer token auth")
                elif self._auth_method is None and self._config.has_env_credentials:
                    from smithy_aws_core.identity import EnvironmentCredentialsResolver

                    config.aws_credentials_identity_resolver = EnvironmentCredentialsResolver()  # type: ignore[no-untyped-call]
                    self._auth_method = "env"
                    logger.info("Bedrock: using SigV4 auth (env vars)")
                elif self._auth_method is None and not profile:
                    session = _credentialed_session(None)
                    if session is not None:
                        config.aws_credentials_identity_resolver = _session_credentials_resolver(
                            session
                        )
                        self._auth_method = "default-chain"
                        logger.info("Bedrock: using SigV4 auth (default credential chain)")

                self._client = BedrockRuntimeClient(config)
            except ImportError:
                logger.warning(
                    "aws-sdk-bedrock-runtime package not installed — Bedrock LLM disabled"
                )
                return None
        return self._client

    async def close(self) -> None:
        """No-op — SDK client doesn't need explicit cleanup."""
