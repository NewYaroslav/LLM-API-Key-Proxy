# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# CRITICAL LOAD-BEARING IMPORT ORDER:
# 1. AIOHTTP_NO_EXTENSIONS=1 must be set before any aiohttp import
# 2. bootstrap.apply_import_time_patches must run before litellm/aiohttp users
# 3. litellm_patches.patch_litellm_finish_reason must run before litellm import
# 4. SSL monkey-patch fires when HTTP_SSL_VERIFY=false
# Do NOT reorder or simplify these imports.

import os
import sys

from . import bootstrap as _bootstrap

apply_import_time_patches = _bootstrap.apply_import_time_patches
apply_import_time_patches()

import asyncio
import logging
from collections.abc import Callable
import httpx
import litellm  # type: ignore[import-untyped]
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Union, TYPE_CHECKING

from ..env_cache import get_provider_env_cache

if TYPE_CHECKING:
    from ..anthropic_compat.models import (
        AnthropicMessagesRequest,
        AnthropicCountTokensRequest,
    )

lib_logger = logging.getLogger("rotator_library")








try:
    DEFAULT_API_KEY_MAX_CONCURRENT_REQUESTS = int(
        get_provider_env_cache().get("API_KEY_MAX_CONCURRENT_REQUESTS", 40)
    )
except ValueError:
    lib_logger.warning(
        "Invalid integer value for API_KEY_MAX_CONCURRENT_REQUESTS env var, using default"
    )
    DEFAULT_API_KEY_MAX_CONCURRENT_REQUESTS = 40

from ..usage_manager import UsageManager

from ..error_types import mask_credential
from ..provider_routing_config import ProviderConfig
from ..http_client_pool import HttpClientPool, close_http_pool
from ..providers import PROVIDER_PLUGINS
from ..model_info_service import get_model_info_service
from ..resilience import ResilienceOrchestrator
from ..credential_manager import CredentialManager
from ..background_refresher import BackgroundRefresher
from ..model_definitions import ModelDefinitions
from ..utils.paths import get_default_root, get_oauth_dir
from .bootstrap import configure_client_logging, configure_litellm_runtime
from ..utils.model_utils import (
    clear_model_match_cache,
    compile_model_patterns,
    extract_provider_from_model,
    normalize_model_string,
    register_model_patterns,
)
from ..utils.provider_locks import ProviderLockManager
from ..utils.provider_registry import get_provider_registry
from ..config import CIRCUIT_BREAKER_PROVIDER_OVERRIDES
from .client_config import ClientConfig


# Import mixin classes for method inheritance
from ._helpers import HelpersMixin
from ._streaming import StreamingMixin
from ._retry import RetryMixin
from ._media import MediaMixin, _IMAGE_PASSTHROUGH_PARAMS, _IMAGE_NATIVE_PROVIDERS
from ._models import ModelsMixin
from ..quota_reporter import QuotaReporter
from ..anthropic_adapter import AnthropicAdapter
from .provider_resolution import ProviderResolver


class RotatingClient(
    HelpersMixin,
    StreamingMixin,
    RetryMixin,
    MediaMixin,
    ModelsMixin,
):
    """
    A client that intelligently rotates and retries API keys using LiteLLM,
    with support for both streaming and non-streaming responses.
    """

    _MODEL_LIST_CACHE_TTL = 300.0

    def __init__(
        self,
        config: Optional[ClientConfig] = None,
        **kwargs,
    ):
        """
        Initialize the RotatingClient with intelligent credential rotation.

        Accepts either a ``ClientConfig`` object via the *config* parameter,
        or individual keyword arguments for backward compatibility.

        Args:
            config: A ``ClientConfig`` instance holding all configuration.
                When provided, any additional *kwargs are ignored (with a warning).
            **kwargs: Individual constructor arguments (backward compatible).
                See :class:`ClientConfig` for field descriptions.
        """
        if config is None:
            config = ClientConfig(**kwargs)
        elif kwargs:
            import warnings
            warnings.warn(
                "RotatingClient: kwargs ignored when config is provided",
                UserWarning,
                stacklevel=2,
            )

        # Resolve data_dir early - this becomes the root for all file operations
        if config.data_dir is not None:
            self.data_dir = Path(config.data_dir).resolve()
        else:
            self.data_dir = get_default_root()

        configure_client_logging(self.data_dir, config.configure_logging)
        configure_litellm_runtime()

        self.max_retries = config.max_retries
        self.global_timeout = config.global_timeout
        self.abort_on_callback_error = config.abort_on_callback_error

        self._init_credential_setup(config.api_keys, config.oauth_credentials)
        self._init_provider_resolver()
        self._init_usage_manager(config.usage_file_path, config.rotation_tolerance)
        self._init_http_pool()
        self._init_resilience()
        self._init_model_patterns(
            config.litellm_provider_params, config.ignore_models, config.whitelist_models,
            config.enable_request_logging, config.max_concurrent_requests_per_key,
        )
        self._init_semaphores()

        # Lazy-init placeholders for composition delegates
        self._quota_reporter_instance: Optional[Any] = None
        self._anthropic_adapter_instance: Optional[Any] = None

    # --- Core methods that remain in this module ---

    @property
    def model_definitions(self):
        if self._model_definitions is None:
            self._model_definitions = ModelDefinitions()
        return self._model_definitions

    @property
    def _model_registry(self):
        if self._lazy_model_registry is None:
            self._lazy_model_registry = get_model_info_service()
        return self._lazy_model_registry

    @_model_registry.setter
    def _model_registry(self, value):
        self._lazy_model_registry = value

    async def _prepare_request_kwargs(
        self,
        base_kwargs: dict[str, Any],
        provider: str,
        credential: str,
        model: str,
        *,
        include_reasoning_effort: bool = False,
    ) -> dict[str, Any]:
        """Clone and normalize per-attempt request kwargs before provider execution."""
        litellm_kwargs = base_kwargs.copy()

        self._strip_client_headers(litellm_kwargs)
        await self._apply_provider_headers(litellm_kwargs, provider, credential)

        if include_reasoning_effort and "reasoning_effort" in base_kwargs:
            litellm_kwargs["reasoning_effort"] = base_kwargs["reasoning_effort"]

        provider_params = self.litellm_provider_params.get(provider)
        if provider_params:
            existing_params = litellm_kwargs.get("litellm_params")
            litellm_kwargs["litellm_params"] = (
                {**provider_params, **existing_params}
                if existing_params
                else provider_params.copy()
            )

        litellm_kwargs["num_retries"] = 0

        get_model_options = self._get_provider_method(provider, "get_model_options")
        if get_model_options:
            model_options = get_model_options(model)
            if model_options:
                for key, value in model_options.items():
                    if key == "reasoning_effort":
                        litellm_kwargs["reasoning_effort"] = value
                    elif key not in litellm_kwargs:
                        litellm_kwargs[key] = value

        return litellm_kwargs

    # --- Pass-through delegators (called from outside this class) ---

    def get_oauth_credentials(self) -> dict[str, list[str]]:
        """Pass-through to self.oauth_credentials. Called by BackgroundRefresher."""
        return self.oauth_credentials

    def _get_provider_method(
        self, provider_name: str, method_name: str, required: bool = False
    ):
        return self._provider_resolver.get_provider_method(
            provider_name, method_name, required
        )

    def _get_provider_instance(self, provider_name: str):
        """Pass-through to ProviderResolver. Called from mixins and usage_manager."""
        return self._provider_resolver.get_provider_instance(provider_name)

    # --- Init sub-methods (decomposed from __init__) ---

    def _init_credential_setup(
        self,
        api_keys: Optional[dict[str, list[str]]],
        oauth_credentials: Optional[dict[str, list[str]]],
    ) -> None:
        """Scan and merge credential sources (api_keys + oauth)."""
        api_keys = api_keys or {}
        oauth_credentials = oauth_credentials or {}

        api_keys = {provider: keys for provider, keys in api_keys.items() if keys}
        oauth_credentials = {
            provider: paths for provider, paths in oauth_credentials.items() if paths
        }

        if not api_keys and not oauth_credentials:
            lib_logger.warning(
                "No provider credentials configured. The client will be unable to make any API requests."
            )

        self.api_keys = api_keys
        if oauth_credentials:
            self.oauth_credentials = oauth_credentials
        else:
            self.credential_manager = CredentialManager(
                dict(os.environ), oauth_dir=get_oauth_dir(self.data_dir)
            )
            self.oauth_credentials = self.credential_manager.discover_and_prepare()
        self.background_refresher = BackgroundRefresher(self)
        self.oauth_providers = set(self.oauth_credentials.keys())

        all_credentials: dict[str, list[str]] = {}
        for provider, keys in api_keys.items():
            all_credentials.setdefault(provider, []).extend(keys)
        for provider, paths in self.oauth_credentials.items():
            all_credentials.setdefault(provider, []).extend(paths)
        self.all_credentials = all_credentials
        self._cred_offset: dict[str, int] = {}
        self._lock_manager = ProviderLockManager()

    def _init_provider_resolver(self) -> None:
        """Initialize provider plugins and the ProviderResolver."""
        self._provider_plugins = PROVIDER_PLUGINS
        self._provider_instances = get_provider_registry()
        self._provider_method_cache: dict[tuple[str, str], Any] = {}
        self.provider_config = ProviderConfig()
        self._provider_resolver = ProviderResolver(
            provider_config=self.provider_config,
            all_credentials=self.all_credentials,
            oauth_providers=self.oauth_providers,
            provider_plugins=self._provider_plugins,
            provider_instances=self._provider_instances,
            provider_method_cache=self._provider_method_cache,
        )

    def _init_usage_manager(
        self,
        usage_file_path: Optional[Union[str, Path]],
        rotation_tolerance: float,
    ) -> None:
        """Build provider configs and construct the UsageManager."""
        from ..client_config import build_all_provider_configs

        provider_configs = build_all_provider_configs(
            self.all_credentials, self._provider_plugins
        )

        if usage_file_path is not None:
            resolved_usage_path = Path(usage_file_path)
        else:
            resolved_usage_path = self.data_dir / "key_usage.json"

        self.usage_manager = UsageManager(
            file_path=resolved_usage_path,
            rotation_tolerance=rotation_tolerance,
            provider_rotation_modes=provider_configs["provider_rotation_modes"],
            provider_plugins=PROVIDER_PLUGINS,
            priority_multipliers=provider_configs["priority_multipliers"],
            priority_multipliers_by_mode=provider_configs[
                "priority_multipliers_by_mode"
            ],
            sequential_fallback_multipliers=provider_configs[
                "sequential_fallback_multipliers"
            ],
            fair_cycle_enabled=provider_configs["fair_cycle_enabled"],
            fair_cycle_tracking_mode=provider_configs["fair_cycle_tracking_mode"],
            fair_cycle_cross_tier=provider_configs["fair_cycle_cross_tier"],
            fair_cycle_duration=provider_configs["fair_cycle_duration"],
            exhaustion_cooldown_threshold=provider_configs[
                "exhaustion_cooldown_threshold"
            ],
            custom_caps=provider_configs["custom_caps"],
            credential_to_provider=self._provider_resolver.build_credential_to_provider_map(),
        )

    def _init_http_pool(self) -> None:
        """Set up HTTP client pool and caches."""
        self._model_list_cache: dict[str, tuple[list[str], float]] = {}
        self._model_fetch_tasks: dict[str, asyncio.Task] = {}
        self._http_pool: Optional[HttpClientPool] = None
        self._pool_initialized = False
        self._provider_endpoints: dict[str, str] = {}

        from cachetools import TTLCache

        self._credential_priority_cache: TTLCache = TTLCache(maxsize=64, ttl=300)
        self._resolve_model_id_cache: TTLCache = TTLCache(maxsize=256, ttl=300)

    def _init_resilience(self) -> None:
        """Wire the ResilienceOrchestrator and expose sub-components."""
        self._resilience = ResilienceOrchestrator(
            provider_overrides=CIRCUIT_BREAKER_PROVIDER_OVERRIDES,
        )
        self.cooldown_manager = self._resilience.cooldown
        self.ip_throttle_detector = self._resilience.ip_throttle
        self.circuit_breaker = self._resilience.circuit_breaker
        self.rate_limiter = self._resilience.rate_limiter

    def _init_model_patterns(
        self,
        litellm_provider_params: Optional[dict[str, Any]],
        ignore_models: Optional[dict[str, list[str]]],
        whitelist_models: Optional[dict[str, list[str]]],
        enable_request_logging: bool,
        max_concurrent_requests_per_key: Optional[dict[str, int]],
    ) -> None:
        """Compile model patterns, store runtime config, validate concurrency limits."""
        self.litellm_provider_params = litellm_provider_params or {}
        self.ignore_models = compile_model_patterns(ignore_models or {})
        self.whitelist_models = compile_model_patterns(whitelist_models or {})
        register_model_patterns(self.ignore_models)
        register_model_patterns(self.whitelist_models)
        clear_model_match_cache()
        self.enable_request_logging = enable_request_logging
        self._model_definitions = None
        self._model_registry = None

        self.max_concurrent_requests_per_key = dict(
            max_concurrent_requests_per_key or {}
        )
        for provider in self.api_keys:
            self.max_concurrent_requests_per_key.setdefault(
                provider, DEFAULT_API_KEY_MAX_CONCURRENT_REQUESTS
            )
        for provider, max_val in self.max_concurrent_requests_per_key.items():
            if max_val < 1:
                lib_logger.warning(
                    "Invalid max_concurrent for '%s': %s. Setting to 1.",
                    provider,
                    max_val,
                )
                self.max_concurrent_requests_per_key[provider] = 1

        self._consecutive_quota_failures: dict[str, int] = {}

    def _init_semaphores(self) -> None:
        """Create global backpressure semaphore."""
        _default_max_concurrent = 128 if sys.platform == "win32" else 256
        try:
            _max_concurrent = int(
                get_provider_env_cache().get(
                    "MAX_CONCURRENT_REQUESTS", str(_default_max_concurrent)
                )
            )
        except ValueError:
            lib_logger.warning(
                "Invalid integer value for MAX_CONCURRENT_REQUESTS env var, using default"
            )
            _max_concurrent = _default_max_concurrent
        self._global_semaphore = asyncio.Semaphore(_max_concurrent)


    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Release reference to the shared HTTP client pool.

        Does NOT close the pool — it is a singleton shared across all
        RotatingClient instances.  Pool lifecycle is managed via
        close_http_pool() during application shutdown.
        """
        try:
            if hasattr(self, "usage_manager") and hasattr(self.usage_manager, "close"):
                await self.usage_manager.close()
        finally:
            await close_http_pool()
            self._http_pool = None
            self._pool_initialized = False

    def _maybe_apply_compaction(
        self,
        kwargs: dict,
        model: str,
        request: Optional[Any] = None,
    ) -> None:
        """Apply context compaction to kwargs['messages'] if enabled.

        Checks CONTEXT_COMPACTION_ENABLED env var and per-request
        X-Context-Compaction header. Mutates kwargs in-place (replaces
        the messages list with the compacted copy).

        Raises ContextOverflowError if compaction cannot fit messages.
        """
        from ..config.defaults import (
            CONTEXT_COMPACTION_ENABLED,
            CONTEXT_COMPACTION_THRESHOLD,
            COMPACTION_KEEP_RECENT_TOOLS,
            COMPACTION_KEEP_RECENT_ASSISTANT,
        )
        from ..context_compactor import CompactionConfig, ContextCompactor
        from ..token_calculator import (
            count_input_tokens_result,
            estimate_input_tokens,
            get_context_window,
        )

        # Determine if compaction is requested
        header_value = None
        if request is not None:
            headers = getattr(request, "headers", None)
            if headers:
                header_value = headers.get("x-context-compaction")
        compaction_enabled = CONTEXT_COMPACTION_ENABLED or (
            isinstance(header_value, str) and header_value.lower() == "auto"
        )
        if not compaction_enabled:
            return

        messages = kwargs.get("messages")
        if not messages:
            return

        # Get context window
        context_window = get_context_window(model, self._model_registry)
        if context_window is None:
            lib_logger.debug(
                "Context compaction skipped: unknown context window for %s", model,
            )
            return

        config = CompactionConfig(
            enabled=True,
            threshold=CONTEXT_COMPACTION_THRESHOLD,
            keep_recent_tools=COMPACTION_KEEP_RECENT_TOOLS,
            keep_recent_assistant=COMPACTION_KEEP_RECENT_ASSISTANT,
        )

        # Build a minimal request_data dict for the compactor
        request_data = {"messages": messages}
        tools = kwargs.get("tools")
        if tools:
            request_data["tools"] = tools

        token_limit = int(context_window * config.threshold)
        estimated_tokens, estimated_bytes = estimate_input_tokens(
            messages=messages,
            tools=tools,
            tool_choice=kwargs.get("tool_choice"),
        )
        if estimated_tokens <= token_limit:
            lib_logger.debug(
                "Context compaction skipped: estimate fits model=%s tokens~=%d "
                "limit=%d bytes=%d",
                model,
                estimated_tokens,
                token_limit,
                estimated_bytes,
            )
            return

        token_count = count_input_tokens_result(
            messages=messages,
            model=model,
            tools=tools,
            tool_choice=kwargs.get("tool_choice"),
            allow_timed_exact=True,
        )
        if not token_count.exact:
            lib_logger.warning(
                "Context compaction skipped: exact token count unavailable for "
                "model=%s tokens~=%d limit=%d bytes=%d reason=%s",
                model,
                token_count.count,
                token_limit,
                token_count.measured_bytes,
                token_count.reason,
            )
            return
        if token_count.count <= token_limit:
            lib_logger.debug(
                "Context compaction skipped: exact count fits model=%s tokens=%d "
                "limit=%d",
                model,
                token_count.count,
                token_limit,
            )
            return

        def _counter(msgs: list, mdl: str) -> int:
            count = count_input_tokens_result(
                msgs,
                mdl,
                allow_timed_exact=True,
            )
            return count.count

        compactor = ContextCompactor(config=config, token_counter=_counter)

        compacted = compactor.compact(
            request_data,
            context_window=context_window,
            model=model,
        )

        # Replace messages with compacted copy
        if compacted["messages"] is not messages:
            kwargs["messages"] = compacted["messages"]
            lib_logger.debug(
                "Context compaction applied: model=%s original=%d compacted=%d messages",
                model, len(messages), len(compacted["messages"]),
            )

    def acompletion(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Union[Any, AsyncGenerator[Any, None]]:
        """
        Dispatcher for completion requests.

        Args:
            request: Optional request object, used for client disconnect checks and logging.
            pre_request_callback: Optional async callback function to be called before each API request attempt.
                The callback will receive the `request` object and the prepared request `kwargs` as arguments.
                This can be used for custom logic such as request validation, logging, or rate limiting.
                If the callback raises an exception, the completion request will be aborted and the exception will propagate.

        Returns:
            The completion response object, or an async generator for streaming responses, or None if all retries fail.
        """
        model = normalize_model_string(kwargs.get("model", ""))
        kwargs["model"] = model
        provider = extract_provider_from_model(model)

        # Apply tiered context compaction if enabled (env var or per-request header)
        self._maybe_apply_compaction(kwargs, model, request)

        # Apply synthetic respond tool injection for matching models
        from ..synthetic_respond_tool import inject_respond_tool, strip_respond_tool_from_history
        kwargs = strip_respond_tool_from_history(kwargs)
        _respond_tool_header = ""
        if request and hasattr(request, "headers"):
            try:
                _respond_tool_header = str(request.headers.get("x-synthetic-respond-tool", "")).lower()
            except Exception:
                _respond_tool_header = ""
        if _respond_tool_header == "true":
            _respond_tool_injected = inject_respond_tool(kwargs)
        elif _respond_tool_header == "false":
            _respond_tool_injected = False
        else:
            _respond_tool_injected = inject_respond_tool(kwargs)
        if _respond_tool_injected:
            kwargs["_respond_tool_injected"] = True

        if self._is_image_only_model(model):
            prompt = self._extract_prompt_from_chat_messages(kwargs.get("messages", []))
            if prompt:
                image_kwargs = {
                    key: kwargs[key]
                    for key in _IMAGE_PASSTHROUGH_PARAMS
                    if key in kwargs
                }
                image_kwargs["model"] = model
                image_kwargs["prompt"] = prompt
                if provider in _IMAGE_NATIVE_PROVIDERS:
                    image_kwargs["_native_provider"] = provider
                return self.aimage_generation(
                    request=request,
                    pre_request_callback=pre_request_callback,
                    **image_kwargs,
                )

        # Remove stream_options for providers that don't support it
        provider_plugin = self._get_provider_instance(provider)
        stream_options_supported = not (
            provider_plugin and getattr(provider_plugin, "stream_options_unsupported", False)
        )

        if not stream_options_supported and "stream_options" in kwargs:
            lib_logger.debug(
                "Removing stream_options for %s provider (not supported)",
                provider,
            )
            kwargs.pop("stream_options", None)

        # Check if provider requires forced streaming for high max_tokens
        # Some providers (e.g., Fireworks) reject non-streaming requests when
        # max_tokens exceeds a threshold with: "Requests with max_tokens > N must have stream=true"
        forced_streaming = False
        stream = kwargs.get("stream")
        if not stream:
            max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
            threshold = getattr(provider_plugin, "stream_required_max_tokens", None) if provider_plugin else None
            if max_tokens and threshold is not None and max_tokens > threshold:
                lib_logger.info(
                    "Forcing stream=true for %s provider (max_tokens=%s > threshold=%s)",
                    provider,
                    max_tokens,
                    threshold,
                )
                kwargs["stream"] = True
                forced_streaming = True

        if stream or forced_streaming:
            if stream_options_supported:
                stream_options = kwargs.get("stream_options")
                if not isinstance(stream_options, dict):
                    stream_options = {}
                    kwargs["stream_options"] = stream_options
                stream_options.setdefault("include_usage", True)

            if forced_streaming:
                # Internally stream but collect into a non-streaming ModelResponse
                return self._forced_streaming_acompletion(
                    request=request,
                    pre_request_callback=pre_request_callback,
                    **kwargs,
                )

            return self._rate_limited_streaming(
                request=request, pre_request_callback=pre_request_callback, **kwargs
            )
        else:
            return self._rate_limited_execute(
                litellm.acompletion,
                request=request,
                pre_request_callback=pre_request_callback,
                **kwargs,
            )

    def aembedding(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """
        Executes an embedding request with retry logic.

        Args:
            request: Optional request object, used for client disconnect checks and logging.
            pre_request_callback: Optional async callback function to be called before each API request attempt.
                The callback will receive the `request` object and the prepared request `kwargs` as arguments.
                This can be used for custom logic such as request validation, logging, or rate limiting.
                If the callback raises an exception, the embedding request will be aborted and the exception will propagate.

        Returns:
            The embedding response object, or None if all retries fail.
        """
        return self._rate_limited_execute(
            litellm.aembedding,
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )




















    # --- Provider-specific API Methods (non-litellm) ---

    async def call_provider_method(
        self,
        provider_name: str,
        method_name: str,
        **kwargs,
    ) -> Any:
        """Call a provider-specific method with credential rotation.

        Picks an available credential for the provider, gets the provider
        instance, and delegates the call. Raises AttributeError if the
        provider or method doesn't exist.
        """
        method = self._get_provider_method(provider_name, method_name, required=True)

        credentials = self.all_credentials.get(provider_name)
        if not credentials:
            raise ValueError(f"No credentials for provider '{provider_name}'")

        http_client = await self._get_http_client_async(streaming=False)

        for credential in credentials:
            try:
                if method is not None:
                    return await method(credential, http_client, **kwargs)
                raise ValueError(f"Provider method '{method_name}' not found for '{provider_name}'")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    if lib_logger.isEnabledFor(logging.WARNING):
                        lib_logger.warning(
                            "Rate limited on credential %s for %s.%s, trying next",
                            mask_credential(credential),
                            provider_name,
                            method_name,
                        )
                    continue
                raise
            except (httpx.HTTPError, TimeoutError, ConnectionError, ValueError, RuntimeError) as e:
                if lib_logger.isEnabledFor(logging.DEBUG):
                    lib_logger.debug(
                        "Failed on credential for %s.%s: %s: %s, trying next",
                        provider_name,
                        method_name,
                        type(e).__name__,
                        e,
                    )
                continue

        raise RuntimeError(
            f"All credentials exhausted for {provider_name}.{method_name}"
        )

    # --- Anthropic API Compatibility Methods ---

    @property
    def quota_reporter(self):
        if self._quota_reporter_instance is None:
            self._quota_reporter_instance = QuotaReporter(
                self.usage_manager,
                self._provider_plugins,
                self._provider_instances,
                self.all_credentials,
            )
        return self._quota_reporter_instance

    async def get_quota_stats(
        self,
        provider_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.quota_reporter.get_quota_stats(provider_filter)

    async def reload_usage_from_disk(self) -> None:
        """Force reload usage data from disk.

        Useful when wanting fresh stats without making external API calls.
        """
        await self.usage_manager.reload_from_disk()

    async def force_refresh_quota(
        self,
        provider: Optional[str] = None,
        credential: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.quota_reporter.force_refresh_quota(provider, credential)

    @property
    def anthropic_adapter(self):
        if self._anthropic_adapter_instance is None:
            self._anthropic_adapter_instance = AnthropicAdapter(
                self.acompletion,
                self.token_count,
                extract_provider_from_model,
                self.all_credentials,
                self.enable_request_logging,
            )
        return self._anthropic_adapter_instance

    async def anthropic_messages(
        self,
        request: "AnthropicMessagesRequest",
        raw_request: Optional[Any] = None,
        pre_request_callback: Optional[Callable] = None,
        raw_body_data: Optional[dict] = None,
    ) -> Any:
        return await self.anthropic_adapter.anthropic_messages(
            request, raw_request, pre_request_callback, raw_body_data
        )

    async def anthropic_count_tokens(
        self,
        request: "AnthropicCountTokensRequest",
    ) -> dict:
        return await self.anthropic_adapter.anthropic_count_tokens(request)
