# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Lifespan management for the FastAPI application.

Extracted from main.py to reduce file size and improve maintainability.
"""

import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncContextManager, Callable, Dict, List, Optional

import orjson

from fastapi import FastAPI
from rotator_library.client import RotatingClient
from rotator_library.providers import PROVIDER_PLUGINS
from rotator_library.client._models import _MODEL_FETCH_BG_TIMEOUT
from rotator_library.credential_manager import CredentialManager
from rotator_library.dns_fix import close_doh_client, close_dns_executor
from rotator_library.model_info_service import init_model_info_service
from proxy_app.batch_manager import EmbeddingBatcher
from proxy_app.config import (
    DEFAULT_GLOBAL_TIMEOUT,
    MAX_GLOBAL_TIMEOUT,
    MIN_GLOBAL_TIMEOUT,
)

logger = logging.getLogger(__name__)


@dataclass
class LifespanConfig:
    """Configuration needed by the lifespan context manager."""

    api_keys: Dict[str, List[str]]
    proxy_api_key: Optional[str]
    bearer_proxy_api_key: Optional[str]
    override_temp_zero: str
    enable_request_logging: bool
    enable_raw_logging: bool
    use_embedding_batcher: bool
    max_concurrent_requests_per_key: Dict[str, int]
    ignore_models: Dict[str, List[str]]
    whitelist_models: Dict[str, List[str]]


# --- Extracted helper functions ---


_WIN_SOCKET_ERRORS = frozenset({64, 121})  # ERROR_NETNAME_DELETED, ERROR_SEM_TIMEOUT


def suppress_connection_reset(loop, original_handler):
    """Create a handler that suppresses noisy transport-level errors on Windows.

    High-TPS providers (fireworks, friendli) forcefully close connections
    after streaming, causing socket.shutdown() to throw in cleanup callbacks.
    Also suppresses WinError 64/121 (network name deleted / semaphore timeout)
    that occur when a client disconnects during accept().
    Scope: only suppress transport-level errors (proactor/send/socket context),
    not arbitrary business logic errors that happen to match the exception type.
    """

    def _handler(loop, context):
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
            msg = str(exc).lower()
            context_msg = context.get("message", "").lower()
            if (
                "send" in msg
                or "socket" in msg
                or "transport" in context_msg
                or "proactor" in context_msg
                or "fatal write error" in context_msg
                or "write error" in context_msg
            ):
                return
        # OSError with winerror 64 (ERROR_NETNAME_DELETED) or 121 (ERROR_SEM_TIMEOUT)
        # occurs when a client disconnects during accept() on Windows proactor loop.
        if isinstance(exc, OSError):
            winerr = getattr(exc, "winerror", None)
            if winerr in _WIN_SOCKET_ERRORS:
                return
        if original_handler:
            original_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    return _handler


def _read_json(path: str):
    """Read and parse a JSON file synchronously."""
    with open(path, "r", encoding="utf-8") as f:
        return orjson.loads(f.read())


def update_metadata(path: str, email: str, timestamp: float):
    """Update credential metadata file with email and timestamp."""
    with open(path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        metadata = data.get("_proxy_metadata", {})
        metadata["email"] = email
        metadata["last_check_timestamp"] = timestamp
        data["_proxy_metadata"] = metadata
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()


async def process_credential(provider: str, path: str, provider_instance):
    """Process a single credential: initialize and fetch user info."""
    try:
        await provider_instance.initialize_token(path)

        if not hasattr(provider_instance, "get_user_info"):
            return (provider, path, None, None)

        user_info = await provider_instance.get_user_info(path)
        email = user_info.get("email") if user_info else None
        return (provider, path, email, None)

    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception(
            "Failed to process OAuth token for %s at '%s': %s",
            provider, path, e,
        )
        return (provider, path, None, e)


async def init_http_pool(client):
    """Initialize HTTP pool with pre-warmed connections."""
    await client._ensure_http_pool()
    return len(client._provider_endpoints)


async def init_model_info():
    """Initialize model info service."""
    return await init_model_info_service()


def _shutdown_step_timeout() -> float:
    try:
        return float(os.getenv("SHUTDOWN_STEP_TIMEOUT", "5"))
    except ValueError:
        return 5.0


async def _safe_await(coro, label: str, timeout: float | None = None) -> None:
    """Await a shutdown coroutine with a bounded timeout."""
    try:
        await asyncio.wait_for(coro, timeout=timeout or _shutdown_step_timeout())
    except asyncio.TimeoutError:
        logger.warning("Timed out while closing %s; continuing shutdown", label)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("Error closing %s: %s", label, e)


async def _safe_close_async(coro_fn, label: str, timeout: float | None = None) -> None:
    """Safely call an async close method, logging errors without propagating."""
    try:
        await _safe_await(coro_fn(), label, timeout=timeout)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("Error closing %s: %s", label, e)


async def _safe_close_sync(close_fn, label: str, timeout: float | None = None) -> None:
    """Safely call a sync close method off the event loop."""
    await _safe_await(asyncio.to_thread(close_fn), label, timeout=timeout)


# --- Lifespan factory ---


def create_lifespan(config: LifespanConfig) -> Callable[[FastAPI], AsyncContextManager[None]]:
    """Create the lifespan async context manager for the FastAPI app.

    Returns an @asynccontextmanager-decorated function that can be passed
    directly to FastAPI(lifespan=...).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage the RotatingClient's lifecycle with the app's lifespan."""
        # Startup guard: warn if PROXY_API_KEY is missing; local/dev open access is allowed.
        if not config.proxy_api_key:
            logger.warning("=" * 70)
            logger.warning("SECURITY: PROXY_API_KEY is not set; proxy authentication is disabled.")
            logger.warning("Your proxy is running WITHOUT authentication — anyone can access it.")
            logger.warning("Set PROXY_API_KEY in .env to require Authorization: Bearer authentication.")
            logger.warning("=" * 70)

        # Suppress noisy ConnectionResetError from Windows ProactorEventLoop
        _original_handler = None
        if sys.platform == "win32":
            loop = asyncio.get_running_loop()
            _original_handler = loop.get_exception_handler()
            loop.set_exception_handler(
                suppress_connection_reset(loop, _original_handler)
            )

        # Bounded executor for OAuth credential file reads to avoid flooding default pool
        _oauth_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="oauth-read")

        # [MODIFIED] Perform skippable OAuth initialization at startup
        skip_oauth_init = (
            os.getenv("SKIP_OAUTH_INIT_CHECK", "false").lower() == "true"
        )

        # The CredentialManager now handles all discovery, including .env overrides.
        # We pass all environment variables to it for this purpose.
        cred_manager = CredentialManager(dict(os.environ))  # type: ignore[arg-type]
        oauth_credentials = cred_manager.discover_and_prepare()

        if not skip_oauth_init and oauth_credentials:
            logger.info(
                "Starting OAuth credential validation and deduplication..."
            )
            processed_emails = {}  # email -> {provider: path}
            credentials_to_initialize = {}  # provider -> [paths]
            final_oauth_credentials = {}

            # --- Pass 1: Pre-initialization Scan & Deduplication ---
            for provider, paths in oauth_credentials.items():
                if provider not in credentials_to_initialize:
                    credentials_to_initialize[provider] = []
                for path in paths:
                    # Skip env-based credentials (virtual paths) - they don't have metadata files
                    if path.startswith("env://"):
                        credentials_to_initialize[provider].append(path)
                        continue

                    try:
                        data = await asyncio.get_running_loop().run_in_executor(
                            _oauth_executor, _read_json, path
                        )
                        metadata = data.get("_proxy_metadata", {})
                        email = metadata.get("email")

                        if email:
                            if email not in processed_emails:
                                processed_emails[email] = {}

                            if provider in processed_emails[email]:
                                original_path = processed_emails[email][provider]
                                logger.warning(
                                    f"Duplicate for '{email}' on '{provider}' found in pre-scan: "
                                    f"'{Path(path).name}'. Original: '{Path(original_path).name}'. Skipping."
                                )
                                continue
                            else:
                                processed_emails[email][provider] = path

                        credentials_to_initialize[provider].append(path)

                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        logger.warning(
                            f"Could not pre-read metadata from '{path}': {e}. "
                            f"Will process during initialization."
                        )
                        credentials_to_initialize[provider].append(path)

            # --- Pass 2: Parallel Initialization of Filtered Credentials ---
            tasks = []
            for provider, paths in credentials_to_initialize.items():
                if not paths:
                    continue

                provider_plugin_class = PROVIDER_PLUGINS.get(provider)
                if provider_plugin_class is None:
                    continue
                if isinstance(provider_plugin_class, tuple):
                    continue  # Skip unresolved lazy entries
                if not isinstance(provider_plugin_class, type):
                    continue

                provider_instance = provider_plugin_class()  # type: ignore[call-non-callable]

                if hasattr(provider_instance, "preload_credentials"):
                    await provider_instance.preload_credentials(paths)

                for path in paths:
                    tasks.append(
                        process_credential(provider, path, provider_instance)
                    )

            # Execute all credential processing tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # --- Pass 3: Sequential Deduplication and Final Assembly ---
            for result in results:
                # Handle exceptions from gather
                if isinstance(result, BaseException):
                    logger.error(
                        f"Credential processing raised exception: {result}"
                    )
                    continue

                provider, path, email, error = result

                # Skip if there was an error
                if error:
                    continue

                # If provider doesn't support get_user_info, add directly
                if email is None:
                    if provider not in final_oauth_credentials:
                        final_oauth_credentials[provider] = []
                    final_oauth_credentials[provider].append(path)
                    continue

                # Handle empty email
                if not email:
                    logger.warning(
                        f"Could not retrieve email for '{path}'. Treating as unique."
                    )
                    if provider not in final_oauth_credentials:
                        final_oauth_credentials[provider] = []
                    final_oauth_credentials[provider].append(path)
                    continue

                # Deduplication check
                if email not in processed_emails:
                    processed_emails[email] = {}

                if (
                    provider in processed_emails[email]
                    and processed_emails[email][provider] != path
                ):
                    original_path = processed_emails[email][provider]
                    logger.warning(
                        f"Duplicate for '{email}' on '{provider}' found post-init: "
                        f"'{Path(path).name}'. Original: '{Path(original_path).name}'. Skipping."
                    )
                    continue
                else:
                    processed_emails[email][provider] = path
                    if provider not in final_oauth_credentials:
                        final_oauth_credentials[provider] = []
                    final_oauth_credentials[provider].append(path)

                    # Update metadata (skip for env-based credentials - they don't have files)
                    if not path.startswith("env://"):
                        try:
                            await asyncio.get_running_loop().run_in_executor(
                                None, update_metadata, path, email, time.time()
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logger.exception(
                                "Failed to update metadata for '%s': %s",
                                path, e,
                            )

            logger.info("OAuth credential processing complete.")
            oauth_credentials = final_oauth_credentials

        # [NEW] Load provider-specific params
        litellm_provider_params = {
            "gemini_cli": {"project_id": os.getenv("GEMINI_CLI_PROJECT_ID")}
        }

        # Load global timeout from environment.
        try:
            global_timeout = int(os.getenv("GLOBAL_TIMEOUT", str(DEFAULT_GLOBAL_TIMEOUT)))
        except ValueError:
            logger.warning(
                "Invalid GLOBAL_TIMEOUT value, using default %d",
                DEFAULT_GLOBAL_TIMEOUT,
            )
            global_timeout = DEFAULT_GLOBAL_TIMEOUT
        if global_timeout < MIN_GLOBAL_TIMEOUT:
            logger.warning(
                "GLOBAL_TIMEOUT=%d is too low, clamping to %d",
                global_timeout,
                MIN_GLOBAL_TIMEOUT,
            )
            global_timeout = MIN_GLOBAL_TIMEOUT
        elif global_timeout > MAX_GLOBAL_TIMEOUT:
            logger.warning(
                "GLOBAL_TIMEOUT=%d is too high, clamping to %d",
                global_timeout,
                MAX_GLOBAL_TIMEOUT,
            )
            global_timeout = MAX_GLOBAL_TIMEOUT

        # The client now uses the root logger configuration
        client = RotatingClient(
            api_keys=config.api_keys,
            oauth_credentials=oauth_credentials,  # Pass OAuth config
            configure_logging=True,
            global_timeout=global_timeout,
            litellm_provider_params=litellm_provider_params,
            ignore_models=config.ignore_models,
            whitelist_models=config.whitelist_models,
            enable_request_logging=config.enable_request_logging,
            max_concurrent_requests_per_key=config.max_concurrent_requests_per_key,
        )

        # [OPTIMIZED] Parallel initialization of HTTP pool, model info service, and background refresher
        # This reduces startup time by ~200-500ms compared to sequential execution
        init_results = await asyncio.gather(
            init_http_pool(client),
            init_model_info(),
            return_exceptions=True,
        )

        endpoint_count = (
            init_results[0]
            if not isinstance(init_results[0], Exception)
            else 0
        )
        model_info_service = (
            init_results[1]
            if not isinstance(init_results[1], Exception)
            else None
        )

        if not isinstance(init_results[0], Exception):
            logger.info(f"HTTP pool initialized with {endpoint_count} endpoints")

        # Log loaded credentials summary (compact, always visible for deployment verification)
        client.background_refresher.start()  # Start the background task
        app.state.rotating_client = client
        app.state.active_streams = 0
        app.state.stream_lock = asyncio.Lock()
        app.state.proxy_api_key = config.proxy_api_key
        app.state.bearer_proxy_api_key = config.bearer_proxy_api_key
        app.state.override_temp_zero = config.override_temp_zero
        app.state.enable_raw_logging = config.enable_raw_logging
        app.state.enable_request_logging = config.enable_request_logging
        app.state.use_embedding_batcher = config.use_embedding_batcher

        # Warn if no provider credentials are configured
        if not client.all_credentials:
            logger.warning("=" * 70)
            logger.warning("NO PROVIDER CREDENTIALS CONFIGURED")
            logger.warning(
                "The proxy is running but cannot serve any LLM requests."
            )
            logger.warning(
                "Launch the credential tool to add API keys or OAuth credentials."
            )
            logger.warning(
                "  * Executable: Run with --add-credential flag"
            )
            logger.warning(
                "  * Source: python src/proxy_app/main.py --add-credential"
            )
            logger.warning("=" * 70)

        import litellm  # type: ignore[import-untyped]

        os.environ["LITELLM_LOG"] = "ERROR"
        litellm.set_verbose = False  # type: ignore[reportPrivateImportUsage]
        if config.use_embedding_batcher:
            batcher = EmbeddingBatcher(client=client)
            app.state.embedding_batcher = batcher
            logger.info("RotatingClient and EmbeddingBatcher initialized.")
        else:
            app.state.embedding_batcher = None
            logger.info(
                "RotatingClient initialized (EmbeddingBatcher disabled)."
            )

        app.state.model_info_service = model_info_service
        if model_info_service:
            logger.info(
                "Model info service started (fetching pricing data in background)."
            )

        # Pre-warm model list cache in background (non-blocking, bounded timeout)
        async def _prewarm_models():
            try:
                await client.get_all_available_models_nonblocking(
                    grouped=True, timeout=_MODEL_FETCH_BG_TIMEOUT
                )
                logger.info("Model prewarm completed (slow providers fetching in background)")
            except Exception as e:
                logger.exception("Model prewarm failed: %s", e)

        asyncio.create_task(_prewarm_models())

        yield

        try:
            # Restore original exception handler on shutdown
            if sys.platform == "win32" and _original_handler is not None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.set_exception_handler(_original_handler)
                except RuntimeError:
                    logger.debug("Suppressed RuntimeError during shutdown handler cleanup")

            # Grace period: allow in-flight streaming responses to complete
            app.state._shutting_down = True
            try:
                logger.info(
                    "Shutdown requested, waiting up to 5s for active streams..."
                )
                for _ in range(50):
                    if not getattr(app.state, "active_streams", 0):
                        break
                    await asyncio.sleep(0.1)
                remaining = getattr(app.state, "active_streams", 0)
                if remaining:
                    logger.warning(
                        "Cancelling %d remaining active streams", remaining
                    )
                    # Cancel remaining in-flight stream generators
                    active_stream_gens = getattr(
                        app.state, "active_stream_gens", None
                    )
                    if active_stream_gens:
                        for stream_gen in list(active_stream_gens):
                            try:
                                if hasattr(stream_gen, "aclose"):
                                    await asyncio.wait_for(
                                        stream_gen.aclose(), timeout=2.0
                                    )
                            except asyncio.TimeoutError:
                                logger.warning(
                                    "Timeout closing stream generator during shutdown"
                                )
                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                logger.exception(
                                    "Error during stream cleanup: %s", e
                                )
                        active_stream_gens.clear()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(
                    "Error waiting for active streams during shutdown: %s", e
                )

            # Stop background/network resources with bounded waits so a stuck
            # provider cleanup cannot keep the process half-alive after Ctrl+C.
            await _safe_await(
                client.background_refresher.stop(), "background refresher"
            )
            await _safe_close_sync(close_doh_client, "DoH client")
            await _safe_close_sync(close_dns_executor, "DNS executor")
            if app.state.embedding_batcher:
                await _safe_await(
                    app.state.embedding_batcher.stop(), "embedding batcher"
                )
            await _safe_await(client.close(), "rotating client")

            # Close litellm's internal aiohttp/httpx sessions to prevent
            # "Unclosed client session" warnings on shutdown
            await _safe_close_async(litellm.close_litellm_async_clients, "litellm async clients")  # type: ignore[reportPrivateImportUsage]

            # Clear litellm's internal httpx handler cache (creates unclosed sessions)
            try:
                from litellm.llms import custom_httpx as _custom_httpx  # type: ignore[import-untyped]

                _handler = getattr(_custom_httpx, "httpx_handler", None)
                if _handler is not None:
                    for _attr in (
                        "_async_client",
                        "_client",
                        "client",
                        "async_client",
                    ):
                        _obj = getattr(_handler, _attr, None)
                        if _obj is not None:
                            if hasattr(_obj, "aclose"):
                                await _safe_close_async(_obj.aclose, f"custom_httpx.{_attr}")
                            elif hasattr(_obj, "close"):
                                await _safe_close_sync(_obj.close, f"custom_httpx.{_attr}")
                    _custom_httpx.httpx_handler = None
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("Error clearing custom_httpx handler: %s", e)

            if (
                hasattr(litellm, "aclient_session")
                and litellm.aclient_session is not None
            ):
                await _safe_close_async(litellm.aclient_session.aclose, "litellm aclient_session")
                litellm.aclient_session = None

            if (
                hasattr(litellm, "client_session")
                and litellm.client_session is not None
            ):
                await _safe_close_sync(litellm.client_session.close, "litellm client_session")
                litellm.client_session = None

            # Stop model info service
            if (
                hasattr(app.state, "model_info_service")
                and app.state.model_info_service
            ):
                await _safe_await(
                    app.state.model_info_service.stop(), "model info service"
                )

            if app.state.embedding_batcher:
                logger.info("RotatingClient and EmbeddingBatcher closed.")
            else:
                logger.info("RotatingClient closed.")
        finally:
            _oauth_executor.shutdown(wait=False)

    return lifespan
