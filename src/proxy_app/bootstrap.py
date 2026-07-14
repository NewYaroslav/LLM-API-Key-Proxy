# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Explicit startup/configuration helpers for the proxy app."""

import argparse
import atexit
import logging
import logging.handlers
import os
import queue
import re
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncContextManager, Callable

from proxy_app.config import DEFAULT_HOST, DEFAULT_PORT
from proxy_app.logging_config import NoLiteLLMLogFilter, RotatorDebugFilter
from proxy_app.ui_constants import SEPARATOR_70

logger = logging.getLogger(__name__)


@dataclass
class BootstrapState:
    root_dir: Path
    load_dotenv: Callable | None
    env_files_found: list[Path]


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="API Key Proxy Server")
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Host to bind the server to. Use 0.0.0.0 to expose to all interfaces.",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to run the server on."
    )
    parser.add_argument(
        "--enable-request-logging",
        action="store_true",
        help="Enable transaction logging in the library (logs request/response with provider correlation).",
    )
    parser.add_argument(
        "--enable-raw-logging",
        action="store_true",
        help="Enable raw I/O logging at proxy boundary (captures unmodified HTTP data, disabled by default).",
    )
    parser.add_argument(
        "--add-credential",
        action="store_true",
        help="Launch the interactive tool to add a new OAuth credential.",
    )
    return parser


def default_args() -> argparse.Namespace:
    return argparse.Namespace(
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        enable_request_logging=False,
        enable_raw_logging=False,
        add_credential=False,
    )


def configure_windows_event_loop_policy() -> None:
    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def configure_windows_console_encoding() -> None:
    if sys.platform != "win32":
        return

    import io

    if hasattr(sys.stdout, "buffer") and sys.stdout.buffer is not None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer") and sys.stderr.buffer is not None:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, io.UnsupportedOperation):
        logger.debug("stdout/stderr reconfigure failed, ignoring", exc_info=True)


def get_root_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path.cwd()


def load_environment(root_dir: Path | None = None) -> BootstrapState:
    root_dir = root_dir or get_root_dir()
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv not installed; .env files will not be loaded")
        return BootstrapState(root_dir=root_dir, load_dotenv=None, env_files_found=[])

    load_dotenv(root_dir / ".env")
    env_files_found = list(root_dir.glob("*.env"))
    for env_file in sorted(env_files_found):
        if env_file.name != ".env":
            load_dotenv(env_file, override=False)

    if env_files_found:
        env_names = [env_file.name for env_file in env_files_found]
        logger.info(
            "Loaded %d .env file(s): %s", len(env_files_found), ", ".join(env_names)
        )

    return BootstrapState(
        root_dir=root_dir, load_dotenv=load_dotenv, env_files_found=env_files_found
    )


def log_startup_banner(args: argparse.Namespace, elapsed: float | None = None) -> None:
    proxy_api_key = os.getenv("PROXY_API_KEY")
    logger.info(SEPARATOR_70)
    logger.info("Starting proxy on %s:%s", args.host, args.port)
    logger.info("Proxy API Key status: %s", "Set" if proxy_api_key else "Not Set")
    logger.info("GitHub: https://github.com/ShmidtS/LLM-API-Key-Proxy")
    logger.info(SEPARATOR_70)
    if elapsed is not None:
        logger.info("Server ready in %.2fs", elapsed)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _rotating_file_handler(
    path: Path,
    *,
    max_bytes_env: str,
    default_max_bytes: int,
    backup_count_env: str,
    default_backup_count: int,
) -> logging.Handler:
    return logging.handlers.RotatingFileHandler(
        path,
        maxBytes=max(1, _env_int(max_bytes_env, default_max_bytes)),
        backupCount=max(1, _env_int(backup_count_env, default_backup_count)),
        encoding="utf-8",
    )


def configure_logging(root_dir: Path) -> None:
    import colorlog
    import litellm  # type: ignore[import-untyped]
    from rotator_library.utils.paths import get_logs_dir

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    log_dir = get_logs_dir(root_dir)

    info_file_handler = _rotating_file_handler(
        log_dir / "proxy.log",
        max_bytes_env="PROXY_LOG_MAX_BYTES",
        default_max_bytes=20 * 1024 * 1024,
        backup_count_env="PROXY_LOG_BACKUP_COUNT",
        default_backup_count=5,
    )
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    info_queue = queue.Queue(-1)
    queued_info_handler = logging.handlers.QueueHandler(info_queue)
    info_queue_listener = logging.handlers.QueueListener(
        info_queue, info_file_handler, respect_handler_level=True
    )
    info_queue_listener.start()
    atexit.register(info_queue_listener.stop)

    debug_file_handler = _rotating_file_handler(
        log_dir / "proxy_debug.log",
        max_bytes_env="PROXY_DEBUG_LOG_MAX_BYTES",
        default_max_bytes=50 * 1024 * 1024,
        backup_count_env="PROXY_DEBUG_LOG_BACKUP_COUNT",
        default_backup_count=3,
    )
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    debug_file_handler.addFilter(RotatorDebugFilter())
    debug_queue = queue.Queue(-1)
    queued_debug_handler = logging.handlers.QueueHandler(debug_queue)
    debug_queue_listener = logging.handlers.QueueListener(
        debug_queue, debug_file_handler, respect_handler_level=True
    )
    debug_queue_listener.start()
    atexit.register(debug_queue_listener.stop)

    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )
    console_handler.addFilter(NoLiteLLMLogFilter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(queued_info_handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(queued_debug_handler)

    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.handlers = []
    litellm_logger.propagate = False


def _parse_env_prefix_map(prefix: str, parser=None):
    result = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            suffix = key[len(prefix) :].lower()
            result[suffix] = parser(value) if parser else value
    return result


def _parse_comma_list(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_max_concurrent(value: str):
    try:
        parsed = int(value)
        return parsed if parsed >= 1 else 1
    except ValueError:
        return 1


def _discover_api_keys() -> dict[str, list[str]]:
    api_keys: dict[str, list[str]] = {}
    for key, value in os.environ.items():
        if "_API_KEY" in key and key != "PROXY_API_KEY":
            match = re.match(r"^([A-Z0-9]+)_API_KEY(?:_\d+)?$", key)
            if match:
                provider = match.group(1).lower()
                api_keys.setdefault(provider, []).append(value)

    provider_credential_aliases = {"nvidia_nim": "nvidia"}
    for old, new in provider_credential_aliases.items():
        if old in api_keys and new not in api_keys:
            api_keys[new] = api_keys.pop(old)
        elif old in api_keys and new in api_keys:
            api_keys[new].extend(api_keys.pop(old))
    return api_keys


def create_lifespan_from_environment(args: argparse.Namespace) -> Callable[[Any], AsyncContextManager[None]]:
    from proxy_app._lifecycle import LifespanConfig, create_lifespan

    proxy_api_key = os.getenv("PROXY_API_KEY")
    enable_request_logging = args.enable_request_logging
    enable_raw_logging = args.enable_raw_logging

    if enable_request_logging:
        logger.info(
            "Transaction logging is enabled (library-level with provider correlation)."
        )
    if enable_raw_logging:
        logger.info("Raw I/O logging is enabled (proxy boundary, unmodified HTTP data).")

    return create_lifespan(
        LifespanConfig(
            api_keys=_discover_api_keys(),
            proxy_api_key=proxy_api_key,
            bearer_proxy_api_key=f"Bearer {proxy_api_key}" if proxy_api_key else None,
            override_temp_zero=os.getenv("OVERRIDE_TEMPERATURE_ZERO", "false").lower(),
            enable_request_logging=enable_request_logging,
            enable_raw_logging=enable_raw_logging,
            use_embedding_batcher=False,
            max_concurrent_requests_per_key=_parse_env_prefix_map(
                "MAX_CONCURRENT_REQUESTS_PER_KEY_", _parse_max_concurrent
            ),
            ignore_models=_parse_env_prefix_map("IGNORE_MODELS_", _parse_comma_list),
            whitelist_models=_parse_env_prefix_map(
                "WHITELIST_MODELS_", _parse_comma_list
            ),
        )
    )


def create_import_safe_lifespan():
    @asynccontextmanager
    async def lifespan(app):
        runtime_lifespan = create_lifespan_from_environment(getattr(app.state, "cli_args", default_args()))
        async with runtime_lifespan(app):
            yield

    return lifespan


def configure_app(app, args: argparse.Namespace) -> None:
    app.state.cli_args = args


def bootstrap(args: argparse.Namespace) -> BootstrapState:
    start_time = time.time()
    configure_windows_event_loop_policy()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    configure_windows_console_encoding()
    state = load_environment()
    log_startup_banner(args)
    configure_logging(state.root_dir)
    logger.debug("Modules loaded in %.2fs", time.time() - start_time)
    return state
