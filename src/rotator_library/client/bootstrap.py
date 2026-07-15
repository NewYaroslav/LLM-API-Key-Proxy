# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import logging
import os
import sys
from pathlib import Path

_IMPORT_PATCHES_APPLIED = False
_LITELLM_RUNTIME_CONFIGURED = False
_AIOHTTP_PATCHED_INIT = None


def _seed_windows_platform_uname() -> None:
    """Avoid Python 3.14 platform.system() WMI hangs during aiohttp import."""
    if sys.platform != "win32":
        return

    import platform

    if getattr(platform, "_uname_cache", None) is not None:
        return

    winver = sys.getwindowsversion()
    platform._uname_cache = platform.uname_result(  # type: ignore[attr-defined]
        "Windows",
        os.environ.get("COMPUTERNAME", ""),
        f"{winver.major}.{winver.minor}",
        f"{winver.major}.{winver.minor}.{winver.build}",
        os.environ.get("PROCESSOR_ARCHITECTURE", ""),
    )


def apply_import_time_patches() -> None:
    """Apply monkey-patches that must run before importing litellm/aiohttp users."""
    global _AIOHTTP_PATCHED_INIT, _IMPORT_PATCHES_APPLIED

    _seed_windows_platform_uname()

    if _IMPORT_PATCHES_APPLIED:
        try:
            from aiohttp import TCPConnector
        except ImportError:
            return
        if TCPConnector.__init__ is _AIOHTTP_PATCHED_INIT:
            return

    # CRITICAL: Apply DNS fix BEFORE importing litellm/aiohttp.
    # This fixes DNS hijacking by VPN/proxy/antivirus that returns wrong IPs.
    from ..dns_fix import apply_dns_fix

    apply_dns_fix()

    # CRITICAL: Apply finish_reason patch BEFORE importing litellm/openai.
    # LiteLLM caches OpenAI models on import, so patch must run first.
    from ..utils.litellm_patches import patch_litellm_finish_reason

    patch_litellm_finish_reason()

    # CRITICAL: Patch aiohttp.TCPConnector to use TLS 1.2 and disable SSL verification.
    # This fixes ConnectionResetError and SSLCertVerificationError with some servers.
    from ..ssl_patch import _patch_aiohttp_connector

    _patch_aiohttp_connector()
    try:
        from aiohttp import TCPConnector
    except ImportError:
        _AIOHTTP_PATCHED_INIT = None
    else:
        _AIOHTTP_PATCHED_INIT = TCPConnector.__init__

    _IMPORT_PATCHES_APPLIED = True


def configure_litellm_runtime() -> None:
    """Configure LiteLLM globals once per process."""
    global _LITELLM_RUNTIME_CONFIGURED

    if _LITELLM_RUNTIME_CONFIGURED:
        return

    import litellm  # type: ignore[import-untyped]

    from ..utils.litellm_patches import suppress_litellm_serialization_warnings

    os.environ["LITELLM_LOG"] = "ERROR"
    litellm.set_verbose = False  # type: ignore[attr-defined]
    litellm.drop_params = True

    # Suppress harmless Pydantic serialization warnings from litellm.
    # See: https://github.com/BerriAI/litellm/issues/11759
    suppress_litellm_serialization_warnings()

    _LITELLM_RUNTIME_CONFIGURED = True


def configure_client_logging(data_dir: str | Path, configure_logging: bool) -> None:
    """Configure rotator_library logging for a client instance."""
    from ..failure_logger import configure_failure_logger
    from ..utils.paths import get_logs_dir

    configure_failure_logger(get_logs_dir(Path(data_dir)))

    lib_logger = logging.getLogger("rotator_library")
    if configure_logging:
        # When True, this allows logs from this library to be handled
        # by the parent application's logging configuration.
        lib_logger.propagate = True
        # Remove any default handlers to prevent duplicate logging.
        if lib_logger.hasHandlers():
            lib_logger.handlers.clear()
            lib_logger.addHandler(logging.NullHandler())
    else:
        lib_logger.propagate = False
