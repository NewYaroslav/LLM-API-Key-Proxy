# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncIterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _seed_windows_platform_uname() -> None:
    """Avoid Python 3.14 platform.system() WMI hangs during aiohttp import."""
    if sys.platform != "win32":
        return

    import os
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


_seed_windows_platform_uname()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from proxy_app.bootstrap import (
    bootstrap,
    create_arg_parser,
    default_args,
)
from proxy_app.config import (
    DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT,
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_UVICORN_BACKLOG,
    env_int,
)
from proxy_app.middleware import SecurityHeadersMiddleware, _NoGzipForSSE
from proxy_app.onboarding import run_onboarding_if_needed


def _register_routes(app: FastAPI) -> None:
    from proxy_app.routes import all_routers

    for router in all_routers:
        app.include_router(router)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from proxy_app.bootstrap import create_lifespan_from_environment

    if not getattr(app.state, "routes_registered", False):
        _register_routes(app)
        app.state.routes_registered = True
    runtime_lifespan = create_lifespan_from_environment(
        getattr(app.state, "cli_args", default_args())
    )
    async with runtime_lifespan(app):
        yield


# --- FastAPI App Setup ---

app = FastAPI(lifespan=lifespan)
app.state.cli_args = default_args()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Accel-Buffering",
        "X-Request-Id",
        "X-Provider",
        "Retry-After",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
    ],
)
app.add_middleware(_NoGzipForSSE)
app.add_middleware(SecurityHeadersMiddleware)


if __name__ == "__main__":
    _register_routes(app)
    app.state.routes_registered = True


@app.api_route("/", methods=["GET", "HEAD"])
async def read_root():
    return {"Status": "API Key Proxy is running"}


def parse_startup_args(args: list[str]) -> argparse.Namespace:
    parser = create_arg_parser()
    return parser.parse_args(args)


def setup_windows_signals() -> None:
    if sys.platform == "win32":
        import signal as _signal

        if hasattr(_signal, "SIGBREAK"):
            _signal.signal(
                _signal.SIGBREAK, lambda *_: _signal.raise_signal(_signal.SIGINT)
            )


def start_uvicorn(parsed) -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=parsed.host,
        port=parsed.port,
        limit_concurrency=env_int(
            "MAX_CONCURRENT_REQUESTS", DEFAULT_MAX_CONCURRENT_REQUESTS
        ),
        backlog=env_int("UVICORN_BACKLOG", DEFAULT_UVICORN_BACKLOG),
        timeout_graceful_shutdown=env_int(
            "TIMEOUT_GRACEFUL_SHUTDOWN", DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT
        ),
    )


def run_server() -> None:
    if len(sys.argv) == 1:
        args = default_args()
        args.startup_state = bootstrap(args)
        from proxy_app.launcher_tui import run_launcher_tui

        run_launcher_tui()
        parsed = parse_startup_args(sys.argv[1:])
        parsed.startup_state = args.startup_state
    else:
        parsed = parse_startup_args(sys.argv[1:])

    run_onboarding_if_needed(app, parsed)
    setup_windows_signals()
    start_uvicorn(parsed)


if __name__ == "__main__":
    run_server()
