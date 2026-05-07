"""
Local Python server entrypoint for manual testing.

Run with:
    conda run -n 314 python test_server.py

Recommended env vars:
    $env:PORT = "8000"
    $env:HOST = "127.0.0.1"
    $env:LLM_PROVIDER = "gateway"   # optional, when you want local agent requests to use ChatGateway
    $env:LLM_MODEL = "coding-default"
    $env:LLM_GATEWAY_BASE_URL = "http://127.0.0.1:8000"
    $env:CRAB_GATEWAY_API_KEY = "test-token"
    $env:LLM_GATEWAY_ROUTES_FILE = "config/gateway_routes.server.json"

Useful routes after startup:
    GET  /docs
    POST /agent/query-stream
    POST /agent/query-stream-delta
    POST /llm/query-stream
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from agent_core import Agent
from agent_core.bootstrap.agent_factory import create_agent as create_runtime_agent
from agent_core.server import create_server


def _configure_logging() -> None:
    debug_flag = (os.getenv("BU_AGENT_SDK_LLM_DEBUG") or "").strip()
    level = logging.DEBUG if debug_flag else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_test_agent() -> Agent:
    """Create a full local runtime agent using the repository bootstrap flow."""
    model = (os.getenv("TEST_SERVER_MODEL") or os.getenv("LLM_MODEL") or "").strip() or None
    root_dir = Path(os.getenv("TEST_SERVER_ROOT_DIR", Path.cwd()))
    agent, _ = create_runtime_agent(model=model, root_dir=root_dir)
    return agent


def build_app():
    """Build the FastAPI app used for local gateway and agent testing."""
    return create_server(
        agent_factory=create_test_agent,
        session_timeout_minutes=int(os.getenv("TEST_SERVER_SESSION_TIMEOUT_MINUTES", "60")),
        max_sessions=int(os.getenv("TEST_SERVER_MAX_SESSIONS", "1000")),
        enable_cleanup_task=True,
    )


app = build_app()


if __name__ == "__main__":
    import uvicorn

    _configure_logging()

    host = (os.getenv("HOST") or "127.0.0.1").strip()
    port = int((os.getenv("PORT") or "8000").strip())

    print(f"Starting BU Agent test server on http://{host}:{port}")
    print(f"API docs: http://{host}:{port}/docs")
    print("Available test endpoints:")
    print("  POST /agent/query-stream")
    print("  POST /agent/query-stream-delta")
    print("  POST /llm/query-stream")
    print("Server-only gateway routes file: config/gateway_routes.server.json")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug" if logging.getLogger().level <= logging.DEBUG else "info",
    )
