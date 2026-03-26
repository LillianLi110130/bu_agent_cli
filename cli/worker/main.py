"""CLI entrypoint for the Python worker prototype."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("cli.worker.main")

WorkerGatewayClient: Any = None
WorkerRunner: Any = None
load_auth_config: Any = None
load_persisted_auth_result: Any = None


def parse_args() -> argparse.Namespace:
    """Parse worker command line arguments."""
    parser = argparse.ArgumentParser(description="BU Agent worker")
    parser.add_argument("--worker-id", required=True, help="Stable worker identifier")
    parser.add_argument("--gateway-base-url", required=True, help="Gateway base URL")
    parser.add_argument("--model", default=None, help="Optional model override")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Optional startup config directory (default: current working directory)",
    )
    parser.add_argument("--root-dir", default=None, help="Optional workspace root directory")
    return parser.parse_args()


async def async_main() -> None:
    """Build the gateway client and runner, then start the worker loop."""
    args = parse_args()

    gateway_client_cls = WorkerGatewayClient
    runner_cls = WorkerRunner
    load_auth_config_fn = load_auth_config
    load_persisted_auth_result_fn = load_persisted_auth_result

    if gateway_client_cls is None:
        from cli.worker.gateway_client import WorkerGatewayClient as gateway_client_cls

    if runner_cls is None:
        from cli.worker.runner import WorkerRunner as runner_cls

    if load_auth_config_fn is None or load_persisted_auth_result_fn is None:
        from cli.worker.auth import load_auth_config as load_auth_config_fn
        from cli.worker.auth import load_persisted_auth_result as load_persisted_auth_result_fn

    base_dir = Path(args.config_dir or Path.cwd()).resolve()
    auth_config = load_auth_config_fn(base_dir=base_dir)
    authorization: str | None = None
    if auth_config.enable_auth:
        persisted_auth = load_persisted_auth_result_fn(base_dir=base_dir)
        if persisted_auth is None:
            raise RuntimeError("Worker auth is enabled but no persisted auth result was found")
        authorization = persisted_auth.authorization

    client_kwargs: dict[str, Any] = {"base_url": args.gateway_base_url}
    if authorization is not None:
        client_kwargs["authorization"] = authorization

    client = gateway_client_cls(**client_kwargs)
    runner = runner_cls(
        worker_id=args.worker_id,
        gateway_client=client,
        model=args.model,
        root_dir=args.root_dir,
    )
    await runner.run_forever()


def cli_main() -> None:
    """Console script entrypoint."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    asyncio.run(async_main())


if __name__ == "__main__":
    cli_main()
