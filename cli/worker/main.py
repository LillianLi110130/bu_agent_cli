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
authenticate_if_enabled: Any = None


def parse_args() -> argparse.Namespace:
    """Parse worker command line arguments."""
    parser = argparse.ArgumentParser(description="BU Agent worker")
    parser.add_argument("--session-key", required=True, help="Bound session key")
    parser.add_argument("--worker-id", required=True, help="Stable worker identifier")
    parser.add_argument("--gateway-base-url", required=True, help="Gateway base URL")
    parser.add_argument("--model", default=None, help="Optional model override")
    parser.add_argument("--root-dir", default=None, help="Optional workspace root directory")
    return parser.parse_args()


async def async_main() -> None:
    """Build the gateway client and runner, then start the worker loop."""
    args = parse_args()

    gateway_client_cls = WorkerGatewayClient
    runner_cls = WorkerRunner
    load_auth_config_fn = load_auth_config
    authenticate_if_enabled_fn = authenticate_if_enabled

    if gateway_client_cls is None:
        from cli.worker.gateway_client import WorkerGatewayClient as gateway_client_cls

    if runner_cls is None:
        from cli.worker.runner import WorkerRunner as runner_cls

    if load_auth_config_fn is None or authenticate_if_enabled_fn is None:
        from cli.worker.auth import authenticate_if_enabled as authenticate_if_enabled_fn
        from cli.worker.auth import load_auth_config as load_auth_config_fn

    base_dir = Path.cwd()
    auth_config = load_auth_config_fn(base_dir=base_dir)
    authorization: str | None = None
    if auth_config.enable_auth:
        authorization = await authenticate_if_enabled_fn(
            config=auth_config,
            base_dir=base_dir,
        )

    client_kwargs: dict[str, Any] = {"base_url": args.gateway_base_url}
    if authorization is not None:
        client_kwargs["authorization"] = authorization

    client = gateway_client_cls(**client_kwargs)
    runner = runner_cls(
        session_key=args.session_key,
        worker_id=args.worker_id,
        gateway_client=client,
        model=args.model,
        root_dir=args.root_dir,
    )
    await runner.run_forever()


def cli_main() -> None:
    """Console script entrypoint."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(async_main())


if __name__ == "__main__":
    cli_main()
