"""CLI entrypoint for the IM gateway runtime."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from agent_core.bootstrap.agent_factory import create_agent, create_llm
from agent_core.bus.queue import MessageBus
from agent_core.channels.manager import ChannelManager
from agent_core.channels.telegram import TelegramChannel
from agent_core.channels.zhaohu import ZhaohuChannel
from agent_core.gateway.config import GatewaySettings
from agent_core.gateway.dispatcher import GatewayDispatcher
from agent_core.gateway.service import GatewayService
from agent_core.heartbeat.service import HeartbeatService
from agent_core.runtime_paths import load_runtime_env
from agent_core.runtime.manager import RuntimeManager

logger = logging.getLogger("agent_core.gateway.main")


def parse_args() -> argparse.Namespace:
    """Parse gateway command line arguments."""
    parser = argparse.ArgumentParser(description="BU Agent IM gateway")
    parser.add_argument("--root-dir", "-r", help="Workspace root directory", default=None)
    parser.add_argument("--model", "-m", help="Model override", default=None)
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=int,
        default=None,
        help="Override heartbeat interval in seconds",
    )
    parser.add_argument(
        "--disable-heartbeat",
        action="store_true",
        help="Disable heartbeat polling even if enabled by environment",
    )
    return parser.parse_args()


async def run_gateway(settings: GatewaySettings) -> None:
    """Build and run the in-process IM gateway."""
    bus = MessageBus()
    runtime_manager = RuntimeManager(
        runtime_factory=lambda: create_agent(model=settings.model, root_dir=settings.root_dir)
    )
    dispatcher = GatewayDispatcher(bus=bus, runtime_manager=runtime_manager)
    gateway_service = GatewayService(dispatcher=dispatcher)

    channel_manager = ChannelManager(bus)
    if settings.telegram_bot_token:
        telegram_config = argparse.Namespace(
            token=settings.telegram_bot_token,
            allow_from=settings.telegram_allow_from or [],
            proxy=settings.telegram_proxy,
        )
        channel_manager.register(TelegramChannel(config=telegram_config, bus=bus))
    else:
        logger.warning("TELEGRAM_BOT_TOKEN is empty; Telegram channel will not start")

    if settings.zhaohu_enabled:
        zhaohu_config = argparse.Namespace(
            host=settings.zhaohu_host,
            port=settings.zhaohu_port,
            webhook_path=settings.zhaohu_webhook_path,
        )
        channel_manager.register(ZhaohuChannel(config=zhaohu_config, bus=bus))
    else:
        logger.info("ZHAOHU_ENABLED is false; Zhaohu channel will not start")

    heartbeat = HeartbeatService(
        workspace=settings.root_dir,
        bus=bus,
        llm=create_llm(settings.model),
        get_delivery_target=lambda: dispatcher.last_active_private_chat,
        interval_seconds=settings.heartbeat_interval_seconds,
        enabled=settings.heartbeat_enabled,
    )

    await gateway_service.start()
    await channel_manager.start_all()
    await heartbeat.start()

    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    finally:
        await heartbeat.stop()
        await channel_manager.stop_all()
        await gateway_service.stop()


def load_settings_from_args(args: argparse.Namespace) -> GatewaySettings:
    """Load .env and resolve gateway settings from CLI arguments plus environment."""
    load_runtime_env()
    root_dir = Path(args.root_dir).resolve() if args.root_dir else Path.cwd().resolve()
    settings = GatewaySettings.from_env(root_dir=root_dir, model=args.model)
    if args.heartbeat_interval_seconds is not None:
        settings.heartbeat_interval_seconds = args.heartbeat_interval_seconds
    if args.disable_heartbeat:
        settings.heartbeat_enabled = False
    return settings


async def async_main() -> None:
    """Resolve CLI/environment config and run the gateway."""
    args = parse_args()
    settings = load_settings_from_args(args)
    await run_gateway(settings)


def cli_main() -> None:
    """Console script entrypoint."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(async_main())


if __name__ == "__main__":
    cli_main()
