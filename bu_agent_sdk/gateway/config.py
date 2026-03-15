"""Configuration helpers for the gateway runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GatewaySettings:
    """Resolved gateway settings from CLI arguments and environment variables."""

    root_dir: Path
    model: str | None
    telegram_bot_token: str = ""
    telegram_allow_from: list[str] | None = None
    telegram_proxy: str | None = None
    zhaohu_enabled: bool = False
    zhaohu_host: str = "0.0.0.0"
    zhaohu_port: int = 18080
    zhaohu_webhook_path: str = "/webhook/zhaohu"
    heartbeat_enabled: bool = True
    heartbeat_interval_seconds: int = 30 * 60

    def __post_init__(self) -> None:
        if self.telegram_allow_from is None:
            self.telegram_allow_from = []
        if not self.zhaohu_webhook_path.startswith("/"):
            self.zhaohu_webhook_path = f"/{self.zhaohu_webhook_path}"

    @classmethod
    def from_env(cls, root_dir: Path, model: str | None) -> "GatewaySettings":
        """Build settings from CLI args plus environment variables."""
        raw_allow_from = os.getenv("TELEGRAM_ALLOW_FROM", "")
        allow_from = [item.strip() for item in raw_allow_from.split(",") if item.strip()]
        heartbeat_enabled = os.getenv("GATEWAY_HEARTBEAT_ENABLED", "true").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        heartbeat_interval_seconds = int(
            os.getenv("GATEWAY_HEARTBEAT_INTERVAL_SECONDS", str(30 * 60))
        )
        zhaohu_enabled = os.getenv("ZHAOHU_ENABLED", "false").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        return cls(
            root_dir=root_dir,
            model=model,
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_allow_from=allow_from,
            telegram_proxy=os.getenv("TELEGRAM_PROXY") or None,
            zhaohu_enabled=zhaohu_enabled,
            zhaohu_host=os.getenv("ZHAOHU_HOST", "0.0.0.0"),
            zhaohu_port=int(os.getenv("ZHAOHU_PORT", "18080")),
            zhaohu_webhook_path=os.getenv("ZHAOHU_WEBHOOK_PATH", "/webhook/zhaohu"),
            heartbeat_enabled=heartbeat_enabled,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
