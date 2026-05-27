"""Send a scheduled task completion notification to the worker gateway."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import httpx


logger = logging.getLogger(__name__)
GATEWAY_URL = "http://127.0.0.1:8888"


@dataclass(frozen=True)
class GatewayConfig:
    """Runtime configuration for gateway completion notifications."""

    gateway_url: str
    worker_id: str
    source: str
    timeout_seconds: float


def load_authorization(*, base_dir: Path | None = None) -> str:
    """Load gateway authorization from ~/.tg_agent/token.json."""
    token_base_dir = base_dir or Path.home()
    token_path = token_base_dir / ".tg_agent" / "token.json"
    payload = json.loads(token_path.read_text(encoding="utf-8"))
    authorization = payload.get("authorization")
    if not isinstance(authorization, str) or not authorization.strip():
        raise ValueError(f"Missing authorization in token file: {token_path}")
    return authorization.strip()


def load_gateway_config(args: argparse.Namespace) -> GatewayConfig:
    """Load gateway config from CLI arguments and the script gateway URL."""
    return GatewayConfig(
        gateway_url=GATEWAY_URL.rstrip("/"),
        worker_id=args.worker_id,
        source=args.source,
        timeout_seconds=float(args.timeout_seconds),
    )


def build_complete_payload(
    *,
    config: GatewayConfig,
    final_content: str,
    final_status: str,
    error_code: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Build the payload expected by /api/worker/complete."""
    return {
        "worker_id": config.worker_id,
        "final_content": final_content,
        "source": config.source,
        "final_status": final_status,
        "error_code": error_code,
        "error_message": error_message,
    }


def send_complete_notification(
    *,
    config: GatewayConfig,
    authorization: str,
    final_content: str,
    final_status: str = "completed",
    error_code: str | None = None,
    error_message: str | None = None,
    dry_run: bool = False,
) -> None:
    """Send the completion notification to the gateway."""
    payload = build_complete_payload(
        config=config,
        final_content=final_content,
        final_status=final_status,
        error_code=error_code,
        error_message=error_message,
    )
    if dry_run:
        logger.info("Dry run gateway completion payload: %s", payload)
        return

    url = f"{config.gateway_url}/api/worker/complete"
    headers = {"Authorization": authorization}
    with httpx.Client(timeout=config.timeout_seconds) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
    logger.info(
        "Sent gateway completion notification worker_id=%s status=%s",
        config.worker_id,
        final_status,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Send a scheduled task completion notification to the worker gateway."
    )
    parser.add_argument("--worker-id", required=True, help="Gateway worker identity.")
    parser.add_argument("--source", default="scheduler", help="Final message source.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="HTTP timeout seconds.",
    )
    parser.add_argument("--content", required=True, help="Final content sent to the gateway.")
    parser.add_argument("--status", default="completed", help="Final status value.")
    parser.add_argument("--error-code", help="Optional stable error code.")
    parser.add_argument("--error-message", help="Optional concise error message.")
    parser.add_argument("--dry-run", action="store_true", help="Log payload without sending it.")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    authorization = load_authorization()
    config = load_gateway_config(args)
    try:
        send_complete_notification(
            config=config,
            authorization=authorization,
            final_content=args.content,
            final_status=args.status,
            error_code=args.error_code,
            error_message=args.error_message,
            dry_run=args.dry_run,
        )
    except Exception:
        logger.exception("Failed to send gateway completion notification")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
