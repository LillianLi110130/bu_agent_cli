from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_ROUTES_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "gateway_routes.server.json"
)


@dataclass(frozen=True)
class GatewayRoute:
    alias: str
    provider: str
    upstream_model: str
    base_url: str | None
    api_key_env: str


def _read_non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _resolve_routes_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    configured = os.getenv("LLM_GATEWAY_ROUTES_FILE")
    if configured:
        return Path(configured)
    return _DEFAULT_ROUTES_PATH


def load_gateway_routes(path: str | Path | None = None) -> dict[str, GatewayRoute]:
    """Load gateway alias routes from a server-only JSON file."""
    routes_path = _resolve_routes_path(path)
    if not routes_path.exists():
        return {}

    try:
        raw = routes_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    route_data = data.get("routes")
    if not isinstance(route_data, dict):
        return {}

    routes: dict[str, GatewayRoute] = {}
    for alias, config in route_data.items():
        alias_name = _read_non_empty_string(alias)
        if alias_name is None or not isinstance(config, dict):
            continue

        upstream_model = _read_non_empty_string(config.get("upstream_model"))
        if upstream_model is None:
            continue

        provider = _read_non_empty_string(config.get("provider")) or "openai"
        base_url = _read_non_empty_string(config.get("base_url"))
        api_key_env = _read_non_empty_string(config.get("api_key_env")) or "OPENAI_API_KEY"

        routes[alias_name] = GatewayRoute(
            alias=alias_name,
            provider=provider.lower(),
            upstream_model=upstream_model,
            base_url=base_url,
            api_key_env=api_key_env,
        )

    return routes
