"""Load MCP server configuration from user settings."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent_core.mcp.types import MCPConfig, MCPServerConfig
from agent_core.runtime_paths import load_user_settings, save_user_settings, user_settings_path

_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PROJECT_MCP_CONFIG = ".tg_agent/mcp.json"


class MCPConfigEditError(RuntimeError):
    """Raised when an MCP config file cannot be edited safely."""


def load_mcp_config(workspace_root: Path | None = None) -> MCPConfig:
    settings = load_user_settings()
    user_config = parse_mcp_config(
        settings,
        source="user",
        user_config_path=str(user_settings_path()),
    )
    if workspace_root is None:
        return user_config

    project_path = workspace_root.resolve() / _PROJECT_MCP_CONFIG
    if not project_path.exists():
        return user_config
    project_config = parse_project_mcp_config(project_path)
    merged = dict(user_config.servers)
    merged.update(project_config.servers)
    return MCPConfig(
        servers=merged,
        project_config_path=str(project_path),
        user_config_path=str(user_settings_path()),
    )


def parse_project_mcp_config(path: Path) -> MCPConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read project MCP config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Project MCP config {path} must be a JSON object")
    return parse_mcp_config(raw, source="project", project_config_path=str(path))


def set_mcp_server_disabled(
    *,
    workspace_root: Path,
    server_name: str,
    source: str,
    disabled: bool,
) -> Path:
    """Persist the disabled flag for an existing MCP server in its source config."""
    normalized_source = source.strip().lower()
    if normalized_source == "project":
        path = workspace_root.resolve() / _PROJECT_MCP_CONFIG
        data = _read_json_object(path, label="Project MCP config")
        _set_disabled_in_config(data, server_name=server_name, disabled=disabled)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path
    if normalized_source == "user":
        data = load_user_settings()
        _set_disabled_in_config(data, server_name=server_name, disabled=disabled)
        return save_user_settings(data)
    raise MCPConfigEditError(f"Unsupported MCP config source: {source}")


def parse_mcp_config(
    settings: dict[str, Any],
    *,
    source: str = "user",
    project_config_path: str | None = None,
    user_config_path: str | None = None,
) -> MCPConfig:
    raw_servers = settings.get("mcpServers", {})
    if raw_servers is None:
        raw_servers = {}
    if not isinstance(raw_servers, dict):
        raise ValueError("settings.mcpServers must be an object")

    servers: dict[str, MCPServerConfig] = {}
    for name, raw in raw_servers.items():
        if not isinstance(name, str) or not _SERVER_NAME_RE.match(name):
            raise ValueError(f"Invalid MCP server name: {name!r}")
        if not isinstance(raw, dict):
            raise ValueError(f"settings.mcpServers.{name} must be an object")
        servers[name] = _parse_server(name, raw, source=source)
    return MCPConfig(
        servers=servers,
        project_config_path=project_config_path,
        user_config_path=user_config_path,
    )


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise MCPConfigEditError(f"{label} does not exist: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MCPConfigEditError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise MCPConfigEditError(f"{label} must be a JSON object: {path}")
    return data


def _set_disabled_in_config(
    data: dict[str, Any],
    *,
    server_name: str,
    disabled: bool,
) -> None:
    raw_servers = data.get("mcpServers")
    if not isinstance(raw_servers, dict):
        raise MCPConfigEditError("mcpServers must be an object")
    raw_server = raw_servers.get(server_name)
    if not isinstance(raw_server, dict):
        raise MCPConfigEditError(f"MCP server is not defined in this config: {server_name}")
    raw_server["disabled"] = disabled


def _parse_server(name: str, raw: dict[str, Any], *, source: str) -> MCPServerConfig:
    server_type = str(raw.get("type") or "stdio").strip()
    if server_type != "stdio":
        raise ValueError(
            f"settings.mcpServers.{name}.type must be 'stdio' in this version"
        )

    command = raw.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"settings.mcpServers.{name}.command must be a non-empty string")

    args = raw.get("args", [])
    if args is None:
        args = []
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ValueError(f"settings.mcpServers.{name}.args must be a list of strings")

    env = raw.get("env")
    normalized_env = None
    if env is not None:
        if not isinstance(env, dict):
            raise ValueError(f"settings.mcpServers.{name}.env must be an object")
        normalized_env = {str(key): str(value) for key, value in env.items()}

    disabled = bool(raw.get("disabled", False))
    return MCPServerConfig(
        name=name,
        type=server_type,
        command=command.strip(),
        args=list(args),
        env=normalized_env,
        disabled=disabled,
        source=source,
    )
