"""Shared data types for MCP support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    type: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    disabled: bool = False
    source: str = "user"


@dataclass(frozen=True)
class MCPConfig:
    servers: dict[str, MCPServerConfig]
    project_config_path: str | None = None
    user_config_path: str | None = None


@dataclass(frozen=True)
class MCPTool:
    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class MCPServerStatus:
    name: str
    state: str
    type: str
    command: str
    args: list[str]
    tool_count: int = 0
    error: str | None = None
