"""Shared data types for LSP support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LSPServerConfig:
    name: str
    command: str
    args: list[str]
    extensions: list[str]
    language_id: str
    root_markers: list[str]
    disabled: bool = False
    env: dict[str, str] | None = None
    settings: dict[str, Any] | None = None
    initialization_options: dict[str, Any] | None = None


@dataclass(frozen=True)
class LSPConfig:
    enabled: bool
    auto_start: bool
    request_timeout_seconds: float
    diagnostics_settle_ms: int
    servers: dict[str, LSPServerConfig]


@dataclass(frozen=True)
class LSPDiagnostic:
    uri: str
    range: dict[str, Any]
    severity: int | None
    code: str | int | None
    source: str | None
    message: str


@dataclass
class SyncedDocument:
    uri: str
    language_id: str
    version: int
    content_hash: str
    path: Path
