"""Load LSP configuration from the existing settings document."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_core.lsp.types import LSPConfig, LSPServerConfig
from agent_core.runtime_paths import load_user_settings

_DEFAULT_SERVERS: dict[str, dict[str, Any]] = {
    "python": {
        "command": "pyright-langserver",
        "args": ["--stdio"],
        "extensions": [".py"],
        "languageId": "python",
        "rootMarkers": ["pyproject.toml", "setup.py", "setup.cfg", ".git"],
    },
    "typescript": {
        "command": "typescript-language-server",
        "args": ["--stdio"],
        "extensions": [".ts", ".tsx", ".js", ".jsx"],
        "languageId": "typescript",
        "rootMarkers": ["tsconfig.json", "package.json", ".git"],
    },
    "java": {
        "command": "jdtls",
        "args": [],
        "extensions": [".java"],
        "languageId": "java",
        "rootMarkers": [
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
            "settings.gradle",
            "settings.gradle.kts",
            ".git",
        ],
    },
}

_DEFAULT_LSP: dict[str, Any] = {
    "enabled": True,
    "autoStart": True,
    "requestTimeoutSeconds": 10,
    "diagnosticsSettleMs": 300,
    "servers": _DEFAULT_SERVERS,
}


def load_lsp_config() -> LSPConfig:
    """Load LSP settings from ``~/.tg_agent/settings.json`` merged with defaults."""
    settings = load_user_settings()
    raw_lsp = settings.get("lsp")
    if raw_lsp is None:
        raw_lsp = {}
    if not isinstance(raw_lsp, Mapping):
        raise ValueError("settings.lsp must be a JSON object when present")
    return parse_lsp_config(raw_lsp)


def parse_lsp_config(raw_lsp: Mapping[str, Any] | None) -> LSPConfig:
    """Parse one LSP settings object merged with built-in defaults."""
    merged = _merge_lsp_document(raw_lsp or {})
    servers_raw = merged.get("servers")
    if not isinstance(servers_raw, Mapping):
        raise ValueError("settings.lsp.servers must be a JSON object")

    servers: dict[str, LSPServerConfig] = {}
    for name, raw_server in servers_raw.items():
        if not isinstance(name, str) or not isinstance(raw_server, Mapping):
            raise ValueError("Each settings.lsp.servers entry must be an object")
        server = _parse_server_config(name.strip(), raw_server)
        servers[server.name] = server

    return LSPConfig(
        enabled=_read_bool(merged.get("enabled"), default=True),
        auto_start=_read_bool(merged.get("autoStart", merged.get("auto_start")), default=True),
        request_timeout_seconds=_read_positive_float(
            merged.get("requestTimeoutSeconds", merged.get("request_timeout_seconds")),
            default=10.0,
        ),
        diagnostics_settle_ms=int(
            _read_positive_float(
                merged.get("diagnosticsSettleMs", merged.get("diagnostics_settle_ms")),
                default=300.0,
            )
        ),
        servers=servers,
    )


def _merge_lsp_document(raw_lsp: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        key: value for key, value in _DEFAULT_LSP.items() if key != "servers"
    }
    merged["servers"] = {
        name: dict(config)
        for name, config in _DEFAULT_SERVERS.items()
    }

    for key, value in raw_lsp.items():
        key = _normalize_lsp_key(str(key))
        if key != "servers":
            merged[key] = value
            continue
        if not isinstance(value, Mapping):
            raise ValueError("settings.lsp.servers must be a JSON object")
        servers = dict(merged["servers"])
        for server_name, server_config in value.items():
            if not isinstance(server_name, str) or not isinstance(server_config, Mapping):
                raise ValueError("Each settings.lsp.servers entry must be an object")
            base = dict(servers.get(server_name, {}))
            base.update(server_config)
            servers[server_name] = base
        merged["servers"] = servers
    return merged


def _normalize_lsp_key(key: str) -> str:
    aliases = {
        "auto_start": "autoStart",
        "request_timeout_seconds": "requestTimeoutSeconds",
        "diagnostics_settle_ms": "diagnosticsSettleMs",
    }
    return aliases.get(key, key)


def _parse_server_config(name: str, raw: Mapping[str, Any]) -> LSPServerConfig:
    if not name:
        raise ValueError("LSP server name must not be empty")
    command = _read_required_string(raw.get("command"), f"settings.lsp.servers.{name}.command")
    args = _read_string_list(raw.get("args", []), f"settings.lsp.servers.{name}.args")
    extensions = _normalize_extensions(
        _read_string_list(raw.get("extensions"), f"settings.lsp.servers.{name}.extensions")
    )
    if not extensions:
        raise ValueError(f"settings.lsp.servers.{name}.extensions must not be empty")
    language_id = _read_required_string(
        raw.get("languageId", raw.get("language_id", name)),
        f"settings.lsp.servers.{name}.languageId",
    )
    root_markers = _read_string_list(
        raw.get("rootMarkers", raw.get("root_markers", [])),
        f"settings.lsp.servers.{name}.rootMarkers",
    )
    return LSPServerConfig(
        name=name,
        command=command,
        args=args,
        extensions=extensions,
        language_id=language_id,
        root_markers=root_markers,
        disabled=_read_bool(raw.get("disabled"), default=False),
        env=_read_string_dict(raw.get("env"), f"settings.lsp.servers.{name}.env"),
        settings=_read_object(
            raw.get("settings"),
            f"settings.lsp.servers.{name}.settings",
        ),
        initialization_options=_read_object(
            raw.get("initializationOptions", raw.get("initialization_options")),
            f"settings.lsp.servers.{name}.initializationOptions",
        ),
    )


def _read_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError("LSP boolean settings must be JSON booleans")
    return value


def _read_positive_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("LSP numeric settings must be numbers")
    if value <= 0:
        raise ValueError("LSP numeric settings must be positive")
    return float(value)


def _read_required_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _read_string_list(value: Any, path: str) -> list[str]:
    if value is None:
        raise ValueError(f"{path} must be a list of strings")
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list of strings")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{path} must contain only non-empty strings")
        result.append(item.strip())
    return result


def _read_string_dict(value: Any, path: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{path} keys must be strings")
        if not isinstance(item, (str, int, float, bool)):
            raise ValueError(f"{path}.{key} must be a scalar value")
        result[key] = str(item)
    return result


def _read_object(value: Any, path: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object")
    return dict(value)


def _normalize_extensions(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        extension = value.lower()
        if not extension.startswith("."):
            extension = "." + extension
        if extension in seen:
            continue
        seen.add(extension)
        normalized.append(extension)
    return normalized
