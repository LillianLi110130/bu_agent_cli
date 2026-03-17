from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from packaging.version import InvalidVersion, Version

from bu_agent_sdk.version import get_cli_version

from .types import PluginCapabilities, PluginCommand, PluginManifest


_PLUGIN_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_RESOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class PluginLoader:
    """Load plugin metadata and resource definitions from disk."""

    def __init__(self, plugin_dir: Path, *, current_cli_version: str | None = None):
        self.plugin_dir = plugin_dir
        self.current_cli_version = current_cli_version or get_cli_version()

    def discover(self) -> list[Path]:
        if not self.plugin_dir.exists():
            return []
        return sorted(path for path in self.plugin_dir.iterdir() if path.is_dir())

    def load_manifest(self, plugin_path: Path) -> PluginManifest:
        manifest_path = plugin_path / "plugin.json"
        if not manifest_path.exists():
            raise ValueError(f"Missing manifest: {manifest_path}")

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid manifest JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError("Manifest must be a JSON object")

        manifest = PluginManifest(
            schema_version=int(payload.get("schema_version", 1)),
            name=str(payload.get("name", "")).strip(),
            version=str(payload.get("version", "")).strip(),
            description=str(payload.get("description", "")).strip(),
            min_cli_version=_optional_str(payload.get("min_cli_version")),
            capabilities=PluginCapabilities.from_dict(payload.get("capabilities")),
        )
        self.validate_manifest(manifest)
        return manifest

    def validate_manifest(self, manifest: PluginManifest) -> None:
        if manifest.schema_version != 1:
            raise ValueError(f"Unsupported schema_version: {manifest.schema_version}")
        if not manifest.name or not _PLUGIN_NAME_RE.fullmatch(manifest.name):
            raise ValueError("Plugin name must be kebab-case")
        if not manifest.version:
            raise ValueError("Plugin version is required")
        if not manifest.description:
            raise ValueError("Plugin description is required")
        if manifest.min_cli_version is not None:
            try:
                current_version = Version(self.current_cli_version)
                minimum_version = Version(manifest.min_cli_version)
            except InvalidVersion as exc:
                raise ValueError(f"Invalid min_cli_version: {manifest.min_cli_version}") from exc
            if current_version < minimum_version:
                raise ValueError(
                    "Plugin requires CLI version "
                    f">={manifest.min_cli_version}; current version is {self.current_cli_version}"
                )

    def load_command(
        self,
        plugin_root: Path,
        plugin_name: str,
        command_path: Path,
    ) -> PluginCommand:
        metadata, body = self._read_markdown_resource(command_path)
        name = str(metadata.get("name", command_path.stem)).strip()
        if not _RESOURCE_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid command name: {name}")

        description = str(metadata.get("description", "")).strip()
        usage = str(metadata.get("usage", f"/{plugin_name}:{name}")).strip()
        category = str(metadata.get("category", "Plugins")).strip() or "Plugins"
        examples = metadata.get("examples") or []
        if not isinstance(examples, list):
            raise ValueError("Command examples must be a list")

        mode = str(metadata.get("mode", "prompt")).strip().lower() or "prompt"
        if mode == "prompt":
            return PluginCommand(
                plugin_name=plugin_name,
                name=name,
                description=description,
                path=command_path,
                mode="prompt",
                content=body,
                usage=usage,
                category=category,
                examples=[str(item) for item in examples],
            )

        if mode == "python":
            script_value = metadata.get("script")
            if not isinstance(script_value, str) or not script_value.strip():
                raise ValueError("Python command requires a non-empty 'script' path")
            script_path = self._resolve_plugin_path(plugin_root, script_value)
            if not script_path.is_file():
                raise ValueError(f"Plugin command script not found: {script_value}")
            return PluginCommand(
                plugin_name=plugin_name,
                name=name,
                description=description,
                path=command_path,
                mode="python",
                content=body,
                script=script_path,
                usage=usage,
                category=category,
                examples=[str(item) for item in examples],
            )

        raise ValueError(f"Unsupported command mode: {mode}")

    def should_load(self, manifest: PluginManifest, kind: str, path: Path) -> bool:
        return manifest.capabilities.allows(kind, path.exists())

    def _read_markdown_resource(self, path: Path) -> tuple[dict, str]:
        raw = path.read_text(encoding="utf-8")
        match = re.match(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)(.*)\Z", raw, re.DOTALL)
        if not match:
            return {}, raw

        metadata = yaml.safe_load(match.group(1)) or {}
        if not isinstance(metadata, dict):
            raise ValueError("Frontmatter must be a mapping")
        return metadata, match.group(2).strip()

    def _resolve_plugin_path(self, plugin_root: Path, path_text: str) -> Path:
        candidate = Path(path_text)
        if candidate.is_absolute():
            raise ValueError("Plugin script path must be relative to the plugin root")

        resolved_root = plugin_root.resolve()
        resolved_path = (plugin_root / candidate).resolve()
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("Plugin script path escapes the plugin root") from exc
        return resolved_path


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
