from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from dotenv import dotenv_values

_TG_AGENT_HOME_DIRNAME = ".tg_agent"


def is_frozen_app() -> bool:
    """Return whether the current process runs from a frozen executable."""
    return bool(getattr(sys, "frozen", False))


def application_root() -> Path:
    """Return the root directory that contains bundled runtime assets."""
    if is_frozen_app():
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            return Path(bundle_root).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _user_home_dir() -> Path:
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if home:
        return Path(home).expanduser().resolve()
    return Path("~").expanduser().resolve()


def tg_agent_home() -> Path:
    """Return the fixed user-level state and configuration directory."""
    return _user_home_dir() / _TG_AGENT_HOME_DIRNAME


def packaged_env_path() -> Path:
    """Return the packaged .env path when present."""
    return application_root() / ".env"


def _read_env_values(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for key, value in dotenv_values(env_path).items():
        if key and value is not None:
            values[key] = value
    return values


def _load_model_preset_api_key_env_names() -> list[str]:
    try:
        from config.model_config import load_model_presets
    except Exception:
        return []

    env_names: list[str] = []
    seen: set[str] = set()
    for preset in load_model_presets().values():
        raw_name = preset.get("api_key_env", "OPENAI_API_KEY")
        if not isinstance(raw_name, str):
            continue
        env_name = raw_name.strip()
        if not env_name or env_name in seen:
            continue
        seen.add(env_name)
        env_names.append(env_name)
    return env_names


def default_runtime_env_values() -> dict[str, str]:
    """Return default CLI runtime env values merged with packaged overrides."""
    values = {}
    values.update(_read_env_values(packaged_env_path()))
    for env_name in _load_model_preset_api_key_env_names():
        values.setdefault(env_name, "")
    return values


def _merge_runtime_env_values(existing_values: dict[str, str] | None = None) -> dict[str, str]:
    merged_values = default_runtime_env_values()
    for key, value in (existing_values or {}).items():
        merged_values[key] = value
    return merged_values


def _render_runtime_env_content(values: dict[str, str]) -> str:
    lines = [
        "# Crab CLI runtime configuration",
        "# Refreshed automatically from packaged defaults on CLI launch.",
        "# Existing user-defined values are preserved when possible.",
        "# Values from the packaged .env are merged here when available.",
    ]
    for key, value in values.items():
        lines.append(f"{key}={value}")
    lines.append("")
    return "\n".join(lines)


def ensure_cli_runtime_state() -> Path:
    """Create default CLI runtime files under ~/.tg_agent without clobbering user values."""
    home_dir = tg_agent_home()
    home_dir.mkdir(parents=True, exist_ok=True)

    env_path = home_dir / ".env"
    merged_env_values = _merge_runtime_env_values(_read_env_values(env_path))
    env_path.write_text(_render_runtime_env_content(merged_env_values), encoding="utf-8")

    packaged_worker_config = application_root() / "tg_crab_worker.json"
    user_worker_config = home_dir / "tg_crab_worker.json"
    if packaged_worker_config.exists() and not user_worker_config.exists():
        shutil.copyfile(packaged_worker_config, user_worker_config)

    return home_dir


def runtime_env_files(cwd: Path | None = None) -> list[Path]:
    """Return env files ordered from lowest to highest precedence."""
    resolved_cwd = (cwd or Path.cwd()).resolve()
    candidate_paths = [
        packaged_env_path(),
        tg_agent_home() / ".env",
        resolved_cwd / ".env",
    ]
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in candidate_paths:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def load_runtime_env(cwd: Path | None = None) -> None:
    """Load runtime env files without overriding shell-provided variables."""
    merged_values: dict[str, str] = {}
    for env_path in runtime_env_files(cwd=cwd):
        for key, value in _read_env_values(env_path).items():
            merged_values[key] = value

    for key, value in merged_values.items():
        os.environ.setdefault(key, value)
