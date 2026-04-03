from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import dotenv_values


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


def tg_agent_home() -> Path:
    """Return the user-level state and configuration directory."""
    configured = os.getenv("TG_AGENT_HOME")
    if configured:
        return Path(configured).expanduser().resolve()

    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if home:
        return (Path(home).expanduser() / ".tg_agent").resolve()
    return Path("~/.tg_agent").expanduser().resolve()


def runtime_env_files(cwd: Path | None = None) -> list[Path]:
    """Return env files ordered from lowest to highest precedence."""
    candidate_paths = [
        tg_agent_home() / ".env",
        (cwd or Path.cwd()).resolve() / ".env",
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
        values = dotenv_values(env_path)
        for key, value in values.items():
            if key and value is not None:
                merged_values[key] = value

    for key, value in merged_values.items():
        os.environ.setdefault(key, value)
