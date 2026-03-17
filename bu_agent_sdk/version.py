from __future__ import annotations

import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path


_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')
_DIST_NAMES = ("tg-agent-cli",)


@lru_cache(maxsize=1)
def get_cli_version() -> str:
    """Return the current CLI version for plugin compatibility checks."""
    for dist_name in _DIST_NAMES:
        try:
            return distribution_version(dist_name)
        except PackageNotFoundError:
            continue

    repo_root = Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"
    if pyproject_path.exists():
        for line in pyproject_path.read_text(encoding="utf-8").splitlines():
            match = _VERSION_RE.match(line.strip())
            if match:
                return match.group(1)

    return "0.0.0"
