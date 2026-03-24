from __future__ import annotations

import ast
import re
from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parent.parent


def _load_project_dependencies() -> dict[str, Requirement]:
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"(?ms)^dependencies\s*=\s*(\[[^\]]*\])", pyproject_text)
    if match is None:
        raise AssertionError("Could not locate [project].dependencies in pyproject.toml")

    dependency_items = ast.literal_eval(match.group(1))
    return {req.name: req for req in map(Requirement, dependency_items)}


def test_package_dependency_constraints_remain_compatible():
    dependencies = _load_project_dependencies()

    assert str(dependencies["httpx"].specifier) == "<0.28,>=0.27"
    assert str(dependencies["markdown-it-py"].specifier) == "<3,>=2.2"
    assert str(dependencies["rich"].specifier) == "<14,>=13"
