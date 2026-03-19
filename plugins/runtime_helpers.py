from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )


def load_payload() -> dict:
    return json.load(sys.stdin)


def ensure_devagent_layout(working_dir: Path) -> Path:
    agents_dir = working_dir / ".devagent" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    return agents_dir


def sync_plugin_subagents(plugin_root: Path, working_dir: Path) -> list[Path]:
    source_dir = plugin_root / "dev_subagents"
    target_dir = ensure_devagent_layout(working_dir)
    copied: list[Path] = []

    if not source_dir.exists():
        return copied

    for agent_file in sorted(source_dir.glob("*.md")):
        target_file = target_dir / agent_file.name
        shutil.copy2(agent_file, target_file)
        copied.append(target_file)

    return copied


def require_devagent() -> bool:
    return shutil.which("devagent") is not None


def read_prompt(plugin_root: Path, relative_path: str) -> str:
    return (plugin_root / relative_path).read_text(encoding="utf-8")


def run_devagent(working_dir: Path, prompt: str) -> int:
    process = subprocess.Popen(
        "devagent --yolo",
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=str(working_dir),
    )
    stdout, stderr = process.communicate(input=prompt)

    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")

    return process.returncode


def render_prompt(template: str, **replacements: str) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def spec_dir(working_dir: Path, spec_name: str) -> Path:
    return working_dir / "docs" / "spec" / spec_name


def require_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]
