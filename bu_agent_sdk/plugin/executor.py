from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from .types import PluginCommand


class PluginExecutionError(RuntimeError):
    """Raised when a plugin command cannot be executed successfully."""


class PluginCommandExecutor:
    """Execute prompt and python plugin commands."""

    async def execute(
        self,
        command: PluginCommand,
        *,
        args: list[str],
        args_text: str,
        working_dir: Path,
    ) -> str:
        if command.mode == "prompt":
            return command.render_prompt(args_text)
        if command.mode == "python":
            return await self.execute_python(
                command,
                args=args,
                args_text=args_text,
                plugin_root=command.plugin_root,
                working_dir=working_dir,
            )
        raise PluginExecutionError(f"Unsupported plugin command mode: {command.mode}")

    async def execute_python(
        self,
        command: PluginCommand,
        *,
        args: list[str],
        args_text: str,
        plugin_root: Path,
        working_dir: Path,
    ) -> str:
        if command.mode != "python" or command.script is None:
            raise PluginExecutionError(f"Command '{command.full_name}' is not a python command")

        payload = {
            "command": command.full_name,
            "plugin": command.plugin_name,
            "name": command.name,
            "args": args,
            "args_text": args_text,
            "plugin_root": str(plugin_root),
            "working_dir": str(working_dir),
        }

        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, str(command.script)],
                cwd=str(working_dir),
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            raise PluginExecutionError(
                f"Failed to start plugin command '{command.full_name}': {exc}"
            ) from exc

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()

        if completed.returncode != 0:
            message = stderr or (
                f"Plugin command '{command.full_name}' failed with exit code {completed.returncode}"
            )
            raise PluginExecutionError(message)

        return stdout
