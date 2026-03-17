import asyncio
import json
import shutil
import uuid
from pathlib import Path

from rich.console import Console

from bu_agent_sdk.agent.registry import AgentRegistry
from bu_agent_sdk.plugin import PluginManager
from cli.at_commands import AtCommandRegistry
from cli.plugins_handler import PluginSlashHandler
from cli.slash_commands import SlashCommandRegistry


def make_workspace(root: Path) -> Path:
    workspace = root / f"case-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def _write_plugin(
    plugin_root: Path,
    *,
    name: str = "review-kit",
    command_body: str = "# Review\n\nFocus: {{args}}",
) -> Path:
    plugin_dir = plugin_root / name
    (plugin_dir / "skills" / "code-review").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "agents").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "commands").mkdir(parents=True, exist_ok=True)

    (plugin_dir / "plugin.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": name,
                "version": "0.1.0",
                "description": "Built-in review helpers",
                "capabilities": {
                    "skills": True,
                    "agents": True,
                    "commands": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (plugin_dir / "skills" / "code-review" / "SKILL.md").write_text(
        """---
name: code-review
description: Review code changes
category: Quality
---

# Code Review
""",
        encoding="utf-8",
    )
    (plugin_dir / "agents" / "reviewer.md").write_text(
        """---
description: Review code changes carefully
mode: subagent
model: reviewer-model
temperature: 0.2
tools:
  read: true
---

# Reviewer
""",
        encoding="utf-8",
    )
    (plugin_dir / "commands" / "review.md").write_text(
        f"""---
name: review
description: Review the current changes
usage: /{name}:review [focus]
category: Review
examples:
  - /{name}:review auth.py
---

{command_body}
""",
        encoding="utf-8",
    )
    return plugin_dir


def _create_manager(workspace: Path, plugin_root: Path) -> PluginManager:
    builtin_skills = workspace / "builtin_skills"
    builtin_agents = workspace / "builtin_agents"
    builtin_skills.mkdir(exist_ok=True)
    builtin_agents.mkdir(exist_ok=True)

    return PluginManager(
        plugin_dir=plugin_root,
        slash_registry=SlashCommandRegistry(),
        skill_registry=AtCommandRegistry(builtin_skills),
        agent_registry=AgentRegistry(builtin_agents),
    )


def test_plugin_manager_loads_plugin_resources():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        plugin_root = workspace / "plugins"
        plugin_root.mkdir()
        _write_plugin(plugin_root)

        manager = _create_manager(workspace, plugin_root)
        manager.load_all()

        plugin = manager.get_plugin("review-kit")
        assert plugin is not None
        assert plugin.status == "loaded"
        assert plugin.skills == ["review-kit:code-review"]
        assert plugin.agents == ["review-kit:reviewer"]
        assert plugin.commands == ["review-kit:review"]
        assert manager.get_command("review-kit:review") is not None

        command = manager.get_command("review-kit:review")
        assert command is not None
        assert "auth.py" in command.render_prompt(["auth.py"])
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_plugin_manager_isolates_failed_plugins():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        plugin_root = workspace / "plugins"
        plugin_root.mkdir()
        _write_plugin(plugin_root, name="good-plugin")
        bad_dir = plugin_root / "broken-plugin"
        bad_dir.mkdir()
        (bad_dir / "plugin.json").write_text("{not-json}", encoding="utf-8")

        manager = _create_manager(workspace, plugin_root)
        manager.load_all()

        good = manager.get_plugin("good-plugin")
        broken = manager.get_plugin("broken-plugin")

        assert good is not None
        assert good.status == "loaded"
        assert broken is not None
        assert broken.status == "failed"
        assert broken.error is not None
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_plugin_reload_updates_prompt_command():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        plugin_root = workspace / "plugins"
        plugin_root.mkdir()
        plugin_dir = _write_plugin(
            plugin_root,
            command_body="# Review\n\nVersion one: {{args}}",
        )

        manager = _create_manager(workspace, plugin_root)
        manager.load_all()

        command = manager.get_command("review-kit:review")
        assert command is not None
        assert "Version one" in command.render_prompt(["auth.py"])

        (plugin_dir / "commands" / "review.md").write_text(
            """---
name: review
description: Review the current changes
usage: /review-kit:review [focus]
category: Review
---

# Review

Version two: {{args}}
""",
            encoding="utf-8",
        )

        manager.reload_all()

        reloaded_command = manager.get_command("review-kit:review")
        assert reloaded_command is not None
        assert "Version two" in reloaded_command.render_prompt(["auth.py"])
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_plugin_slash_handler_supports_list_show_and_reload():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        plugin_root = workspace / "plugins"
        plugin_root.mkdir()
        _write_plugin(plugin_root)
        manager = _create_manager(workspace, plugin_root)
        manager.load_all()

        console = Console(record=True, width=120)
        handler = PluginSlashHandler(manager=manager, console=console)

        list_result = asyncio.run(handler.handle(["list"]))
        show_result = asyncio.run(handler.handle(["show", "review-kit"]))
        reload_result = asyncio.run(handler.handle(["reload"]))

        output = console.export_text()
        assert list_result.handled is True
        assert show_result.handled is True
        assert reload_result.reloaded is True
        assert "review-kit" in output
        assert "/review-kit:review" in output
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
