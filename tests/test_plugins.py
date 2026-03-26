import asyncio
import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from agent_core.agent.registry import AgentRegistry
from agent_core.plugin import (
    PluginCommandExecutor,
    PluginExecutionError,
    PluginManager,
)
from agent_core.version import get_cli_version
from cli.app import ClaudeCodeCLI
from cli.at_commands import AtCommandRegistry
from cli.plugins_handler import PluginSlashHandler
from cli.slash_commands import SlashCommandRegistry


def make_workspace(root: Path) -> Path:
    workspace = root / f"case-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def _write_manifest(
    plugin_dir: Path,
    *,
    manifest_name: str,
    min_cli_version: str | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "name": manifest_name,
        "version": "0.1.0",
        "description": "Built-in review helpers",
        "capabilities": {
            "skills": True,
            "agents": True,
            "commands": True,
        },
    }
    if min_cli_version is not None:
        payload["min_cli_version"] = min_cli_version
    (plugin_dir / "plugin.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_shared_resources(plugin_dir: Path) -> None:
    (plugin_dir / "skills" / "code-review").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "agents").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "commands").mkdir(parents=True, exist_ok=True)

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


def _write_prompt_command(
    plugin_dir: Path,
    *,
    plugin_name: str,
    command_body: str = "# Review\n\nFocus: {{args}}",
) -> None:
    (plugin_dir / "commands" / "review.md").write_text(
        f"""---
name: review
description: Review the current changes
usage: /{plugin_name}:review [focus]
category: Review
examples:
  - /{plugin_name}:review auth.py
---

{command_body}
""",
        encoding="utf-8",
    )


def _write_python_command(
    plugin_dir: Path,
    *,
    plugin_name: str,
    command_name: str = "summarize",
    script_rel: str = "scripts/summarize.py",
    command_body: str = "Generate a workspace summary.",
    script_body: str = (
        "import json\n"
        "import sys\n"
        "payload = json.load(sys.stdin)\n"
        "print(f\"Summary for {payload['args_text']} in {payload['working_dir']}\")\n"
    ),
    create_script: bool = True,
) -> None:
    script_path = plugin_dir / Path(script_rel)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "commands" / f"{command_name}.md").write_text(
        f"""---
name: {command_name}
description: Summarize the current workspace
usage: /{plugin_name}:{command_name} [target]
category: Review
examples:
  - /{plugin_name}:{command_name}
  - /{plugin_name}:{command_name} src/auth.py
mode: python
script: {script_rel}
---

{command_body}
""",
        encoding="utf-8",
    )
    if create_script:
        script_path.write_text(script_body, encoding="utf-8")


def _write_plugin(
    plugin_root: Path,
    *,
    dir_name: str = "review-kit",
    manifest_name: str | None = None,
    prompt_command_body: str = "# Review\n\nFocus: {{args}}",
    include_python_command: bool = False,
    python_script_rel: str = "scripts/summarize.py",
    python_script_body: str | None = None,
    create_python_script: bool = True,
    min_cli_version: str | None = None,
) -> Path:
    plugin_dir = plugin_root / dir_name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    resolved_name = manifest_name or dir_name

    _write_shared_resources(plugin_dir)
    _write_manifest(
        plugin_dir,
        manifest_name=resolved_name,
        min_cli_version=min_cli_version,
    )
    _write_prompt_command(
        plugin_dir,
        plugin_name=resolved_name,
        command_body=prompt_command_body,
    )
    if include_python_command:
        _write_python_command(
            plugin_dir,
            plugin_name=resolved_name,
            script_rel=python_script_rel,
            script_body=python_script_body
            or (
                "import json\n"
                "import sys\n"
                "payload = json.load(sys.stdin)\n"
                "print(f\"Summary for {payload['args_text']} in {payload['working_dir']}\")\n"
            ),
            create_script=create_python_script,
        )
    return plugin_dir


def _create_manager(
    workspace: Path,
    plugin_root: Path,
    *,
    workspace_plugin_root: Path | None = None,
    current_cli_version: str | None = None,
) -> PluginManager:
    builtin_skills = workspace / "builtin_skills"
    builtin_agents = workspace / "builtin_agents"
    builtin_skills.mkdir(exist_ok=True)
    builtin_agents.mkdir(exist_ok=True)

    plugin_dirs = [("builtin", plugin_root)]
    if workspace_plugin_root is not None:
        plugin_dirs.append(("workspace", workspace_plugin_root))

    return PluginManager(
        plugin_dir=None if workspace_plugin_root is not None else plugin_root,
        plugin_dirs=plugin_dirs if workspace_plugin_root is not None else None,
        slash_registry=SlashCommandRegistry(),
        skill_registry=AtCommandRegistry(builtin_skills),
        agent_registry=AgentRegistry(builtin_agents),
        current_cli_version=current_cli_version or get_cli_version(),
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

        command = manager.get_command("review-kit:review")
        assert command is not None
        assert command.mode == "prompt"
        assert "auth flow" in command.render_prompt("auth flow")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_plugin_manager_loads_python_command_and_executor_receives_payload():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        plugin_root = workspace / "plugins"
        plugin_root.mkdir()
        _write_plugin(
            plugin_root,
            include_python_command=True,
            python_script_body=(
                "import json\n"
                "import os\n"
                "import sys\n"
                "from pathlib import Path\n"
                "payload = json.load(sys.stdin)\n"
                "print(json.dumps({"
                "\"args_text\": payload['args_text'], "
                "\"plugin_root_name\": Path(payload['plugin_root']).name, "
                '"cwd": os.getcwd()'
                "}))\n"
            ),
        )

        manager = _create_manager(workspace, plugin_root)
        manager.load_all()

        command = manager.get_command("review-kit:summarize")
        assert command is not None
        assert command.mode == "python"
        assert command.script is not None

        executor = PluginCommandExecutor()
        output = asyncio.run(
            executor.execute(
                command,
                args=["src/auth.py"],
                args_text="src/auth.py",
                working_dir=workspace,
            )
        )
        payload = json.loads(output)
        assert payload["args_text"] == "src/auth.py"
        assert payload["plugin_root_name"] == "review-kit"
        assert Path(payload["cwd"]) == workspace
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
        _write_plugin(plugin_root, dir_name="good-plugin")
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
            prompt_command_body="# Review\n\nVersion one: {{args}}",
        )

        manager = _create_manager(workspace, plugin_root)
        manager.load_all()

        command = manager.get_command("review-kit:review")
        assert command is not None
        assert "Version one" in command.render_prompt("auth.py")

        _write_prompt_command(
            plugin_dir,
            plugin_name="review-kit",
            command_body="# Review\n\nVersion two: {{args}}",
        )

        manager.reload_all()

        reloaded_command = manager.get_command("review-kit:review")
        assert reloaded_command is not None
        assert "Version two" in reloaded_command.render_prompt("auth.py")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_plugin_manager_rejects_missing_python_script():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        plugin_root = workspace / "plugins"
        plugin_root.mkdir()
        _write_plugin(
            plugin_root,
            include_python_command=True,
            create_python_script=False,
        )

        manager = _create_manager(workspace, plugin_root)
        manager.load_all()

        plugin = manager.get_plugin("review-kit")
        assert plugin is not None
        assert plugin.status == "failed"
        assert plugin.error is not None
        assert "script not found" in plugin.error.lower()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_plugin_manager_rejects_python_escape_path():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        plugin_root = workspace / "plugins"
        plugin_root.mkdir()
        (plugin_root / "outside.py").write_text("print('outside')\n", encoding="utf-8")
        _write_plugin(
            plugin_root,
            include_python_command=True,
            python_script_rel="../outside.py",
            create_python_script=False,
        )

        manager = _create_manager(workspace, plugin_root)
        manager.load_all()

        plugin = manager.get_plugin("review-kit")
        assert plugin is not None
        assert plugin.status == "failed"
        assert plugin.error is not None
        assert "escapes the plugin root" in plugin.error.lower()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_plugin_manager_enforces_min_cli_version():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        plugin_root = workspace / "plugins"
        plugin_root.mkdir()
        _write_plugin(
            plugin_root,
            min_cli_version="999.0.0",
        )

        manager = _create_manager(workspace, plugin_root, current_cli_version="0.1.0")
        manager.load_all()

        plugin = manager.get_plugin("review-kit")
        assert plugin is not None
        assert plugin.status == "failed"
        assert plugin.error is not None
        assert "requires cli version" in plugin.error.lower()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_plugin_manager_rejects_duplicate_manifest_name():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        plugin_root = workspace / "plugins"
        plugin_root.mkdir()
        _write_plugin(plugin_root, dir_name="dup-one", manifest_name="dup")
        _write_plugin(plugin_root, dir_name="dup-two", manifest_name="dup")

        manager = _create_manager(workspace, plugin_root)
        manager.load_all()

        good = manager.get_plugin("dup")
        duplicate = manager.get_plugin("dup-two")
        assert good is not None
        assert good.status == "loaded"
        assert duplicate is not None
        assert duplicate.status == "failed"
        assert duplicate.error is not None
        assert "duplicate plugin name" in duplicate.error.lower()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_plugin_executor_reports_python_failure():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        plugin_root = workspace / "plugins"
        plugin_root.mkdir()
        _write_plugin(
            plugin_root,
            include_python_command=True,
            python_script_body=(
                "import sys\n" "sys.stderr.write('boom')\n" "raise SystemExit(2)\n"
            ),
        )

        manager = _create_manager(workspace, plugin_root)
        manager.load_all()

        command = manager.get_command("review-kit:summarize")
        assert command is not None

        executor = PluginCommandExecutor()
        with pytest.raises(PluginExecutionError, match="boom"):
            asyncio.run(
                executor.execute(
                    command,
                    args=[],
                    args_text="",
                    working_dir=workspace,
                )
            )
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
        _write_plugin(plugin_root, include_python_command=True)
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
        assert "/review-kit:review (prompt)" in output
        assert "/review-kit:summarize (python)" in output
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_workspace_plugin_overrides_builtin_plugin():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        builtin_root = workspace / "builtin_plugins"
        workspace_plugin_root = workspace / ".tg_agent" / "plugins"
        builtin_root.mkdir()
        workspace_plugin_root.mkdir(parents=True)

        _write_plugin(
            builtin_root,
            prompt_command_body="# Review\n\nBuiltin version: {{args}}",
        )
        _write_plugin(
            workspace_plugin_root,
            prompt_command_body="# Review\n\nWorkspace version: {{args}}",
        )

        manager = _create_manager(
            workspace,
            builtin_root,
            workspace_plugin_root=workspace_plugin_root,
        )
        manager.load_all()

        plugin = manager.get_plugin("review-kit")
        assert plugin is not None
        assert plugin.status == "loaded"
        assert plugin.source == "workspace"
        assert plugin.path == workspace_plugin_root / "review-kit"

        builtin_plugin = manager.get_plugin_from_source("review-kit", "builtin")
        assert builtin_plugin is not None
        assert builtin_plugin.source == "builtin"

        command = manager.get_command("review-kit:review")
        assert command is not None
        assert "Workspace version" in command.render_prompt("auth.py")
        assert "Builtin version" not in command.render_prompt("auth.py")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_workspace_broken_plugin_overrides_builtin_without_fallback():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        builtin_root = workspace / "builtin_plugins"
        workspace_plugin_root = workspace / ".tg_agent" / "plugins"
        builtin_root.mkdir()
        workspace_plugin_root.mkdir(parents=True)

        _write_plugin(builtin_root)
        broken_workspace_plugin = workspace_plugin_root / "review-kit"
        broken_workspace_plugin.mkdir()
        (broken_workspace_plugin / "plugin.json").write_text("{not-json}", encoding="utf-8")

        manager = _create_manager(
            workspace,
            builtin_root,
            workspace_plugin_root=workspace_plugin_root,
        )
        manager.load_all()

        plugin = manager.get_plugin("review-kit")
        assert plugin is not None
        assert plugin.status == "failed"
        assert plugin.source == "workspace"
        assert plugin.error is not None
        assert "invalid manifest json" in plugin.error.lower()
        assert manager.get_command("review-kit:review") is None
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_plugin_slash_handler_can_copy_builtin_plugin_into_workspace():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        builtin_root = workspace / "builtin_plugins"
        workspace_plugin_root = workspace / ".tg_agent" / "plugins"
        builtin_root.mkdir()
        _write_plugin(builtin_root)

        manager = _create_manager(
            workspace,
            builtin_root,
            workspace_plugin_root=workspace_plugin_root,
        )
        manager.load_all()

        console = Console(record=True, width=120)
        handler = PluginSlashHandler(manager=manager, console=console)

        copy_result = asyncio.run(handler.handle(["copy", "review-kit"]))
        reload_result = asyncio.run(handler.handle(["reload"]))

        copied_plugin = workspace_plugin_root / "review-kit"
        output = console.export_text()
        assert copy_result.handled is True
        assert reload_result.reloaded is True
        assert copied_plugin.exists()
        assert "已将插件复制到工作区" in output
        assert "编辑完成后请运行 /plugins reload。" in output

        plugin = manager.get_plugin("review-kit")
        assert plugin is not None
        assert plugin.source == "workspace"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_plugin_slash_handler_copy_rejects_existing_workspace_plugin():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        builtin_root = workspace / "builtin_plugins"
        workspace_plugin_root = workspace / ".tg_agent" / "plugins"
        builtin_root.mkdir()
        workspace_plugin_root.mkdir(parents=True)
        _write_plugin(builtin_root)
        _write_plugin(workspace_plugin_root)

        manager = _create_manager(
            workspace,
            builtin_root,
            workspace_plugin_root=workspace_plugin_root,
        )
        manager.load_all()

        console = Console(record=True, width=120)
        handler = PluginSlashHandler(manager=manager, console=console)

        result = asyncio.run(handler.handle(["copy", "review-kit"]))
        output = console.export_text()
        assert result.handled is True
        assert "Workspace plugin already exists" in output
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_plugin_slash_handler_uninstall_removes_builtin_even_when_workspace_overrides():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        builtin_root = workspace / "builtin_plugins"
        workspace_plugin_root = workspace / ".tg_agent" / "plugins"
        builtin_root.mkdir()
        workspace_plugin_root.mkdir(parents=True)
        _write_plugin(builtin_root)
        _write_plugin(workspace_plugin_root)

        manager = _create_manager(
            workspace,
            builtin_root,
            workspace_plugin_root=workspace_plugin_root,
        )
        manager.load_all()

        console = Console(record=True, width=120)
        handler = PluginSlashHandler(manager=manager, console=console)

        result = asyncio.run(handler.handle(["uninstall", "review-kit", "--force"]))
        manager.reload_all()

        output = console.export_text()
        assert result.handled is True
        assert result.reloaded is True
        assert not (builtin_root / "review-kit").exists()
        assert (workspace_plugin_root / "review-kit").exists()
        assert "卸载成功" in output

        plugin = manager.get_plugin("review-kit")
        assert plugin is not None
        assert plugin.source == "workspace"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_plugin_slash_handler_uninstall_rejects_workspace_only_plugin():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        builtin_root = workspace / "builtin_plugins"
        workspace_plugin_root = workspace / ".tg_agent" / "plugins"
        builtin_root.mkdir()
        workspace_plugin_root.mkdir(parents=True)
        _write_plugin(workspace_plugin_root)

        manager = _create_manager(
            workspace,
            builtin_root,
            workspace_plugin_root=workspace_plugin_root,
        )
        manager.load_all()

        console = Console(record=True, width=120)
        handler = PluginSlashHandler(manager=manager, console=console)

        result = asyncio.run(handler.handle(["uninstall", "review-kit", "--force"]))
        output = console.export_text()

        assert result.handled is True
        assert result.reloaded is False
        assert "未找到内置插件：review-kit" in output
        assert (workspace_plugin_root / "review-kit").exists()

        plugin = manager.get_plugin("review-kit")
        assert plugin is not None
        assert plugin.source == "workspace"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_cli_runs_python_plugin_command_and_prints_output_without_agent(monkeypatch):
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        plugin_root = workspace / "plugins"
        plugin_root.mkdir()
        _write_plugin(
            plugin_root,
            include_python_command=True,
            python_script_body=(
                "import json\n"
                "import sys\n"
                "payload = json.load(sys.stdin)\n"
                "print(f\"Focus: {payload['args_text']}\")\n"
            ),
        )
        manager = _create_manager(workspace, plugin_root)
        manager.load_all()

        captured: list[str] = []
        console = Console(record=True, width=120)

        async def fake_run_agent(user_input, has_image: bool = False):
            captured.append(user_input)

        agent = SimpleNamespace(
            llm=SimpleNamespace(model="test-model", base_url=None),
            messages=[],
            system_prompt="",
            clear_history=lambda: None,
            register_hook=lambda hook: None,
        )
        context = SimpleNamespace(working_dir=workspace, subagent_manager=None)
        monkeypatch.setattr("cli.interactive_input.PromptSession", lambda: SimpleNamespace())

        cli = ClaudeCodeCLI(
            agent=agent,
            context=context,
            slash_registry=SlashCommandRegistry(),
            at_registry=AtCommandRegistry(workspace / "missing-skills"),
            agent_registry=AgentRegistry(workspace / "missing-agents"),
            plugin_manager=manager,
        )
        cli._console = console
        cli._run_agent = fake_run_agent  # type: ignore[method-assign]

        handled = asyncio.run(cli._handle_slash_command("/review-kit:summarize src/auth.py"))
        assert handled is True
        assert captured == []
        assert "Focus: src/auth.py" in console.export_text()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
