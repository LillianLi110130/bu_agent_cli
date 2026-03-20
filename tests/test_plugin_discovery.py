import json
import shutil
import uuid
from pathlib import Path

from bu_agent_sdk.agent.registry import AgentRegistry
from bu_agent_sdk.plugin import PluginManager
from bu_agent_sdk.version import get_cli_version
from cli.at_commands import AtCommandRegistry
from cli.slash_commands import SlashCommandRegistry


def make_workspace(root: Path) -> Path:
    workspace = root / f"case-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def _write_manifest(plugin_dir: Path, *, manifest_name: str) -> None:
    payload = {
        "schema_version": 1,
        "name": manifest_name,
        "version": "0.1.0",
        "description": "Built-in review helpers",
        "capabilities": {
            "skills": False,
            "agents": False,
            "commands": False,
        },
    }
    (plugin_dir / "plugin.json").write_text(json.dumps(payload), encoding="utf-8")


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
        current_cli_version=get_cli_version(),
    )


def test_plugin_manager_ignores_cache_and_non_plugin_directories():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    try:
        plugin_root = workspace / "plugins"
        plugin_root.mkdir()

        valid_plugin = plugin_root / "review-kit"
        valid_plugin.mkdir()
        _write_manifest(valid_plugin, manifest_name="review-kit")

        (plugin_root / "__pycache__").mkdir()
        (plugin_root / ".cache").mkdir()
        (plugin_root / "scratch").mkdir()

        manager = _create_manager(workspace, plugin_root)
        manager.load_all()

        plugins = manager.list_plugins()

        assert [plugin.name for plugin in plugins] == ["review-kit"]
        assert manager.get_plugin("__pycache__") is None
        assert manager.get_plugin(".cache") is None
        assert manager.get_plugin("scratch") is None
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
