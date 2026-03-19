from pathlib import Path

from bu_agent_sdk.agent.registry import AgentRegistry
from bu_agent_sdk.plugin import PluginManager
from bu_agent_sdk.version import get_cli_version
from cli.at_commands import AtCommandRegistry
from cli.slash_commands import SlashCommandRegistry


def test_workflow_plugins_load_expected_commands():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)

    builtin_skills = temp_root / "workflow_builtin_skills"
    builtin_agents = temp_root / "workflow_builtin_agents"
    builtin_skills.mkdir(exist_ok=True)
    builtin_agents.mkdir(exist_ok=True)

    manager = PluginManager(
        plugin_dir=repo_root / "plugins",
        slash_registry=SlashCommandRegistry(),
        skill_registry=AtCommandRegistry(builtin_skills),
        agent_registry=AgentRegistry(builtin_agents),
        current_cli_version=get_cli_version(),
    )

    manager.load_all()

    frontend = manager.get_plugin("frontend-workflow")
    ta_workflow = manager.get_plugin("ta-workflow")

    assert frontend is not None
    assert frontend.status == "loaded"
    assert set(frontend.commands) == {
        "frontend-workflow:design",
        "frontend-workflow:requirement",
        "frontend-workflow:tasks",
    }

    assert ta_workflow is not None
    assert ta_workflow.status == "loaded"
    assert set(ta_workflow.commands) == {
        "ta-workflow:decompose",
        "ta-workflow:ta",
    }

    assert manager.get_command("frontend-workflow:init") is None
