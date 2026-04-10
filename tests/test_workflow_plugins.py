from pathlib import Path

from agent_core.agent.registry import AgentRegistry
from agent_core.plugin import PluginManager
from agent_core.version import get_cli_version
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
    knowledge_wiki = manager.get_plugin("knowledge-wiki")
    ta_workflow = manager.get_plugin("ta-workflow")
    tgcrab_frontend = manager.get_plugin("tgcrab-frontend")

    assert frontend is not None
    assert frontend.status == "loaded"
    assert set(frontend.commands) == {
        "frontend-workflow:design",
        "frontend-workflow:requirement",
        "frontend-workflow:tasks",
    }

    assert knowledge_wiki is not None
    assert knowledge_wiki.status == "loaded"
    assert set(knowledge_wiki.commands) == {
        "knowledge-wiki:ask",
        "knowledge-wiki:ingest",
        "knowledge-wiki:init",
        "knowledge-wiki:lint",
        "knowledge-wiki:reindex",
        "knowledge-wiki:status",
    }

    assert ta_workflow is not None
    assert ta_workflow.status == "loaded"
    assert set(ta_workflow.commands) == {
        "ta-workflow:decompose",
        "ta-workflow:ta",
    }

    assert tgcrab_frontend is not None
    assert tgcrab_frontend.status == "loaded"
    assert set(tgcrab_frontend.commands) == {
        "tgcrab-frontend:design",
        "tgcrab-frontend:requirement",
        "tgcrab-frontend:tasks",
    }

    assert manager.get_command("frontend-workflow:init") is None
