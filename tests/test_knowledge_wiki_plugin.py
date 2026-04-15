import asyncio
import shutil
import uuid
from pathlib import Path

from agent_core.agent.registry import AgentRegistry
from agent_core.plugin import PluginCommandExecutor, PluginManager
from agent_core.version import get_cli_version
from cli.at_commands import AtCommandRegistry
from cli.slash_commands import SlashCommandRegistry


def _create_manager(repo_root: Path) -> PluginManager:
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    builtin_skills = temp_root / "knowledge_wiki_builtin_skills"
    builtin_agents = temp_root / "knowledge_wiki_builtin_agents"
    builtin_skills.mkdir(exist_ok=True)
    builtin_agents.mkdir(exist_ok=True)
    return PluginManager(
        plugin_dir=repo_root / "plugins",
        slash_registry=SlashCommandRegistry(),
        skill_registry=AtCommandRegistry(builtin_skills),
        agent_registry=AgentRegistry(builtin_agents),
        current_cli_version=get_cli_version(),
    )


def make_workspace(root: Path) -> Path:
    workspace = root / f"case-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def test_knowledge_wiki_plugin_registers_expected_resources():
    repo_root = Path(__file__).resolve().parent.parent
    manager = _create_manager(repo_root)

    manager.load_all()

    plugin = manager.get_plugin("knowledge-wiki")
    assert plugin is not None
    assert plugin.status == "loaded"
    assert plugin.skills == ["knowledge-wiki:llm-wiki"]
    assert plugin.agents == ["knowledge-wiki:maintainer"]
    assert set(plugin.commands) == {
        "knowledge-wiki:ask",
        "knowledge-wiki:ingest",
        "knowledge-wiki:init",
        "knowledge-wiki:lint",
        "knowledge-wiki:reindex",
        "knowledge-wiki:status",
    }


def test_knowledge_wiki_init_status_and_reindex_commands_work():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace = make_workspace(temp_root)
    manager = _create_manager(repo_root)
    manager.load_all()
    executor = PluginCommandExecutor()

    init_command = manager.get_command("knowledge-wiki:init")
    status_command = manager.get_command("knowledge-wiki:status")
    reindex_command = manager.get_command("knowledge-wiki:reindex")

    assert init_command is not None
    assert status_command is not None
    assert reindex_command is not None

    try:
        init_output = asyncio.run(
            executor.execute(
                init_command,
                args=[],
                args_text="",
                working_dir=workspace,
            )
        )
        assert "Knowledge wiki initialized." in init_output
        assert (workspace / ".tg_agent" / "knowledge" / "default" / "wiki" / "index.md").exists()

        status_output = asyncio.run(
            executor.execute(
                status_command,
                args=[],
                args_text="",
                working_dir=workspace,
            )
        )
        assert "Knowledge wiki status" in status_output
        assert "Indexed pages" in status_output

        concept_page = (
            workspace
            / ".tg_agent"
            / "knowledge"
            / "default"
            / "wiki"
            / "concepts"
            / "llm-wiki.md"
        )
        concept_page.write_text(
            """---
id: concept-llm-wiki
type: concept
title: LLM Wiki
source_ids:
  - src-001
updated_at: 2026-04-10
status: active
confidence: medium
---

# LLM Wiki

Persistent wiki layer for source-backed knowledge.

Related: [Overview](../overview.md)
""",
            encoding="utf-8",
        )

        reindex_output = asyncio.run(
            executor.execute(
                reindex_command,
                args=[],
                args_text="",
                working_dir=workspace,
            )
        )
        assert "Knowledge wiki reindexed." in reindex_output
        pages_json = workspace / ".tg_agent" / "knowledge" / "default" / "state" / "pages.json"
        links_json = workspace / ".tg_agent" / "knowledge" / "default" / "state" / "links.json"
        index_md = workspace / ".tg_agent" / "knowledge" / "default" / "wiki" / "index.md"
        assert pages_json.exists()
        assert links_json.exists()
        assert index_md.exists()
        assert "LLM Wiki" in index_md.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
