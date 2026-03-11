import asyncio
import json
import shutil
import uuid
from pathlib import Path

from cli.ralph_commands import RalphSlashHandler
from cli.ralph_service import RalphService
from cli.slash_commands import SlashCommandRegistry


def make_workspace(root: Path) -> Path:
    workspace = root / f"case-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def test_ralph_service_init_spec():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace_root = make_workspace(temp_root)
    try:
        service = RalphService(workspace_root=workspace_root, script_root=repo_root)

        result = asyncio.run(service.init_spec(spec_name="demo"))

        assert result.success is True
        spec_dir = workspace_root / "docs" / "spec" / "demo"
        assert spec_dir.exists()
        assert (spec_dir / "plan" / "plan.json").exists()
        assert json.loads((spec_dir / "plan" / "plan.json").read_text(encoding="utf-8")) == []
        assert (spec_dir / "requirement").exists()
        assert (spec_dir / "implement").exists()
        assert (spec_dir / "logs").exists()
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)


def test_ralph_service_init_agent():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace_root = make_workspace(temp_root)
    try:
        service = RalphService(workspace_root=workspace_root, script_root=repo_root)

        result = asyncio.run(service.init_agent())

        assert result.success is True
        assert (workspace_root / ".devagent" / "commands" / "ralph" / "implement.md").exists()
        assert (workspace_root / ".devagent" / "agents" / "code-reviewer.md").exists()
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)


def test_ralph_handler_status_without_runs():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace_root = make_workspace(temp_root)
    try:
        handler = RalphSlashHandler(workspace_root=workspace_root)

        handled = asyncio.run(handler.handle(["status"]))

        assert handled is True
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)


def test_ralph_decompose_prompt_includes_ta_outputs():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace_root = make_workspace(temp_root)
    try:
        prompt_dir = workspace_root / ".devagent" / "commands" / "ralph"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        source_prompt = repo_root / ".devagent" / "commands" / "ralph" / "DECOMPOSE_TASK.md"
        shutil.copy2(source_prompt, prompt_dir / "DECOMPOSE_TASK.md")

        service = RalphService(workspace_root=workspace_root, script_root=repo_root)
        paths = service._resolve_paths("demo")

        prompt = service._build_decompose_prompt(paths, description="focus on payment flow")

        assert "requirements_file:" in prompt
        assert "design_file:" in prompt
        assert "task_file:" in prompt
        assert "demo-requirements.md" in prompt
        assert "demo-design.md" in prompt
        assert "demo-task.md" in prompt
        assert "focus on payment flow" in prompt
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)


def test_slash_registry_contains_ralph():
    registry = SlashCommandRegistry()
    command = registry.get("ralph")

    assert command is not None
    assert command.name == "ralph"
    assert "init-spec" in command.usage
    assert "ta" in command.usage
