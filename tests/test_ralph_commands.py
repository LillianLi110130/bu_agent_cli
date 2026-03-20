import asyncio
import io
import json
import shutil
import uuid
from pathlib import Path

from rich.console import Console

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
        assert (spec_dir / "input").exists()
        assert (spec_dir / "artifacts").exists()
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
        assert (workspace_root / ".devagent" / "agents").exists()
        assert not (workspace_root / ".devagent" / "agents" / "code-reviewer.md").exists()
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


def test_ralph_handler_run_requires_spec_name_or_plan_file():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace_root = make_workspace(temp_root)
    output = io.StringIO()
    try:
        handler = RalphSlashHandler(
            workspace_root=workspace_root,
            console=Console(file=output, force_terminal=False, color_system=None),
        )

        handled = asyncio.run(handler.handle(["run"]))

        assert handled is True
        rendered = output.getvalue()
        assert "/ralph run requires <spec_name> or --plan-file." in rendered
        assert "Usage: /ralph run <spec_name> <optional flags>" in rendered
        assert "/ralph run --plan-file <path> <optional flags>" in rendered
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)


def test_ralph_handler_dry_run_requires_spec_name_or_plan_file():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace_root = make_workspace(temp_root)
    output = io.StringIO()
    try:
        handler = RalphSlashHandler(
            workspace_root=workspace_root,
            console=Console(file=output, force_terminal=False, color_system=None),
        )

        handled = asyncio.run(handler.handle(["dry-run"]))

        assert handled is True
        rendered = output.getvalue()
        assert "/ralph dry-run requires <spec_name> or --plan-file." in rendered
        assert "Usage: /ralph dry-run <spec_name> <optional flags>" in rendered
        assert "/ralph dry-run --plan-file <path> <optional flags>" in rendered
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)


def test_ralph_service_resolves_spec_execution_paths_from_input_and_plan():
    repo_root = Path(__file__).resolve().parent.parent
    temp_root = repo_root / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    workspace_root = make_workspace(temp_root)
    try:
        service = RalphService(workspace_root=workspace_root, script_root=repo_root)
        paths = service._resolve_paths("demo")

        assert paths.requirement_dir == workspace_root / "docs" / "spec" / "demo" / "input"
        assert paths.plan_file == workspace_root / "docs" / "spec" / "demo" / "plan" / "plan.json"
        assert paths.log_dir == workspace_root / "docs" / "spec" / "demo" / "logs"
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)


def test_slash_registry_contains_ralph():
    registry = SlashCommandRegistry()
    command = registry.get("ralph")

    assert command is not None
    assert command.name == "ralph"
    assert command.usage == "/ralph <init-spec|init-agent|dry-run|run|status|cancel> ..."
    assert all("/ralph ta" not in example for example in command.examples)
    assert all("/ralph decompose" not in example for example in command.examples)
