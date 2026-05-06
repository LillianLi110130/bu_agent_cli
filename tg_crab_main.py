"""
Crab CLI - An interactive coding assistant with file operations.

Includes bash, file operations (read/write/edit), search (glob/grep),
todo management, and task completion - all with dependency injection
for secure filesystem access.

Usage:
    py -3.10 tg_crab_main.py
    py -3.10 tg_crab_main.py --model gpt-4o
    py -3.10 tg_crab_main.py --root-dir ./other-project

Environment Variables:
    LLM_MODEL: Model to use (default: GLM-4.7)
    LLM_BASE_URL: LLM API base URL (default: https://open.bigmodel.cn/api/coding/paas/v4)
    OPENAI_API_KEY: API key for OpenAI-compatible APIs
"""

import argparse
import asyncio
import inspect
import logging
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from agent_core import Agent
from agent_core.agent import (
    AgentHook,
    AuditHook,
    BashFileTaskGuardHook,
    ExcelReadGuardHook,
    HumanApprovalHook,
    SubagentCompletionHook,
    build_default_approval_policy,
)
from agent_core.agent.config import AgentConfig
from agent_core.agent.registry import AgentRegistry, default_agent_sources
from agent_core.task import SubagentTaskManager
from agent_core.team import (
    TeamRuntime,
    is_team_experiment_enabled,
    team_experiment_disabled_message,
)
from agent_core.team.runtime import TEAM_WORKER_INTERNAL_FLAG
from agent_core.bootstrap.agent_factory import build_project_context
from agent_core.llm import ChatOpenAI
from agent_core.plugin import PluginManager
from agent_core.runtime_paths import (
    application_root,
    ensure_cli_runtime_state,
    is_frozen_app,
    load_runtime_env,
    tg_agent_home,
)
from agent_core.skill.discovery import default_skill_dirs
from agent_core.skill.review import SkillReviewHook, SkillReviewRunner
from agent_core.skill.runtime_service import SkillRuntimeService
from cli.app import TGAgentCLI
from cli.at_commands import AtCommand, AtCommandRegistry
from cli.im_bridge import FileBridgeStore, resolve_session_binding_id
from cli.session_runtime import CLISessionRuntime
from cli.slash_commands import SlashCommandRegistry
from cli.worker.auth import (
    _get_package_config_dir,
    _resolve_auth_config_path,
    authenticate_startup,
    load_auth_config,
)
from cli.worker.gateway_client import WorkerGatewayClient
from tools import ALL_TOOLS, SandboxContext, get_sandbox_context
from tools.sandbox import get_current_agent
from tools.skills import get_skill_runtime_service, skill_list, skill_manage, skill_view

# =============================================================================
# Prompt & Skills Loading
# =============================================================================

# Directory paths
_SCRIPT_DIR = application_root()
_PROMPTS_DIR = _SCRIPT_DIR / "agent_core" / "prompts"
_SKILLS_DIR = _SCRIPT_DIR / "skills"
_AGENTS_DIR = _PROMPTS_DIR / "agents"
_PLUGINS_DIR = _SCRIPT_DIR / "plugins"
_INTERNAL_WORKER_FLAG = "--run-worker-internal"
CLI_TOOLS = [
    *ALL_TOOLS,
    skill_list,
    skill_view,
    skill_manage,
]

_SKILL_MANAGEMENT_GUIDANCE = """
## Skill Learning

After completing a complex task (5+ tool calls), fixing a tricky error, or discovering
a reusable workflow, tool/API/platform pitfall, or stable user preference, consider saving
the approach as a user-level skill with `skill_manage` so it can be reused next time.

When using a user-level skill and finding it outdated, incomplete, or wrong, patch it
immediately with `skill_manage(action="patch")`. Do not modify builtin, plugin, workspace,
or project-private skills. `skill_manage` may only write under `~/.tg_agent/skills`;
never delete skills and do not create draft skills.

When the user asks to create, update, optimize, refine, or fix a skill, inspect
visible skills with `skill_list`/`skill_view` and write changes with `skill_manage`. Do not
use generic `read`, `write`, or `edit` tools to modify skills under `~/.tg_agent/skills`, 
because that bypasses skill validation and skill index refresh.

Only use generic file editing tools for skills when the user explicitly asks to modify a
repository/workspace skill file, or when `skill_manage` cannot write that skill source.
"""


@dataclass(slots=True)
class RuntimeRegistries:
    slash_registry: SlashCommandRegistry
    skill_registry: AtCommandRegistry
    agent_registry: AgentRegistry
    plugin_manager: PluginManager


def _format_skills(skills: list[AtCommand]) -> str:
    """Format skills list into a readable string for the prompt."""
    if not skills:
        return "No skills available."

    skills_formatted = "\n".join(
        (f"- {skill.name}\n" f"  - Path: {skill.path}\n" f"  - Desc: {skill.description}")
        for skill in sorted(skills, key=lambda item: item.name)
    )
    return skills_formatted


def _load_prompt_template(template_name: str = "system.md") -> str:
    """Load a prompt template from the prompts directory."""
    template_path = _PROMPTS_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _get_windows_system_info(release: str) -> str:
    """Format Windows version information."""
    version = release.split(".")[0] if "." in release else release
    return f"Windows {version}"


def _get_linux_distro_info() -> str | None:
    """Return Linux distribution info via the optional distro package."""
    try:
        import distro
    except ImportError:
        return None

    distro_name = distro.name()
    distro_version = distro.version()
    if distro_version:
        return f"{distro_name} {distro_version}"
    return distro_name


def _extract_pretty_name(os_release_content: str) -> str | None:
    """Extract PRETTY_NAME from /etc/os-release content."""
    for line in os_release_content.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip('"')
    return None


def _get_linux_os_release_info() -> str | None:
    """Return Linux distribution info from /etc/os-release when available."""
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            return _extract_pretty_name(f.read())
    except (IOError, OSError):
        return None


def _get_linux_system_info(release: str) -> str:
    """Format Linux distribution or kernel version information."""
    distro_info = _get_linux_distro_info()
    if distro_info:
        return distro_info

    os_release_info = _get_linux_os_release_info()
    if os_release_info:
        return os_release_info

    return f"Linux {release}"


def _get_macos_system_info() -> str:
    """Format macOS version information."""
    import platform

    version = platform.mac_ver()[0]
    if version:
        return f"macOS {version}"
    return "macOS"


def _get_system_info() -> str:
    """收集并格式化系统信息（仅操作系统名称和版本）"""
    import platform

    system = platform.system()
    release = platform.release()

    if system == "Windows":
        return _get_windows_system_info(release)
    if system == "Linux":
        return _get_linux_system_info(release)
    if system == "Darwin":
        return _get_macos_system_info()
    return f"{system} {release}"


def create_runtime_registries(
    *,
    workspace_root: Path,
    plugin_dir: Path | None = None,
    plugin_dirs: list[tuple[str, Path]] | None = None,
    skills_dir: Path | None = None,
) -> RuntimeRegistries:
    """Create shared registries for built-ins and workspace plugins."""
    slash_registry = SlashCommandRegistry()
    resolved_skill_dirs = default_skill_dirs(
        workspace_root=workspace_root,
        builtin_skills_dir=skills_dir or _SKILLS_DIR,
    )
    skill_registry = AtCommandRegistry(skill_dirs=resolved_skill_dirs)
    agent_registry = AgentRegistry(
        agent_sources=default_agent_sources(
            workspace_root,
            builtin_agents_dir=_AGENTS_DIR,
        )
    )
    resolved_plugin_dirs = plugin_dirs or [
        ("builtin", plugin_dir or _PLUGINS_DIR),
        ("workspace", workspace_root / ".tg_agent" / "plugins"),
    ]
    plugin_manager = PluginManager(
        plugin_dir=None,
        plugin_dirs=resolved_plugin_dirs,
        slash_registry=slash_registry,
        skill_registry=skill_registry,
        agent_registry=agent_registry,
    )
    plugin_manager.load_all()
    return RuntimeRegistries(
        slash_registry=slash_registry,
        skill_registry=skill_registry,
        agent_registry=agent_registry,
        plugin_manager=plugin_manager,
    )


def _build_system_prompt(
    working_dir: Path,
    skill_registry: AtCommandRegistry,
    agent_registry: AgentRegistry,
) -> str:
    """Build the system prompt by loading template and injecting skills."""
    from string import Template

    # Load skills and Format skills
    skills = skill_registry.get_all()
    skills_text = _format_skills(skills)

    # Load subagents
    callable_agents = agent_registry.list_callable_agents()

    # Format subagents
    agents_text = ""
    if callable_agents:
        agents_lines = []
        for agent_name in callable_agents:
            config = agent_registry.get_config(agent_name)
            if config:
                agents_lines.append(f"- {agent_name}: {config.description}")
        agents_text = "\n".join(agents_lines)
    else:
        agents_text = "No subagents available."

    # Get system information
    system_info_text = _get_system_info()

    template_str = _load_prompt_template("system.md")

    template = Template(template_str)
    prompt = template.substitute(
        SKILLS=skills_text,
        WORKING_DIR=str(working_dir),
        SUBAGENTS=agents_text,
        SYSTEM_INFO=system_info_text,
        PROJECT_CONTEXT=build_project_context(),
    )

    return f"{prompt.rstrip()}\n\n{_SKILL_MANAGEMENT_GUIDANCE.strip()}\n"


console = Console()

_DEFAULT_WORKER_HOST = (
    os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME") or socket.gethostname() or "local"
)
_DEFAULT_IM_WORKER_ID = os.getenv("BU_AGENT_IM_WORKER_ID", f"worker-{_DEFAULT_WORKER_HOST}")


@dataclass
class WorkerProcessHandle:
    """Track the spawned worker process and its redirected log file."""

    process: asyncio.subprocess.Process
    log_file: Any


def _load_cli_runtime_env() -> None:
    """Load CLI-only runtime env files."""
    ensure_cli_runtime_state()
    load_runtime_env()


def _configure_llm_debug_logging() -> None:
    """Enable OpenAI tool-call debug logs when explicitly requested."""
    if not (os.getenv("BU_AGENT_SDK_LLM_DEBUG") or os.getenv("bu_agent_sdk_LLM_DEBUG")):
        return

    log_path_raw = os.getenv("BU_AGENT_SDK_LLM_DEBUG_FILE")
    log_path = (
        Path(log_path_raw).expanduser()
        if log_path_raw
        else ensure_cli_runtime_state() / "logs" / "llm-debug.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    debug_logger = logging.getLogger("agent_core.llm.openai")
    debug_logger.setLevel(logging.INFO)
    debug_logger.propagate = False

    for handler in list(debug_logger.handlers):
        if getattr(handler, "_bu_agent_llm_debug_handler", False):
            debug_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    file_handler._bu_agent_llm_debug_handler = True  # type: ignore[attr-defined]
    debug_logger.addHandler(file_handler)

def _should_run_internal_worker(argv: list[str] | None = None) -> bool:
    """Return whether the current process should dispatch to the worker entrypoint."""
    return _INTERNAL_WORKER_FLAG in (argv or sys.argv[1:])


def _should_run_internal_team_worker(argv: list[str] | None = None) -> bool:
    """Return whether the current process should dispatch to a team worker."""
    return TEAM_WORKER_INTERNAL_FLAG in (argv or sys.argv[1:])


def _strip_internal_worker_flag(argv: list[str] | None = None) -> list[str]:
    """Return argv without the internal worker dispatch flag."""
    return [arg for arg in (argv or sys.argv[1:]) if arg != _INTERNAL_WORKER_FLAG]


def _strip_internal_team_worker_flag(argv: list[str] | None = None) -> list[str]:
    """Return argv without the internal team worker dispatch flag."""
    return [arg for arg in (argv or sys.argv[1:]) if arg != TEAM_WORKER_INTERNAL_FLAG]


def _build_worker_process_command(
    *,
    worker_id: str,
    gateway_base_url: str,
    config_dir: Path,
    root_dir: Path,
    model: str | None = None,
) -> list[str]:
    """Build the worker subprocess command for source and frozen runtimes."""
    if is_frozen_app():
        command = [sys.executable, _INTERNAL_WORKER_FLAG]
    else:
        command = [sys.executable, "-m", "cli.worker.main"]

    command.extend(
        [
            "--worker-id",
            worker_id,
            "--gateway-base-url",
            gateway_base_url,
            "--config-dir",
            str(config_dir),
            "--root-dir",
            str(root_dir),
        ]
    )
    if model:
        command.extend(["--model", model])
    return command


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Crab CLI - Interactive coding assistant")
    parser.add_argument(
        "--model",
        "-m",
        help="LLM model to use (default: from LLM_MODEL env var or GLM-4.7)",
    )
    parser.add_argument(
        "--root-dir",
        "-r",
        help="Workspace root for sandbox (default: current working directory)",
    )
    parser.add_argument(
        "--local-bridge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Route local terminal input through the file-backed bridge queue (default: enabled)",
    )
    parser.add_argument(
        "--im-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable worker bridge mode for a remote IM session (default: enabled)",
    )
    parser.add_argument(
        "--im-gateway-base-url",
        default=None,
        help="Gateway base URL for the worker (default: from tg_crab_worker.json)",
    )
    args = parser.parse_args()
    args.config_dir = Path.cwd().resolve()
    resolved_config_path = _resolve_auth_config_path(base_dir=args.config_dir)
    if resolved_config_path is not None:
        args.config_source_dir = resolved_config_path.parent.resolve()
    else:
        args.config_source_dir = _get_package_config_dir().resolve()
    auth_config = load_auth_config(base_dir=args.config_dir)
    if args.im_enable:
        args.local_bridge = True
        args.im_worker_id = _DEFAULT_IM_WORKER_ID
        args.im_gateway_base_url = args.im_gateway_base_url or auth_config.gateway_base_url
    return args


def _build_bridge_store(
    *,
    args: argparse.Namespace,
    ctx: SandboxContext,
) -> FileBridgeStore | None:
    """Create the file-backed bridge store for local or IM-enabled runs."""
    if not (args.local_bridge or args.im_enable):
        return None
    binding_source = args.im_worker_id or "local-cli"
    session_binding_id = resolve_session_binding_id(binding_source)
    return FileBridgeStore(root_dir=ctx.working_dir, session_binding_id=session_binding_id)


async def _start_im_worker_process(
    *,
    args: argparse.Namespace,
    ctx: SandboxContext,
) -> WorkerProcessHandle | None:
    """Start the background worker process when IM mode is enabled."""
    if not args.im_enable:
        return None

    missing = [
        name
        for name, value in (
            ("worker_id", args.im_worker_id),
            ("gateway_base_url", args.im_gateway_base_url),
        )
        if not value
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"IM worker mode requires: {missing_str}")

    command = _build_worker_process_command(
        worker_id=str(args.im_worker_id),
        gateway_base_url=str(args.im_gateway_base_url),
        config_dir=Path(getattr(args, "config_dir", Path.cwd())).resolve(),
        root_dir=ctx.working_dir,
        model=str(args.model) if args.model else None,
    )

    bridge_store = _build_bridge_store(args=args, ctx=ctx)
    if bridge_store is None:
        raise RuntimeError("Bridge store must exist before starting the IM worker")
    bridge_store.initialize()
    worker_log_path = bridge_store.logs_dir / "worker.log"
    worker_log_file = worker_log_path.open("a", encoding="utf-8", buffering=1)

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(_SCRIPT_DIR if not is_frozen_app() else Path.cwd().resolve()),
        stdout=worker_log_file,
        stderr=worker_log_file,
    )
    await asyncio.sleep(0.25)
    if process.returncode is not None:
        worker_log_file.flush()
        worker_log_file.close()
        raise RuntimeError(
            "IM worker exited during startup. "
            f"Check the worker log for details: {worker_log_path}"
        )
    return WorkerProcessHandle(process=process, log_file=worker_log_file)


async def _stop_im_worker_process(handle: WorkerProcessHandle | None) -> None:
    """Terminate the worker subprocess if it is running."""
    if handle is None:
        return

    process = handle.process
    if process.returncode is not None:
        handle.log_file.close()
        return

    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
    finally:
        handle.log_file.close()


async def _mark_worker_offline(
    *,
    worker_id: str | None,
    gateway_base_url: str | None,
) -> None:
    """Best-effort offline notification sent by the parent CLI process."""
    if not worker_id or not gateway_base_url:
        return

    client = WorkerGatewayClient(base_url=gateway_base_url)
    try:
        await client.offline(worker_id=worker_id)
    except Exception:
        pass
    finally:
        await client.aclose()


async def _close_llm_runtime(llm: Any) -> None:
    """Best-effort close for LLM runtimes that own async clients/transports."""
    close = getattr(llm, "close", None)
    if close is None or not callable(close):
        return

    result = close()
    if inspect.isawaitable(result):
        await result


async def _authenticate_worker_startup(args: argparse.Namespace) -> None:
    """Authenticate before starting the CLI and worker when auth is enabled."""
    base_dir = Path(getattr(args, "config_dir", Path.cwd())).resolve()
    auth_config = load_auth_config(base_dir=base_dir)
    if not auth_config.enable_auth:
        return

    auth_result = await authenticate_startup(
        config=auth_config,
        base_dir=base_dir,
    )
    args.im_worker_id = auth_result.user_id


def create_llm(model: str | None = None) -> ChatOpenAI:
    """Create LLM instance based on environment or model parameter."""
    model = model or (os.getenv("LLM_MODEL") or "").strip() or "GLM-5.1"
    base_url = (os.getenv("LLM_BASE_URL") or "").strip() or "https://open.bigmodel.cn/api/coding/paas/v4"
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or "OPENAI_API_KEY"

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def build_agent_hooks() -> list[AgentHook]:
    """Build runtime hooks for agent instances.

    Keep the default set non-invasive for CLI usage. The built-in
    FinishGuardHook is already attached inside Agent, so this helper
    only adds optional extra hooks.
    """
    hooks: list[AgentHook] = [
        HumanApprovalHook(policy=build_default_approval_policy()),
        BashFileTaskGuardHook(),
        ExcelReadGuardHook(),
        SubagentCompletionHook(),
        AuditHook(),
    ]

    return hooks


def create_agent(
    model: str | None,
    root_dir: Path | str | None = None,
    agent_config: AgentConfig | None = None,
    runtime_registries: RuntimeRegistries | None = None,
) -> tuple[Agent, SandboxContext, RuntimeRegistries]:
    """Create configured Agent and SandboxContext.

    Returns:
        Tuple of (Agent, SandboxContext, RuntimeRegistries)
    """
    ctx = SandboxContext.create(root_dir)
    llm = create_llm(model)
    runtime = runtime_registries or create_runtime_registries(
        workspace_root=ctx.working_dir,
    )

    system_prompt = _build_system_prompt(
        ctx.working_dir,
        skill_registry=runtime.skill_registry,
        agent_registry=runtime.agent_registry,
    )
    def system_prompt_builder() -> str:
        return _build_system_prompt(
            ctx.working_dir,
            skill_registry=runtime.skill_registry,
            agent_registry=runtime.agent_registry,
        )
    skill_runtime_service = SkillRuntimeService(
        skill_registry=runtime.skill_registry,
        plugin_manager=runtime.plugin_manager,
        system_prompt_builder=system_prompt_builder,
    )

    subagent_executor = SubagentTaskManager(
        registry=runtime.agent_registry,
        all_tools=CLI_TOOLS,
        context=ctx,
        skill_registry=runtime.skill_registry,
    )
    ctx.subagent_executor = subagent_executor
    if is_team_experiment_enabled():
        ctx.team_runtime = TeamRuntime(
            teams_root=tg_agent_home() / "teams",
            workspace_root=ctx.root_dir,
        )

    agent = Agent(
        llm=llm,
        tools=CLI_TOOLS,
        system_prompt=system_prompt,
        dependency_overrides={
            get_sandbox_context: lambda: ctx,
            get_current_agent: lambda: agent,
            get_skill_runtime_service: lambda: skill_runtime_service,
        },
        agent_config=agent_config,
        hooks=build_agent_hooks(),
    )
    skill_runtime_service.bind_agent(agent)
    setattr(agent, "_skill_runtime_service", skill_runtime_service)
    setattr(ctx, "skill_runtime_service", skill_runtime_service)
    agent.register_hook(
        SkillReviewHook(
            runner=SkillReviewRunner(service=skill_runtime_service),
        )
    )

    subagent_executor.set_main_agent(agent)
    ctx.current_agent = agent

    return agent, ctx, runtime


async def main():
    """Main entry point."""
    _load_cli_runtime_env()
    _configure_llm_debug_logging()
    args = parse_args()
    await _authenticate_worker_startup(args)

    agent, ctx, runtime = create_agent(
        model=args.model,
        root_dir=args.root_dir,
    )
    session_runtime = CLISessionRuntime.create_for_context(ctx)
    bridge_store = _build_bridge_store(args=args, ctx=ctx)
    worker_process = await _start_im_worker_process(args=args, ctx=ctx)
    if bridge_store is not None:
        console.print(
            "[dim]桥接会话：[/] "
            f"[cyan]{bridge_store.session_binding_id}[/cyan] "
            f"[dim]->[/dim] {bridge_store.bridge_dir}"
        )
    if args.im_enable:
        console.print(
            "[dim]IM 工作进程：[/] "
            f"worker=[cyan]{args.im_worker_id}[/cyan] "
            f"gateway=[cyan]{args.im_gateway_base_url}[/cyan]"
        )
        if bridge_store is not None:
            console.print(f"[dim]工作日志：[/] {bridge_store.logs_dir / 'worker.log'}")
    cli = TGAgentCLI(
        agent=agent,
        context=ctx,
        slash_registry=runtime.slash_registry,
        at_registry=runtime.skill_registry,
        agent_registry=runtime.agent_registry,
        plugin_manager=runtime.plugin_manager,
        system_prompt_builder=lambda: _build_system_prompt(
            ctx.working_dir,
            skill_registry=runtime.skill_registry,
            agent_registry=runtime.agent_registry,
        ),
        skill_runtime_service=getattr(agent, "_skill_runtime_service", None),
        bridge_store=bridge_store,
        session_runtime=session_runtime,
    )

    try:
        await cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]再见！[/yellow]")
    finally:
        if ctx.shell_task_manager is not None:
            await ctx.shell_task_manager.shutdown(cancel_running=True)
        if ctx.subagent_executor is not None:
            await ctx.subagent_executor.shutdown(cancel_running=True)
        await _close_llm_runtime(agent.llm)
        await _mark_worker_offline(
            worker_id=args.im_worker_id,
            gateway_base_url=args.im_gateway_base_url,
        )
        await _stop_im_worker_process(worker_process)


def cli_main():
    """Console script entry point."""
    if _should_run_internal_team_worker():
        if not is_team_experiment_enabled():
            console.print(f"[red]{team_experiment_disabled_message()}[/red]")
            raise SystemExit(2)
        from cli.team.worker import main as team_worker_main

        sys.argv = [sys.argv[0], *_strip_internal_team_worker_flag()]
        team_worker_main()
        return

    if _should_run_internal_worker():
        from cli.worker.main import cli_main as worker_cli_main

        sys.argv = [sys.argv[0], *_strip_internal_worker_flag()]
        worker_cli_main()
        return

    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
