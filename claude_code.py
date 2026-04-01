"""
Claude Code CLI - An interactive coding assistant with file operations.

Includes bash, file operations (read/write/edit), search (glob/grep),
todo management, and task completion - all with dependency injection
for secure filesystem access.

Usage:
    py -3.10 claude_code.py
    py -3.10 claude_code.py --model gpt-4o
    py -3.10 claude_code.py --root-dir ./other-project

Environment Variables:
    LLM_MODEL: Model to use (default: GLM-4.7)
    LLM_BASE_URL: LLM API base URL (default: https://open.bigmodel.cn/api/coding/paas/v4)
    OPENAI_API_KEY: API key for OpenAI-compatible APIs
"""

import argparse
import asyncio
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
    build_default_approval_policy,
)
from agent_core.agent.config import AgentConfig
from agent_core.agent.registry import AgentRegistry
from agent_core.llm import ChatOpenAI
from agent_core.plugin import PluginManager
from agent_core.skill.discovery import default_skill_dirs
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

# =============================================================================
# Prompt & Skills Loading
# =============================================================================

# Directory paths
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROMPTS_DIR = _SCRIPT_DIR / "agent_core" / "prompts"
_SKILLS_DIR = _SCRIPT_DIR / "skills"
_AGENTS_DIR = _PROMPTS_DIR / "agents"
_PLUGINS_DIR = _SCRIPT_DIR / "plugins"


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
    agents_dir: Path | None = None,
) -> RuntimeRegistries:
    """Create shared registries for built-ins and workspace plugins."""
    slash_registry = SlashCommandRegistry()
    resolved_skill_dirs = default_skill_dirs(
        workspace_root=workspace_root,
        builtin_skills_dir=skills_dir or _SKILLS_DIR,
    )
    skill_registry = AtCommandRegistry(skill_dirs=resolved_skill_dirs)
    agent_registry = AgentRegistry(agents_dir or _AGENTS_DIR)
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
    )

    return prompt


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Claude Code CLI - Interactive coding assistant")
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

    command = [
        sys.executable,
        "-m",
        "cli.worker.main",
        "--worker-id",
        str(args.im_worker_id),
        "--gateway-base-url",
        str(args.im_gateway_base_url),
        "--config-dir",
        str(Path(getattr(args, "config_dir", Path.cwd())).resolve()),
        "--root-dir",
        str(ctx.working_dir),
    ]
    if args.model:
        command.extend(["--model", str(args.model)])

    bridge_store = _build_bridge_store(args=args, ctx=ctx)
    if bridge_store is None:
        raise RuntimeError("Bridge store must exist before starting the IM worker")
    bridge_store.initialize()
    worker_log_path = bridge_store.logs_dir / "worker.log"
    worker_log_file = worker_log_path.open("a", encoding="utf-8", buffering=1)

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(_SCRIPT_DIR),
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
    model = model or os.getenv("LLM_MODEL", "GLM-4.7")
    base_url = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4")
    api_key = os.getenv("OPENAI_API_KEY", "OPENAI_API_KEY")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def build_agent_hooks(*, mode: str) -> list[AgentHook]:
    """Build runtime hooks for agent instances.

    Keep the default set non-invasive for CLI usage. The built-in
    FinishGuardHook is already attached inside Agent, so this helper
    only adds optional extra hooks.
    """
    hooks: list[AgentHook] = [
        HumanApprovalHook(policy=build_default_approval_policy(mode)),
        BashFileTaskGuardHook(),
        ExcelReadGuardHook(),
        AuditHook(),
    ]

    return hooks


def create_agent(
    model: str | None,
    root_dir: Path | str | None = None,
    mode: str = "primary",
    agent_config: AgentConfig | None = None,
    runtime_registries: RuntimeRegistries | None = None,
) -> tuple[Agent, SandboxContext, RuntimeRegistries]:
    """Create configured Agent and SandboxContext.

    Returns:
        Tuple of (Agent, SandboxContext, RuntimeRegistries)
    """
    from agent_core.agent.subagent_manager import SubagentManager

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

    subagent_manager = SubagentManager(
        agent_factory=_create_subagent_factory,
        registry=runtime.agent_registry,
        all_tools=ALL_TOOLS,
        workspace=ctx.working_dir,
        context=ctx,
    )
    ctx.subagent_manager = subagent_manager

    agent = Agent(
        llm=llm,
        tools=ALL_TOOLS,
        system_prompt=system_prompt,
        dependency_overrides={get_sandbox_context: lambda: ctx},
        mode=mode,
        agent_config=agent_config,
        hooks=build_agent_hooks(mode=mode),
    )

    if subagent_manager:
        subagent_manager.set_main_agent(agent)

    return agent, ctx, runtime


def _create_subagent_factory(config: AgentConfig, parent_ctx: Any, all_tools: list) -> Agent:
    """Factory function to create subagent instances."""
    from config.model_config import get_model_config

    model, base_url, api_key = get_model_config(config.model)

    llm = ChatOpenAI(
        model=model, api_key=api_key, base_url=base_url, temperature=config.temperature
    )

    # 为子代理添加系统信息到系统提示词
    system_info_text = _get_system_info()
    system_prompt = (
        f"{config.system_prompt}\n\n## System Information\n\n"
        f"The current environment: {system_info_text}"
    )

    agent = Agent(
        llm=llm,
        tools=all_tools,
        system_prompt=system_prompt,
        mode="subagent",
        agent_config=config,
        dependency_overrides={get_sandbox_context: lambda: parent_ctx},
        hooks=build_agent_hooks(mode="subagent"),
    )
    return agent


async def main():
    """Main entry point."""
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
        bridge_store=bridge_store,
        session_runtime=session_runtime,
    )

    try:
        await cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]再见！[/yellow]")
    finally:
        await _mark_worker_offline(
            worker_id=args.im_worker_id,
            gateway_base_url=args.im_gateway_base_url,
        )
        await _stop_im_worker_process(worker_process)


def cli_main():
    """Console script entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
