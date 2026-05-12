"""Agent factory for filesystem-backed team member processes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from agent_core import Agent
from agent_core.agent.config import AgentConfig
from agent_core.llm import ChatOpenAI
from agent_core.skill.runtime_service import SkillRuntimeService
from cli.at_commands import AtCommandRegistry
from config.model_config import get_model_config
from tools import SandboxContext, get_sandbox_context
from tools.sandbox import get_current_agent
from tools.skills import get_skill_runtime_service

_TEAM_MEMBER_BLOCKED_TOOL_NAMES = {
    "delegate",
    "delegate_parallel",
    "skill_manage",
    "team_create",
    "team_spawn_member",
    "team_create_task",
    "team_update_task",
    "team_shutdown",
}


def _bootstrap_module() -> Any:
    """Return the loaded TgAgent bootstrap module without double-loading script mode."""
    main_module = sys.modules.get("__main__")
    if main_module is not None and hasattr(main_module, "create_runtime_registries"):
        return main_module

    import tg_crab_main

    return tg_crab_main


def _without_team_member_blocked_tools(tools: list[Any]) -> list[Any]:
    return [
        tool
        for tool in tools
        if getattr(tool, "name", None) not in _TEAM_MEMBER_BLOCKED_TOOL_NAMES
    ]


def _load_agent_type_skill_text(
    *,
    skill_registry: AtCommandRegistry,
    skill_names: list[str],
) -> str:
    if not skill_names:
        return ""
    parts: list[str] = []
    for skill_name in skill_names:
        skill = skill_registry.get(skill_name)
        if skill is None:
            continue
        try:
            parts.append(f"### @{skill.name}\n{skill.load_content().strip()}")
        except Exception:
            continue
    return "\n\n".join(parts)


def build_team_member_system_prompt(
    *,
    base_system_prompt: str,
    team_prompt: str,
    agent_config: AgentConfig | None,
    skill_registry: AtCommandRegistry,
) -> str:
    parts = [part.strip() for part in (base_system_prompt, team_prompt) if part.strip()]
    if agent_config is not None and agent_config.system_prompt.strip():
        parts.append(f"## Agent Type Instructions\n\n{agent_config.system_prompt.strip()}")
        skills_text = _load_agent_type_skill_text(
            skill_registry=skill_registry,
            skill_names=agent_config.skills,
        )
        if skills_text:
            parts.append(f"## Loaded Agent Type Skills\n\n{skills_text}")
    return "\n\n".join(parts)


def _resolve_team_member_llm(
    *,
    base_llm: ChatOpenAI,
    configured_model: str | None,
) -> ChatOpenAI:
    normalized = (configured_model or "").strip()
    if not normalized or normalized.lower() == "inherit":
        return base_llm

    model, base_url, api_key = get_model_config(normalized)
    return ChatOpenAI(
        model=model,
        api_key=api_key or base_llm.api_key,
        base_url=base_url or base_llm.base_url,
        temperature=base_llm.temperature,
        frequency_penalty=base_llm.frequency_penalty,
        reasoning_effort=base_llm.reasoning_effort,
        seed=base_llm.seed,
        service_tier=base_llm.service_tier,
        top_p=base_llm.top_p,
        parallel_tool_calls=base_llm.parallel_tool_calls,
        prompt_cache_key=base_llm.prompt_cache_key,
        prompt_cache_retention=base_llm.prompt_cache_retention,
        max_retries=base_llm.max_retries,
        timeout=base_llm.timeout,
        default_headers=base_llm.default_headers,
        default_query=base_llm.default_query,
    )


def create_team_member_agent(
    *,
    model: str | None,
    root_dir: Path | str | None,
    team_prompt: str,
    agent_type: str | None = None,
    runtime_registries: Any | None = None,
) -> tuple[Agent, SandboxContext, Any]:
    """Create an independent team member agent through the CLI bootstrap path."""
    bootstrap = _bootstrap_module()
    ctx = SandboxContext.create(root_dir)
    base_llm = bootstrap.create_llm(model)
    runtime = runtime_registries or bootstrap.create_runtime_registries(
        workspace_root=ctx.working_dir,
    )
    agent_config = runtime.agent_registry.get_config(agent_type) if agent_type else None
    if agent_type and agent_config is None:
        raise ValueError(f"Agent '{agent_type}' not found")

    base_system_prompt = bootstrap._build_system_prompt(
        ctx.working_dir,
        skill_registry=runtime.skill_registry,
        agent_registry=runtime.agent_registry,
    )

    def system_prompt_builder() -> str:
        return build_team_member_system_prompt(
            base_system_prompt=bootstrap._build_system_prompt(
                ctx.working_dir,
                skill_registry=runtime.skill_registry,
                agent_registry=runtime.agent_registry,
            ),
            team_prompt=team_prompt,
            agent_config=agent_config,
            skill_registry=runtime.skill_registry,
        )

    system_prompt = build_team_member_system_prompt(
        base_system_prompt=base_system_prompt,
        team_prompt=team_prompt,
        agent_config=agent_config,
        skill_registry=runtime.skill_registry,
    )
    skill_runtime_service = SkillRuntimeService(
        skill_registry=runtime.skill_registry,
        plugin_manager=runtime.plugin_manager,
        system_prompt_builder=system_prompt_builder,
    )

    agent = Agent(
        llm=_resolve_team_member_llm(
            base_llm=base_llm,
            configured_model=agent_config.model if agent_config is not None else None,
        ),
        tools=_without_team_member_blocked_tools(bootstrap.CLI_TOOLS),
        system_prompt=system_prompt,
        max_iterations=agent_config.max_turns if agent_config and agent_config.max_turns else 200,
        dependency_overrides={
            get_sandbox_context: lambda: ctx,
            get_current_agent: lambda: agent,
            get_skill_runtime_service: lambda: skill_runtime_service,
        },
        agent_config=agent_config,
        hooks=bootstrap.build_agent_hooks(),
        runtime_role="team_member",
    )
    skill_runtime_service.bind_agent(agent)
    setattr(agent, "_skill_runtime_service", skill_runtime_service)
    setattr(ctx, "skill_runtime_service", skill_runtime_service)
    ctx.current_agent = agent
    return agent, ctx, runtime
