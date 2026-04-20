from __future__ import annotations

import json
from typing import Annotated

from agent_core.skill.manager import SkillManagementError
from agent_core.skill.runtime_service import SkillRuntimeService
from agent_core.tools import Depends, tool


def get_skill_runtime_service() -> SkillRuntimeService:
    """Dependency injection marker. Override this in the agent."""
    raise RuntimeError("get_skill_runtime_service() must be overridden via dependency_overrides")


@tool(
    "List currently visible skills with source and writability metadata.",
    context_policy="trim",
    context_max_inline_chars=6000,
)
async def skill_list(
    service: Annotated[SkillRuntimeService, Depends(get_skill_runtime_service)],
) -> str:
    skills = [
        {
            "name": skill.name,
            "description": skill.description,
            "category": skill.category,
            "path": str(skill.path),
            "source": service.source_of(skill),
            "writable": service.is_writable(skill),
        }
        for skill in service.list()
    ]
    return json.dumps({"skills": skills}, ensure_ascii=False, indent=2)


@tool(
    "Read the full SKILL.md for a visible skill by name.",
    context_policy="trim",
    context_max_inline_chars=20000,
)
async def skill_view(
    name: str,
    service: Annotated[SkillRuntimeService, Depends(get_skill_runtime_service)],
) -> str:
    try:
        return service.view(name)
    except Exception as exc:
        return f"Error: {exc}"


@tool(
    "Manage reusable user-level skills. Skills are procedural memory: compact "
    "instructions for recurring task types. Create a skill when a task produced a reusable "
    "workflow, overcame non-obvious errors, revealed a tool/API/platform pitfall, or captured "
    "a user-preferred way to do this kind of work. Update a user-level skill when existing "
    "instructions were incomplete, stale, wrong, missing verification steps, or missing pitfalls "
    "discovered during execution. Prefer patch for small corrections. Use edit only for major "
    "rewrites. Do not create skills for simple one-off tasks, temporary progress, or facts better "
    "suited for memory. A good skill includes trigger conditions, exact steps or commands, "
    "pitfalls, and verification. Only write under ~/.tg_agent/skills. Never delete skills.",
)
async def skill_manage(
    action: str,
    name: str,
    service: Annotated[SkillRuntimeService, Depends(get_skill_runtime_service)],
    content: str | None = None,
    old_string: str | None = None,
    new_string: str | None = None,
    file_path: str | None = None,
    replace_all: bool = False,
) -> str:
    try:
        normalized_action = action.strip().lower()
        if normalized_action == "delete":
            return "Error: delete is disabled. skill_manage may only create or update user skills."
        if normalized_action == "create":
            if not content:
                return "Error: content is required for create."
            result = service.manager.create(name=name, content=content)
        elif normalized_action == "patch":
            if old_string is None or new_string is None:
                return "Error: old_string and new_string are required for patch."
            result = service.manager.patch(
                name=name,
                old_string=old_string,
                new_string=new_string,
                replace_all=replace_all,
            )
        elif normalized_action == "edit":
            if not content:
                return "Error: content is required for edit."
            result = service.manager.edit(name=name, content=content)
        elif normalized_action == "write_file":
            if not file_path or content is None:
                return "Error: file_path and content are required for write_file."
            result = service.manager.write_file(name=name, file_path=file_path, content=content)
        elif normalized_action == "remove_file":
            if not file_path:
                return "Error: file_path is required for remove_file."
            result = service.manager.remove_file(name=name, file_path=file_path)
        else:
            return (
                "Error: unsupported action. Use create, patch, edit, write_file, or remove_file."
            )

        reload_result = service.reload(refresh_agent_prompt=True)
        return "\n".join(
            [
                result.message,
                f"Path: {result.path}",
                f"Skills reloaded: {reload_result.format_summary()}.",
                (
                    "Current agent skill index refreshed."
                    if reload_result.refreshed_agent_prompt
                    else "Current agent skill index was not refreshed."
                ),
            ]
        )
    except SkillManagementError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error managing skill: {exc}"
