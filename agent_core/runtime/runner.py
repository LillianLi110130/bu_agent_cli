"""Unified runner that turns a call request into an executing Agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_core.agent import Agent
from agent_core.llm.base import BaseChatModel
from agent_core.llm.factory import create_chat_model
from agent_core.llm.messages import AssistantMessage, BaseMessage, ToolMessage

if TYPE_CHECKING:
    from agent_core.task.local_agent_task import SubagentCallRequest


_DELEGATION_TOOL_NAMES = {"delegate", "delegate_parallel"}


class AgentCallRunner:
    """Build and execute a concrete agent instance for one call request."""

    def __init__(
        self,
        *,
        registry: Any,
        all_tools: list[Any],
        skill_registry: Any | None = None,
    ) -> None:
        self._registry = registry
        self._all_tools = list(all_tools)
        self._skill_registry = skill_registry

    def build_execution(
        self,
        *,
        parent_agent: Agent,
        request: "SubagentCallRequest",
    ) -> tuple[Agent, str]:
        if request.subagent_type:
            child = self.build_named_agent(
                parent_agent=parent_agent,
                agent_type=request.subagent_type,
                runtime_role="subagent",
            )
            return child, request.prompt
        return self._build_fork_agent(parent_agent=parent_agent, request=request)

    def build_named_agent(
        self,
        *,
        parent_agent: Agent,
        agent_type: str,
        runtime_role: str = "subagent",
    ) -> Agent:
        config = self._registry.get_config(agent_type)
        if config is None:
            raise ValueError(f"Agent '{agent_type}' not found")

        llm = self._resolve_named_llm(parent_agent.llm, config.model)
        system_prompt = config.system_prompt
        skills_text = self._load_skill_text(config.skills)
        if skills_text:
            system_prompt = f"{system_prompt}\n\n## Loaded Skills\n\n{skills_text}"

        child = Agent(
            llm=llm,
            tools=self._without_delegation_tools(self._all_tools),
            system_prompt=system_prompt,
            max_iterations=config.max_turns or parent_agent.max_iterations,
            dependency_overrides=dict(parent_agent.dependency_overrides or {}),
            agent_config=config,
            hooks=list(parent_agent.hooks),
            runtime_role=runtime_role,
        )
        child.dependency_overrides = self._bind_current_agent(child.dependency_overrides, child)
        return child

    def _build_fork_agent(
        self,
        *,
        parent_agent: Agent,
        request: "SubagentCallRequest",
    ) -> tuple[Agent, str]:
        child = Agent(
            llm=parent_agent.llm,
            tools=self._without_delegation_tools(parent_agent.tools),
            system_prompt=parent_agent.system_prompt,
            max_iterations=parent_agent.max_iterations,
            tool_choice=parent_agent.tool_choice,
            compaction=parent_agent.compaction,
            dependency_overrides=dict(parent_agent.dependency_overrides or {}),
            hooks=list(parent_agent.hooks),
            runtime_role="subagent",
            is_fork_child=True,
        )
        child.dependency_overrides = self._bind_current_agent(child.dependency_overrides, child)
        child.load_history(self._build_fork_messages(parent_agent))
        return child, self._build_fork_child_message(request)

    def _without_delegation_tools(self, tools: list[Any]) -> list[Any]:
        return [tool for tool in tools if getattr(tool, "name", None) not in _DELEGATION_TOOL_NAMES]

    def _bind_current_agent(self, overrides: dict | None, child: Agent) -> dict:
        from tools.sandbox import get_current_agent

        resolved = dict(overrides or {})
        resolved[get_current_agent] = lambda: child
        return resolved

    def _resolve_named_llm(
        self,
        parent_llm: BaseChatModel,
        configured_model: str | None,
    ) -> BaseChatModel:
        """Resolve the effective model for a named subagent definition."""
        normalized = (configured_model or "").strip()
        if not normalized or normalized.lower() == "inherit":
            return parent_llm
        return self._resolve_llm(parent_llm, normalized)

    def _build_fork_messages(self, parent_agent: Agent) -> list[BaseMessage]:
        """Copy the parent conversation history for a forked child execution."""
        copied_messages = [message.model_copy(deep=True) for message in parent_agent.messages]
        return self._strip_incomplete_trailing_tool_block(copied_messages)

    def _strip_incomplete_trailing_tool_block(
        self,
        messages: list[BaseMessage],
    ) -> list[BaseMessage]:
        """Remove a trailing assistant tool-call block when its tool results are incomplete.

        Forked children inherit the parent's in-flight history snapshot. If the parent
        is currently executing a tool, the copied history may end with an assistant
        message that declares `tool_calls` but does not yet have the matching trailing
        tool messages. OpenAI-compatible backends reject this shape, so trim or
        downgrade that trailing block before the forked child starts.
        """
        if not messages:
            return messages

        last_index = len(messages) - 1
        trailing_tool_messages: list[ToolMessage] = []
        while last_index >= 0 and isinstance(messages[last_index], ToolMessage):
            trailing_tool_messages.append(messages[last_index])
            last_index -= 1

        if last_index < 0:
            return messages

        assistant_message = messages[last_index]
        if not isinstance(assistant_message, AssistantMessage) or not assistant_message.tool_calls:
            return messages

        expected_tool_ids = {tool_call.id for tool_call in assistant_message.tool_calls}
        received_tool_ids = {
            tool_message.tool_call_id
            for tool_message in trailing_tool_messages
            if tool_message.tool_call_id in expected_tool_ids
        }

        if received_tool_ids == expected_tool_ids:
            return messages

        sanitized = list(messages[:last_index])
        if assistant_message.content is not None or assistant_message.refusal is not None:
            sanitized.append(
                AssistantMessage(
                    content=assistant_message.content,
                    name=assistant_message.name,
                    refusal=assistant_message.refusal,
                    tool_calls=None,
                    cache=assistant_message.cache,
                )
            )
        return sanitized

    def _build_fork_child_message(self, request: "SubagentCallRequest") -> str:
        """Build the directive injected as the fork child's first message."""
        description = request.description.strip()
        prompt = request.prompt.strip()
        guidance = [
            "You are a forked child agent running inside a delegated execution.",
            "",
            "Rules:",
            "1. You are not the primary agent.",
            "2. Execute the assigned task directly; do not ask the user follow-up questions unless you are blocked.",
            "3. Do not call `delegate` or `delegate_parallel`, and do not create additional subagents.",
            "4. Do not narrate between tool calls unless necessary for progress.",
            "5. Prefer acting with tools over discussing plans.",
            "6. When finished, return a concise result summary only.",
            "",
            f"Task summary: {description}",
            "Task:",
            prompt,
        ]
        return "\n".join(guidance).strip()

    def _resolve_llm(
        self,
        parent_llm: BaseChatModel,
        override_model: str | None,
    ) -> BaseChatModel:
        if not override_model:
            return parent_llm
        return create_chat_model(override_model, fallback_llm=parent_llm)

    def _load_skill_text(self, skill_names: list[str]) -> str:
        if not skill_names or self._skill_registry is None:
            return ""
        parts: list[str] = []
        for skill_name in skill_names:
            skill = self._skill_registry.get(skill_name)
            if skill is None:
                continue
            try:
                parts.append(f"### @{skill.name}\n{skill.load_content().strip()}")
            except Exception:
                continue
        return "\n\n".join(parts)
