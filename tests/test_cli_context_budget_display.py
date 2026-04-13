from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import cli.app as app_module
from agent_core import Agent
from agent_core.agent.hooks import HookContext
from agent_core.agent.runtime_events import ContextMaintenanceRequested
from agent_core.agent.runtime_state import AgentRunState
from agent_core.llm.messages import BaseMessage, Function, ToolCall, UserMessage
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk, ChatInvokeUsage
from agent_core.tools.decorator import tool
from cli.app import TGAgentCLI, _CLIContextBudgetHook, _CLIContextBudgetSnapshot
from cli.slash_commands import SlashCommandRegistry
from tools import SandboxContext


class _DummyPrompter:
    def __init__(self, console):
        self.console = console


class FakeLLM:
    def __init__(self, responses: list[ChatInvokeCompletion], *, model: str = "fake-model"):
        self.responses = list(responses)
        self.invocations: list[list[BaseMessage]] = []
        self.model = model

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> ChatInvokeCompletion:
        del tools, tool_choice, kwargs
        self.invocations.append(list(messages))
        if not self.responses:
            raise AssertionError("No scripted response left for FakeLLM")
        return self.responses.pop(0)

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        del messages, tools, tool_choice, kwargs
        if False:
            yield ChatInvokeCompletionChunk()


def _usage(prompt_tokens: int, completion_tokens: int = 10) -> ChatInvokeUsage:
    return ChatInvokeUsage(
        prompt_tokens=prompt_tokens,
        prompt_cached_tokens=0,
        prompt_cache_creation_tokens=0,
        prompt_image_tokens=0,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _make_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent: Agent,
) -> TGAgentCLI:
    monkeypatch.setattr(app_module, "InteractivePrompter", _DummyPrompter)
    context = SandboxContext.create(tmp_path)
    cli = TGAgentCLI(
        agent=agent,
        context=context,
        slash_registry=SlashCommandRegistry(),
    )
    monkeypatch.setattr(cli, "_start_loading", lambda message="思考中": None)
    monkeypatch.setattr(cli, "_stop_loading", lambda loading: None)
    monkeypatch.setattr(cli, "_maybe_inject_agents_md", lambda: None)
    return cli


@pytest.mark.asyncio
async def test_context_budget_toolbar_defaults_to_100_left(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent(llm=FakeLLM([]), tools=[])
    cli = _make_cli(tmp_path, monkeypatch, agent)

    await cli._refresh_empty_context_budget_display()

    toolbar = cli._render_context_budget_toolbar()
    assert toolbar == "上下文 100% left · 0k/128.0k tokens · fake-model"


@pytest.mark.asyncio
async def test_reset_command_restores_context_budget_toolbar_to_100_left(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent(llm=FakeLLM([]), tools=[])
    cli = _make_cli(tmp_path, monkeypatch, agent)
    cli._last_context_budget = _CLIContextBudgetSnapshot(
        model="fake-model",
        estimated_tokens=100,
        context_limit=1000,
        remaining_tokens=900,
        context_utilization=0.1,
        remaining_ratio=0.9,
        message_count=1,
    )

    handled = await cli._handle_slash_command("/reset")

    assert handled is True
    assert cli._render_context_budget_toolbar() == (
        "上下文 100% left · 0k/128.0k tokens · fake-model"
    )


@pytest.mark.asyncio
async def test_switch_model_preset_refreshes_context_budget_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent(llm=FakeLLM([]), tools=[])
    cli = _make_cli(tmp_path, monkeypatch, agent)
    await cli._refresh_empty_context_budget_display()
    assert "0k/128.0k tokens · fake-model" in cli._render_context_budget_toolbar()
    assert agent._context._budget_engine is not None
    agent._context._budget_engine._context_limit_cache["small-model"] = 64_000
    printed: list[_CLIContextBudgetSnapshot] = []
    monkeypatch.setattr(
        cli,
        "_print_context_budget_status",
        lambda snapshot: printed.append(snapshot),
    )

    async def fake_switch_model_preset(
        preset_name: str,
        *,
        manual: bool = True,
        auto_state=None,
    ) -> bool:
        del preset_name, manual, auto_state
        agent.llm.model = "small-model"
        return True

    monkeypatch.setattr(
        cli._model_switch_service,
        "switch_model_preset",
        fake_switch_model_preset,
    )

    switched = await cli._switch_model_preset("small")

    assert switched is True
    assert cli._render_context_budget_toolbar() == (
        "上下文 100% left · 0k/64.0k tokens · small-model"
    )
    assert printed
    assert printed[-1].model == "small-model"
    assert printed[-1].context_limit == 64_000


@pytest.mark.asyncio
async def test_run_agent_updates_context_budget_after_llm_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent(
        llm=FakeLLM([ChatInvokeCompletion(content="done", usage=_usage(100))]),
        tools=[],
    )
    cli = _make_cli(tmp_path, monkeypatch, agent)
    printed: list[_CLIContextBudgetSnapshot] = []
    monkeypatch.setattr(
        cli,
        "_print_context_budget_status",
        lambda snapshot: printed.append(snapshot),
    )

    final_content = await cli._run_agent("hello", has_image=False)

    assert final_content == "done"
    assert cli._last_context_budget is not None
    assert any(snapshot.trigger == "post_llm_response" for snapshot in printed)
    assert "fake-model" in cli._render_context_budget_toolbar()
    assert "left" in cli._render_context_budget_toolbar()


@pytest.mark.asyncio
async def test_run_agent_updates_context_budget_after_each_tool_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @tool("Return a payload")
    async def echo_tool() -> str:
        return "tool payload"

    agent = Agent(
        llm=FakeLLM(
            [
                ChatInvokeCompletion(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            function=Function(name="echo_tool", arguments="{}"),
                        )
                    ],
                    usage=_usage(80),
                ),
                ChatInvokeCompletion(content="done", usage=_usage(110)),
            ]
        ),
        tools=[echo_tool],
    )
    cli = _make_cli(tmp_path, monkeypatch, agent)
    printed: list[_CLIContextBudgetSnapshot] = []
    monkeypatch.setattr(
        cli,
        "_print_context_budget_status",
        lambda snapshot: printed.append(snapshot),
    )

    final_content = await cli._run_agent("hello", has_image=False)

    assert final_content == "done"
    assert any(snapshot.trigger == "post_tool_result" for snapshot in printed)


@pytest.mark.asyncio
async def test_context_budget_hook_preserves_post_compaction_trigger(
    tmp_path: Path,
) -> None:
    agent = Agent(
        llm=FakeLLM([ChatInvokeCompletion(content="done")]),
        tools=[],
    )
    agent._context.add_message(UserMessage(content="hello"))
    assert agent._context._budget_engine is not None
    agent._context._budget_engine.note_trigger("post_compaction")
    state = AgentRunState(query_mode="stream", max_iterations=5)
    ctx = HookContext(agent=agent, state=state)
    hook = _CLIContextBudgetHook()

    await hook.after_event(
        ContextMaintenanceRequested(
            response=ChatInvokeCompletion(content="done"),
            iteration=1,
        ),
        ctx,
        [],
    )

    assert len(ctx.ui_events) == 1
    snapshot = ctx.ui_events[0]
    assert isinstance(snapshot, _CLIContextBudgetSnapshot)
    assert snapshot.trigger == "post_compaction"
