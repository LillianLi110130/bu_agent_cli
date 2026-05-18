from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from agent_core.agent import Agent
from agent_core.llm.messages import BaseMessage, ToolMessage
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from tools import ALL_TOOLS
from tools.sandbox import SandboxContext, get_sandbox_context
from tools.todos import (
    ACTIVE_TODO_SNAPSHOT_HEADER,
    get_todo_store,
    hydrate_todo_store_from_messages,
    todo,
)


class FakeLLM:
    def __init__(self, responses: list[ChatInvokeCompletion]):
        self.responses = list(responses)
        self.invocations: list[list[BaseMessage]] = []
        self.model = "fake-model"

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
        self.invocations.append(list(messages))
        if not self.responses:
            raise AssertionError("No scripted response left for FakeLLM")
        return self.responses.pop(0)

    async def ainvoke_streaming(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> ChatInvokeCompletion:
        return await self.ainvoke(messages, tools=tools, tool_choice=tool_choice, **kwargs)

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        if False:
            yield ChatInvokeCompletionChunk()


async def _execute_todo_async(ctx: SandboxContext, **kwargs) -> dict:
    raw = await todo.execute(_overrides={get_sandbox_context: lambda: ctx}, **kwargs)
    return json.loads(raw)


@pytest.mark.asyncio
async def test_todo_read_empty_returns_full_snapshot(tmp_path):
    ctx = SandboxContext.create(tmp_path)

    result = await _execute_todo_async(ctx)

    assert result == {
        "todos": [],
        "summary": {
            "total": 0,
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "cancelled": 0,
        },
        "warnings": [],
    }


@pytest.mark.asyncio
async def test_todo_replace_normalizes_and_returns_warnings(tmp_path):
    ctx = SandboxContext.create(tmp_path)

    result = await _execute_todo_async(
        ctx,
        todos=[
            {"content": "Inspect code", "status": "pending"},
            {"id": "2", "content": "", "status": "bad-status"},
            {"id": "3", "content": "Run tests", "status": "in_progress"},
            {"id": "4", "content": "Write docs", "status": "in_progress"},
        ],
    )

    assert result["todos"] == [
        {"id": "1", "content": "Inspect code", "status": "pending"},
        {"id": "2", "content": "(no description)", "status": "pending"},
        {"id": "3", "content": "Run tests", "status": "pending"},
        {"id": "4", "content": "Write docs", "status": "in_progress"},
    ]
    assert result["summary"] == {
        "total": 4,
        "pending": 3,
        "in_progress": 1,
        "completed": 0,
        "cancelled": 0,
    }
    assert any("assigned id 1" in warning for warning in result["warnings"])
    assert any("invalid status" in warning for warning in result["warnings"])
    assert any("Multiple in_progress" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_todo_merge_updates_without_losing_content_and_appends(tmp_path):
    ctx = SandboxContext.create(tmp_path)
    await _execute_todo_async(
        ctx,
        todos=[
            {"id": "1", "content": "Inspect code", "status": "in_progress"},
            {"id": "2", "content": "Run tests", "status": "pending"},
        ],
    )

    result = await _execute_todo_async(
        ctx,
        todos=[
            {"id": "1", "status": "completed"},
            {"content": "missing id", "status": "pending"},
            {"id": "3", "content": "Write summary", "status": "pending"},
        ],
        merge=True,
    )

    assert result["todos"] == [
        {"id": "1", "content": "Inspect code", "status": "completed"},
        {"id": "2", "content": "Run tests", "status": "pending"},
        {"id": "3", "content": "Write summary", "status": "pending"},
    ]
    assert any("without id" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_finish_guard_uses_todo_store(tmp_path):
    ctx = SandboxContext.create(tmp_path)
    get_todo_store(ctx.session_id).write(
        [{"id": "1", "content": "Run tests", "status": "pending"}],
        merge=False,
    )
    llm = FakeLLM(
        [
            ChatInvokeCompletion(content="premature"),
            ChatInvokeCompletion(content="final"),
        ]
    )
    agent = Agent(llm=llm, tools=[], use_streaming=False)
    setattr(agent, "_sandbox_context", ctx)

    result = await agent.query("do work")

    assert result == "final"
    assert len(llm.invocations) == 2
    assert any("unfinished todo items" in message.text for message in agent.messages)


@pytest.mark.asyncio
async def test_compaction_injects_active_todo_snapshot(tmp_path):
    ctx = SandboxContext.create(tmp_path)
    get_todo_store(ctx.session_id).write(
        [
            {"id": "1", "content": "Done task", "status": "completed"},
            {"id": "2", "content": "Current task", "status": "in_progress"},
            {"id": "3", "content": "Future task", "status": "pending"},
            {"id": "4", "content": "Obsolete task", "status": "cancelled"},
        ],
        merge=False,
    )
    agent = Agent(llm=FakeLLM([]), tools=[])
    setattr(agent, "_sandbox_context", ctx)

    async def fake_maintain_budget(self, llm, *, trigger=None):
        return SimpleNamespace(trigger="post_compaction")

    agent._context.maintain_budget = MethodType(fake_maintain_budget, agent._context)

    await agent._maintain_context_from_budget(trigger="post_response")

    injected = agent.messages[-1].text
    assert injected.startswith(ACTIVE_TODO_SNAPSHOT_HEADER)
    assert "Current task" in injected
    assert "Future task" in injected
    assert "Done task" not in injected
    assert "Obsolete task" not in injected


def test_hydrate_todo_store_from_latest_tool_result(tmp_path):
    ctx = SandboxContext.create(tmp_path)
    older = ToolMessage(
        tool_call_id="call-old",
        tool_name="todo",
        content=json.dumps(
            {
                "todos": [{"id": "1", "content": "Old", "status": "pending"}],
                "summary": {},
                "warnings": [],
            }
        ),
    )
    newer = ToolMessage(
        tool_call_id="call-new",
        tool_name="todo",
        content=json.dumps(
            {
                "todos": [{"id": "2", "content": "New", "status": "in_progress"}],
                "summary": {},
                "warnings": [],
            }
        ),
    )

    hydrated = hydrate_todo_store_from_messages(ctx.session_id, [older, newer])

    assert hydrated is True
    assert get_todo_store(ctx.session_id).read() == [
        {"id": "2", "content": "New", "status": "in_progress"}
    ]


def test_old_todo_tools_are_not_registered():
    tool_names = {tool.name for tool in ALL_TOOLS}

    assert "todo" in tool_names
    assert "todo_read" not in tool_names
    assert "todo_write" not in tool_names


def test_todo_tool_schema_documents_usage_contract():
    definition = todo.definition

    assert "complex tasks with 3+ steps" in definition.description
    assert "merge=false" in definition.description
    assert "merge=true" in definition.description
    assert "pending, in_progress, completed, cancelled" in definition.description
    assert "full current list" in definition.description


def test_system_prompt_documents_when_to_use_todo():
    prompt = Path("agent_core/prompts/system.md").read_text(encoding="utf-8")

    assert "## Todo 任务跟踪" in prompt
    assert "预计需要 3 个以上步骤" in prompt
    assert "不要为了形式化而使用 todo" in prompt
