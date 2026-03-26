from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agent_core.agent import Agent, build_default_approval_policy
from agent_core.agent.events import ToolCallEvent
from agent_core.agent.runtime_events import ToolCallRequested
from agent_core.agent.tool_args import (
    parse_tool_arguments_for_display,
    parse_tool_arguments_for_execution,
)
from agent_core.llm.messages import BaseMessage, Function, ToolCall
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from agent_core.tools.decorator import tool


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

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        if False:
            yield ChatInvokeCompletionChunk()


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (r'{"file_path":"D:\中文\文件.txt"}', r"D:\中文\文件.txt"),
        (r'{"file_path":"C:\temp\中文.txt"}', r"C:\temp\中文.txt"),
    ],
)
def test_parse_tool_arguments_for_execution_repairs_windows_style_paths(
    arguments: str,
    expected: str,
):
    parsed = parse_tool_arguments_for_execution(arguments)

    assert parsed["file_path"] == expected


def test_parse_tool_arguments_for_display_preserves_valid_json_escapes():
    parsed = parse_tool_arguments_for_display(r'{"text":"line1\nline2","path":"D:\中文\文件.txt"}')

    assert parsed["text"] == "line1\nline2"
    assert parsed["path"] == r"D:\中文\文件.txt"


@pytest.mark.anyio
async def test_agent_query_executes_tool_with_repaired_windows_style_path():
    seen: list[str] = []

    @tool("Echo file path")
    async def echo_path(file_path: str) -> str:
        seen.append(file_path)
        return file_path

    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=Function(
                            name="echo_path",
                            arguments=r'{"file_path":"D:\中文\文件.txt"}',
                        ),
                    )
                ]
            ),
            ChatInvokeCompletion(content="done"),
        ]
    )
    agent = Agent(llm=llm, tools=[echo_path])

    result = await agent.query("echo the file path")

    assert result == "done"
    assert seen == [r"D:\中文\文件.txt"]


@pytest.mark.anyio
async def test_query_stream_emits_repaired_tool_call_arguments():
    @tool("Echo file path")
    async def echo_path(file_path: str) -> str:
        return file_path

    llm = FakeLLM(
        [
            ChatInvokeCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        function=Function(
                            name="echo_path",
                            arguments=r'{"file_path":"D:\中文\文件.txt"}',
                        ),
                    )
                ]
            ),
            ChatInvokeCompletion(content="done"),
        ]
    )
    agent = Agent(llm=llm, tools=[echo_path])

    events = [event async for event in agent.query_stream("echo the file path")]
    tool_events = [event for event in events if isinstance(event, ToolCallEvent)]

    assert len(tool_events) == 1
    assert tool_events[0].args["file_path"] == r"D:\中文\文件.txt"


def test_approval_policy_uses_repaired_tool_arguments_for_command_preview():
    policy = build_default_approval_policy("primary")
    event = ToolCallRequested(
        tool_call=ToolCall(
            id="call-1",
            function=Function(
                name="bash",
                arguments=r'{"command":"type D:\中文\文件.txt"}',
            ),
        ),
        iteration=1,
    )

    request = policy(event, None)  # type: ignore[arg-type]

    assert request is not None
    assert request.arguments["command"] == r"type D:\中文\文件.txt"
    assert request.command_preview == r"type D:\中文\文件.txt"
