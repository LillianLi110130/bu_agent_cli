import json

import pytest

from bu_agent_sdk.agent.compaction import CompactionConfig
from bu_agent_sdk.agent.events import FinalResponseEvent
from bu_agent_sdk.agent.service import Agent
from bu_agent_sdk.hooks import HookManager, create_hook_api
from bu_agent_sdk.llm.messages import Function, ToolCall
from bu_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from bu_agent_sdk.tools.decorator import tool


class FakeLLM:
    def __init__(self, responses: list[ChatInvokeCompletion], model: str = "fake-model"):
        self.model = model
        self._responses = responses
        self._index = 0

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    @property
    def model_name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
        if self._index >= len(self._responses):
            raise AssertionError("FakeLLM has no more responses configured")
        response = self._responses[self._index]
        self._index += 1
        return response


@pytest.mark.asyncio
async def test_hook_manager_priority_and_tolerance():
    manager = HookManager()
    calls: list[str] = []

    async def low_priority(**kwargs):
        calls.append("low")

    async def middle_priority(**kwargs):
        calls.append("middle")
        raise RuntimeError("boom")

    async def high_priority(**kwargs):
        calls.append("high")

    manager.register("on_model_start", low_priority, priority=50)
    manager.register("on_model_start", middle_priority, priority=20)
    manager.register("on_model_start", high_priority, priority=10)

    # Should not raise when one hook fails.
    await manager.trigger("on_model_start")

    assert calls == ["high", "middle", "low"]


@pytest.mark.asyncio
async def test_hook_decorator_api_registers_hooks():
    manager = HookManager()
    hook = create_hook_api(manager)
    calls: list[str] = []

    @hook.on_session_start(priority=5)
    async def session_start(session_id: str, message, agent):
        calls.append(f"start:{session_id}")

    await manager.trigger(
        "on_session_start",
        session_id="session_1",
        message="hello",
        agent=None,
    )

    assert calls == ["start:session_1"]


@pytest.mark.asyncio
async def test_query_stream_triggers_all_lifecycle_hooks():
    @tool("Echo tool")
    async def echo(value: str) -> str:
        return f"echo:{value}"

    usage_first = ChatInvokeUsage(
        prompt_tokens=3,
        prompt_cached_tokens=0,
        prompt_cache_creation_tokens=0,
        prompt_image_tokens=0,
        completion_tokens=4,
        total_tokens=7,
    )
    usage_second = ChatInvokeUsage(
        prompt_tokens=2,
        prompt_cached_tokens=0,
        prompt_cache_creation_tokens=0,
        prompt_image_tokens=0,
        completion_tokens=2,
        total_tokens=4,
    )

    llm = FakeLLM(
        responses=[
            ChatInvokeCompletion(
                content="I will call a tool",
                thinking="thinking-1",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=Function(
                            name="echo",
                            arguments=json.dumps({"value": "ping"}),
                        ),
                    )
                ],
                usage=usage_first,
            ),
            ChatInvokeCompletion(
                content="all done",
                thinking=None,
                tool_calls=[],
                usage=usage_second,
            ),
        ]
    )

    manager = HookManager()
    hook = create_hook_api(manager)
    hook_events: list[str] = []
    session_end_payload: dict | None = None
    tool_end_payload: dict | None = None

    @hook.on_session_start()
    async def on_session_start(session_id: str, message, agent):
        hook_events.append("on_session_start")
        assert session_id.startswith("session_")
        assert message == "hello"

    @hook.on_model_start()
    async def on_model_start(model_name: str, messages, tool_count: int, iteration: int, agent):
        hook_events.append("on_model_start")
        assert model_name == "fake-model"
        assert tool_count == 1
        assert iteration in (1, 2)
        assert messages

    @hook.on_model_end()
    async def on_model_end(
        model_name: str,
        content: str,
        thinking: str | None,
        tool_calls,
        usage: dict,
        iteration: int,
        agent,
    ):
        hook_events.append("on_model_end")
        assert model_name == "fake-model"
        assert isinstance(tool_calls, list)
        assert isinstance(usage, dict)
        if iteration == 1:
            assert content == "I will call a tool"
            assert thinking == "thinking-1"
            assert usage["total_tokens"] == 7
        if iteration == 2:
            assert content == "all done"

    @hook.on_tool_start()
    async def on_tool_start(tool_name: str, args: dict, tool_call_id: str, iteration: int, agent):
        hook_events.append("on_tool_start")
        assert tool_name == "echo"
        assert args == {"value": "ping"}
        assert tool_call_id == "call_1"
        assert iteration == 1

    @hook.on_tool_end()
    async def on_tool_end(
        tool_name: str,
        args: dict,
        result: str,
        is_error: bool,
        tool_call_id: str,
        duration_ms: float,
        iteration: int,
        agent,
    ):
        nonlocal tool_end_payload
        hook_events.append("on_tool_end")
        tool_end_payload = {
            "tool_name": tool_name,
            "args": args,
            "result": result,
            "is_error": is_error,
            "tool_call_id": tool_call_id,
            "duration_ms": duration_ms,
            "iteration": iteration,
        }

    @hook.on_session_end()
    async def on_session_end(
        session_id: str,
        final_response: str,
        total_tokens: int,
        duration_ms: float,
        agent,
    ):
        nonlocal session_end_payload
        hook_events.append("on_session_end")
        session_end_payload = {
            "session_id": session_id,
            "final_response": final_response,
            "total_tokens": total_tokens,
            "duration_ms": duration_ms,
        }

    agent = Agent(
        llm=llm,
        tools=[echo],
        compaction=CompactionConfig(enabled=False),
        hook_manager=manager,
    )

    streamed_events = [event async for event in agent.query_stream("hello")]

    assert isinstance(streamed_events[-1], FinalResponseEvent)
    assert streamed_events[-1].content == "all done"

    assert hook_events == [
        "on_session_start",
        "on_model_start",
        "on_model_end",
        "on_tool_start",
        "on_tool_end",
        "on_model_start",
        "on_model_end",
        "on_session_end",
    ]

    assert tool_end_payload is not None
    assert tool_end_payload["tool_name"] == "echo"
    assert tool_end_payload["args"] == {"value": "ping"}
    assert tool_end_payload["result"] == "echo:ping"
    assert tool_end_payload["is_error"] is False
    assert tool_end_payload["tool_call_id"] == "call_1"
    assert tool_end_payload["iteration"] == 1
    assert tool_end_payload["duration_ms"] >= 0

    assert session_end_payload is not None
    assert session_end_payload["session_id"].startswith("session_")
    assert session_end_payload["final_response"] == "all done"
    assert session_end_payload["total_tokens"] == 11
    assert session_end_payload["duration_ms"] >= 0
