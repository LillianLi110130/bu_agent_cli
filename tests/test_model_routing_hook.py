from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agent_core.agent import Agent, ModelRoutingHook
from agent_core.llm.messages import (
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
)
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk


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


class StubModelSwitchService:
    def __init__(self):
        self.calls: list[bool] = []

    async def ensure_model_for_turn(self, *, has_image: bool, auto_state) -> bool:
        self.calls.append(has_image)
        return True


@pytest.mark.anyio
async def test_model_routing_hook_detects_text_turn():
    service = StubModelSwitchService()
    llm = FakeLLM([ChatInvokeCompletion(content="done")])
    agent = Agent(llm=llm, tools=[])
    agent.register_hook(ModelRoutingHook(service=service, auto_state=object()))

    result = await agent.query("hello")

    assert result == "done"
    assert service.calls == [False]


@pytest.mark.anyio
async def test_model_routing_hook_detects_image_turn():
    service = StubModelSwitchService()
    llm = FakeLLM([ChatInvokeCompletion(content="done")])
    agent = Agent(llm=llm, tools=[])
    agent.register_hook(ModelRoutingHook(service=service, auto_state=object()))

    payload = [
        ContentPartTextParam(text="describe"),
        ContentPartImageParam(
            image_url=ImageURL(url="data:image/png;base64,AAAA", media_type="image/png")
        ),
    ]
    events = [event async for event in agent.query_stream(payload)]

    assert events[-1].content == "done"
    assert service.calls == [True]
