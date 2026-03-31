from __future__ import annotations

from collections.abc import AsyncIterator
import types

import pytest

from agent_core.agent import Agent
from agent_core.agent.service import ModelSwitchPreflightResult
from agent_core.llm.messages import (
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    UserMessage,
)
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from cli.model_switch_service import ModelAutoState, ModelSwitchService


class FakeLLM:
    def __init__(self, model: str = "text-model", base_url: str | None = None):
        self.model = model
        self.base_url = base_url
        self.invocations: list[list[BaseMessage]] = []

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
        return ChatInvokeCompletion(content="done")

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        if False:
            yield ChatInvokeCompletionChunk()


class StubConsole:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, message: str) -> None:
        self.messages.append(message)


def _make_service(
    *,
    current_model: str = "text-model",
    current_base_url: str | None = None,
    presets: dict | None = None,
    console: StubConsole | None = None,
) -> tuple[Agent, ModelSwitchService, StubConsole]:
    agent = Agent(llm=FakeLLM(model=current_model, base_url=current_base_url), tools=[])
    output = console or StubConsole()
    service = ModelSwitchService(
        agent=agent,
        model_presets=presets
        or {
            "text": {"model": "text-model", "api_key_env": "TEST_TEXT_KEY", "vision": False},
            "vision": {"model": "vision-model", "api_key_env": "TEST_VISION_KEY", "vision": True},
        },
        default_model_preset="text",
        auto_vision_preset="vision",
        image_summary_preset="vision",
        console=output,
    )
    return agent, service, output


def _image_part() -> ContentPartImageParam:
    return ContentPartImageParam(
        image_url=ImageURL(url="data:image/png;base64,AAAA", media_type="image/png")
    )


@pytest.mark.asyncio
async def test_prepare_text_model_image_memory_replaces_images_with_text_memory():
    agent, service, _console = _make_service(current_model="vision-model")
    agent.load_history(
        [
            UserMessage(
                content=[
                    ContentPartTextParam(text="请看这个报错"),
                    _image_part(),
                    ContentPartTextParam(text="帮我总结"),
                ]
            )
        ]
    )

    service._resolve_image_summary_llm = lambda: (None, None, "未配置")  # type: ignore[method-assign]

    await service.prepare_text_model_image_memory(manual=True)

    message = agent.messages[0]
    assert isinstance(message, UserMessage)
    assert isinstance(message.content, list)
    assert all(getattr(part, "type", None) != "image_url" for part in message.content)
    image_memory_parts = [
        part.text
        for part in message.content
        if getattr(part, "type", None) == "text" and part.text.startswith("[ImageSummary]")
    ]
    assert len(image_memory_parts) == 1
    assert "未配置可用视觉摘要模型" in image_memory_parts[0]


@pytest.mark.asyncio
async def test_switch_model_preset_restores_context_when_image_memory_preparation_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_TEXT_KEY", "text-key")
    agent, service, console = _make_service(current_model="vision-model")
    original_message = UserMessage(
        content=[
            ContentPartTextParam(text="带图片的请求"),
            _image_part(),
        ]
    )
    agent.load_history([original_message.model_copy(deep=True)])

    async def fail_prepare(*, manual: bool) -> None:
        raise RuntimeError("boom")

    service.prepare_text_model_image_memory = fail_prepare  # type: ignore[method-assign]

    switched = await service.switch_model_preset("text")

    assert switched is False
    assert any("切换前准备图片记忆失败" in message for message in console.messages)
    restored_message = agent.messages[0]
    assert isinstance(restored_message, UserMessage)
    assert isinstance(restored_message.content, list)
    assert any(getattr(part, "type", None) == "image_url" for part in restored_message.content)
    assert agent.llm.model == "vision-model"


@pytest.mark.asyncio
async def test_ensure_model_for_turn_auto_switches_to_vision_and_back():
    _agent, service, console = _make_service(current_model="text-model")
    auto_state = ModelAutoState(sticky_preset="text")
    calls: list[str] = []

    async def fake_switch(
        self,
        preset_name: str,
        *,
        manual: bool = True,
        auto_state: ModelAutoState | None = None,
    ) -> bool:
        calls.append(preset_name)
        if preset_name == "vision":
            self._agent.llm.model = "vision-model"
        else:
            self._agent.llm.model = "text-model"
        return True

    service.switch_model_preset = types.MethodType(fake_switch, service)

    first_turn = await service.ensure_model_for_turn(has_image=True, auto_state=auto_state)
    second_turn = await service.ensure_model_for_turn(has_image=False, auto_state=auto_state)

    assert first_turn is True
    assert second_turn is True
    assert calls == ["vision", "text"]
    assert auto_state.auto_switched is False
    assert auto_state.auto_from_preset is None
    assert any("自动切换到视觉预设：vision" in message for message in console.messages)
    assert any("自动切回文本预设：text" in message for message in console.messages)


@pytest.mark.asyncio
async def test_switch_model_preset_updates_llm_and_manual_auto_state(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_TEXT_KEY", "text-key")
    agent, service, console = _make_service(current_model="vision-model", current_base_url="https://old")
    auto_state = ModelAutoState(sticky_preset="vision", auto_switched=True, auto_from_preset="text")

    class FakeChatOpenAI:
        def __init__(self, *, model: str, api_key: str, base_url: str | None = None):
            self.model = model
            self.api_key = api_key
            self.base_url = base_url

    async def fake_preflight(model: str) -> ModelSwitchPreflightResult:
        return ModelSwitchPreflightResult(
            ok=True,
            target_model=model,
            estimated_tokens=10,
            threshold=100,
            context_limit=1000,
            threshold_utilization=0.1,
            compacted=True,
        )

    monkeypatch.setattr("cli.model_switch_service.ChatOpenAI", FakeChatOpenAI)
    agent.preflight_model_switch = fake_preflight  # type: ignore[method-assign]
    service.prepare_text_model_image_memory = (  # type: ignore[method-assign]
        lambda *, manual: __import__("asyncio").sleep(0)
    )

    switched = await service.switch_model_preset("text", auto_state=auto_state)

    assert switched is True
    assert agent.llm.model == "text-model"
    assert agent.llm.base_url == "https://old"
    assert agent._context._compaction_service is not None
    assert agent._context._compaction_service.llm is agent.llm
    assert auto_state.sticky_preset == "text"
    assert auto_state.auto_switched is False
    assert auto_state.auto_from_preset is None
    assert any("已在切换前压缩上下文" in message for message in console.messages)


def test_set_llm_updates_compaction_service_reference():
    agent, _service, _console = _make_service(current_model="text-model")
    new_llm = FakeLLM(model="replacement-model", base_url="https://example.test")

    agent.set_llm(new_llm)

    assert agent.llm is new_llm
    assert agent._context._compaction_service is not None
    assert agent._context._compaction_service.llm is new_llm
