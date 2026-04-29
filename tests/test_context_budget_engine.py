from __future__ import annotations

from collections.abc import AsyncIterator
from types import MethodType, SimpleNamespace

import pytest

from agent_core.agent import Agent, CompactionConfig, ContextBudgetEngine
from agent_core.agent.budget import BudgetAssessment
from agent_core.agent.context import ContextManager
from agent_core.llm.messages import (
    AssistantMessage,
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    ToolMessage,
    UserMessage,
)
from agent_core.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk, ChatInvokeUsage


class FakeLLM:
    def __init__(
        self,
        *,
        responses: list[ChatInvokeCompletion] | None = None,
        stream_chunks: list[ChatInvokeCompletionChunk] | None = None,
        model: str = "fake-model",
    ) -> None:
        self.responses = list(responses or [])
        self.stream_chunks = list(stream_chunks or [])
        self.invocations: list[list[BaseMessage]] = []
        self.stream_invocations: list[list[BaseMessage]] = []
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
        return await self.ainvoke(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    async def astream(
        self,
        messages: list[BaseMessage],
        tools=None,
        tool_choice=None,
        **kwargs,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        self.stream_invocations.append(list(messages))
        for chunk in list(self.stream_chunks):
            yield chunk


def _usage(prompt_tokens: int, completion_tokens: int = 10) -> ChatInvokeUsage:
    return ChatInvokeUsage(
        prompt_tokens=prompt_tokens,
        prompt_cached_tokens=0,
        prompt_cache_creation_tokens=0,
        prompt_image_tokens=0,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _has_image_part(messages: list[BaseMessage]) -> bool:
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        if any(getattr(part, "type", None) == "image_url" for part in content):
            return True
    return False


@pytest.mark.asyncio
async def test_context_budget_engine_uses_real_prompt_baseline_plus_appended_messages():
    engine = ContextBudgetEngine(config=CompactionConfig())
    prompt_messages = [UserMessage(content="analyze this file")]

    engine.record_usage(model="baseline-model", messages=prompt_messages, usage=_usage(120))

    current_messages = [
        *prompt_messages,
        AssistantMessage(content="I will inspect the code"),
        ToolMessage(tool_call_id="call-1", tool_name="read", content="long tool output"),
    ]
    assessment = await engine.assess(model="baseline-model", messages=current_messages)

    incremental = engine.estimate_tokens_for_messages(current_messages[len(prompt_messages) :])
    assert assessment.baseline_prompt_tokens == 120
    assert assessment.incremental_tokens == incremental
    assert assessment.estimated_tokens == 120 + incremental
    assert assessment.token_estimate_source == "provider_baseline_plus_delta"


@pytest.mark.asyncio
async def test_context_budget_engine_reestimates_full_history_when_model_changes():
    engine = ContextBudgetEngine(config=CompactionConfig())
    prompt_messages = [UserMessage(content="inspect repo")]

    engine.record_usage(model="baseline-model", messages=prompt_messages, usage=_usage(120))

    current_messages = [
        *prompt_messages,
        AssistantMessage(content="Looking into it"),
        ToolMessage(tool_call_id="call-1", tool_name="read", content="tool output payload"),
    ]

    assessment = await engine.assess(model="other-model", messages=current_messages)

    full_estimate = engine.estimate_tokens_for_messages(current_messages)
    assert assessment.baseline_prompt_tokens == 120
    assert assessment.incremental_tokens == full_estimate
    assert assessment.estimated_tokens == full_estimate
    assert assessment.token_estimate_source == "local_full"


def test_context_budget_engine_does_not_count_base64_image_payload_as_text_tokens():
    engine = ContextBudgetEngine(config=CompactionConfig())
    image_data = "A" * 240_000
    message = UserMessage(
        content=[
            ContentPartTextParam(text="describe"),
            ContentPartImageParam(
                image_url=ImageURL(
                    url=f"data:image/png;base64,{image_data}",
                    media_type="image/png",
                )
            ),
        ]
    )

    raw_json_estimate = len(message.model_dump_json()) // 4 + 4
    estimate = engine.estimate_tokens_for_messages([message])

    assert raw_json_estimate > 50_000
    assert estimate < 5_000
    assert estimate >= 2_048


@pytest.mark.asyncio
async def test_preflight_compaction_preserves_trailing_image_message_outside_text_compaction():
    llm = FakeLLM(
        responses=[
            ChatInvokeCompletion(
                content="<summary>older context</summary><working_state>{}</working_state>"
            )
        ]
    )
    agent = Agent(llm=llm, tools=[])
    image_message = UserMessage(
        content=[
            ContentPartTextParam(text="what is this image"),
            ContentPartImageParam(
                image_url=ImageURL(
                    url="data:image/png;base64,AAAA",
                    media_type="image/png",
                )
            ),
        ]
    )
    agent.load_history([UserMessage(content="older context"), image_message])

    compacted = await agent._compact_messages_now()

    assert compacted is True
    assert len(llm.invocations) == 1
    assert not _has_image_part(llm.invocations[0])
    assert agent.messages[-1] == image_message


@pytest.mark.asyncio
async def test_context_maintain_budget_does_not_run_sliding_window_before_compaction():
    context = ContextManager(
        messages=[
            UserMessage(content="first user"),
            AssistantMessage(content="assistant"),
            ToolMessage(tool_call_id="call-1", tool_name="read", content="tool result"),
        ],
        sliding_window_messages=1,
    )
    llm = SimpleNamespace(model="fake-model")
    slide_calls: list[int] = []
    compact_calls: list[str | None] = []

    async def fake_assess_budget(self, *, model: str, usage=None, trigger=None):
        return BudgetAssessment(
            model=model,
            context_limit=1000,
            warn_threshold=100,
            compact_threshold=800,
            hard_threshold=920,
            baseline_prompt_tokens=0,
            incremental_tokens=150,
            estimated_tokens=150,
            message_count=len(self._messages),
            warn_threshold_ratio=0.1,
            compact_threshold_ratio=0.8,
            hard_threshold_ratio=0.92,
            threshold_utilization=150 / 800,
            context_utilization=150 / 1000,
            trigger=trigger,
        )

    async def fake_slide(self, keep_count: int, llm, pin_roles=("system", "developer"), buffer: int = 10):
        del keep_count, llm, pin_roles, buffer
        slide_calls.append(1)
        return True

    async def fake_compact(self, llm, usage=None, *, trigger=None):
        del llm, usage
        compact_calls.append(trigger)
        return False

    context.assess_budget = MethodType(fake_assess_budget, context)
    context.apply_sliding_window_by_messages = MethodType(fake_slide, context)
    context.check_and_compact = MethodType(fake_compact, context)

    assessment = await context.maintain_budget(llm, trigger="post_response")

    assert assessment.estimated_tokens == 150
    assert slide_calls == []
    assert compact_calls == []


@pytest.mark.asyncio
async def test_query_runtime_loop_uses_unified_budget_maintenance_trigger():
    llm = FakeLLM(responses=[ChatInvokeCompletion(content="done", usage=_usage(64))])
    agent = Agent(llm=llm, tools=[])
    triggers: list[str | None] = []

    async def fake_maintain_budget(self, llm, *, trigger=None):
        triggers.append(trigger)
        return await self.assess_budget(model=llm.model, trigger=trigger)

    agent._context.maintain_budget = MethodType(fake_maintain_budget, agent._context)

    result = await agent.query("hello")

    assert result == "done"
    assert triggers == ["post_response"]


@pytest.mark.asyncio
async def test_query_stream_delta_uses_unified_budget_maintenance_trigger():
    llm = FakeLLM(
        stream_chunks=[
            ChatInvokeCompletionChunk(delta="done"),
            ChatInvokeCompletionChunk(usage=_usage(64)),
        ]
    )
    agent = Agent(llm=llm, tools=[])
    triggers: list[str | None] = []

    async def fake_maintain_budget(self, llm, *, trigger=None):
        triggers.append(trigger)
        return await self.assess_budget(model=llm.model, trigger=trigger)

    agent._context.maintain_budget = MethodType(fake_maintain_budget, agent._context)

    events = [event async for event in agent.query_stream_delta("hello")]

    assert events[-1].content == "done"
    assert triggers == ["post_stream_response"]


@pytest.mark.asyncio
async def test_preflight_model_switch_reuses_budget_engine_estimate(monkeypatch: pytest.MonkeyPatch):
    llm = FakeLLM(model="current-model")
    agent = Agent(llm=llm, tools=[], compaction=CompactionConfig(threshold_ratio=0.8))
    prompt_messages = [UserMessage(content="inspect repo")]
    agent.load_history(prompt_messages)
    agent._context.record_prompt_usage(
        model=agent.llm.model,
        messages=prompt_messages,
        usage=_usage(180),
    )
    agent._context.add_message(AssistantMessage(content="Inspecting"))
    agent._context.add_message(
        ToolMessage(tool_call_id="call-1", tool_name="read", content="tool output payload")
    )

    compact_calls: list[bool] = []

    async def fake_compact_now() -> bool:
        compact_calls.append(True)
        return False

    assert agent._context._budget_engine is not None
    agent._context._budget_engine._context_limit_cache["target-model"] = 220
    agent._compact_messages_now = fake_compact_now  # type: ignore[method-assign]

    preflight = await agent.preflight_model_switch("target-model", utilization_limit=0.5)

    expected_total = agent._context._budget_engine.estimate_tokens_for_messages(agent.messages)
    assert preflight.ok is False
    assert preflight.estimated_tokens == expected_total
    assert preflight.threshold == int(220 * 0.8)
    assert compact_calls == [True]
    assert "automatic compaction failed" in (preflight.reason or "")
