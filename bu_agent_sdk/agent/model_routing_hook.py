"""Runtime hook for automatic model routing before LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bu_agent_sdk.agent.hooks import BaseAgentHook, HookAction, HookDecision
from bu_agent_sdk.agent.runtime_events import LLMCallRequested, RuntimeEvent
from bu_agent_sdk.llm.messages import UserMessage


@dataclass
class ModelRoutingHook(BaseAgentHook):
    """Auto-switch models before LLM calls based on current input modality."""

    service: Any = None
    auto_state: Any = None
    priority: int = 15

    async def before_event(
        self,
        event: RuntimeEvent,
        ctx,
    ) -> HookDecision | None:
        if not isinstance(event, LLMCallRequested):
            return None

        has_image = False
        for message in reversed(ctx.agent._context.get_messages()):
            if not isinstance(message, UserMessage):
                continue
            if isinstance(message.content, list):
                has_image = any(getattr(part, "type", None) == "image_url" for part in message.content)
            break

        ok = await self.service.ensure_model_for_turn(
            has_image=has_image,
            auto_state=self.auto_state,
        )
        if ok:
            return None

        return HookDecision(
            action=HookAction.ABORT,
            reason="automatic model routing failed",
        )
