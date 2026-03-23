"""Worker runtime factory for default and test-only execution modes."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from bu_agent_sdk import Agent
from bu_agent_sdk.bootstrap.agent_factory import create_agent
from bu_agent_sdk.llm.messages import BaseMessage
from bu_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeCompletionChunk
from tools import SandboxContext

logger = logging.getLogger("cli.worker.runtime_factory")


class EchoLLM:
    """Minimal echo-only LLM used by worker e2e tests."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.model = "echo-worker"

    @property
    def provider(self) -> str:
        return "echo"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools: list[Any] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion:
        del tools, tool_choice, kwargs
        latest_user_message = _extract_latest_user_message(messages)
        return ChatInvokeCompletion(content=f"{self.prefix}{latest_user_message}")

    async def astream(
        self,
        messages: list[BaseMessage],
        tools: list[Any] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatInvokeCompletionChunk]:
        del tools, tool_choice, kwargs
        latest_user_message = _extract_latest_user_message(messages)
        yield ChatInvokeCompletionChunk(delta=f"{self.prefix}{latest_user_message}")


def create_worker_runtime(
    model: str | None,
    root_dir: Path | str | None = None,
) -> tuple[Any, SandboxContext]:
    """Create the worker runtime, allowing test-only echo mode via environment."""
    runtime_mode = os.getenv("BU_AGENT_WORKER_RUNTIME_MODE", "default").strip().lower()

    if runtime_mode in {"", "default", "agent"}:
        logger.info(f"Using default worker runtime mode={runtime_mode or 'default'}")
        return create_agent(model=model, root_dir=root_dir)

    if runtime_mode == "echo":
        echo_prefix = os.getenv("BU_AGENT_WORKER_ECHO_PREFIX", "echo:")
        logger.info(f"Using echo worker runtime with prefix={echo_prefix}")
        return _create_echo_runtime(prefix=echo_prefix, root_dir=root_dir)

    raise ValueError(f"Unsupported worker runtime mode: {runtime_mode}")


def _create_echo_runtime(
    prefix: str,
    root_dir: Path | str | None,
) -> tuple[Agent, SandboxContext]:
    """Create a deterministic echo runtime for cross-process e2e tests."""
    context = SandboxContext.create(root_dir)
    agent = Agent(
        llm=EchoLLM(prefix=prefix),
        tools=[],
        system_prompt="Echo worker runtime",
    )
    return agent, context


def _extract_latest_user_message(messages: list[BaseMessage]) -> str:
    """Return the latest user-visible text content from the conversation."""
    for message in reversed(messages):
        if getattr(message, "role", None) != "user":
            continue
        if hasattr(message, "text"):
            return str(message.text)
        content = getattr(message, "content", "")
        return str(content)
    return ""
