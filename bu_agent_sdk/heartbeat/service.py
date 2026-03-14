"""Periodic heartbeat service for proactive gateway task execution."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from bu_agent_sdk.bus.events import InboundMessage
from bu_agent_sdk.bus.queue import MessageBus
from bu_agent_sdk.llm.base import ToolDefinition
from bu_agent_sdk.llm.messages import SystemMessage, UserMessage

if TYPE_CHECKING:
    from bu_agent_sdk.llm.base import BaseChatModel

logger = logging.getLogger("bu_agent_sdk.heartbeat")

_HEARTBEAT_TOOL = ToolDefinition(
    name="heartbeat",
    description="Report whether the heartbeat task queue should run now.",
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["skip", "run"],
                "description": "skip = nothing to do, run = tasks should be executed",
            },
            "tasks": {
                "type": "string",
                "description": "Natural-language summary of the tasks to execute when action=run.",
            },
        },
        "required": ["action"],
    },
)


class HeartbeatService:
    """Periodically inspect HEARTBEAT.md and publish work into the bus."""

    def __init__(
        self,
        workspace: Path,
        bus: MessageBus,
        llm: "BaseChatModel",
        get_delivery_target: Callable[[], tuple[str, str] | None],
        interval_seconds: int = 30 * 60,
        enabled: bool = True,
    ) -> None:
        self.workspace = workspace
        self.bus = bus
        self.llm = llm
        self.get_delivery_target = get_delivery_target
        self.interval_seconds = interval_seconds
        self.enabled = enabled
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def heartbeat_file(self) -> Path:
        """Return the HEARTBEAT.md path in the configured workspace."""
        return self.workspace / "HEARTBEAT.md"

    def _read_heartbeat_file(self) -> str | None:
        if not self.heartbeat_file.exists():
            return None
        content = self.heartbeat_file.read_text(encoding="utf-8").strip()
        return content or None

    async def _decide(self, content: str) -> tuple[str, str]:
        """Use the LLM to decide whether the heartbeat should run tasks now."""
        response = await self.llm.ainvoke(
            messages=[
                SystemMessage(
                    content=(
                        "You are a heartbeat agent. "
                        "Always call the heartbeat tool to report your decision."
                    )
                ),
                UserMessage(
                    content=(
                        "Review the following HEARTBEAT.md content and decide "
                        "whether there are tasks to execute now.\n\n"
                        f"{content}"
                    )
                ),
            ],
            tools=[_HEARTBEAT_TOOL],
            tool_choice="required",
        )
        if not response.has_tool_calls:
            return "skip", ""

        arguments = response.tool_calls[0].function.arguments
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("Failed to decode heartbeat tool arguments; skipping tick")
                return "skip", ""
        else:
            parsed_arguments = arguments

        action = parsed_arguments.get("action", "skip")
        tasks = parsed_arguments.get("tasks", "")
        return action, tasks

    async def start(self) -> None:
        """Start the background heartbeat loop."""
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return
        if self._running and self._task is not None and not self._task.done():
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the background heartbeat loop."""
        self._running = False
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run_loop(self) -> None:
        """Run the periodic heartbeat loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_seconds)
                if self._running:
                    await self.trigger_now()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Heartbeat loop error: {exc}")

    async def trigger_now(self) -> bool:
        """Run one heartbeat cycle immediately and publish work if needed."""
        content = self._read_heartbeat_file()
        if not content:
            return False

        action, tasks = await self._decide(content)
        if action != "run" or not tasks:
            return False

        target = self.get_delivery_target()
        if target is None:
            logger.info("Heartbeat skipped because no delivery target is available")
            return False

        channel, chat_id = target
        await self.bus.publish_inbound(
            InboundMessage(
                channel=channel,
                sender_id="heartbeat",
                chat_id=chat_id,
                content=tasks,
                metadata={"heartbeat": True},
                origin="heartbeat",
                session_key_override="heartbeat",
            )
        )
        return True
