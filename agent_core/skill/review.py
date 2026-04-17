from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from agent_core.agent.hooks import BaseAgentHook
from agent_core.agent.runtime_events import RunFailed, RunFinished, ToolResultReceived
from agent_core.agent.service import Agent
from agent_core.llm.messages import BaseMessage, ToolMessage
from agent_core.skill.runtime_service import SkillRuntimeService
from tools.skills import get_skill_runtime_service, skill_list, skill_manage, skill_view

logger = logging.getLogger("agent_core.skill.review")
_SKILL_MANAGE_SUCCESS_RE = re.compile(r"\ASkill (?P<action>created|patched|edited): (?P<name>.+)")
_SKILL_SUPPORT_SUCCESS_RE = re.compile(
    r"\ASkill support file (?P<action>written|removed): (?P<name>.+)"
)

SKILL_REVIEW_PROMPT = """Review the conversation above and decide whether any reusable procedural
knowledge should be saved as a skill.

Save or update a skill only if one of these is true:
- The task required a non-trivial workflow, multiple tool calls, debugging, or trial and error.
- A specific sequence of steps was discovered that would help solve similar tasks later.
- The user corrected the approach or expressed a reusable preference about how this task type
  should be handled.
- A tool, API, library, platform, or environment had a pitfall, workaround, command, or constraint
  worth preserving.
- An existing skill was used and found to be incomplete, stale, or wrong.

Do not save:
- One-off task progress.
- Simple facts better suited for memory.
- Temporary TODO state.
- Generic advice that is already obvious.

You may only create or update user-level skills under ~/.tg_agent/skills.
Never modify builtin skills, plugin skills, workspace skills, or project-private skills.
Never delete skills.
Do not create draft skills.

If a relevant user-level skill exists, update it with the new lesson. Prefer targeted patches over
full rewrites.
If no relevant user-level skill exists and the workflow is reusable, create a concise new
user-level skill.
If nothing is worth saving, respond exactly: Nothing to save."""


@dataclass(frozen=True, slots=True)
class SkillReviewChange:
    action: str
    name: str
    path: str | None = None


@dataclass(frozen=True, slots=True)
class SkillReviewResult:
    final_response: str
    changes: list[SkillReviewChange] = field(default_factory=list)


class SkillReviewRunner:
    """Run the hidden skill review agent after complex primary turns."""

    def __init__(
        self,
        *,
        service: SkillRuntimeService,
        max_iterations: int = 8,
    ) -> None:
        self.service = service
        self.max_iterations = max_iterations

    async def run(
        self,
        main_agent: Agent,
        messages_snapshot: list[BaseMessage],
    ) -> SkillReviewResult:
        review_agent = Agent(
            llm=main_agent.llm,
            tools=[skill_list, skill_view, skill_manage],
            system_prompt=main_agent.system_prompt,
            max_iterations=self.max_iterations,
            dependency_overrides={get_skill_runtime_service: lambda: self.service},
            mode="skill_review",
            hooks=[],
        )
        review_agent.load_history(list(messages_snapshot))
        final_response = await review_agent.query(SKILL_REVIEW_PROMPT)
        return SkillReviewResult(
            final_response=final_response,
            changes=_extract_skill_manage_changes(review_agent.messages),
        )


def _extract_skill_manage_changes(messages: list[BaseMessage]) -> list[SkillReviewChange]:
    changes: list[SkillReviewChange] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.tool_name != "skill_manage" or message.is_error:
            continue
        content = message.text.strip()
        if not content or content.startswith("Error:"):
            continue
        change = _parse_skill_manage_success(content)
        if change is not None:
            changes.append(change)
    return changes


def _parse_skill_manage_success(content: str) -> SkillReviewChange | None:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return None

    match = _SKILL_MANAGE_SUCCESS_RE.match(lines[0])
    support_match = _SKILL_SUPPORT_SUCCESS_RE.match(lines[0])
    if match is None and support_match is None:
        return None

    active_match = match or support_match
    action = active_match.group("action")
    name = active_match.group("name").strip()
    path = None
    for line in lines[1:]:
        if line.startswith("Path:"):
            path = line.split(":", 1)[1].strip()
            break
    return SkillReviewChange(action=action, name=name, path=path)


@dataclass(kw_only=True)
class SkillReviewHook(BaseAgentHook):
    """Trigger background skill review after enough tool work in the primary loop."""

    runner: SkillReviewRunner
    interval: int = 10
    enabled: bool = True
    on_changes: Callable[[list[SkillReviewChange]], None] | None = None
    on_nothing_to_save: Callable[[], None] | None = None
    priority: int = 980

    _iters_since_skill: int = 0
    _task: asyncio.Task | None = None

    async def after_event(self, event, ctx, emitted_events):
        del emitted_events
        if not self.enabled:
            return None

        if isinstance(event, ToolResultReceived):
            self._record_tool_result(event)
            return None

        if isinstance(event, RunFailed):
            return None

        if isinstance(event, RunFinished):
            self._maybe_start_review(event, ctx.agent)
        return None

    def _record_tool_result(self, event: ToolResultReceived) -> None:
        tool_name = event.tool_call.function.name
        if tool_name == "skill_manage" and not event.tool_result.is_error:
            self._iters_since_skill = 0
            return
        self._iters_since_skill += 1

    def _maybe_start_review(self, event: RunFinished, agent: Agent) -> None:
        if not self._should_trigger(event, agent):
            return

        self._iters_since_skill = 0
        snapshot = agent.messages
        task = asyncio.create_task(self.runner.run(agent, snapshot))
        task.add_done_callback(self._handle_background_result)
        self._task = task

    def _should_trigger(self, event: RunFinished, agent: Agent) -> bool:
        if getattr(agent, "mode", "primary") != "primary":
            return False
        if "skill_manage" not in getattr(agent, "_tool_map", {}):
            return False
        if not (event.final_response or "").strip():
            return False
        if event.final_response.strip() == "[Cancelled by user]":
            return False
        if self._iters_since_skill < max(1, self.interval):
            return False
        if self._task is not None and not self._task.done():
            return False
        return True

    def _handle_background_result(self, task: asyncio.Task) -> None:
        try:
            result = task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Background skill review failed")
            return
        logger.debug("Background skill review completed: %s", result.final_response)
        if result.changes and self.on_changes is not None:
            self.on_changes(result.changes)
            return
        if _is_nothing_to_save(result.final_response) and self.on_nothing_to_save is not None:
            self.on_nothing_to_save()


def _is_nothing_to_save(final_response: str) -> bool:
    return final_response.strip() == "Nothing to save."
