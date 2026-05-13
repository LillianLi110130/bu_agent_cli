"""Standalone teammate process for filesystem-backed agent teams."""

from __future__ import annotations

import argparse
import asyncio
import signal
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agent_core.agent.events import FinalResponseEvent
from agent_core.team.messaging import TeamMessenger
from agent_core.team.models import TeamMessage
from agent_core.team.models import TeamTask, utc_now_iso
from agent_core.team.protocol import (
    MODEL_CONTEXT_MESSAGE_TYPES,
    normalize_message_type,
)
from agent_core.team.runtime import TeamRuntime
from agent_core.team.store import TeamStore
from agent_core.team.task_board import TaskBoard
from cli.team.factory import create_team_member_agent
from cli.team.heartbeat import TeamHeartbeatHook
from cli.team.inbox import (
    TeamInboxAttachmentHook,
    TeamInboxBuffer,
    format_team_messages_for_context,
)
from cli.team.auto_prompt import TEAM_LANGUAGE_RULES
from cli.team.event_log import TeamAgentEventLogHook
from cli.session_runtime import CLISessionRuntime

CreateAgentFactory = Callable[..., tuple[Any, Any, Any]]
CANCELLED_BY_USER_RESPONSE = "[Cancelled by user]"


class TeamWorker:
    def __init__(
        self,
        *,
        teams_root: Path,
        team_id: str,
        member_id: str,
        agent_type: str,
        workspace: Path,
        poll_interval: float = 1.0,
        create_agent_factory: CreateAgentFactory | None = None,
    ) -> None:
        self.store = TeamStore(teams_root)
        self.team_id = team_id
        self.member_id = member_id
        self.agent_type = agent_type
        self.workspace = workspace
        self.poll_interval = poll_interval
        self._create_agent_factory = create_agent_factory or self._default_create_agent
        self._stopping = False
        self.team_dir = self.store.team_dir(team_id)
        self.inbox_buffer = TeamInboxBuffer()
        self.messenger = TeamMessenger(
            team_id=team_id,
            team_dir=self.team_dir,
            append_event=lambda event_type, actor, payload: self.store.append_event(
                team_id,
                event_type,
                actor=actor,
                payload=payload,
            ),
        )
        self.task_board = TaskBoard(self.team_dir)
        self._heartbeat_status: str | None = None
        self._heartbeat_extra: dict[str, Any] = {}
        self._last_non_idle_status: str | None = None
        self._idle_notification_sent = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runtime_tasks: set[asyncio.Task] = set()

    async def run(self) -> None:
        try:
            self._loop = asyncio.get_running_loop()
            agent, ctx = self._build_member_agent()
            agent.register_hook(
                TeamInboxAttachmentHook(
                    buffer=self.inbox_buffer,
                    member_id=self.member_id,
                )
            )
            agent.register_hook(
                TeamHeartbeatHook(
                    write_heartbeat=self._write_heartbeat,
                    get_state=self._get_heartbeat_state,
                )
            )
            agent.register_hook(
                TeamAgentEventLogHook(
                    team_id=self.team_id,
                    member_id=self.member_id,
                    event_log_path=self.team_dir / "sessions" / self.member_id / "events.jsonl",
                )
            )
            agent.bind_session_runtime(self._create_member_session_runtime(ctx))
            self.store.ensure_mailbox(self.team_id, self.member_id)
            self._install_signal_handlers()
            self._send_to_lead("started", f"{self.member_id} started as {self.agent_type}")

            self._runtime_tasks = {
                asyncio.create_task(self._message_pump(), name=f"{self.member_id}:message_pump"),
                asyncio.create_task(self._work_loop(agent), name=f"{self.member_id}:work_loop"),
            }
            await asyncio.gather(*self._runtime_tasks)
        except asyncio.CancelledError:
            self._stopping = True
        except Exception as exc:
            self._write_heartbeat("failed", error=str(exc))
            self.store.mark_member_status(self.team_id, self.member_id, "failed")
            self._send_to_lead(
                "worker_failed",
                str(exc),
                metadata={"error": str(exc)},
            )
            raise
        finally:
            self._cancel_runtime_tasks()

        self._write_heartbeat("stopped")
        self.store.mark_member_status(self.team_id, self.member_id, "stopped")
        self._send_to_lead("stopped", f"{self.member_id} stopped")

    def _build_member_agent(self):
        agent, ctx, _runtime = self._create_agent_factory(
            None,
            self.workspace,
            agent_type=self.agent_type,
            team_prompt=self._build_team_system_prompt(),
        )
        ctx.team_runtime = TeamRuntime(
            teams_root=self.store.teams_root,
            workspace_root=self.workspace,
        )
        return agent, ctx

    def _build_team_system_prompt(self) -> str:
        team_context = self._build_team_context_prompt()
        return (
            "## Agent Team Runtime\n\n"
            "You are a teammate in a multi-process TgAgent team. The primary CLI process "
            "is the team lead and is responsible for coordinating final user-facing output.\n\n"
            f"- Team ID: {self.team_id}\n"
            f"- Member ID: {self.member_id}\n"
            f"- Workspace: {self.workspace}\n\n"
            f"{team_context}\n\n"
            f"{TEAM_LANGUAGE_RULES}\n\n"
            "Team rules:\n"
            "- Work on tasks claimed from the shared team task board.\n"
            "- You may coordinate with other teammates by sending `message` via `team_send_message` "
            "when it directly helps your assigned task.\n"
            "- Keep cross-teammate coordination narrow and factual. Do not reassign work, create "
            "tasks, alter dependencies, or become the orchestrator; ask the lead for those decisions.\n"
            "- Prefer reporting important coordination decisions back to the lead so the primary "
            "CLI can keep the user-facing plan coherent.\n"
            "- Treat `message` and `clarification_response` mailbox messages as coordination "
            "instructions that enter your model context.\n"
            "- `task_assignment`, `task_update`, `message_ack`, `status_check`, and "
            "`shutdown_request` are runtime protocol messages handled outside normal task context.\n"
            "- Do not call delegate or delegate_parallel, and do not spawn other agents.\n"
            "- Skills are read-only for teammates: you may list or view skills, but you must "
            "not create, edit, patch, manage, or persist skills.\n"
            "- Do not run skill review or attempt to save lessons learned as skills. Report "
            "reusable lessons to the team lead instead.\n"
            "- If a skill asks for user interaction, validation, approval, or clarification, "
            "send a message to the team lead with the exact question using type "
            "`clarification_request` and wait; "
            "do not ask the end user directly.\n"
            "- Use message type `clarification_request` for these blocker questions. Include "
            "metadata with `question`, `options` when useful, `recommended` when you have a "
            "safe default, `blocking: true`, and `task_id` when the question is task-specific.\n"
            "- If a skill suggests writing planning/design files that are outside your assigned "
            "task, ask the team lead first.\n"
            "- Return concise task results intended for the team lead to aggregate.\n"
            "- Keep file changes within the assigned task and write scope."
        )

    def _build_team_context_prompt(self) -> str:
        try:
            config = self.store.load_config(self.team_id)
            members_path = self.team_dir / "members.json"
        except Exception:
            return (
                "Team context:\n"
                "- Lead: lead\n"
                "- Members registry: unavailable at startup"
            )

        return "\n".join(
            [
                "Team context:",
                f"- Goal: {config.goal}",
                "- Lead: lead",
                f"- You: {self.member_id} (agent_type={self.agent_type})",
                f"- Members registry: {members_path}",
                "- Read the members registry when you need the latest teammate list, "
                "agent types, process ids, or statuses.",
            ]
        )

    async def _message_pump(self) -> None:
        while not self._stopping:
            await self._receive_messages_once()
            await asyncio.sleep(self.poll_interval)

    async def _receive_messages_once(self) -> bool:
        handled = False
        for message in self.messenger.receive(self.member_id):
            message_type = normalize_message_type(message.type)
            if message_type == "shutdown_request":
                self._ack_message(message)
                self._stopping = True
                self._cancel_runtime_tasks()
                return True
            if message_type == "message_ack":
                handled = True
                continue
            if message_type == "task_assignment":
                self._ack_message(message)
                self._reset_idle_notification()
                self._write_heartbeat(
                    "idle",
                    heartbeat_reason="task_assignment",
                    task_id=message.metadata.get("task_id"),
                )
                handled = True
                continue
            if message_type == "task_update":
                self._ack_message(message)
                self._reset_idle_notification()
                self._write_heartbeat(
                    "idle",
                    heartbeat_reason="task_update",
                    task_id=message.metadata.get("task_id"),
                )
                handled = True
                continue
            if message_type == "status_check":
                self._ack_message(message)
                self._send_to_lead(
                    "status_response",
                    "status",
                    metadata={
                        "heartbeat": self.store.read_heartbeat(self.team_id, self.member_id),
                    },
                )
                handled = True
                continue

            if message_type in MODEL_CONTEXT_MESSAGE_TYPES:
                self._reset_idle_notification()
                await self.inbox_buffer.push(message)
            else:
                self._reset_idle_notification()
                await self.inbox_buffer.push(message)
            handled = True
            self._ack_message(message)
        pending = await self.inbox_buffer.count()
        if pending:
            self._last_non_idle_status = "message_pending"
            self._write_heartbeat("message_pending", pending=pending)
        return handled

    def _ack_message(self, message: TeamMessage) -> None:
        if normalize_message_type(message.type) == "message_ack":
            return
        if message.sender == self.member_id:
            return
        self.send_message(
            recipient=message.sender,
            body=f"Received message: {message.message_id}",
            type="message_ack",
            metadata={"message_id": message.message_id},
        )

    async def _work_loop(self, agent) -> None:
        while not self._stopping:
            if await self.inbox_buffer.has_messages():
                await self._submit_pending_messages(agent)
                continue

            task = self.task_board.claim_next(self.member_id)
            if task is not None:
                self._reset_idle_notification()
                await self._run_task(agent, task)
                continue

            self._mark_idle(reason="no_claimable_task")
            await asyncio.sleep(self.poll_interval)

    async def _submit_pending_messages(self, agent) -> None:
        messages = await self.inbox_buffer.drain()
        if not messages:
            return
        self._set_heartbeat_state("processing_messages", pending=0)
        prompt = format_team_messages_for_context(
            messages,
            member_id=self.member_id,
            idle=True,
        )
        try:
            result = await self._query_agent_stream(agent, prompt)
            self._send_to_lead(
                "message",
                result,
                metadata={
                    "message_ids": [message.message_id for message in messages],
                    "status": "processed",
                },
            )
        except Exception as exc:
            self._send_to_lead(
                "message",
                str(exc),
                metadata={
                    "message_ids": [message.message_id for message in messages],
                    "status": "blocked",
                    "error": str(exc),
                },
            )
        finally:
            self._clear_heartbeat_state()

    async def _run_task(self, agent, task: TeamTask) -> None:
        self._set_heartbeat_state("working", task_id=task.task_id)
        prompt = self._build_task_prompt(task)
        try:
            result = await self._query_agent_stream(agent, prompt)
            if result.strip() == CANCELLED_BY_USER_RESPONSE:
                self.task_board.block_task(
                    task.task_id,
                    self.member_id,
                    "Task was cancelled by user",
                )
                self._send_to_lead(
                    "task_blocked_notification",
                    "Task was cancelled by user",
                    metadata={
                        "task_id": task.task_id,
                        "title": task.title,
                        "cancelled": True,
                    },
                )
                return
            self.task_board.complete_task(task.task_id, self.member_id, result)
            self._send_to_lead(
                "task_done_notification",
                result,
                metadata={"task_id": task.task_id, "title": task.title},
            )
        except Exception as exc:
            self.task_board.block_task(task.task_id, self.member_id, str(exc))
            self._send_to_lead(
                "task_blocked_notification",
                str(exc),
                metadata={"task_id": task.task_id, "title": task.title},
            )
        finally:
            self._clear_heartbeat_state()

    def _build_task_prompt(self, task: TeamTask) -> str:
        scope = "\n".join(f"- {item}" for item in task.write_scope) or "(none)"
        return (
            "You are a teammate in a multi-process agent team.\n"
            f"Team member: {self.member_id}\n"
            f"Task ID: {task.task_id}\n"
            f"Title: {task.title}\n\n"
            "Task description:\n"
            f"{task.description}\n\n"
            "Write scope:\n"
            f"{scope}\n\n"
            f"{TEAM_LANGUAGE_RULES}\n\n"
            "Complete the task directly. If you need user interaction, approval, validation, "
            "or clarification, send a `clarification_request` message to the team lead with "
            "the exact question, options, recommendation, blocking=true, and this task_id. "
            "Do not ask the end user directly. Return a concise result for the team lead."
        )

    async def _query_agent_stream(self, agent, prompt: str) -> str:
        final_response = ""
        async for event in agent.query_stream(prompt):
            if isinstance(event, FinalResponseEvent):
                final_response = event.content
        return final_response

    def _send_to_lead(
        self,
        type: str,
        body: str,
        *,
        metadata: dict | None = None,
    ) -> None:
        self.send_message(
            recipient="lead",
            body=body,
            type=type,
            metadata=metadata,
        )

    def send_message(
        self,
        *,
        recipient: str,
        body: str,
        type: str = "message",
        metadata: dict | None = None,
    ) -> None:
        self.messenger.deliver(
            sender=self.member_id,
            recipient=recipient,
            type=type,
            body=body,
            metadata=metadata,
        )

    def _write_heartbeat(self, status: str, **extra) -> None:
        payload = {
            "team_id": self.team_id,
            "member_id": self.member_id,
            "status": status,
            "updated_at": utc_now_iso(),
            **extra,
        }
        self.store.write_heartbeat(self.team_id, self.member_id, payload)

    def _set_heartbeat_state(self, status: str, **extra: Any) -> None:
        if status != "idle":
            self._last_non_idle_status = status
        self._heartbeat_status = status
        self._heartbeat_extra = dict(extra)
        self._write_heartbeat(status, heartbeat_reason="state_entered", **self._heartbeat_extra)

    def _clear_heartbeat_state(self) -> None:
        self._heartbeat_status = None
        self._heartbeat_extra = {}

    def _get_heartbeat_state(self) -> tuple[str | None, dict[str, Any]]:
        return self._heartbeat_status, dict(self._heartbeat_extra)

    def _reset_idle_notification(self) -> None:
        self._idle_notification_sent = False

    def _mark_idle(self, *, reason: str) -> None:
        previous_status = self._last_non_idle_status or "started"
        self._write_heartbeat("idle", heartbeat_reason=reason)
        if self._idle_notification_sent:
            return
        self._idle_notification_sent = True
        self._send_to_lead(
            "idle_notification",
            f"{self.member_id} is idle and standing by.",
            metadata={
                "reason": reason,
                "previous_status": previous_status,
            },
        )

    def _install_signal_handlers(self) -> None:
        def stop(_signum, _frame) -> None:
            self._stopping = True
            self._cancel_runtime_tasks()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

    def _cancel_runtime_tasks(self) -> None:
        tasks = [task for task in self._runtime_tasks if not task.done()]
        if not tasks:
            return
        loop = self._loop
        for task in tasks:
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(task.cancel)
            else:
                task.cancel()

    def _create_member_session_runtime(self, ctx) -> CLISessionRuntime:
        now = datetime.now().astimezone()
        member_session_dir = (self.team_dir / "sessions" / self.member_id).resolve()
        checkpoints_dir = member_session_dir / "checkpoints"
        artifacts_dir = member_session_dir / "artifacts"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        ctx.add_allowed_dir(member_session_dir)
        ctx.add_runtime_managed_dir(checkpoints_dir)
        ctx.add_runtime_managed_dir(artifacts_dir)

        runtime = CLISessionRuntime(
            session_id=ctx.session_id,
            root_dir=member_session_dir,
            sessions_dir=self.team_dir / "sessions",
            rollout_dir=member_session_dir,
            checkpoints_dir=checkpoints_dir,
            artifacts_dir=artifacts_dir,
            working_state_path=member_session_dir / "working_state.json",
            meta_path=member_session_dir / "meta.json",
            started_at=now.isoformat(timespec="seconds"),
            last_active_at=now.isoformat(timespec="seconds"),
        )
        runtime.write_meta()
        return runtime

    @staticmethod
    def _default_create_agent(
        model: str | None,
        workspace: Path,
        *,
        agent_type: str | None,
        team_prompt: str,
    ):
        return create_team_member_agent(
            model=model,
            root_dir=workspace,
            agent_type=agent_type,
            team_prompt=team_prompt,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one team member process.")
    parser.add_argument("--teams-root", required=True)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--member-id", required=True)
    parser.add_argument("--agent-type", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    return parser.parse_args()


async def amain() -> None:
    args = parse_args()
    worker = TeamWorker(
        teams_root=Path(args.teams_root),
        team_id=args.team_id,
        member_id=args.member_id,
        agent_type=args.agent_type,
        workspace=Path(args.workspace),
        poll_interval=args.poll_interval,
    )
    await worker.run()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
