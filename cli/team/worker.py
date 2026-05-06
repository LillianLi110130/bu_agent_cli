"""Standalone teammate process for filesystem-backed agent teams."""

from __future__ import annotations

import argparse
import asyncio
import signal
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agent_core.team.messaging import TeamMessenger
from agent_core.team.models import TeamMessage
from agent_core.team.models import TeamTask, utc_now_iso
from agent_core.team.store import TeamStore
from agent_core.team.task_board import TaskBoard
from cli.team.factory import create_team_member_agent
from cli.team.inbox import (
    TeamInboxAttachmentHook,
    TeamInboxBuffer,
    format_team_messages_for_context,
)
from cli.session_runtime import CLISessionRuntime

CreateAgentFactory = Callable[..., tuple[Any, Any, Any]]


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

    async def run(self) -> None:
        agent, ctx = self._build_member_agent()
        agent.register_hook(
            TeamInboxAttachmentHook(
                buffer=self.inbox_buffer,
                member_id=self.member_id,
            )
        )
        agent.bind_session_runtime(self._create_member_session_runtime(ctx))
        self.store.ensure_mailbox(self.team_id, self.member_id)
        self._install_signal_handlers()
        self._send_to_lead("started", f"{self.member_id} started as {self.agent_type}")

        await asyncio.gather(
            self._message_pump(),
            self._work_loop(agent),
        )

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
        return agent, ctx

    def _build_team_system_prompt(self) -> str:
        return (
            "## Agent Team Runtime\n\n"
            "You are a teammate in a multi-process TgAgent team. The primary CLI process "
            "is the team lead and is responsible for coordinating final user-facing output.\n\n"
            f"- Team ID: {self.team_id}\n"
            f"- Member ID: {self.member_id}\n"
            f"- Workspace: {self.workspace}\n\n"
            "Team rules:\n"
            "- Work on tasks claimed from the shared team task board.\n"
            "- Treat mailbox messages from the lead as coordination instructions.\n"
            "- Do not call delegate or delegate_parallel, and do not spawn other agents.\n"
            "- Return concise task results intended for the team lead to aggregate.\n"
            "- Keep file changes within the assigned task and write scope."
        )

    async def _message_pump(self) -> None:
        while not self._stopping:
            await self._receive_messages_once()
            await asyncio.sleep(self.poll_interval)

    async def _receive_messages_once(self) -> bool:
        handled = False
        for message in self.messenger.receive(self.member_id):
            if message.type == "shutdown":
                self._stopping = True
                return True
            await self.inbox_buffer.push(message)
            handled = True
            self._ack_message(message)
        if handled:
            self._write_heartbeat("message_pending", pending=await self.inbox_buffer.count())
        return handled

    def _ack_message(self, message: TeamMessage) -> None:
        if message.type in {"message_ack", "note_ack"}:
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
            self._write_heartbeat("running")

            if await self.inbox_buffer.has_messages():
                await self._submit_pending_messages(agent)
                continue

            task = self.task_board.claim_next(self.member_id)
            if task is not None:
                await self._run_task(agent, task)
                continue

            await asyncio.sleep(self.poll_interval)

    async def _submit_pending_messages(self, agent) -> None:
        messages = await self.inbox_buffer.drain()
        if not messages:
            return
        self._write_heartbeat("processing_messages", pending=0)
        prompt = format_team_messages_for_context(
            messages,
            member_id=self.member_id,
            idle=True,
        )
        try:
            result = await agent.query(prompt)
            self._send_to_lead(
                "messages_processed",
                result,
                metadata={"message_ids": [message.message_id for message in messages]},
            )
        except Exception as exc:
            self._send_to_lead(
                "messages_blocked",
                str(exc),
                metadata={"message_ids": [message.message_id for message in messages]},
            )

    async def _run_task(self, agent, task: TeamTask) -> None:
        self._write_heartbeat("working", task_id=task.task_id)
        prompt = self._build_task_prompt(task)
        try:
            result = await agent.query(prompt)
            self.task_board.complete_task(task.task_id, self.member_id, result)
            self._send_to_lead(
                "task_completed",
                result,
                metadata={"task_id": task.task_id, "title": task.title},
            )
        except Exception as exc:
            self.task_board.block_task(task.task_id, self.member_id, str(exc))
            self._send_to_lead(
                "task_blocked",
                str(exc),
                metadata={"task_id": task.task_id, "title": task.title},
            )

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
            "Complete the task directly. Return a concise result for the team lead."
        )

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
        type: str = "note",
        metadata: dict | None = None,
        reply_to: str | None = None,
    ) -> None:
        self.messenger.send_message(
            sender=self.member_id,
            recipient=recipient,
            type=type,
            body=body,
            metadata=metadata,
            reply_to=reply_to,
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

    def _install_signal_handlers(self) -> None:
        def stop(_signum, _frame) -> None:
            self._stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

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
