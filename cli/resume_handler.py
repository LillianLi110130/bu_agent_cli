"""Presentation and picker state for the CLI /resume command."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markup import escape as rich_escape

from cli.session_store import (
    CLISessionStore,
    SessionContextSnapshot,
    SessionMeta,
    SessionStoreError,
    workspace_identity,
)


@dataclass(frozen=True, slots=True)
class ResumePickResult:
    handled: bool
    selected_session_id: str | None = None


class ResumeSlashHandler:
    """Handle /resume list rendering, picker input, and resume result display."""

    def __init__(
        self,
        *,
        store: CLISessionStore | None,
        console: Console,
        workspace_dir: Path,
    ) -> None:
        self._store = store
        self._console = console
        self._workspace_dir = workspace_dir
        self._pick_active = False
        self._pick_options: list[SessionMeta] = []

    @property
    def pick_active(self) -> bool:
        return self._pick_active

    def bind_console(self, console: Console) -> None:
        self._console = console

    def bind_store(self, store: CLISessionStore | None) -> None:
        self._store = store

    def start_pick_mode(self, *, current_session_id: str) -> None:
        """List resumable sessions for the current workspace and enter picker mode."""
        if self._store is None:
            self._console.print("[yellow]会话历史存储不可用，无法使用 /resume。[/yellow]")
            return

        _workspace_root, workspace_key = workspace_identity(self._workspace_dir)
        sessions = self._store.list_sessions(
            workspace_key=workspace_key,
            exclude_session_id=current_session_id,
            limit=20,
        )
        if not sessions:
            self._console.print("[yellow]当前工作区暂无可恢复的其他会话。[/yellow]")
            return

        self._pick_options = sessions
        self._pick_active = True

        self._console.print()
        self._console.print("[bold cyan]选择要恢复的会话：[/bold cyan]")
        self._console.print()
        for idx, session in enumerate(sessions, 1):
            title = session.title or "Untitled session"
            title = title.replace("\n", " ").strip()
            if len(title) > 40:
                title = f"{title[:40]}..."
            age = _format_relative_time(session.updated_at)
            self._console.print(
                f"  {idx}. [cyan]{rich_escape(title)}[/cyan]   "
                f"[dim]{session.message_count} messages   {age}[/dim]"
            )
        self._console.print("[dim]输入编号后回车即可恢复，输入 q 可取消。[/dim]")

    def handle_pick_input(self, user_input: str) -> ResumePickResult:
        """Handle one line of input while in numbered resume-pick mode."""
        if not self._pick_active:
            return ResumePickResult(handled=False)

        value = user_input.strip()
        if not value:
            self._console.print("[dim]请输入编号，或输入 q 取消。[/dim]")
            return ResumePickResult(handled=True)

        if value.lower() in {"q", "quit", "cancel", "exit"}:
            self.clear_pick()
            self._console.print("[yellow]已取消会话恢复。[/yellow]")
            return ResumePickResult(handled=True)

        if not value.isdigit():
            self._console.print("[red]选择无效，请输入编号，或输入 q 取消。[/red]")
            return ResumePickResult(handled=True)

        index = int(value)
        if index < 1 or index > len(self._pick_options):
            max_index = len(self._pick_options)
            self._console.print(
                f"[red]选择超出范围，请输入 1-{max_index}，或输入 q 取消。[/red]"
            )
            return ResumePickResult(handled=True)

        session = self._pick_options[index - 1]
        return ResumePickResult(handled=True, selected_session_id=session.id)

    def clear_pick(self) -> None:
        self._pick_active = False
        self._pick_options = []

    def print_resume_result(
        self,
        meta: SessionMeta,
        snapshot: SessionContextSnapshot,
    ) -> None:
        title = rich_escape(meta.title or "Untitled session")
        self._console.print()
        self._console.print(f"[green]已恢复：[/green]{title}")
        if snapshot.compacted:
            self._console.print("[yellow]该会话曾触发上下文压缩，已恢复压缩后的可继续上下文。[/yellow]")
            return

        if self._store is None:
            return
        try:
            rounds = self._store.recent_user_assistant_rounds(meta.id, limit=10)
        except SessionStoreError:
            self._console.print("[yellow]历史预览数据损坏，已跳过展示。[/yellow]")
            return

        if not rounds:
            self._console.print("[dim]该会话暂无可展示的历史对话。[/dim]")
            return

        self._console.print()
        self._console.print("[bold cyan]最近 10 轮对话：[/bold cyan]")
        for item in rounds:
            user_text = _truncate_preview_text(item.user)
            assistant_text = _truncate_preview_text(item.assistant or "(no assistant text)")
            self._console.print(f"[bold]User:[/bold] {rich_escape(user_text)}")
            self._console.print(f"[bold]Assistant:[/bold] {rich_escape(assistant_text)}")
            self._console.print()


def _truncate_preview_text(text: str, *, max_chars: int = 1000) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars]}..."


def _format_relative_time(timestamp: float) -> str:
    elapsed = max(0, int(time.time() - timestamp))
    if elapsed < 60:
        return "just now"
    minutes = elapsed // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days == 1:
        return "yesterday"
    return f"{days}d ago"
