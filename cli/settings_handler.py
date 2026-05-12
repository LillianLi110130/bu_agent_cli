"""Interactive handler for the /settings slash command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markup import escape as rich_escape

from agent_core.runtime_paths import (
    clear_default_workspace,
    get_default_workspace,
    set_default_workspace,
)


@dataclass(frozen=True, slots=True)
class SettingsInputResult:
    handled: bool


class SettingsSlashHandler:
    """处理 /settings 的用户级设置交互菜单。"""

    _STATE_ROOT = "root"
    _STATE_WORKSPACE = "workspace"
    _STATE_WORKSPACE_EDIT = "workspace_edit"
    _STATE_WORKSPACE_CLEAR_CONFIRM = "workspace_clear_confirm"

    def __init__(self, *, console: Console, workspace_dir: Path) -> None:
        self._console = console
        self._workspace_dir = workspace_dir
        self._state: str | None = None

    @property
    def active(self) -> bool:
        return self._state is not None

    def bind_console(self, console: Console) -> None:
        self._console = console

    def start(self) -> None:
        self._state = self._STATE_ROOT
        self._render_root_menu()

    def handle_input(self, user_input: str) -> SettingsInputResult:
        if self._state is None:
            return SettingsInputResult(handled=False)

        value = user_input.strip()
        if self._state == self._STATE_ROOT:
            return self._handle_root_input(value)
        if self._state == self._STATE_WORKSPACE:
            return self._handle_workspace_input(value)
        if self._state == self._STATE_WORKSPACE_EDIT:
            return self._handle_workspace_edit_input(value)
        if self._state == self._STATE_WORKSPACE_CLEAR_CONFIRM:
            return self._handle_workspace_clear_confirmation(value)
        self.clear()
        return SettingsInputResult(handled=True)

    def clear(self) -> None:
        self._state = None

    def _handle_root_input(self, value: str) -> SettingsInputResult:
        if value in {"1"}:
            self._state = self._STATE_WORKSPACE
            self._render_workspace_menu()
            return SettingsInputResult(handled=True)
        if value.lower() in {"q", "quit", "cancel", "exit"}:
            self.clear()
            self._console.print("[yellow]已退出设置。[/yellow]")
            return SettingsInputResult(handled=True)
        self._console.print("[red]请输入 1 管理默认工作区，或输入 q 退出。[/red]")
        return SettingsInputResult(handled=True)

    def _handle_workspace_input(self, value: str) -> SettingsInputResult:
        current = self._safe_get_default_workspace()
        if value == "1":
            self._state = self._STATE_WORKSPACE_EDIT
            self._render_workspace_edit_prompt(current)
            return SettingsInputResult(handled=True)
        if value == "2" and current is not None:
            self._state = self._STATE_WORKSPACE_CLEAR_CONFIRM
            self._console.print("[bold cyan]确认清除默认工作区吗？[/bold cyan] [dim](y/N，q 取消)[/dim]")
            return SettingsInputResult(handled=True)
        if (value == "2" and current is None) or (value == "3" and current is not None):
            self._state = self._STATE_ROOT
            self._render_root_menu()
            return SettingsInputResult(handled=True)
        if (value == "3" and current is None) or (value == "4" and current is not None):
            self._render_workspace_help(current)
            return SettingsInputResult(handled=True)
        if value.lower() in {"q", "quit", "cancel", "exit"}:
            self._state = self._STATE_ROOT
            self._render_root_menu()
            return SettingsInputResult(handled=True)

        valid = "1, 2, 3" if current is None else "1, 2, 3, 4"
        self._console.print(f"[red]请输入 {valid} 之一，或输入 q 返回上一级。[/red]")
        return SettingsInputResult(handled=True)

    def _handle_workspace_edit_input(self, value: str) -> SettingsInputResult:
        if value.lower() in {"q", "quit", "cancel", "exit"}:
            self._state = self._STATE_WORKSPACE
            self._render_workspace_menu()
            return SettingsInputResult(handled=True)
        if not value:
            self._console.print("[red]请输入目录路径，或输入 q 取消。[/red]")
            return SettingsInputResult(handled=True)

        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (self._workspace_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()

        if not candidate.exists():
            self._console.print(
                f"[red]路径不存在：[/red] {rich_escape(str(candidate))}\n"
                "[dim]请输入已存在的目录，或输入 q 取消。[/dim]"
            )
            return SettingsInputResult(handled=True)
        if not candidate.is_dir():
            self._console.print(
                f"[red]该路径不是目录：[/red] {rich_escape(str(candidate))}\n"
                "[dim]请输入目录路径，或输入 q 取消。[/dim]"
            )
            return SettingsInputResult(handled=True)

        saved = set_default_workspace(candidate)
        self._console.print(
            f"[green]默认工作区已保存：[/green] {rich_escape(str(saved))}\n"
            "[dim]该设置将在下次启动时生效，当前会话的工作区不会变化。[/dim]"
        )
        self._state = self._STATE_WORKSPACE
        self._render_workspace_menu()
        return SettingsInputResult(handled=True)

    def _handle_workspace_clear_confirmation(self, value: str) -> SettingsInputResult:
        if value.lower() in {"q", "quit", "cancel", "exit", "n", "no"}:
            self._state = self._STATE_WORKSPACE
            self._render_workspace_menu()
            return SettingsInputResult(handled=True)
        if value.lower() in {"y", "yes"}:
            clear_default_workspace()
            self._console.print(
                "[green]默认工作区已清除。[/green]\n"
                "[dim]之后在未传入 --root-dir 时，将回退为当前已有的启动行为。[/dim]"
            )
            self._state = self._STATE_WORKSPACE
            self._render_workspace_menu()
            return SettingsInputResult(handled=True)
        self._console.print("[red]请输入 y 确认清除，或输入 n/q 取消。[/red]")
        return SettingsInputResult(handled=True)

    def _render_root_menu(self) -> None:
        self._console.print()
        self._console.print("[bold cyan]设置[/bold cyan]")
        self._console.print("  1. 默认工作区")
        self._console.print("[dim]请输入编号，或输入 q 退出设置。[/dim]")

    def _render_workspace_menu(self) -> None:
        current = self._safe_get_default_workspace()
        current_text = rich_escape(str(current)) if current is not None else "未设置"
        effective_text = (
            "启动时若未传入 --root-dir，将使用这里配置的工作区。"
            if current is not None
            else "当前未配置默认工作区，启动行为保持不变。"
        )
        self._console.print()
        self._console.print("[bold cyan]默认工作区[/bold cyan]")
        self._console.print(f"当前值：{current_text}")
        self._console.print(f"[dim]{effective_text}[/dim]")
        self._console.print("  1. 编辑")
        if current is not None:
            self._console.print("  2. 清除")
            self._console.print("  3. 返回")
            self._console.print("  4. 帮助")
        else:
            self._console.print("  2. 返回")
            self._console.print("  3. 帮助")
        self._console.print("[dim]请输入编号，或输入 q 返回上一级。[/dim]")

    def _render_workspace_edit_prompt(self, current: Path | None) -> None:
        self._console.print()
        self._console.print("[bold cyan]编辑默认工作区[/bold cyan]")
        if current is not None:
            self._console.print(f"当前值：{rich_escape(str(current))}")
        self._console.print(
            f"[dim]请输入目录路径。相对路径会基于 "
            f"{rich_escape(str(self._workspace_dir))} 解析。输入 q 可取消。[/dim]"
        )

    def _render_workspace_help(self, current: Path | None) -> None:
        current_text = rich_escape(str(current)) if current is not None else "未设置"
        self._console.print()
        self._console.print("[bold cyan]默认工作区说明[/bold cyan]")
        self._console.print(f"当前值：{current_text}")
        self._console.print("[dim]该设置只影响后续 CLI 启动时的默认工作区。[/dim]")
        self._console.print("[dim]当前会话的工作区不会因此改变。[/dim]")
        self._console.print("[dim]如果启动时传入了 --root-dir，会优先使用显式参数。[/dim]")

    @staticmethod
    def _safe_get_default_workspace() -> Path | None:
        try:
            return get_default_workspace()
        except ValueError:
            return None
