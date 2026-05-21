"""Prompt-toolkit bottom-input UI for the interactive CLI."""

from __future__ import annotations

import asyncio
import shutil
from html import escape as html_escape
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import ThreadedCompleter, merge_completers
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from cli.at_commands import AtCommandCompleter, parse_at_command
from cli.slash_commands import SlashCommandCompleter, parse_slash_command

if TYPE_CHECKING:
    from cli.app import TGAgentCLI


class TGAgentTUI:
    """Keep only the input prompt anchored while normal output uses terminal scrollback."""

    def __init__(self, cli: TGAgentCLI) -> None:
        self._cli = cli
        self._active_task: asyncio.Task[None] | None = None
        self._background_tasks: list[asyncio.Task[Any]] = []
        self._prompt_task: asyncio.Task[str] | None = None
        self._session = self._build_session()
        self._stop_requested = False
        self._cancel_event: asyncio.Event | None = None
        self._suspend_prompt_for_active_task = False
        self._prompt_spinner_index = 0
        self._prompt_spinner_active = False
        self._last_console_size = cli._console.size
        self._resize_repaint_at: float | None = None

    async def run(self) -> None:
        old_tui_enabled = self._cli._fixed_input_tui_enabled
        self._cli._fixed_input_tui_enabled = True
        self._cli._set_terminal_ui_invalidator(self._invalidate_prompt)

        self._cli._print_welcome()
        await self._cli._refresh_empty_context_budget_display()
        self._start_background_tasks()

        try:
            await self._run_loop()
        finally:
            await self._shutdown_background_tasks()
            await self._cancel_prompt_task()
            self._cli._cancel_terminal_approval_prompt()
            await self._cancel_active_task()
            self._cli._set_terminal_activity_status(None)
            self._cli._set_terminal_ui_invalidator(None)
            self._cli._fixed_input_tui_enabled = old_tui_enabled

    def _build_session(self) -> PromptSession:
        kb = KeyBindings()
        input_locked_filter = Condition(self._should_lock_input)
        cancel_available_filter = Condition(self._can_cancel_from_prompt)

        @kb.add("enter")
        def _submit_or_complete(event) -> None:  # noqa: ANN001
            buffer = event.current_buffer
            if self._should_lock_input():
                event.app.invalidate()
                return

            complete_state = buffer.complete_state
            if complete_state and buffer.text.lstrip().startswith("@"):
                completion = complete_state.current_completion
                if completion is None and complete_state.completions:
                    completion = complete_state.completions[0]
                if completion is not None:
                    buffer.apply_completion(completion)
                    skill_name, message = parse_at_command(buffer.text)
                    if skill_name and not message and not buffer.text.endswith(" "):
                        buffer.insert_text(" ")
                    return

            buffer.validate_and_handle()

        @kb.add("c-j")
        def _newline(event) -> None:  # noqa: ANN001
            if self._should_lock_input():
                event.app.invalidate()
                return
            event.current_buffer.insert_text("\n")

        @kb.add("q", filter=cancel_available_filter)
        def _cancel_running_task(event) -> None:  # noqa: ANN001
            if self._cancel_event is not None and not self._cancel_event.is_set():
                self._cancel_event.set()
                self._cli._request_foreground_subagent_cancel_from_terminal()
                self._cli._request_active_team_shutdown_from_cancel()
                event.app.invalidate()
                return

            if self._cli._cancel_active_bridge_run_from_terminal():
                self._cli._request_foreground_subagent_cancel_from_terminal()
                self._cli._request_active_team_shutdown_from_cancel()
                event.app.invalidate()
                return

            event.app.invalidate()

        @kb.add("c-c")
        def _cancel_or_clear(event) -> None:  # noqa: ANN001
            if self._should_lock_input():
                event.app.invalidate()
                return
            event.current_buffer.reset()

        @kb.add("c-d")
        def _exit(event) -> None:  # noqa: ANN001
            if self._should_lock_input():
                event.app.invalidate()
                return
            event.app.exit(exception=EOFError)

        slash_completer = SlashCommandCompleter(self._cli._slash_registry)
        at_completer = AtCommandCompleter(self._cli._at_registry)
        threaded_completer = ThreadedCompleter(
            merge_completers([slash_completer, at_completer])
        )
        style = Style.from_dict(
            {
                "": "#e5e7eb",
                "prompt": "#e5e7eb",
                "completion-menu.completion": "bg:#008888 #ffffff",
                "completion-menu.completion.current": "bg:#ffffff #000000",
                "completion-menu.meta.completion": "bg:#00aaaa #000000",
                "completion-menu.meta.current": "bg:#00ffff #000000",
                "completion-menu": "bg:#008888 #ffffff",
                "bottom-toolbar": "noreverse bg:default #777777",
                "bottom-toolbar.text": "noreverse bg:default #777777",
            }
        )
        session = PromptSession(
            message=lambda: HTML(self._render_prompt_message()),
            key_bindings=kb,
            completer=threaded_completer,
            complete_while_typing=True,
            auto_suggest=AutoSuggestFromHistory(),
            style=style,
            enable_history_search=True,
            multiline=True,
            erase_when_done=True,
            reserve_space_for_menu=0,
            bottom_toolbar=lambda: HTML(self._render_bottom_toolbar()),
        )
        session.default_buffer.read_only = input_locked_filter
        session.app.layout.current_window.height = self._input_window_height
        session.default_buffer.on_text_changed += self._invalidate_prompt_layout
        return session

    def _separator_markup(self) -> str:
        return f"<ansibrightblack><b>{self._separator()}</b></ansibrightblack>"

    def _render_prompt_message(self) -> str:
        status = self._cli._get_terminal_activity_status()
        lines = []
        approval_lines = self._cli._terminal_approval_prompt_lines()
        if approval_lines:
            for line in approval_lines:
                lines.append(f"<ansiyellow>{html_escape(line)}</ansiyellow>")
            lines.append("")
        if status:
            frame = self._prompt_spinner_frame()
            lines.append(f'<style fg="#c084fc">{frame} {status}...</style>')
            lines.append("")
        lines.append(self._separator_markup())
        lines.append("<ansiblue>>> </ansiblue>")
        return "\n".join(lines)

    def _prompt_spinner_frame(self) -> str:
        frames = ["-", "\\", "|", "/"]
        return frames[self._prompt_spinner_index % len(frames)]

    def _render_bottom_toolbar(self) -> str:
        status = self._cli._render_context_budget_toolbar()
        approval_toolbar = self._cli._terminal_approval_toolbar()
        if approval_toolbar is not None:
            status = f"<ansiyellow>{html_escape(approval_toolbar)}</ansiyellow> · {status}"
        elif self._should_lock_input():
            status = f"按 q 取消当前执行 · {status}"
        return f"{self._separator_markup()}\n{status}"

    @staticmethod
    def _separator() -> str:
        columns = shutil.get_terminal_size((80, 20)).columns
        return "─" * max(20, columns - 1)

    def _input_window_height(self) -> Dimension:
        text = self._session.default_buffer.text if hasattr(self, "_session") else ""
        columns = shutil.get_terminal_size((80, 20)).columns
        prompt_width = get_cwidth(">>> ")
        available_width = max(1, columns - prompt_width - 1)
        visual_lines = 0
        for logical_line in text.split("\n") or [""]:
            line_width = get_cwidth(logical_line)
            visual_lines += max(1, (line_width + available_width - 1) // available_width)
        height = min(max(visual_lines, 1), 6)
        height += self._completion_menu_height()
        return Dimension.exact(height)

    def _completion_menu_height(self) -> int:
        if not hasattr(self, "_session"):
            return 0

        buffer = self._session.default_buffer
        stripped_text = buffer.text.lstrip()
        if not stripped_text.startswith(("/", "@")):
            return 0

        complete_state = buffer.complete_state
        if complete_state is None:
            return 0

        return min(max(len(complete_state.completions), 1), 8)

    def _invalidate_prompt_layout(self, _) -> None:  # noqa: ANN001
        if hasattr(self, "_session"):
            self._session.app.invalidate()

    async def _run_loop(self) -> None:
        if self._cli._bridge_store is not None:
            should_continue = await self._cli._drain_bridge_queue()
            if not should_continue:
                return

        while True:
            self._schedule_resize_repaint_if_needed()
            self._repaint_after_resize_if_ready()

            if self._prompt_task is None and (
                self._active_task is None or not self._suspend_prompt_for_active_task
            ):
                self._prompt_task = asyncio.create_task(self._prompt_async())

            wait_set = set()
            if self._prompt_task is not None:
                wait_set.add(self._prompt_task)
            if self._active_task is not None:
                wait_set.add(self._active_task)
            if not wait_set:
                continue

            done, _ = await asyncio.wait(
                wait_set,
                timeout=self._cli.BRIDGE_POLL_INTERVAL_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if self._active_task is None:
                should_continue = await self._drain_background_work()
                if not should_continue:
                    return

            if self._active_task is not None and self._active_task in done:
                await self._finish_active_task()
                if self._stop_requested:
                    return

            if self._prompt_task is None or self._prompt_task not in done:
                continue

            try:
                user_input = await self._prompt_task
            except asyncio.CancelledError:
                if self._stop_requested:
                    return
                continue
            except EOFError:
                self._cli._console.print("\n[yellow]再见！[/yellow]")
                return
            except KeyboardInterrupt:
                self._prompt_task = None
                continue
            finally:
                self._prompt_task = None

            should_continue = await self._handle_user_input(user_input)
            if not should_continue:
                return

    def _schedule_resize_repaint_if_needed(self) -> None:
        current_size = self._cli._console.size
        if current_size == self._last_console_size:
            return
        self._last_console_size = current_size
        self._resize_repaint_at = asyncio.get_running_loop().time() + 0.2

    def _repaint_after_resize_if_ready(self) -> None:
        if self._resize_repaint_at is None:
            return
        if asyncio.get_running_loop().time() < self._resize_repaint_at:
            return
        self._resize_repaint_at = None
        self._cli._repaint_output_history(preserve_activity=True)
        self._invalidate_prompt()

    async def _prompt_async(self) -> str:
        with patch_stdout(raw=True):
            return await self._session.prompt_async()

    async def _handle_user_input(self, user_input: str) -> bool:
        user_input = user_input.strip()
        if self._cli._has_terminal_approval_prompt():
            self._cli._submit_terminal_approval_input(user_input)
            return True

        if not user_input:
            return True

        if self._active_task is not None and not self._active_task.done():
            self._cli._console.print(
                "\n[yellow]当前已有任务在运行，请等待完成后再提交。[/yellow]"
            )
            return True

        self._cli._print_user_input_record(user_input)

        if self._cli._is_immediate_local_exit_input(user_input):
            self._cli._console.print("[yellow]再见！[/yellow]")
            return False

        self._cancel_event = asyncio.Event()
        self._suspend_prompt_for_active_task = self._should_suspend_prompt_for_input(user_input)
        self._active_task = asyncio.create_task(self._execute_input(user_input))
        return True

    def _is_input_locked(self) -> bool:
        return self._active_task is not None and not self._active_task.done()

    def _should_lock_input(self) -> bool:
        if self._cli._has_terminal_approval_prompt():
            return False
        return self._is_input_locked() or self._cli._has_active_bridge_run()

    def _can_cancel_from_prompt(self) -> bool:
        if self._cli._has_terminal_approval_prompt():
            return False
        return self._is_input_locked() or self._cli._has_active_bridge_run()

    @staticmethod
    def _should_suspend_prompt_for_input(user_input: str) -> bool:
        command = parse_slash_command(user_input)
        if command.name != "agents":
            return False
        subcommand = command.args[0].lower() if command.args else ""
        return subcommand in {"create", "edit", "delete", "reload"}

    async def _execute_input(self, text: str) -> None:
        try:
            if self._cli._bridge_store is not None:
                self._cli._enqueue_local_bridge_input(text)
                should_continue = await self._cli._drain_bridge_queue(
                    cancel_event=self._cancel_event
                )
            else:
                outcome = await self._cli._execute_input_text(
                    text,
                    source="local",
                    cancel_event=self._cancel_event,
                )
                should_continue = outcome.continue_running

            if not should_continue:
                self._stop_requested = True
                await self._cancel_prompt_task()
        except asyncio.CancelledError:
            self._cli._console.print("[yellow]当前运行已取消。[/yellow]")
        except Exception as exc:
            self._cli._console.print(f"[red]执行失败：{exc}[/red]")

    async def _finish_active_task(self) -> None:
        if self._active_task is None:
            return
        task = self._active_task
        self._active_task = None
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._cancel_event = None
            self._suspend_prompt_for_active_task = False

    async def _drain_background_work(self) -> bool:
        if self._cli._bridge_store is not None:
            return await self._cli._drain_bridge_queue()
        return await self._cli._drain_team_auto_trigger_queue()

    def _start_background_tasks(self) -> None:
        self._background_tasks = [
            asyncio.create_task(self._cli._consume_subagent_notifications()),
            asyncio.create_task(self._cli._consume_team_inbox_auto_triggers()),
            asyncio.create_task(self._prompt_spinner_loop()),
        ]

    async def _shutdown_background_tasks(self) -> None:
        for task in self._background_tasks:
            task.cancel()
        for task in self._background_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._background_tasks.clear()
        self._prompt_spinner_active = False

    async def _prompt_spinner_loop(self) -> None:
        while True:
            if self._cli._get_terminal_activity_status():
                self._prompt_spinner_active = True
                self._prompt_spinner_index += 1
                self._invalidate_prompt()
                await asyncio.sleep(0.18)
                continue

            if self._prompt_spinner_active:
                self._prompt_spinner_active = False
                self._prompt_spinner_index = 0
                self._invalidate_prompt()
            await asyncio.sleep(0.08)

    def _invalidate_prompt(self) -> None:
        if hasattr(self, "_session"):
            self._session.app.invalidate()

    async def _cancel_prompt_task(self) -> None:
        if self._prompt_task is None or self._prompt_task.done():
            self._prompt_task = None
            return
        self._prompt_task.cancel()
        try:
            await self._prompt_task
        except (asyncio.CancelledError, EOFError, KeyboardInterrupt):
            pass
        finally:
            self._prompt_task = None

    async def _cancel_active_task(self) -> None:
        if self._active_task is None or self._active_task.done():
            self._active_task = None
            return
        self._active_task.cancel()
        try:
            await self._active_task
        except asyncio.CancelledError:
            pass
        finally:
            self._active_task = None
            self._cancel_event = None
            self._suspend_prompt_for_active_task = False
