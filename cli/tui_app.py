"""Prompt-toolkit bottom-input UI for the interactive CLI."""

from __future__ import annotations

import asyncio
import shutil
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import ThreadedCompleter, merge_completers
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.text import Text

from cli.at_commands import AtCommandCompleter, parse_at_command
from cli.slash_commands import SlashCommandCompleter

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
        self._cancel_requested = False

    async def run(self) -> None:
        old_tui_enabled = self._cli._fixed_input_tui_enabled
        self._cli._fixed_input_tui_enabled = True

        self._cli._print_welcome()
        await self._cli._refresh_empty_context_budget_display()
        self._start_background_tasks()

        try:
            await self._run_loop()
        finally:
            await self._shutdown_background_tasks()
            await self._cancel_prompt_task()
            await self._cancel_active_task()
            self._cli._fixed_input_tui_enabled = old_tui_enabled

    def _build_session(self) -> PromptSession:
        kb = KeyBindings()

        @kb.add("enter")
        def _submit_or_complete(event) -> None:  # noqa: ANN001
            buffer = event.current_buffer
            if self._active_task is not None and not self._active_task.done():
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
            event.current_buffer.insert_text("\n")

        @kb.add("escape", "enter")
        def _escape_newline(event) -> None:  # noqa: ANN001
            event.current_buffer.insert_text("\n")

        @kb.add("c-c")
        def _cancel_or_clear(event) -> None:  # noqa: ANN001
            if self._active_task is not None and not self._active_task.done():
                if self._cancel_event is not None and not self._cancel_event.is_set():
                    self._cancel_requested = True
                    self._cancel_event.set()
                    self._cli._console.print("\n[yellow]正在取消当前执行...[/yellow]")
                    event.app.invalidate()
                    return

                self._active_task.cancel()
                self._cli._console.print("\n[yellow]取消未及时完成，已强制中断。[/yellow]")
                return
            event.current_buffer.reset()

        @kb.add("c-d")
        def _exit(event) -> None:  # noqa: ANN001
            event.app.exit(exception=EOFError)

        slash_completer = SlashCommandCompleter(self._cli._slash_registry)
        at_completer = AtCommandCompleter(self._cli._at_registry)
        threaded_completer = ThreadedCompleter(
            merge_completers([slash_completer, at_completer])
        )
        style = Style.from_dict(
            {
                "completion-menu.completion": "bg:#008888 #ffffff",
                "completion-menu.completion.current": "bg:#ffffff #000000",
                "completion-menu.meta.completion": "bg:#00aaaa #000000",
                "completion-menu.meta.current": "bg:#00ffff #000000",
                "completion-menu": "bg:#008888 #ffffff",
                "bottom-toolbar": "noreverse bg:default #777777",
                "bottom-toolbar.text": "noreverse bg:default #777777",
            }
        )
        return PromptSession(
            message=lambda: HTML(
                f"{self._separator_markup()}\n"
                "<ansiblue>>> </ansiblue>"
            ),
            key_bindings=kb,
            completer=threaded_completer,
            complete_while_typing=True,
            auto_suggest=AutoSuggestFromHistory(),
            style=style,
            enable_history_search=True,
            multiline=True,
            erase_when_done=True,
            bottom_toolbar=lambda: HTML(self._render_bottom_toolbar()),
        )

    def _separator_markup(self) -> str:
        return f"<ansibrightblack><b>{self._separator()}</b></ansibrightblack>"

    def _render_bottom_toolbar(self) -> str:
        status = self._cli._render_context_budget_toolbar()
        if self._active_task is not None and not self._active_task.done():
            if self._cancel_requested:
                status = f"<ansiyellow>正在取消当前执行...</ansiyellow> · {status}"
            else:
                status = f"<ansiyellow>Ctrl+C 取消当前执行</ansiyellow> · {status}"
        return f"{self._separator_markup()}\n{status}"

    @staticmethod
    def _separator() -> str:
        columns = shutil.get_terminal_size((80, 20)).columns
        # Avoid filling the last terminal column. Many terminals set an
        # auto-wrap flag when a character lands in the final column, and that
        # flag is exactly what makes resize redraws leave messy separator
        # fragments in scrollback.
        return "─" * max(20, columns - 1)

    async def _run_loop(self) -> None:
        if self._cli._bridge_store is not None:
            should_continue = await self._cli._drain_bridge_queue()
            if not should_continue:
                return

        while True:
            if self._prompt_task is None:
                self._prompt_task = asyncio.create_task(self._prompt_async())

            wait_set = {self._prompt_task}
            if self._active_task is not None:
                wait_set.add(self._active_task)

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

            if self._prompt_task not in done:
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

    async def _prompt_async(self) -> str:
        with patch_stdout(raw=True):
            return await self._session.prompt_async()

    async def _handle_user_input(self, user_input: str) -> bool:
        user_input = user_input.strip()
        if not user_input:
            return True

        if self._active_task is not None and not self._active_task.done():
            self._cli._console.print(
                "\n[yellow]当前已有任务在运行，请等待完成后再提交。[/yellow]"
            )
            return True

        self._print_user_input_record(user_input)

        if self._cli._is_immediate_local_exit_input(user_input):
            self._cli._console.print("[yellow]再见！[/yellow]")
            return False

        self._cancel_event = asyncio.Event()
        self._cancel_requested = False
        self._active_task = asyncio.create_task(self._execute_input(user_input))
        return True

    def _print_user_input_record(self, user_input: str) -> None:
        self._cli._console.print()
        lines = user_input.splitlines() or [""]
        columns = max(20, shutil.get_terminal_size((80, 20)).columns - 1)
        for index, line in enumerate(lines):
            prefix = ">>> " if index == 0 else "... "
            content = f"{prefix}{line}"
            self._cli._console.print(
                Text(content.ljust(columns), style="white on grey23"),
                overflow="crop",
                crop=True,
            )
        self._cli._console.print()

    async def _execute_input(self, text: str) -> None:
        try:
            if self._cli._bridge_store is not None:
                self._cli._enqueue_local_bridge_input(text)
                should_continue = await self._cli._drain_bridge_queue()
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
            self._cancel_requested = False

    async def _drain_background_work(self) -> bool:
        if self._cli._bridge_store is not None:
            return await self._cli._drain_bridge_queue()
        return await self._cli._drain_team_auto_trigger_queue()

    def _start_background_tasks(self) -> None:
        self._background_tasks = [
            asyncio.create_task(self._cli._consume_subagent_notifications()),
            asyncio.create_task(self._cli._consume_team_inbox_auto_triggers()),
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
            self._cancel_requested = False
