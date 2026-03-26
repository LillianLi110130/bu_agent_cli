"""
Agents Slash Command Handler

Handles the /agents slash command for managing agent configurations.
"""

import asyncio
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_core.agent.config import AgentConfig
from agent_core.agent.registry import AgentRegistry, get_agent_registry

from cli.interactive_input import InteractivePrompter, get_editor_command


class AgentSlashHandler:
    """Handler for /agents slash command.

    Manages agent configurations through CLI commands.
    """

    AVAILABLE_TOOLS = [
        "bash",
        "read",
        "write",
        "edit",
        "glob_search",
        "grep",
        "todo_read",
        "todo_write",
        "done",
        "run_subagent",
        "run_parallel_subagents",
    ]
    MODE_LABELS = {
        "primary": "主智能体",
        "subagent": "子智能体",
        "all": "通用",
    }

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        console: Console | None = None,
        prompter: InteractivePrompter | None = None,
        agents_dir: Path | None = None,
    ):
        """Initialize the handler."""
        self.console = console or Console()
        self.prompter = prompter or InteractivePrompter(self.console)
        self.agents_dir = agents_dir or (
            Path(__file__).parent.parent / "agent_core" / "prompts" / "agents"
        )
        self.registry = registry or get_agent_registry(self.agents_dir)

    async def handle(self, args: list[str]) -> bool:
        """Handle /agents command."""
        if not args:
            return await self._list()

        subcommand = args[0].lower()
        sub_args = args[1:]

        handlers = {
            "list": self._list,
            "ls": self._list,
            "show": self._show,
            "info": self._show,
            "create": self._create,
            "new": self._create,
            "add": self._create,
            "delete": self._delete,
            "remove": self._delete,
            "rm": self._delete,
            "del": self._delete,
            "edit": self._edit,
            "update": self._edit,
            "reload": self._reload,
        }

        handler = handlers.get(subcommand)
        if handler:
            return await handler(sub_args)

        if self.registry.get_config(subcommand):
            return await self._show([subcommand])

        self.console.print(f"[red]未知子命令：{subcommand}[/red]")
        self.console.print("[dim]可用子命令：list、show、create、delete、edit、reload[/dim]")
        return True

    async def _list(self, args: list[str] | None = None) -> bool:
        """List all agents."""
        args = args or []

        mode = None
        for arg in args:
            if arg in ("primary", "subagent", "all"):
                mode = arg
                break

        agent_names = self.registry.list_agents(mode=mode)

        if not agent_names:
            self.console.print("[yellow]未找到智能体[/yellow]")
            return True

        table = Table(title="可用智能体")
        table.add_column("名称", style="cyan")
        table.add_column("模式", style="blue")
        table.add_column("模型", style="dim")
        table.add_column("描述", style="white")

        for name in agent_names:
            config = self.registry.get_config(name)
            if not config:
                continue

            mode_style = {
                "primary": "green",
                "subagent": "blue",
                "all": "magenta",
            }.get(config.mode, "white")
            desc = (
                config.description[:50] + "..."
                if len(config.description) > 50
                else config.description
            )

            table.add_row(
                name,
                f"[{mode_style}]{self.MODE_LABELS.get(config.mode, config.mode)}[/{mode_style}]",
                config.model or "默认",
                desc,
            )

        self.console.print()
        self.console.print(table)
        self.console.print()
        self.console.print("[dim]使用 /agents show <name> 查看详情[/dim]")
        return True

    async def _show(self, args: list[str]) -> bool:
        """Show agent details."""
        if not args:
            self.console.print("[red]用法：/agents show <name>[/red]")
            return True

        name = args[0]
        config = self.registry.get_config(name)

        if not config:
            self.console.print(f"[red]未找到智能体：{name}[/red]")
            self.console.print("[dim]使用 /agents list 查看可用智能体[/dim]")
            return True

        details = self._build_agent_details(config)

        panel = Panel(
            "\n".join(details),
            title=f"[bold blue]智能体：{config.name}[/bold blue]",
            border_style="bright_blue",
            padding=(1, 2),
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()
        return True

    def _build_agent_details(self, config: AgentConfig) -> list[str]:
        """Build the display lines for the agent details panel."""
        details = self._build_basic_agent_details(config)
        self._append_tool_details(details, config)
        self._append_prompt_preview(details, config)
        self._append_source_path(details, config)
        return details

    def _build_basic_agent_details(self, config: AgentConfig) -> list[str]:
        """Build the basic metadata section for an agent."""
        mode_color = self._get_mode_color(config.mode)
        details = [
            f"[bold cyan]名称：[/] {config.name}",
            (
                f"[bold cyan]模式：[/] "
                f"[{mode_color}]{self.MODE_LABELS.get(config.mode, config.mode)}[/{mode_color}]"
            ),
            f"[bold cyan]描述：[/] {config.description}",
        ]

        if config.model:
            details.append(f"[bold cyan]模型：[/] {config.model}")
        if config.temperature is not None:
            details.append(f"[bold cyan]温度：[/] {config.temperature}")
        return details

    def _append_tool_details(self, details: list[str], config: AgentConfig) -> None:
        """Append configured tool information to the detail lines."""
        if not config.tools:
            return

        from tools import ALL_TOOLS

        all_tool_names = {tool.name for tool in ALL_TOOLS}
        enabled_tools = [tool for tool, enabled in config.tools.items() if enabled]
        disabled_tools = [tool for tool, enabled in config.tools.items() if not enabled]

        details.append("")
        details.append("[bold cyan]工具：[/]")
        self._append_enabled_tool_details(details, enabled_tools, all_tool_names)

        if disabled_tools:
            details.append("")
            details.append("[bold cyan]已禁用工具：[/]")
            for tool in disabled_tools:
                details.append(f"  [red]✗[/red] {tool}")

    def _append_enabled_tool_details(
        self,
        details: list[str],
        enabled_tools: list[str],
        all_tool_names: set[str],
    ) -> None:
        """Append enabled tools, marking tools that are not currently registered."""
        if not enabled_tools:
            details.append("  [dim]未显式启用[/dim]")
            return

        for tool in enabled_tools:
            marker = "[green]✓[/green]" if tool in all_tool_names else "[yellow]?[/yellow]"
            details.append(f"  {marker} {tool}")

    def _append_prompt_preview(self, details: list[str], config: AgentConfig) -> None:
        """Append a truncated system prompt preview."""
        details.append("")
        details.append("[bold cyan]系统提示词：[/]")

        prompt_preview = config.system_prompt[:200]
        if len(config.system_prompt) > 200:
            prompt_preview += "..."
        details.append(f"[dim]{prompt_preview}[/dim]")

    def _append_source_path(self, details: list[str], config: AgentConfig) -> None:
        """Append the backing file path when it exists on disk."""
        file_path = config.source_path or (self.agents_dir / f"{config.name}.md")
        if not file_path.exists():
            return

        details.append("")
        details.append(f"[bold cyan]文件：[/] [dim]{file_path}[/dim]")

    def _get_mode_color(self, mode: str) -> str:
        """Return the display color for a mode."""
        return {
            "primary": "green",
            "subagent": "blue",
            "all": "magenta",
        }.get(mode, "white")

    async def _create(self, args: list[str]) -> bool:
        """Create a new agent."""
        name = args[0] if args else None

        if not name:
            name = await self.prompter.prompt_text("智能体名称")
            if not name:
                self.console.print("[red]名称不能为空[/red]")
                return True

        if self.registry.get_config(name):
            self.console.print(f"[red]智能体已存在：{name}[/red]")
            return True

        return await self._interactive_create(name)

    async def _interactive_create(self, name: str) -> bool:
        """Interactively create a new agent."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]正在创建新智能体：{name}[/bold cyan]\n"
                f"[dim]直接回车可接受默认值[/dim]",
                border_style="cyan",
            )
        )

        description = await self.prompter.prompt_text(
            "[1/5] 描述",
            optional=True,
            default="",
        )
        mode = await self.prompter.prompt_choice(
            "[2/5] 模式（subagent/primary/all）",
            choices=["subagent", "primary", "all"],
            default="subagent",
        )
        model = await self.prompter.prompt_text(
            "[3/5] 模型",
            optional=True,
            default="",
        )
        temperature = await self.prompter.prompt_number(
            "[4/5] 温度",
            default=0.7,
            min_value=0.0,
            max_value=2.0,
        )

        self.console.print()
        self.console.print("[5/5] 选择工具：")
        tools = await self.prompter.prompt_multiselect(
            "可用工具",
            choices=self.AVAILABLE_TOOLS,
        )

        self.console.print()
        system_prompt = await self.prompter.prompt_multiline(
            "系统提示词",
            default=f"You are a {name} agent.",
        )

        self.console.print()
        self.console.print(
            Panel(
                f"[bold]摘要：[/bold]\n"
                f"  名称： [cyan]{name}[/cyan]\n"
                f"  模式： {mode}\n"
                f"  模型： {model or '默认'}\n"
                f"  温度： {temperature}\n"
                f"  工具： {', '.join(tools) if tools else '全部'}\n"
                f"  系统提示词： [dim]{system_prompt[:100]}...[/dim]",
                title="[bold blue]确认创建[/bold blue]",
                border_style="blue",
            )
        )

        confirmed = await self.prompter.prompt_yes_no("确认创建该智能体吗？", default=True)
        if not confirmed:
            self.console.print("[yellow]已取消[/yellow]")
            return True

        tools_dict = {tool: True for tool in tools} if tools else None
        await self._write_agent_file(
            name=name,
            description=description,
            mode=mode,
            model=model,
            temperature=temperature,
            tools=tools_dict,
            system_prompt=system_prompt,
        )

        self._reload_registry()
        self.console.print(f"[green]✓ 已创建智能体：{name}[/green]")
        return True

    async def _write_agent_file(
        self,
        name: str,
        description: str,
        mode: str,
        model: str | None,
        temperature: float,
        tools: dict[str, bool] | None,
        system_prompt: str,
    ):
        """Write agent configuration to file."""
        file_path = self.agents_dir / f"{name}.md"

        front_matter: dict[str, Any] = {
            "description": description,
            "mode": mode,
        }
        if model:
            front_matter["model"] = model
        if temperature is not None:
            front_matter["temperature"] = temperature
        if tools:
            front_matter["tools"] = tools

        content = (
            f"---\n"
            f"{yaml.dump(front_matter, default_flow_style=False, sort_keys=False)}"
            f"---\n\n{system_prompt}"
        )
        file_path.write_text(content, encoding="utf-8")

    async def _delete(self, args: list[str]) -> bool:
        """Delete an agent."""
        if not args:
            self.console.print("[red]用法：/agents delete <name>[/red]")
            return True

        name = args[0]
        config = self.registry.get_config(name)
        if not config:
            self.console.print(f"[red]未找到智能体：{name}[/red]")
            return True

        file_path = self.agents_dir / f"{name}.md"
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]名称：[/] {config.name}\n"
                f"[bold]模式：[/] {self.MODE_LABELS.get(config.mode, config.mode)}\n"
                f"[bold]描述：[/] {config.description}\n"
                f"[bold]文件：[/] [dim]{file_path}[/dim]",
                title=f"[red]删除智能体：{name}[/red]",
                border_style="red",
            )
        )

        confirmed = await self.prompter.prompt_yes_no(f"确认删除智能体“{name}”吗？", default=False)
        if not confirmed:
            self.console.print("[yellow]已取消[/yellow]")
            return True

        if file_path.exists():
            file_path.unlink()
            self._reload_registry()
            self.console.print(f"[green]✓ 已删除智能体：{name}[/green]")
        else:
            self.console.print(f"[red]未找到文件：{file_path}[/red]")

        return True

    async def _edit(self, args: list[str]) -> bool:
        """Edit an agent."""
        if not args:
            self.console.print("[red]用法：/agents edit <name>[/red]")
            return True

        name = args[0]
        config = self.registry.get_config(name)
        if not config:
            self.console.print(f"[red]未找到智能体：{name}[/red]")
            return True

        use_editor = "--editor" in args or "-e" in args
        if use_editor:
            return await self._edit_in_editor(name)

        return await self._interactive_edit(name, config)

    async def _interactive_edit(self, name: str, config: AgentConfig) -> bool:
        """Interactively edit an agent."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]正在编辑智能体：{name}[/bold cyan]",
                border_style="cyan",
            )
        )

        self.console.print()
        self.console.print("[bold]请选择要编辑的字段：[/bold]")
        self.console.print("  [1] 描述")
        self.console.print("  [2] 模式")
        self.console.print("  [3] 模型")
        self.console.print("  [4] 温度")
        self.console.print("  [5] 系统提示词")
        self.console.print("  [6] 工具")
        self.console.print("  [7] 在外部编辑器中打开")
        self.console.print("  [0] 取消")

        while True:
            try:
                from prompt_toolkit.formatted_text import HTML

                choice_input = await self.prompter._session.prompt_async(
                    HTML("<ansiblue>请选择 [0-7]：</ansiblue> ")
                )

                if not choice_input:
                    self.console.print("[yellow]已取消[/yellow]")
                    return True

                choice = choice_input.strip()
                if choice == "0":
                    self.console.print("[yellow]已取消[/yellow]")
                    return True

                if choice == "1":
                    config.description = await self.prompter.prompt_edit("描述", config.description)
                elif choice == "2":
                    config.mode = await self.prompter.prompt_choice(
                        "模式（subagent/primary/all）",
                        ["subagent", "primary", "all"],
                        config.mode,
                    )
                elif choice == "3":
                    new_model = await self.prompter.prompt_text(
                        "模型",
                        config.model or "",
                        optional=True,
                    )
                    config.model = new_model or None
                elif choice == "4":
                    current_temp = config.temperature or 0.7
                    config.temperature = await self.prompter.prompt_number(
                        "温度",
                        current_temp,
                        0.0,
                        2.0,
                    )
                elif choice == "5":
                    self.console.print("\n[dim]当前系统提示词：[/dim]")
                    prompt_preview = config.system_prompt[:300]
                    if len(config.system_prompt) > 300:
                        prompt_preview += "..."
                    self.console.print(prompt_preview)
                    self.console.print()
                    config.system_prompt = await self.prompter.prompt_multiline(
                        "新的系统提示词",
                        config.system_prompt,
                    )
                elif choice == "6":
                    current_tools = list(config.tools.keys()) if config.tools else []
                    new_tools = await self.prompter.prompt_multiselect(
                        "选择工具",
                        self.AVAILABLE_TOOLS,
                        current_tools,
                    )
                    config.tools = {tool: True for tool in new_tools}
                elif choice == "7":
                    return await self._edit_in_editor(name)
                else:
                    self.console.print("[red]无效选项[/red]")
                    continue

                await self._write_agent_file(
                    name=config.name,
                    description=config.description,
                    mode=config.mode,
                    model=config.model,
                    temperature=config.temperature or 0.7,
                    tools=config.tools,
                    system_prompt=config.system_prompt,
                )
                self._reload_registry()
                self.console.print(f"[green]✓ 已更新智能体：{name}[/green]")
                return True
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[yellow]已取消[/yellow]")
                return True

    async def _edit_in_editor(self, name: str) -> bool:
        """Edit agent in external editor."""
        file_path = self.agents_dir / f"{name}.md"
        if not file_path.exists():
            self.console.print(f"[red]未找到智能体文件：{file_path}[/red]")
            return True

        editor = get_editor_command()
        if not editor:
            self.console.print("[red]未配置编辑器[/red]")
            self.console.print("[dim]请设置 EDITOR 环境变量[/dim]")
            return True

        self.console.print(f"[dim]正在使用 {editor} 打开 {file_path}...[/dim]")

        try:
            process = await asyncio.create_subprocess_exec(editor, str(file_path))
            await process.wait()
            self._reload_registry()
            self.console.print(f"[green]✓ 已重新加载智能体：{name}[/green]")
            return True
        except Exception as e:
            self.console.print(f"[red]打开编辑器失败：{e}[/red]")
            return True

    async def _reload(self, _args: list[str] | None = None) -> bool:
        """Reload all agent configurations."""
        self._reload_registry()
        count = len(self.registry.list_agents())
        self.console.print(f"[green]✓ 已重新加载 {count} 个智能体[/green]")
        return True

    def _reload_registry(self):
        """Reload the agent registry."""
        global _global_registry
        import agent_core.agent.registry as registry_module

        _global_registry = AgentRegistry(self.agents_dir)
        registry_module._global_registry = _global_registry
        self.registry = _global_registry
