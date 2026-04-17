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
from agent_core.agent.registry import AgentRegistry, default_agent_sources, get_agent_registry

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
        "delegate",
        "task_status",
        "task_cancel",
    ]
    WRITABLE_SCOPES = {"workspace", "user"}
    SOURCE_LABELS = {
        "workspace": "项目",
        "user": "用户",
        "builtin": "内置",
    }

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        console: Console | None = None,
        prompter: InteractivePrompter | None = None,
        workspace_root: Path | None = None,
        agents_dir: Path | None = None,
    ):
        """Initialize the handler."""
        self.console = console or Console()
        self.prompter = prompter or InteractivePrompter(self.console)
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.user_agents_dir = Path.home() / ".tg_agent" / "agents"
        self.workspace_agents_dir = self.workspace_root / ".tg_agent" / "agents"
        self.builtin_agents_dir = agents_dir or (
            Path(__file__).parent.parent / "agent_core" / "prompts" / "agents"
        )
        self.registry = registry or get_agent_registry(
            self.workspace_root,
            builtin_agents_dir=self.builtin_agents_dir,
        )

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
        scope_filter = self._parse_scope_filter(args)
        agent_names = self.registry.list_agents()

        if not agent_names:
            self.console.print("[yellow]未找到智能体[/yellow]")
            return True

        table = Table(title="可用智能体")
        table.add_column("名称", style="cyan")
        table.add_column("来源", style="magenta")
        table.add_column("模型", style="dim")
        table.add_column("描述", style="white")

        for name in agent_names:
            config = self.registry.get_config(name)
            if not config:
                continue
            if scope_filter and not self._matches_scope_filter(name, config, scope_filter):
                continue

            desc = (
                config.description[:50] + "..."
                if len(config.description) > 50
                else config.description
            )

            table.add_row(
                name,
                self._source_label_for_config(name, config),
                config.model or "默认",
                desc,
            )

        if table.row_count == 0:
            label = self._format_source_label(scope_filter) if scope_filter else "指定来源"
            self.console.print(f"[yellow]未找到来自 {label} 的智能体[/yellow]")
            return True

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
        details = [
            f"[bold cyan]名称：[/] {config.name}",
            f"[bold cyan]描述：[/] {config.description}",
            f"[bold cyan]来源：[/] {self._source_label_for_config(config.name, config)}",
        ]

        if config.model:
            details.append(f"[bold cyan]模型：[/] {config.model}")
        if config.temperature is not None:
            details.append(f"[bold cyan]温度：[/] {config.temperature}")
        details.append(f"[bold cyan]优先级：[/] {config.source_priority}")
        return details

    def _append_tool_details(self, details: list[str], config: AgentConfig) -> None:
        """Append configured tool information to the detail lines."""
        from tools import ALL_TOOLS

        all_tool_names = {tool.name for tool in ALL_TOOLS}
        enabled_tools = list(config.tools or [])
        disabled_tools = list(config.disallowed_tools or [])

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
        file_path = config.source_path
        if file_path is None:
            return
        if not file_path.exists():
            return

        details.append("")
        details.append(f"[bold cyan]文件：[/] [dim]{file_path}[/dim]")

    def _parse_scope_filter(self, args: list[str]) -> str | None:
        """Parse optional scope filter from command args."""
        scope_aliases = {
            "workspace": "workspace",
            "project": "workspace",
            "项目": "workspace",
            "user": "user",
            "用户": "user",
            "builtin": "builtin",
            "内置": "builtin",
            "plugin": "plugin",
            "插件": "plugin",
        }
        for arg in args:
            normalized = scope_aliases.get(arg.lower())
            if normalized:
                return normalized
        return None

    def _format_source_label(self, source_scope: str) -> str:
        """Format source scope for display."""
        if source_scope in self.SOURCE_LABELS:
            return self.SOURCE_LABELS[source_scope]
        if source_scope.startswith("plugin:"):
            return f"插件({source_scope.split(':', 1)[1]})"
        if source_scope not in {"workspace", "user", "builtin"}:
            return f"插件({source_scope})"
        return source_scope

    def _source_label_for_config(self, name: str, config: AgentConfig) -> str:
        """Format source label with plugin awareness."""
        if self._is_plugin_agent(name, config):
            return "插件"
        return self._format_source_label(config.source_scope)

    def _matches_scope_filter(self, name: str, config: AgentConfig, scope_filter: str) -> bool:
        """Check whether a config matches the requested scope filter."""
        if scope_filter == "plugin":
            return self._is_plugin_agent(name, config)
        return config.source_scope == scope_filter and not self._is_plugin_agent(name, config)

    def _is_plugin_agent(self, name: str, config: AgentConfig) -> bool:
        """Best-effort detection for plugin-provided agents."""
        if ":" in name:
            return True
        source_path = config.source_path
        if source_path is None:
            return False
        return "plugins" in source_path.parts

    def _resolve_target_agents_dir(self, scope: str) -> Path:
        """Resolve writable target directory for create/update flows."""
        if scope == "workspace":
            return self.workspace_agents_dir
        if scope == "user":
            return self.user_agents_dir
        raise ValueError(f"Unsupported writable scope: {scope}")

    def _is_writable_agent(self, name: str, config: AgentConfig) -> bool:
        """Whether the agent can be modified through /agents."""
        return config.source_scope in self.WRITABLE_SCOPES and not self._is_plugin_agent(name, config)

    async def _create(self, args: list[str]) -> bool:
        """Create a new agent."""
        args = list(args)
        scope = "workspace"
        if "--scope" in args:
            index = args.index("--scope")
            if index + 1 >= len(args):
                self.console.print("[red]用法：/agents create <name> [--scope workspace|user][/red]")
                return True
            scope = args[index + 1].lower()
            del args[index : index + 2]
        elif "--user" in args:
            scope = "user"
            args.remove("--user")
        elif "--workspace" in args:
            scope = "workspace"
            args.remove("--workspace")

        if scope not in self.WRITABLE_SCOPES:
            self.console.print("[red]仅支持创建到 workspace 或 user 来源[/red]")
            return True

        name = args[0] if args else None

        if not name:
            name = await self.prompter.prompt_text("智能体名称")
            if not name:
                self.console.print("[red]名称不能为空[/red]")
                return True

        if self.registry.get_config(name):
            self.console.print(f"[red]智能体已存在：{name}[/red]")
            return True

        return await self._interactive_create(name, scope)

    async def _interactive_create(self, name: str, scope: str) -> bool:
        """Interactively create a new agent."""
        target_dir = self._resolve_target_agents_dir(scope)
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]正在创建新智能体：{name}[/bold cyan]\n"
                f"[dim]写入位置：{self._format_source_label(scope)} ({target_dir})[/dim]\n"
                f"[dim]直接回车可接受默认值[/dim]",
                border_style="cyan",
            )
        )

        description = await self.prompter.prompt_text(
            "[1/6] 描述",
            optional=True,
            default="",
        )
        model = await self.prompter.prompt_text(
            "[2/6] 模型",
            optional=True,
            default="",
        )
        temperature = await self.prompter.prompt_number(
            "[3/6] 温度",
            default=0.7,
            min_value=0.0,
            max_value=2.0,
        )

        self.console.print()
        self.console.print("[4/6] 选择允许工具：")
        tools = await self.prompter.prompt_multiselect("可用工具", choices=self.AVAILABLE_TOOLS)

        self.console.print()
        self.console.print("[5/6] 选择禁用工具：")
        disallowed_tools = await self.prompter.prompt_multiselect(
            "禁用工具",
            choices=self.AVAILABLE_TOOLS,
        )

        self.console.print()
        self.console.print("[6/6] 编辑系统提示词：")
        system_prompt = await self.prompter.prompt_multiline(
            "系统提示词",
            default=f"You are a {name} agent.",
        )

        self.console.print()
        self.console.print(
            Panel(
                f"[bold]摘要：[/bold]\n"
                f"  名称： [cyan]{name}[/cyan]\n"
                f"  来源： {self._format_source_label(scope)}\n"
                f"  模型： {model or '默认'}\n"
                f"  温度： {temperature}\n"
                f"  允许工具： {', '.join(tools) if tools else '全部'}\n"
                f"  禁用工具： {', '.join(disallowed_tools) if disallowed_tools else '无'}\n"
                f"  系统提示词： [dim]{system_prompt[:100]}...[/dim]",
                title="[bold blue]确认创建[/bold blue]",
                border_style="blue",
            )
        )

        confirmed = await self.prompter.prompt_yes_no("确认创建该智能体吗？", default=True)
        if not confirmed:
            self.console.print("[yellow]已取消[/yellow]")
            return True

        await self._write_agent_file(
            target_dir=target_dir,
            name=name,
            description=description,
            model=model,
            temperature=temperature,
            tools=tools or None,
            disallowed_tools=disallowed_tools,
            system_prompt=system_prompt,
        )

        self._reload_registry()
        self.console.print(f"[green]✓ 已创建智能体：{name}[/green]")
        return True

    async def _write_agent_file(
        self,
        target_dir: Path,
        name: str,
        description: str,
        model: str | None,
        temperature: float,
        tools: list[str] | None,
        disallowed_tools: list[str] | None,
        system_prompt: str,
    ):
        """Write agent configuration to file."""
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{name}.md"

        front_matter: dict[str, Any] = {
            "description": description,
        }
        if model:
            front_matter["model"] = model
        if temperature is not None:
            front_matter["temperature"] = temperature
        if tools:
            front_matter["tools"] = tools
        if disallowed_tools:
            front_matter["disallowedTools"] = disallowed_tools

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

        if not self._is_writable_agent(name, config):
            self.console.print(
                f"[yellow]来源为 {self._source_label_for_config(name, config)} 的智能体不支持通过 /agents delete 删除。[/yellow]"
            )
            return True

        file_path = config.source_path
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]名称：[/] {config.name}\n"
                f"[bold]来源：[/] {self._source_label_for_config(name, config)}\n"
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

        if not self._is_writable_agent(name, config):
            self.console.print(
                f"[yellow]来源为 {self._source_label_for_config(name, config)} 的智能体不支持通过 /agents edit 修改。[/yellow]"
            )
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
                f"[bold cyan]正在编辑智能体：{name}[/bold cyan]\n"
                f"[dim]来源：{self._source_label_for_config(name, config)}[/dim]",
                border_style="cyan",
            )
        )

        self.console.print()
        self.console.print("[bold]请选择要编辑的字段：[/bold]")
        self.console.print("  [1] 描述")
        self.console.print("  [2] 模型")
        self.console.print("  [3] 温度")
        self.console.print("  [4] 系统提示词")
        self.console.print("  [5] 允许工具")
        self.console.print("  [6] 禁用工具")
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
                    new_model = await self.prompter.prompt_text(
                        "模型",
                        config.model or "",
                        optional=True,
                    )
                    config.model = new_model or None
                elif choice == "3":
                    current_temp = config.temperature or 0.7
                    config.temperature = await self.prompter.prompt_number(
                        "温度",
                        current_temp,
                        0.0,
                        2.0,
                    )
                elif choice == "4":
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
                elif choice == "5":
                    current_tools = list(config.tools or [])
                    new_tools = await self.prompter.prompt_multiselect(
                        "选择允许工具",
                        self.AVAILABLE_TOOLS,
                        current_tools,
                    )
                    config.tools = new_tools
                elif choice == "6":
                    current_disallowed = list(config.disallowed_tools or [])
                    config.disallowed_tools = await self.prompter.prompt_multiselect(
                        "选择禁用工具",
                        self.AVAILABLE_TOOLS,
                        current_disallowed,
                    )
                elif choice == "7":
                    return await self._edit_in_editor(name)
                else:
                    self.console.print("[red]无效选项[/red]")
                    continue

                await self._write_agent_file(
                    target_dir=config.source_path.parent,
                    name=config.name,
                    description=config.description,
                    model=config.model,
                    temperature=config.temperature or 0.7,
                    tools=config.tools,
                    disallowed_tools=config.disallowed_tools,
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
        config = self.registry.get_config(name)
        if config is None or config.source_path is None:
            self.console.print(f"[red]未找到智能体文件：{name}[/red]")
            return True
        if not self._is_writable_agent(name, config):
            self.console.print(
                f"[yellow]来源为 {self._source_label_for_config(name, config)} 的智能体不支持通过 /agents edit 修改。[/yellow]"
            )
            return True

        file_path = config.source_path
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

        _global_registry = AgentRegistry(
            agent_sources=default_agent_sources(
                self.workspace_root,
                builtin_agents_dir=self.builtin_agents_dir,
            )
        )
        registry_module._global_registry = _global_registry
        self.registry = _global_registry
