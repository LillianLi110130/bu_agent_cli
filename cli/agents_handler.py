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

from bu_agent_sdk.agent.config import AgentConfig
from bu_agent_sdk.agent.registry import AgentRegistry, get_agent_registry

from cli.interactive_input import InteractivePrompter, get_editor_command


class AgentSlashHandler:
    """Handler for /agents slash command.

    Manages agent configurations through CLI commands.
    """

    # Available tools for selection
    AVAILABLE_TOOLS = [
        "bash", "read", "write", "edit",
        "glob_search", "grep", "todo_read", "todo_write",
        "done", "run_subagent", "run_parallel_subagents",
    ]

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        console: Console | None = None,
        prompter: InteractivePrompter | None = None,
        agents_dir: Path | None = None,
    ):
        """Initialize the handler.

        Args:
            registry: AgentRegistry instance
            console: Rich console instance
            prompter: InteractivePrompter instance
            agents_dir: Path to agents directory
        """
        self.console = console or Console()
        self.prompter = prompter or InteractivePrompter(self.console)
        self.agents_dir = agents_dir or (Path(__file__).parent.parent / "bu_agent_sdk" / "prompts" / "agents")
        self.registry = registry or get_agent_registry(self.agents_dir)

    async def handle(self, args: list[str]) -> bool:
        """Handle /agents command.

        Args:
            args: Command arguments

        Returns:
            True if command was handled
        """
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

        # Check if it's an agent name (show details)
        if self.registry.get_config(subcommand):
            return await self._show([subcommand])

        self.console.print(f"[red]Unknown subcommand: {subcommand}[/red]")
        self.console.print("[dim]Available: list, show, create, delete, edit, reload[/dim]")
        return True

    # ==================== List Command ====================

    async def _list(self, args: list[str] | None = None) -> bool:
        """List all agents."""
        if args is None:
            args = []

        # Parse mode filter
        mode = None
        for arg in args:
            if arg in ("primary", "subagent", "all"):
                mode = arg
                break

        agent_names = self.registry.list_agents(mode=mode)

        if not agent_names:
            self.console.print("[yellow]No agents found[/yellow]")
            return True

        # Create table
        table = Table(title="Available Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Mode", style="blue")
        table.add_column("Model", style="dim")
        table.add_column("Description", style="white")

        for name in agent_names:
            config = self.registry.get_config(name)
            if not config:
                continue

            # Color mode
            mode_style = {
                "primary": "green",
                "subagent": "blue",
                "all": "magenta",
            }.get(config.mode, "white")

            # Truncate description
            desc = config.description[:50] + "..." if len(config.description) > 50 else config.description

            table.add_row(
                name,
                f"[{mode_style}]{config.mode}[/{mode_style}]",
                config.model or "default",
                desc,
            )

        self.console.print()
        self.console.print(table)
        self.console.print()
        self.console.print("[dim]Use /agents show <name> for details[/dim]")

        return True

    # ==================== Show Command ====================

    async def _show(self, args: list[str]) -> bool:
        """Show agent details."""
        if not args:
            self.console.print("[red]Usage: /agents show <name>[/red]")
            return True

        name = args[0]
        config = self.registry.get_config(name)

        if not config:
            self.console.print(f"[red]Agent not found: {name}[/red]")
            self.console.print("[dim]Use /agents list to see available agents[/dim]")
            return True

        # Build details panel
        details = []

        # Basic info
        mode_color = {
            "primary": "green",
            "subagent": "blue",
            "all": "magenta",
        }.get(config.mode, "white")

        details.append(f"[bold cyan]Name:[/] {config.name}")
        details.append(f"[bold cyan]Mode:[/] [{mode_color}]{config.mode}[/{mode_color}]")
        details.append(f"[bold cyan]Description:[/] {config.description}")

        if config.model:
            details.append(f"[bold cyan]Model:[/] {config.model}")
        if config.temperature is not None:
            details.append(f"[bold cyan]Temperature:[/] {config.temperature}")

        # Tools
        from tools import ALL_TOOLS
        all_tool_names = {t.name for t in ALL_TOOLS}

        if config.tools:
            enabled_tools = [t for t, e in config.tools.items() if e]
            disabled_tools = [t for t, e in config.tools.items() if not e]

            details.append("")
            details.append("[bold cyan]Tools:[/]")

            if enabled_tools:
                for tool in enabled_tools:
                    marker = "[green]✓[/green]" if tool in all_tool_names else "[yellow]?[/yellow]"
                    details.append(f"  {marker} {tool}")
            else:
                details.append("  [dim]None explicitly enabled[/dim]")

            if disabled_tools:
                details.append("")
                details.append("[bold cyan]Disabled Tools:[/]")
                for tool in disabled_tools:
                    details.append(f"  [red]✗[/red] {tool}")

        # System prompt preview
        details.append("")
        details.append("[bold cyan]System Prompt:[/]")
        prompt_preview = config.system_prompt[:200]
        if len(config.system_prompt) > 200:
            prompt_preview += "..."
        details.append(f"[dim]{prompt_preview}[/dim]")

        # File path
        file_path = self.agents_dir / f"{config.name}.md"
        if file_path.exists():
            details.append("")
            details.append(f"[bold cyan]File:[/] [dim]{file_path}[/dim]")

        # Create panel
        panel = Panel(
            "\n".join(details),
            title=f"[bold blue]Agent: {config.name}[/bold blue]",
            border_style="bright_blue",
            padding=(1, 2),
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

        return True

    # ==================== Create Command ====================

    async def _create(self, args: list[str]) -> bool:
        """Create a new agent."""
        name = args[0] if args else None

        if not name:
            name = await self.prompter.prompt_text("Agent name")
            if not name:
                self.console.print("[red]Name is required[/red]")
                return True

        # Check if already exists
        if self.registry.get_config(name):
            self.console.print(f"[red]Agent already exists: {name}[/red]")
            return True

        return await self._interactive_create(name)

    async def _interactive_create(self, name: str) -> bool:
        """Interactively create a new agent.

        Args:
            name: Agent name
        """
        self.console.print()
        self.console.print(Panel(
            f"[bold cyan]Creating new agent: {name}[/bold cyan]\n"
            f"[dim]Press Enter to accept defaults[/dim]",
            border_style="cyan",
        ))

        # Gather inputs
        description = await self.prompter.prompt_text(
            "[1/5] Description",
            optional=True,
            default="",
        )

        mode = await self.prompter.prompt_choice(
            "[2/5] Mode",
            choices=["subagent", "primary", "all"],
            default="subagent",
        )

        model = await self.prompter.prompt_text(
            "[3/5] Model",
            optional=True,
            default="",
        )

        temperature = await self.prompter.prompt_number(
            "[4/5] Temperature",
            default=0.7,
            min_value=0.0,
            max_value=2.0,
        )

        self.console.print()
        self.console.print("[5/5] Select tools:")
        tools = await self.prompter.prompt_multiselect(
            "Available tools",
            choices=self.AVAILABLE_TOOLS,
        )

        # System prompt
        self.console.print()
        system_prompt = await self.prompter.prompt_multiline(
            "System prompt",
            default=f"You are a {name} agent.",
        )

        # Confirm
        self.console.print()
        self.console.print(Panel(
            f"[bold]Summary:[/bold]\n"
            f"  Name: [cyan]{name}[/cyan]\n"
            f"  Mode: {mode}\n"
            f"  Model: {model or 'default'}\n"
            f"  Temperature: {temperature}\n"
            f"  Tools: {', '.join(tools) if tools else 'all'}\n"
            f"  System Prompt: [dim]{system_prompt[:100]}...[/dim]",
            title="[bold blue]Confirm Creation[/bold blue]",
            border_style="blue",
        ))

        confirmed = await self.prompter.prompt_yes_no("Create this agent?", default=True)

        if not confirmed:
            self.console.print("[yellow]Cancelled[/yellow]")
            return True

        # Build tools dict
        tools_dict = None
        if tools:
            tools_dict = {t: True for t in tools}

        # Write to file
        await self._write_agent_file(
            name=name,
            description=description,
            mode=mode,
            model=model,
            temperature=temperature,
            tools=tools_dict,
            system_prompt=system_prompt,
        )

        # Reload registry
        self._reload_registry()

        self.console.print(f"[green]✓ Agent created: {name}[/green]")
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

        # Build front matter
        front_matter = {
            "description": description,
            "mode": mode,
        }
        if model:
            front_matter["model"] = model
        if temperature is not None:
            front_matter["temperature"] = temperature
        if tools:
            front_matter["tools"] = tools

        # Write file
        content = f"---\n{yaml.dump(front_matter, default_flow_style=False, sort_keys=False)}---\n\n{system_prompt}"
        file_path.write_text(content, encoding="utf-8")

    # ==================== Delete Command ====================

    async def _delete(self, args: list[str]) -> bool:
        """Delete an agent."""
        if not args:
            self.console.print("[red]Usage: /agents delete <name>[/red]")
            return True

        name = args[0]
        config = self.registry.get_config(name)

        if not config:
            self.console.print(f"[red]Agent not found: {name}[/red]")
            return True

        # Show info
        file_path = self.agents_dir / f"{name}.md"

        self.console.print()
        self.console.print(Panel(
            f"[bold]Name:[/] {config.name}\n"
            f"[bold]Mode:[/] {config.mode}\n"
            f"[bold]Description:[/] {config.description}\n"
            f"[bold]File:[/] [dim]{file_path}[/dim]",
            title=f"[red]Delete Agent: {name}[/red]",
            border_style="red",
        ))

        # Confirm
        confirmed = await self.prompter.prompt_yes_no(
            f"Delete agent '{name}'?",
            default=False,
        )

        if not confirmed:
            self.console.print("[yellow]Cancelled[/yellow]")
            return True

        # Delete file
        if file_path.exists():
            file_path.unlink()
            self._reload_registry()
            self.console.print(f"[green]✓ Agent deleted: {name}[/green]")
        else:
            self.console.print(f"[red]File not found: {file_path}[/red]")

        return True

    # ==================== Edit Command ====================

    async def _edit(self, args: list[str]) -> bool:
        """Edit an agent."""
        if not args:
            self.console.print("[red]Usage: /agents edit <name>[/red]")
            return True

        name = args[0]
        config = self.registry.get_config(name)

        if not config:
            self.console.print(f"[red]Agent not found: {name}[/red]")
            return True

        # Check for --editor flag
        use_editor = "--editor" in args or "-e" in args

        if use_editor:
            return await self._edit_in_editor(name)

        return await self._interactive_edit(name, config)

    async def _interactive_edit(self, name: str, config: AgentConfig) -> bool:
        """Interactively edit an agent.

        Args:
            name: Agent name
            config: Current config
        """
        self.console.print()
        self.console.print(Panel(
            f"[bold cyan]Editing agent: {name}[/bold cyan]",
            border_style="cyan",
        ))

        # Show options
        self.console.print()
        self.console.print("[bold]Select field to edit:[/bold]")
        self.console.print("  [1] Description")
        self.console.print("  [2] Mode")
        self.console.print("  [3] Model")
        self.console.print("  [4] Temperature")
        self.console.print("  [5] System Prompt")
        self.console.print("  [6] Tools")
        self.console.print("  [7] Open in external editor")
        self.console.print("  [0] Cancel")

        while True:
            try:
                from prompt_toolkit.formatted_text import HTML

                choice_input = await self.prompter._session.prompt_async(
                    HTML("<ansiblue>Choice [0-7]:</ansiblue> ")
                )

                if not choice_input:
                    self.console.print("[yellow]Cancelled[/yellow]")
                    return True

                choice = choice_input.strip()

                if choice == "0":
                    self.console.print("[yellow]Cancelled[/yellow]")
                    return True

                if choice == "1":
                    new_value = await self.prompter.prompt_edit(
                        "Description", config.description
                    )
                    config.description = new_value

                elif choice == "2":
                    new_value = await self.prompter.prompt_choice(
                        "Mode", ["subagent", "primary", "all"], config.mode
                    )
                    config.mode = new_value

                elif choice == "3":
                    new_value = await self.prompter.prompt_text(
                        "Model", config.model or "", optional=True
                    )
                    config.model = new_value or None

                elif choice == "4":
                    current_temp = config.temperature or 0.7
                    new_value = await self.prompter.prompt_number(
                        "Temperature", current_temp, 0.0, 2.0
                    )
                    config.temperature = new_value

                elif choice == "5":
                    self.console.print("\n[dim]Current system prompt:[/dim]")
                    prompt_preview = config.system_prompt[:300]
                    if len(config.system_prompt) > 300:
                        prompt_preview += "..."
                    self.console.print(prompt_preview)
                    self.console.print()

                    new_value = await self.prompter.prompt_multiline(
                        "New system prompt",
                        config.system_prompt,
                    )
                    config.system_prompt = new_value

                elif choice == "6":
                    current_tools = list(config.tools.keys()) if config.tools else []
                    new_tools = await self.prompter.prompt_multiselect(
                        "Select tools",
                        self.AVAILABLE_TOOLS,
                        current_tools,
                    )
                    config.tools = {t: True for t in new_tools}

                elif choice == "7":
                    return await self._edit_in_editor(name)

                else:
                    self.console.print("[red]Invalid choice[/red]")
                    continue

                # Write changes to file
                await self._write_agent_file(
                    name=config.name,
                    description=config.description,
                    mode=config.mode,
                    model=config.model,
                    temperature=config.temperature or 0.7,
                    tools=config.tools,
                    system_prompt=config.system_prompt,
                )

                # Reload registry
                self._reload_registry()

                self.console.print(f"[green]✓ Agent updated: {name}[/green]")
                return True

            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[yellow]Cancelled[/yellow]")
                return True

    async def _edit_in_editor(self, name: str) -> bool:
        """Edit agent in external editor.

        Args:
            name: Agent name
        """
        file_path = self.agents_dir / f"{name}.md"

        if not file_path.exists():
            self.console.print(f"[red]Agent file not found: {file_path}[/red]")
            return True

        editor = get_editor_command()
        if not editor:
            self.console.print("[red]No editor configured[/red]")
            self.console.print("[dim]Set EDITOR environment variable[/dim]")
            return True

        self.console.print(f"[dim]Opening {file_path} in {editor}...[/dim]")

        try:
            process = await asyncio.create_subprocess_exec(
                editor,
                str(file_path),
            )
            await process.wait()

            # Reload registry
            self._reload_registry()

            self.console.print(f"[green]✓ Agent reloaded: {name}[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]Failed to open editor: {e}[/red]")
            return True

    # ==================== Reload Command ====================

    async def _reload(self, _args: list[str] | None = None) -> bool:
        """Reload all agent configurations."""
        self._reload_registry()
        count = len(self.registry.list_agents())
        self.console.print(f"[green]✓ Reloaded {count} agent(s)[/green]")
        return True

    def _reload_registry(self):
        """Reload the agent registry."""
        global _global_registry
        import bu_agent_sdk.agent.registry as registry_module
        _global_registry = AgentRegistry(self.agents_dir)
        registry_module._global_registry = _global_registry
        self.registry = _global_registry
