"""Slash command handler for MCP management."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from agent_core.mcp.config import MCPConfigEditError, set_mcp_server_disabled
from agent_core.mcp.manager import MCPManager


_QUIT_INPUTS = {"q", "quit", "cancel", "exit"}
_BACK_INPUTS = {"b", "back"}


class MCPSlashHandler:
    """Handle the /mcp slash command group."""

    def __init__(self, *, manager: MCPManager | None, console: Console) -> None:
        self.manager = manager
        self.console = console
        self._state: str | None = None
        self._server_order: list[str] = []
        self._selected_server: str | None = None

    @property
    def active(self) -> bool:
        return self._state is not None

    def bind_console(self, console: Console) -> None:
        self.console = console

    async def handle(self, args: list[str]) -> bool:
        if self.manager is None:
            self.console.print("[yellow]MCP manager 未初始化。[/yellow]")
            return True

        if not args:
            self.start()
            return True

        if args[0].lower() == "status":
            self._print_status()
            return True

        command = args[0].lower()
        if command == "tools":
            server = args[1] if len(args) > 1 else None
            self._print_tools(server)
            return True

        if command == "instructions":
            server = args[1] if len(args) > 1 else None
            self._print_instructions(server)
            return True

        if command == "stop":
            if len(args) != 2:
                self.console.print("[red]用法：/mcp stop <server>[/red]")
                return True
            await self.manager.stop_server(args[1])
            self.console.print(f"[green]MCP server 已停止：[/green]{args[1]}")
            return True

        if command in {"restart", "reconnect"}:
            if len(args) != 2:
                self.console.print("[red]用法：/mcp reconnect <server>[/red]")
                return True
            await self._restart(args[1])
            return True

        if command == "disable":
            if len(args) != 2:
                self.console.print("[red]用法：/mcp disable <server>[/red]")
                return True
            await self._disable(args[1])
            return True

        if command == "enable":
            if len(args) != 2:
                self.console.print("[red]用法：/mcp enable <server>[/red]")
                return True
            await self._enable(args[1])
            return True

        self.console.print(
            "[yellow]/mcp 仅支持 status、tools [server]、instructions [server]、stop <server>、restart/reconnect <server>、disable <server>、enable <server>。[/yellow]"
        )
        return True

    def start(self) -> None:
        if self.manager is None:
            self.console.print("[yellow]MCP manager 未初始化。[/yellow]")
            return
        self._state = "servers"
        self._selected_server = None
        self._render_server_menu()

    async def handle_input(self, user_input: str) -> bool:
        if self.manager is None or self._state is None:
            return False
        value = user_input.strip()
        if self._state == "servers":
            return self._handle_server_menu_input(value)
        if self._state == "server":
            return await self._handle_server_action_input(value)
        self.clear()
        return True

    def clear(self) -> None:
        self._state = None
        self._selected_server = None
        self._server_order = []

    async def _restart(self, server_name: str) -> None:
        try:
            client = await self.manager.restart_server(server_name)
        except Exception as exc:
            self.console.print(f"[red]MCP server 重启失败：{exc}[/red]")
            return
        self.console.print(
            "[green]MCP server 已重启：[/green]"
            f"{server_name} [dim]tools={len(client.tools)}[/dim]"
        )

    async def _disable(self, server_name: str) -> None:
        server = self._server_status(server_name)
        if server is None:
            self.console.print(f"[red]未找到 MCP server：{server_name}[/red]")
            return
        source = str(server.get("source") or "")
        if not source:
            self.console.print(f"[red]无法判断配置来源：{server_name}[/red]")
            return
        try:
            if server.get("state") == "running":
                await self.manager.stop_server(server_name)
            path = set_mcp_server_disabled(
                workspace_root=self.manager.workspace_root,
                server_name=server_name,
                source=source,
                disabled=True,
            )
            self.manager.reload_config()
        except (MCPConfigEditError, Exception) as exc:
            self.console.print(f"[red]MCP server 禁用失败：{exc}[/red]")
            return
        self.console.print(f"[green]MCP server 已禁用：[/green]{server_name} [dim]{path}[/dim]")

    async def _enable(self, server_name: str) -> None:
        server = self._server_status(server_name)
        if server is None:
            self.console.print(f"[red]未找到 MCP server：{server_name}[/red]")
            return
        source = str(server.get("source") or "")
        was_running = server.get("state") == "running"
        try:
            path = set_mcp_server_disabled(
                workspace_root=self.manager.workspace_root,
                server_name=server_name,
                source=source,
                disabled=False,
            )
            self.manager.reload_config()
        except (MCPConfigEditError, Exception) as exc:
            self.console.print(f"[red]MCP server 启用失败：{exc}[/red]")
            return
        try:
            client = await self.manager.start_server(server_name)
        except Exception as exc:
            self.console.print(
                f"[yellow]MCP server 已启用，但启动失败：{exc}[/yellow]\n"
                f"[dim]配置已保存：{path}[/dim]"
            )
            return
        if was_running:
            self.console.print(
                f"[green]MCP server 已启用，且已在运行：[/green]"
                f"{server_name} [dim]tools={len(client.tools)} config={path}[/dim]"
            )
            return
        self.console.print(
            f"[green]MCP server 已启用并启动：[/green]"
            f"{server_name} [dim]tools={len(client.tools)} config={path}[/dim]"
        )

    def _handle_server_menu_input(self, value: str) -> bool:
        lowered = value.lower()
        if lowered in _QUIT_INPUTS:
            self.clear()
            self.console.print("[yellow]已退出 MCP 管理。[/yellow]")
            return True
        if lowered in {"r", "refresh"}:
            self._render_server_menu()
            return True
        if not value.isdigit():
            self.console.print("[red]请输入 server 编号，或输入 r 刷新，q 退出。[/red]")
            return True
        index = int(value) - 1
        if index < 0 or index >= len(self._server_order):
            self.console.print("[red]server 编号无效。[/red]")
            return True
        self._selected_server = self._server_order[index]
        self._state = "server"
        self._render_selected_server_menu()
        return True

    async def _handle_server_action_input(self, value: str) -> bool:
        server_name = self._selected_server
        if server_name is None:
            self._state = "servers"
            self._render_server_menu()
            return True

        lowered = value.lower()
        if lowered in _QUIT_INPUTS:
            self.clear()
            self.console.print("[yellow]已退出 MCP 管理。[/yellow]")
            return True
        if lowered in _BACK_INPUTS:
            self._state = "servers"
            self._selected_server = None
            self._render_server_menu()
            return True

        server = self._server_status(server_name)
        if server is None:
            self.console.print(f"[red]MCP server 不存在：{server_name}[/red]")
            self._state = "servers"
            self._render_server_menu()
            return True

        if server.get("state") == "disabled" or server.get("disabled") is True:
            if value == "1":
                await self._enable(server_name)
                self._render_selected_server_menu()
                return True
            self.console.print("[red]该 server 已禁用。请输入 1 启用，或输入 b 返回。[/red]")
            return True

        if value == "1":
            self._print_tools(server_name)
            self._render_selected_server_menu()
            return True
        if value == "2":
            self._print_instructions(server_name)
            self._render_selected_server_menu()
            return True
        if value == "3":
            await self._disable(server_name)
            self._render_selected_server_menu()
            return True
        if value == "4":
            await self._restart(server_name)
            self._render_selected_server_menu()
            return True

        self.console.print("[red]请输入 1-4，或输入 b 返回、q 退出。[/red]")
        return True

    def _print_status(self) -> None:
        status = self.manager.status()
        servers = status.get("servers", [])
        clients = status.get("clients", [])
        running = [client.get("name") for client in clients if isinstance(client, dict)]
        self.console.print("[bold cyan]MCP Service[/bold cyan]")
        self.console.print(f"  workspace root:  {status.get('workspaceRoot')}")
        self.console.print(f"  user config:     {status.get('userConfigPath') or '-'}")
        self.console.print(f"  project config:  {status.get('projectConfigPath') or '-'}")
        self.console.print(f"  running clients: {', '.join(running) if running else 'none'}")
        self.console.print()

        table = Table(title="Registered Servers")
        table.add_column("Server")
        table.add_column("State")
        table.add_column("Source")
        table.add_column("Type")
        table.add_column("Tools")
        table.add_column("Instr")
        table.add_column("Command")
        for server in servers:
            if not isinstance(server, dict):
                continue
            command = " ".join(
                [str(server.get("command", "-")), *[str(arg) for arg in server.get("args", [])]]
            )
            table.add_row(
                str(server.get("name", "-")),
                str(server.get("state", "-")),
                str(server.get("source", "-")),
                str(server.get("type", "-")),
                str(server.get("toolCount", 0)),
                "yes" if server.get("instructions") else "-",
                command,
            )
            error = server.get("error")
            if error:
                table.add_row("", "[red]error[/red]", "", "", "", "", str(error))
        self.console.print(table)
        self.console.print()
        self.console.print("[bold cyan]Usage[/bold cyan]")
        self.console.print("  /mcp status")
        self.console.print("  /mcp tools")
        self.console.print("  /mcp tools codegraph")
        self.console.print("  /mcp instructions codegraph")
        self.console.print("  /mcp reconnect codegraph")
        self.console.print("  /mcp disable codegraph")
        self.console.print("  /mcp enable codegraph")

    def _render_server_menu(self) -> None:
        servers = self._server_statuses()
        self._server_order = [str(server.get("name")) for server in servers]
        self.console.print()
        self.console.print("[bold cyan]MCP Servers[/bold cyan]")
        if not servers:
            self.console.print("[dim]没有配置 MCP server。[/dim]")
            self.console.print("[dim]输入 q 退出。[/dim]")
            return
        for index, server in enumerate(servers, 1):
            name = str(server.get("name", "-"))
            state = str(server.get("state", "-"))
            source = str(server.get("source", "-"))
            tools = str(server.get("toolCount", 0))
            marker = "×" if state == "disabled" else "●" if state == "running" else "·"
            self.console.print(
                f"  {index}. {marker} {name} "
                f"[dim]state={state} source={source} tools={tools}[/dim]"
            )
        self.console.print("[dim]请输入编号选择 server，r 刷新，q 退出。[/dim]")

    def _render_selected_server_menu(self) -> None:
        server_name = self._selected_server
        if server_name is None:
            self._state = "servers"
            self._render_server_menu()
            return
        server = self._server_status(server_name)
        if server is None:
            self.console.print(f"[red]MCP server 不存在：{server_name}[/red]")
            self._state = "servers"
            self._render_server_menu()
            return
        state = str(server.get("state", "-"))
        source = str(server.get("source", "-"))
        self.console.print()
        self.console.print(f"[bold cyan]MCP Server: {server_name}[/bold cyan]")
        self.console.print(f"[dim]state={state} source={source}[/dim]")
        if state == "disabled" or server.get("disabled") is True:
            self.console.print("  1. Enable")
            self.console.print("[dim]请输入 1 启用，b 返回，q 退出。[/dim]")
            return
        self.console.print("  1. Tools")
        self.console.print("  2. Instructions")
        self.console.print("  3. Disable")
        self.console.print("  4. Reconnect")
        self.console.print("[dim]请输入编号，b 返回，q 退出。[/dim]")

    def _print_tools(self, server_name: str | None) -> None:
        tools = self.manager.list_tools(server_name)
        title = f"MCP Tools: {server_name}" if server_name else "MCP Tools"
        table = Table(title=title)
        table.add_column("Server")
        table.add_column("Tool")
        table.add_column("Description")
        for tool in tools:
            table.add_row(tool.server_name, tool.name, tool.description[:120])
        self.console.print(table)
        if not tools:
            self.console.print("[dim]没有已发现的 MCP tools。server 可能仍在启动或启动失败。[/dim]")

    def _print_instructions(self, server_name: str | None) -> None:
        instructions = self.manager.instructions(server_name)
        if not instructions:
            self.console.print("[dim]没有可用的 MCP server instructions。[/dim]")
            return
        for server, text in instructions.items():
            self.console.print(f"[bold cyan]MCP Instructions: {server}[/bold cyan]")
            self.console.print(text)

    def _server_statuses(self) -> list[dict]:
        status = self.manager.status()
        servers = status.get("servers", [])
        if not isinstance(servers, list):
            return []
        return [server for server in servers if isinstance(server, dict)]

    def _server_status(self, server_name: str) -> dict | None:
        for server in self._server_statuses():
            if server.get("name") == server_name:
                return server
        return None
