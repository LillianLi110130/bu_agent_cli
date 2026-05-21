from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.markdown import Markdown

from agent_core.plugin import PluginManager


@dataclass(slots=True)
class PluginSlashResult:
    handled: bool = True
    reloaded: bool = False


class PluginSlashHandler:
    """Handle /plugins slash command operations."""

    def __init__(
        self,
        manager: PluginManager,
        console: Console | None = None,
        markdown_output_callback: Callable[[str], None] | None = None,
    ):
        self._manager = manager
        self._console = console or Console()
        self._markdown_output_callback = markdown_output_callback

    def _emit_markdown(self, content: str) -> None:
        self._console.print(Markdown(content, style="#e5e7eb"))
        if self._markdown_output_callback is not None:
            self._markdown_output_callback(content)

    async def handle(self, args: list[str]) -> PluginSlashResult:
        if not args:
            return await self._list()

        subcommand = args[0].lower()
        sub_args = args[1:]

        if subcommand in {"list", "ls"}:
            return await self._list()
        if subcommand == "show":
            return await self._show(sub_args)
        if subcommand == "copy":
            return await self._copy(sub_args)
        if subcommand == "reload":
            return await self._reload()
        if subcommand == "install":
            return await self._install(sub_args)
        if subcommand == "uninstall":
            return await self._uninstall(sub_args)

        self._console.print(f"[red]未知子命令：{subcommand}[/red]")
        self._console.print("[dim]可用子命令：list、show、copy、reload、install、uninstall[/dim]")
        return PluginSlashResult()

    async def _list(self) -> PluginSlashResult:
        plugins = self._manager.list_plugins()
        if not plugins:
            self._console.print("[yellow]未找到插件。[/yellow]")
            return PluginSlashResult()

        lines = [
            "## 插件",
            "",
            "| 名称 | 来源 | 状态 | 版本 | 资源 | 描述 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]

        for plugin in plugins:
            resources = (
                f"技能={len(plugin.skills)} 智能体={len(plugin.agents)} 命令={len(plugin.commands)}"
            )
            status_text = {
                "loaded": "已加载",
                "failed": "加载失败",
            }.get(plugin.status, plugin.status)
            version = plugin.version or "-"
            description = plugin.description or (plugin.error or "")
            lines.append(
                "| "
                + " | ".join(
                    [
                        self._escape_markdown_table_cell(plugin.name),
                        self._escape_markdown_table_cell(plugin.source),
                        self._escape_markdown_table_cell(status_text),
                        self._escape_markdown_table_cell(version),
                        self._escape_markdown_table_cell(resources),
                        self._escape_markdown_table_cell(description),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "- 使用 `/plugins show <name>` 查看详情。",
                "- 使用 `/plugins copy <name>` 复制内置插件到工作区。",
                "- 使用 `/plugins reload` 重新加载插件。",
                "- 使用 `/plugins install <plugin_path>` 从本地路径安装插件。",
                "- 使用 `/plugins uninstall <plugin_name> --force` 卸载插件。",
            ]
        )

        self._console.print()
        self._emit_markdown("\n".join(lines))
        self._console.print()
        return PluginSlashResult()

    async def _show(self, args: list[str]) -> PluginSlashResult:
        if not args:
            self._console.print("[red]用法：/plugins show <name>[/red]")
            return PluginSlashResult()

        plugin = self._manager.get_plugin(args[0])
        if plugin is None:
            self._console.print(f"[red]未找到插件：{args[0]}[/red]")
            return PluginSlashResult()

        lines = [
            f"## 插件：{plugin.name}",
            "",
            f"- **名称**：`{plugin.name}`",
            f"- **来源**：{plugin.source}",
            f"- **状态**：{plugin.status}",
            f"- **版本**：`{plugin.version or '-'}`",
            f"- **路径**：`{plugin.path}`",
        ]

        if plugin.description:
            lines.append(f"- **描述**：{plugin.description}")
        if plugin.error:
            lines.append(f"- **错误**：{plugin.error}")
        if plugin.skills:
            lines.extend(["", "### 技能"])
            lines.extend(f"- `{skill}`" for skill in plugin.skills)
        if plugin.agents:
            lines.extend(["", "### 智能体"])
            lines.extend(f"- `{agent}`" for agent in plugin.agents)
        if plugin.commands:
            lines.extend(["", "### 命令"])
            command_labels = []
            for item in plugin.commands:
                command = self._manager.get_command(item)
                mode_suffix = f" ({command.mode})" if command is not None else ""
                command_labels.append(f"`/{item}`{mode_suffix}")
            lines.extend(f"- {label}" for label in command_labels)
        if plugin.warnings:
            lines.extend(["", "### 警告"])
            lines.extend(f"- {warning}" for warning in plugin.warnings)

        self._console.print()
        self._emit_markdown("\n".join(lines))
        self._console.print()
        return PluginSlashResult()

    @staticmethod
    def _escape_markdown_table_cell(value: str) -> str:
        return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")

    async def _copy(self, args: list[str]) -> PluginSlashResult:
        if not args:
            self._console.print("[red]用法：/plugins copy <name>[/red]")
            return PluginSlashResult()

        try:
            target_dir = self._manager.copy_builtin_plugin(args[0])
        except FileExistsError as exc:
            self._console.print(f"[red]{exc}[/red]")
            return PluginSlashResult()
        except ValueError as exc:
            self._console.print(f"[red]{exc}[/red]")
            return PluginSlashResult()

        self._console.print(f"[green]已将插件复制到工作区：[/green] [dim]{target_dir}[/dim]")
        self._console.print("[dim]编辑完成后请运行 /plugins reload。[/dim]")
        return PluginSlashResult()

    async def _reload(self) -> PluginSlashResult:
        plugins = self._manager.reload_all()
        loaded = sum(1 for plugin in plugins if plugin.status == "loaded")
        failed = sum(1 for plugin in plugins if plugin.status != "loaded")
        self._console.print(f"[green]插件已重新加载。[/green] 成功={loaded} 失败={failed}")
        return PluginSlashResult(reloaded=True)

    async def _install(self, args: list[str]) -> PluginSlashResult:
        if not args:
            self._console.print("[red]用法：/plugins install <plugin_path>[/red]")
            self._console.print("[dim]示例：/plugins install /path/to/my-plugin[/dim]")
            return PluginSlashResult()

        source_path_str = " ".join(args)
        source_path = Path(source_path_str).expanduser().resolve()

        if not source_path.exists():
            self._console.print(f"[red]路径不存在：{source_path_str}[/red]")
            return PluginSlashResult()

        plugin_json_path = source_path / "plugin.json"
        if not plugin_json_path.exists():
            self._console.print(f"[red]在以下路径中未找到 plugin.json：{source_path_str}[/red]")
            self._console.print("[dim]有效插件目录必须包含 plugin.json 文件。[/dim]")
            return PluginSlashResult()

        try:
            import json

            manifest_data = json.loads(plugin_json_path.read_text(encoding="utf-8"))
            plugin_name = manifest_data.get("name", "")
            if not plugin_name:
                self._console.print("[red]plugin.json 无效：缺少 name 字段[/red]")
                return PluginSlashResult()
        except json.JSONDecodeError as e:
            self._console.print(f"[red]plugin.json 无效：{e}[/red]")
            return PluginSlashResult()

        dest_path = self._manager.plugin_dir / plugin_name
        if dest_path.exists():
            self._console.print(f"[yellow]插件“{plugin_name}”已存在：{dest_path}[/yellow]")
            self._console.print("[dim]如需重新加载该插件，请运行 /plugins reload。[/dim]")
            return PluginSlashResult()

        try:
            self._console.print(f"[dim]正在从 {source_path} 复制插件...[/dim]")
            shutil.copytree(source_path, dest_path)
            self._console.print(f"[green]插件“{plugin_name}”安装成功。[/green]")
            self._console.print(f"[dim]位置：{dest_path}[/dim]")
        except Exception as e:
            self._console.print(f"[red]复制插件失败：{e}[/red]")
            return PluginSlashResult()

        self._console.print("[dim]正在重新加载插件...[/dim]")
        return PluginSlashResult(reloaded=True)

    async def _uninstall(self, args: list[str]) -> PluginSlashResult:
        """Uninstall a plugin by removing its directory."""
        if not args:
            self._console.print("[red]用法：/plugins uninstall <plugin_name>[/red]")
            self._console.print("[dim]示例：/plugins uninstall my-plugin[/dim]")
            return PluginSlashResult()

        plugin_name = args[0]
        plugin = self._manager.get_plugin_from_source(plugin_name, "builtin")

        if plugin is None:
            self._console.print(f"[red]未找到内置插件：{plugin_name}[/red]")
            self._console.print("[dim]使用 /plugins list 查看可用插件及其来源。[/dim]")
            return PluginSlashResult()

        plugin_path = plugin.path
        if not plugin_path.exists():
            self._console.print(f"[red]未找到插件目录：{plugin_path}[/red]")
            return PluginSlashResult()

        try:
            plugin_path.resolve().relative_to(self._manager.plugin_dir.resolve())
        except ValueError:
            self._console.print(f"[red]插件路径不在 plugins 目录下：{plugin_path}[/red]")
            return PluginSlashResult()

        force = "--force" in args or "-f" in args
        if not force:
            self._console.print(f"[yellow]即将卸载插件：{plugin_name}[/yellow]")
            self._console.print(f"[dim]路径：{plugin_path}[/dim]")
            self._console.print("[dim]在非交互模式下可使用 --force 跳过确认。[/dim]")
            self._console.print("[red]请添加 --force 参数以确认卸载。[/red]")
            return PluginSlashResult()

        try:
            shutil.rmtree(plugin_path)
            self._console.print(f"[green]插件“{plugin_name}”卸载成功。[/green]")
        except Exception as e:
            self._console.print(f"[red]卸载插件失败：{e}[/red]")
            return PluginSlashResult()

        self._console.print("[dim]正在重新加载插件...[/dim]")
        return PluginSlashResult(reloaded=True)
