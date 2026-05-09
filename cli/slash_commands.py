"""Slash command system for Crab CLI."""

from collections import OrderedDict
from dataclasses import dataclass, field
import re
from typing import Callable

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document


@dataclass
class SlashCommand:
    """Metadata for a slash command."""

    name: str
    description: str
    usage: str = ""
    handler: Callable[[], str] | Callable[[str], str] | None = None
    is_builtin: bool = True
    category: str = "通用"
    examples: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.usage:
            self.usage = f"/{self.name}"


class SlashCommandRegistry:
    """Registry for slash commands with autocomplete support."""

    def __init__(self):
        self._commands: OrderedDict[str, SlashCommand] = OrderedDict()
        self._register_default_commands()

    def _register_default_commands(self):
        """Register default built-in commands."""
        default_commands = [
            SlashCommand(
                name="help",
                description="显示可用命令与帮助信息",
                usage="/help",
                examples=["/help"],
                category="通用",
            ),
            SlashCommand(
                name="exit",
                description="退出 CLI",
                usage="/exit",
                examples=["/exit", "/quit"],
                category="通用",
            ),
            SlashCommand(
                name="quit",
                description="退出 CLI（/exit 的别名）",
                usage="/quit",
                examples=["/quit"],
                category="通用",
            ),
            SlashCommand(
                name="pwd",
                description="显示当前工作目录",
                usage="/pwd",
                examples=["/pwd"],
                category="文件系统",
            ),
            SlashCommand(
                name="clear",
                description="清空终端屏幕",
                usage="/clear",
                examples=["/clear"],
                category="通用",
            ),
            SlashCommand(
                name="model",
                description="打开编号选择器，或切换到已配置的模型预设",
                usage="/model [show|list|<preset>]",
                examples=[
                    "/model",
                    "/model show",
                    "/model list",
                    "/model glm",
                ],
                category="设置",
            ),
            SlashCommand(
                name="approval",
                description="控制高风险工具调用的人类审批开关",
                usage="/approval [on|off|status]",
                examples=[
                    "/approval on",
                    "/approval off",
                    "/approval status",
                ],
                category="设置",
            ),
            SlashCommand(
                name="history",
                description="显示命令历史",
                usage="/history",
                examples=["/history"],
                category="通用",
            ),
            SlashCommand(
                name="skills",
                description="列出、查看和重新加载技能",
                usage="/skills [list|reload|show <name>|review]",
                examples=[
                    "/skills",
                    "/skills reload",
                    "/skills show calculator",
                    "/skills review",
                ],
                category="技能",
            ),
            SlashCommand(
                name="memory",
                description="查看持久记忆和 memory review 记录",
                usage="/memory [list|review]",
                examples=[
                    "/memory",
                    "/memory list",
                    "/memory review",
                ],
                category="记忆",
            ),
            SlashCommand(
                name="reset",
                description="重置会话上下文",
                usage="/reset",
                examples=["/reset"],
                category="会话",
            ),
            SlashCommand(
                name="init",
                description="生成 TGAGENTS.md",
                usage="/init",
                examples=["/init"],
                category="会话",
            ),
            SlashCommand(
                name="tasks",
                description="列出所有后台任务",
                usage="/tasks",
                examples=["/tasks"],
                category="后台任务",
            ),
            SlashCommand(
                name="task",
                description="查看指定后台任务的详情",
                usage="/task <task_id>",
                examples=["/task abc123"],
                category="后台任务",
            ),
            SlashCommand(
                name="task_cancel",
                description="取消正在运行的后台任务",
                usage="/task_cancel <task_id>",
                examples=["/task_cancel abc123"],
                category="后台任务",
            ),
            SlashCommand(
                name="allow",
                description="将目录加入沙箱允许列表",
                usage="/allow <path>",
                examples=["/allow /path/to/project", "/allow .."],
                category="文件系统",
            ),
            SlashCommand(
                name="allowed",
                description="列出沙箱中所有已允许的目录",
                usage="/allowed",
                examples=["/allowed"],
                category="文件系统",
            ),
            SlashCommand(
                name="agents",
                description="管理智能体配置",
                usage="/agents [list|show|create|edit|delete|reload]",
                examples=[
                    "/agents",
                    "/agents list workspace",
                    "/agents show code_reviewer",
                    "/agents create my_agent",
                    "/agents create my_agent --scope user",
                    "/agents edit code_reviewer",
                    "/agents delete old_agent",
                    "/agents reload",
                ],
                category="智能体",
            ),
            SlashCommand(
                name="team",
                description="管理多进程 agent team，或让 lead 自动编排",
                usage="/team [auto <goal> [--name <name>]|create <goal> [--name <name>]|use|list|spawn|task|tasks|members|inbox|send|status|stop|shutdown]",
                examples=[
                    '/team auto "修复当前项目里的类型错误" --name fix-types',
                    "/team create 分析当前项目 --name analysis-team",
                    "/team spawn <team_id> explorer-1 --agent explore",
                    '/team task "分析 subagent runtime" --to explorer-1',
                    "/team tasks",
                    "/team inbox",
                    "/team shutdown",
                ],
                category="智能体",
            ),
            SlashCommand(
                name="ralph",
                description="运行 Ralph 工作流命令",
                usage="/ralph <init-spec|init-agent|dry-run|run|status|cancel> ...",
                examples=[
                    "/ralph init-spec my_spec",
                    "/ralph init-agent",
                    "/ralph dry-run my_spec",
                    "/ralph run my_spec --silent",
                    "/ralph status",
                    "/ralph cancel abc123",
                ],
                category="Ralph",
            ),
            SlashCommand(
                name="plugins",
                description="列出、查看、复制、安装、卸载和重载内置插件",
                usage="/plugins [list|show|copy|reload|install|uninstall]",
                examples=[
                    "/plugins",
                    "/plugins list",
                    "/plugins show review-kit",
                    "/plugins copy review-kit",
                    "/plugins reload",
                    "/plugins install /path/to/my-plugin",
                    "/plugins uninstall my-plugin",
                    "/plugins uninstall my-plugin --force",
                ],
                category="插件",
            ),
        ]

        for cmd in default_commands:
            self.register(cmd)

    def register(self, command: SlashCommand) -> None:
        self._commands[command.name] = command

    def unregister(self, name: str) -> None:
        if name in self._commands:
            del self._commands[name]

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name)

    def get_all(self) -> list[SlashCommand]:
        return list(self._commands.values())

    def get_by_category(self) -> dict[str, list[SlashCommand]]:
        categories: dict[str, list[SlashCommand]] = {}
        for cmd in self._commands.values():
            categories.setdefault(cmd.category, []).append(cmd)
        return categories

    def match_prefix(self, prefix: str) -> list[SlashCommand]:
        prefix_lower = prefix.lower()
        return [cmd for cmd in self._commands.values() if cmd.name.lower().startswith(prefix_lower)]


class SlashCommandCompleter(Completer):
    """Completer for slash commands."""

    def __init__(self, registry: SlashCommandRegistry):
        self._registry = registry

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> iter:
        text = document.text_before_cursor
        if not text or text[0] != "/":
            return

        command_part = text[1:].split()[0] if len(text) > 1 else ""
        matching_commands = self._registry.match_prefix(command_part)

        for cmd in matching_commands:
            if command_part == cmd.name:
                insert_text = cmd.name + " "
            else:
                insert_text = cmd.name

            display = f"/{cmd.name}"
            display_meta = cmd.description
            slash_pos = text.find("/")
            cmd_word_start = slash_pos + 1
            while cmd_word_start < len(text) and text[cmd_word_start] == " ":
                cmd_word_start += 1

            cursor_pos = len(text)
            start_position = cmd_word_start - cursor_pos

            yield Completion(
                insert_text,
                start_position=start_position,
                display=display,
                display_meta=display_meta,
            )


def is_slash_command(text: str) -> bool:
    return text.strip().startswith("/")


@dataclass(slots=True)
class ParsedSlashCommand:
    name: str
    args: list[str]
    args_text: str = ""


def parse_slash_command(text: str) -> ParsedSlashCommand:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return ParsedSlashCommand(name="", args=[], args_text="")

    command_body = stripped[1:].lstrip()
    if not command_body:
        return ParsedSlashCommand(name="", args=[], args_text="")

    match = re.match(r"^(?P<name>\S+)(?P<rest>[\s\S]*)$", command_body)
    if match is None:
        return ParsedSlashCommand(name="", args=[], args_text="")

    command_name = match.group("name")
    args_text = match.group("rest").lstrip()
    args = args_text.split() if args_text else []
    return ParsedSlashCommand(name=command_name, args=args, args_text=args_text)
