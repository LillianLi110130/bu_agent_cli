"""Dangerous shell command detection for bash tool approvals."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Pattern

from agent_core.agent.runtime_events import ToolCallRequested
from agent_core.agent.tool_args import ToolArgumentsError, parse_tool_arguments_for_execution

if TYPE_CHECKING:
    from agent_core.agent.hitl import HumanApprovalRequest
    from agent_core.agent.hooks import HookContext


CommandSafetyAction = Literal["allow", "ask", "block"]
CommandSafetySeverity = Literal["low", "medium", "high", "critical"]
CommandSafetyRuleAction = Literal["ask", "block"]

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CommandSafetyRule:
    id: str
    pattern: Pattern[str]
    description: str
    category: str
    severity: CommandSafetySeverity
    action: CommandSafetyRuleAction
    session_label: str


@dataclass(frozen=True)
class CommandSafetyFinding:
    rule_id: str
    description: str
    category: str
    severity: CommandSafetySeverity
    action: CommandSafetyRuleAction
    session_label: str

    @property
    def approval_key(self) -> str:
        return f"safety:{self.rule_id}"


@dataclass(frozen=True)
class CommandSafetyResult:
    action: CommandSafetyAction
    findings: tuple[CommandSafetyFinding, ...] = ()

    @property
    def blocked_findings(self) -> tuple[CommandSafetyFinding, ...]:
        return tuple(finding for finding in self.findings if finding.action == "block")

    @property
    def ask_findings(self) -> tuple[CommandSafetyFinding, ...]:
        return tuple(finding for finding in self.findings if finding.action == "ask")


def _rule(
    rule_id: str,
    pattern: str,
    description: str,
    category: str,
    severity: CommandSafetySeverity,
    action: CommandSafetyRuleAction,
    session_label: str,
) -> CommandSafetyRule:
    return CommandSafetyRule(
        id=rule_id,
        pattern=re.compile(pattern, re.IGNORECASE | re.DOTALL),
        description=description,
        category=category,
        severity=severity,
        action=action,
        session_label=session_label,
    )


DANGEROUS_COMMAND_RULES: tuple[CommandSafetyRule, ...] = (
    _rule(
        "fork_bomb",
        r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
        "Fork bomb pattern can rapidly exhaust system processes.",
        "process",
        "critical",
        "block",
        "fork bomb",
    ),
    _rule(
        "rm_recursive_broad_scope",
        r"\brm\s+-[^\s;|&]*r[^\s;|&]*\s+(?:/|\*|\.|\.\.)(?:\s|$|[;&|])",
        "Recursive deletion targets root, wildcard, current directory, or parent directory.",
        "filesystem",
        "critical",
        "block",
        "rm 递归大范围删除",
    ),
    _rule(
        "mkfs_filesystem",
        r"\bmkfs(?:\.[a-z0-9_+-]+)?\b",
        "Filesystem formatting command can destroy disk contents.",
        "disk",
        "critical",
        "block",
        "mkfs",
    ),
    _rule(
        "dd_write_block_device",
        r"\bdd\b(?=.*\bof\s*=\s*/dev/(?:sd|hd|vd|nvme|disk))",
        "dd writes directly to a block device.",
        "disk",
        "critical",
        "block",
        "dd 写裸设备",
    ),
    _rule(
        "redirect_block_device",
        r">\s*/dev/(?:sd|hd|vd|nvme|disk)",
        "Shell redirection writes directly to a block device.",
        "disk",
        "critical",
        "block",
        "重定向写裸设备",
    ),
    _rule(
        "powershell_disk_destroy",
        r"\b(format-volume|clear-disk|remove-partition)\b",
        "PowerShell disk or partition destructive operation.",
        "disk",
        "critical",
        "block",
        "PowerShell 磁盘破坏",
    ),
    _rule(
        "remote_script_to_shell",
        r"\b(curl|wget|iwr|irm|invoke-webrequest|invoke-restmethod)\b.*\|\s*"
        r"(?:ba)?sh\b|\b(curl|wget)\b.*\|\s*(?:zsh|ksh)\b|"
        r"\b(iwr|irm|invoke-webrequest|invoke-restmethod)\b.*\|\s*(iex|invoke-expression)\b",
        "Remote content is piped directly into a shell or expression evaluator.",
        "remote_execution",
        "critical",
        "block",
        "远程脚本直接执行",
    ),
    _rule(
        "kill_all_processes",
        r"\bkill\s+-9\s+-1\b",
        "Command attempts to force kill all accessible processes.",
        "process",
        "critical",
        "block",
        "kill -9 -1",
    ),
    _rule(
        "kill_agent_process",
        r"\b(pkill|killall|taskkill)\b(?=.*\b(crab|tg_crab_main\.py|cli\.worker|gateway)\b)",
        "Command attempts to terminate the agent, worker, or gateway process.",
        "self_protection",
        "critical",
        "block",
        "终止 agent 相关进程",
    ),
    _rule(
        "delete_tg_agent_dir_rm",
        r"\brm\b(?=[^;&|]*(?:-[^\s;|&]*r|--recursive\b))"
        r"[^;&|]*\s(?:--\s+)?[\"']?(?:[^\"'\s;&|]+[\\/])?\.tg_agent[\\/]?[\"']?"
        r"(?:\s|$|[;&|])",
        "Command deletes the agent runtime/state directory.",
        "self_protection",
        "critical",
        "block",
        "删除 .tg_agent",
    ),
    _rule(
        "delete_tg_agent_dir_cmd",
        r"\b(?:rmdir|rd)\b(?=[^;&|]*/s\b)"
        r"[^;&|]*\s[\"']?(?:[^\"'\s;&|]+[\\/])?\.tg_agent[\\/]?[\"']?"
        r"(?:\s|$|[;&|])",
        "Command deletes the agent runtime/state directory.",
        "self_protection",
        "critical",
        "block",
        "删除 .tg_agent",
    ),
    _rule(
        "delete_tg_agent_dir_powershell",
        r"\b(?:remove-item|rm|del|erase|rd|rmdir)\b"
        r"(?=[^;&|]*(?:-recurse\b|\brecurse\b|/s\b))"
        r"[^;&|]*[\"']?(?:[^\"'\s;&|]+[\\/])?\.tg_agent[\\/]?[\"']?"
        r"(?:\s|$|[;&|])",
        "Command deletes the agent runtime/state directory.",
        "self_protection",
        "critical",
        "block",
        "删除 .tg_agent",
    ),
    _rule(
        "rm_recursive",
        r"\brm\b(?=[^;&|]*(?:-[^\s;|&]*r|--recursive\b))[^;&|]*\s+\S+",
        "Recursive deletion can remove user files or project state.",
        "filesystem",
        "high",
        "ask",
        "rm 递归删除",
    ),
    _rule(
        "powershell_recursive_delete",
        r"\b(remove-item|rm|del|erase|rd|rmdir)\b"
        r"(?=[^;&|]*(?:-recurse\b|\brecurse\b|/s\b))",
        "Recursive deletion can remove user files or project state.",
        "filesystem",
        "high",
        "ask",
        "Remove-Item 递归删除",
    ),
    _rule(
        "find_delete",
        r"\bfind\b.*(?:-delete|-exec\s+(?:/\S*/)?rm\b)",
        "find is used to delete files in bulk.",
        "filesystem",
        "high",
        "ask",
        "find 批量删除",
    ),
    _rule(
        "xargs_rm",
        r"\bxargs\b.*\brm\b",
        "xargs is used to delete files in bulk.",
        "filesystem",
        "high",
        "ask",
        "xargs rm",
    ),
    _rule(
        "git_reset_hard",
        r"\bgit\s+reset\s+--hard\b",
        "git reset --hard may destroy uncommitted changes.",
        "source_control",
        "high",
        "ask",
        "git reset --hard",
    ),
    _rule(
        "git_clean_force",
        r"\bgit\s+clean\s+-[^\s;|&]*f",
        "git clean -f may delete untracked user files.",
        "source_control",
        "high",
        "ask",
        "git clean -f",
    ),
    _rule(
        "git_push_force",
        r"\bgit\s+push\b(?=[^;&|]*(?:--force(?:-with-lease)?\b|\s-f(?:\s|$|[;&|])))",
        "git force push may overwrite remote history.",
        "source_control",
        "high",
        "ask",
        "git push --force",
    ),
    _rule(
        "git_branch_delete_force",
        r"\bgit\s+branch\s+-D\b",
        "git branch -D forcibly deletes a branch.",
        "source_control",
        "medium",
        "ask",
        "git branch -D",
    ),
    _rule(
        "world_writable_permissions",
        r"\bchmod\s+(?:-[^\s]+\s+)*(?:777|666|[ao]\+[rwx]*w)\b",
        "Command makes files or directories broadly writable.",
        "permissions",
        "high",
        "ask",
        "chmod 全局可写",
    ),
    _rule(
        "recursive_chown_root",
        r"\bchown\s+(?:-[^\s]*r[^\s]*|--recursive)\s+root\b",
        "Recursive chown to root can make files inaccessible.",
        "permissions",
        "high",
        "ask",
        "chown -R root",
    ),
    _rule(
        "write_etc",
        r"(?:>\s*/etc/|\btee\b.*\s/etc/|\b(?:cp|mv|install)\b.*\s/etc/|"
        r"\bsed\s+(?:-[^\s]*i|--in-place)\b.*\s/etc/)",
        "Command writes to system configuration under /etc.",
        "system_config",
        "high",
        "ask",
        "写入 /etc",
    ),
    _rule(
        "systemctl_destructive",
        r"\bsystemctl\s+(?:-[^\s]+\s+)*(?:stop|restart|disable|mask)\b",
        "Command changes or stops system services.",
        "service",
        "high",
        "ask",
        "systemctl 服务变更",
    ),
    _rule(
        "force_kill_process",
        r"\bpkill\s+-9\b|\btaskkill\b(?=.*(?:/f|-f))|\bstop-process\b(?=.*-force\b)",
        "Command forcefully terminates processes.",
        "process",
        "high",
        "ask",
        "强制终止进程",
    ),
    _rule(
        "shell_command_via_c",
        r"\b(?:bash|sh|zsh|ksh)\s+-[^\s]*c(?:\s|$)",
        "Shell -c can hide or compose risky operations.",
        "execution",
        "medium",
        "ask",
        "shell -c",
    ),
    _rule(
        "inline_interpreter_execution",
        r"\b(?:python[23]?|perl|ruby|node)\s+-[ec]\s+",
        "Inline interpreter execution can hide risky operations.",
        "execution",
        "medium",
        "ask",
        "解释器 inline 执行",
    ),
    _rule(
        "interpreter_heredoc",
        r"\b(?:python[23]?|perl|ruby|node)\s+<<",
        "Interpreter heredoc executes script content from stdin.",
        "execution",
        "medium",
        "ask",
        "解释器 heredoc 执行",
    ),
    _rule(
        "chmod_execute_script",
        r"\bchmod\s+\+x\b.*[;&|]+\s*\./",
        "Command makes a local script executable and immediately runs it.",
        "execution",
        "medium",
        "ask",
        "chmod +x 后执行脚本",
    ),
)


def normalize_command(command: str) -> str:
    normalized = _ANSI_ESCAPE_RE.sub("", command)
    normalized = normalized.replace("\x00", "")
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = normalized.lower()
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def check_dangerous_command(command: str) -> CommandSafetyResult:
    normalized = normalize_command(command)
    if not normalized:
        return CommandSafetyResult(action="allow")

    findings: list[CommandSafetyFinding] = []
    for rule in DANGEROUS_COMMAND_RULES:
        if not rule.pattern.search(normalized):
            continue
        findings.append(
            CommandSafetyFinding(
                rule_id=rule.id,
                description=rule.description,
                category=rule.category,
                severity=rule.severity,
                action=rule.action,
                session_label=rule.session_label,
            )
        )

    if not findings:
        return CommandSafetyResult(action="allow")

    if any(finding.action == "block" for finding in findings):
        return CommandSafetyResult(action="block", findings=tuple(findings))
    return CommandSafetyResult(action="ask", findings=tuple(findings))


def command_preview(command: str, max_chars: int = 200) -> str:
    preview = _WHITESPACE_RE.sub(" ", command.replace("\x00", "")).strip()
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 3].rstrip() + "..."


def format_findings(findings: tuple[CommandSafetyFinding, ...]) -> str:
    return "\n".join(f"- {finding.rule_id}: {finding.description}" for finding in findings)


def safety_session_label(findings: tuple[CommandSafetyFinding, ...]) -> str:
    if not findings:
        return "该风险"
    labels = []
    seen = set()
    for finding in findings:
        if finding.session_label in seen:
            continue
        labels.append(finding.session_label)
        seen.add(finding.session_label)
    return "、".join(labels)


def build_command_safety_approval_policy():
    """Build a mandatory approval policy for ask-level bash safety findings."""

    from agent_core.agent.hitl import HumanApprovalRequest

    def policy(event: ToolCallRequested, ctx: "HookContext") -> HumanApprovalRequest | None:
        del ctx
        if event.tool_call.function.name != "bash":
            return None
        try:
            args = parse_tool_arguments_for_execution(event.tool_call.function.arguments)
        except ToolArgumentsError:
            return None

        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return None

        result = check_dangerous_command(command)
        if result.action != "ask":
            return None

        findings = result.ask_findings
        risk_level = _highest_severity(findings)
        label = safety_session_label(findings)
        reason = (
            "该 bash 命令命中安全审查规则，需要人工确认。\n"
            f"Matched safety rule(s):\n{format_findings(findings)}"
        )
        return HumanApprovalRequest(
            tool_name="bash",
            tool_call_id=event.tool_call.id,
            arguments=args,
            reason=reason,
            risk_level=risk_level,
            command_preview=command_preview(command, max_chars=200),
            approval_kind="safety",
            approval_keys=tuple(finding.approval_key for finding in findings),
            session_approval_label=label,
        )

    return policy


def _highest_severity(findings: tuple[CommandSafetyFinding, ...]) -> CommandSafetySeverity:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    if not findings:
        return "medium"
    return max((finding.severity for finding in findings), key=lambda item: order[item])
