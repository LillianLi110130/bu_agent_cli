"""Permission decisions for tool calls."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agent_core.agent.command_safety import (
    CommandSafetyFinding,
    CommandSafetySeverity,
    check_dangerous_command,
    command_preview,
    format_findings,
    safety_session_label,
)
from agent_core.agent.hitl import HumanApprovalRequest
from agent_core.agent.tool_args import ToolArgumentsError, parse_tool_arguments_for_execution

if TYPE_CHECKING:
    from agent_core.agent.hooks import HookContext
    from agent_core.agent.runtime_events import ToolCallRequested


PermissionDecisionType = Literal["allow", "ask", "deny"]


@dataclass(frozen=True)
class PermissionReason:
    source: str
    rule_id: str | None
    message: str
    category: str | None = None
    severity: str | None = None
    guidance: str | None = None


@dataclass(frozen=True)
class PermissionDecision:
    decision: PermissionDecisionType
    reasons: tuple[PermissionReason, ...] = ()
    approval_request: HumanApprovalRequest | None = None

    @classmethod
    def allow(cls) -> "PermissionDecision":
        return cls(decision="allow")


class PermissionEngine:
    """Evaluate tool calls before execution."""

    def evaluate_tool_call(
        self,
        event: "ToolCallRequested",
        ctx: "HookContext",
    ) -> PermissionDecision:
        if event.tool_call.function.name != "bash":
            return PermissionDecision.allow()

        try:
            args = parse_tool_arguments_for_execution(event.tool_call.function.arguments)
        except ToolArgumentsError:
            return PermissionDecision.allow()

        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return PermissionDecision.allow()

        safety_result = check_dangerous_command(command)
        safety_reasons = _reasons_from_findings(safety_result.findings)
        git_read_allowed = _is_allowed_git_read_command(command, ctx)
        file_task_reasons = () if git_read_allowed else _evaluate_bash_file_task(command)
        reasons = (*safety_reasons, *file_task_reasons)

        if safety_result.action == "block" or file_task_reasons:
            return PermissionDecision(decision="deny", reasons=reasons)

        if safety_result.action == "allow":
            return PermissionDecision.allow()

        findings = safety_result.ask_findings
        request = HumanApprovalRequest(
            tool_name="bash",
            tool_call_id=event.tool_call.id,
            arguments=args,
            reason=(
                "该 bash 命令命中安全审查规则，需要人工确认。\n"
                f"Matched safety rule(s):\n{format_findings(findings)}"
            ),
            risk_level=_highest_severity(findings),
            command_preview=command_preview(command, max_chars=200),
            approval_kind="safety",
            approval_keys=tuple(finding.approval_key for finding in findings),
            session_approval_label=safety_session_label(findings),
        )
        return PermissionDecision(
            decision="ask",
            reasons=_reasons_from_findings(findings),
            approval_request=request,
        )


_BASH_FILE_DISCOVERY_RE = re.compile(
    r"(?ix)"
    r"(?:^|[;&|]\s*)"
    r"(?:"
    r"dir\b|"
    r"ls\b|"
    r"tree\b|"
    r"fd\b|"
    r"where\b|"
    r"get-childitem\b|"
    r"gci\b|"
    r"git(?:\s+[^;&|\s]+)*\s+ls-files\b"
    r")"
)
_BASH_TEXT_SEARCH_RE = re.compile(
    r"(?ix)"
    r"(?:^|[;&|]\s*)"
    r"(?:"
    r"find\b(?![^;&|]*(?:-delete|-exec\s+(?:/\S*/)?rm\b))|"
    r"findstr\b|"
    r"grep\b|"
    r"rg\b|"
    r"select-string\b|"
    r"git(?:\s+[^;&|\s]+)*\s+grep\b"
    r")"
)
_BASH_FILE_READ_RE = re.compile(
    r"(?ix)"
    r"(?:^|[;&|]\s*)"
    r"(?:"
    r"type\b|"
    r"cat\b|"
    r"more\b|"
    r"get-content\b|"
    r"gc\b|"
    r"git(?:\s+[^;&|\s]+)*\s+show\b|"
    r"git(?:\s+[^;&|\s]+)*\s+diff\b"
    r")"
)
_SHELL_TASK_LOG_PATH_RE = re.compile(
    r"(?i)(?:^|[\\/\s\"'])(?:\.tg_agent|\.tgagent)[\\/]"
    r"shell_tasks[\\/][^\\/\s\"']+[\\/]([A-Za-z0-9_-]+)\.log\b"
)
_HIDDEN_EXECUTION_RE = re.compile(
    r"(?ix)"
    r"\b(?:bash|sh|zsh|ksh)\s+-[^\s]*c(?:\s|$)|"
    r"\b(?:python[23]?|perl|ruby|node)\s+-[ec]\s+|"
    r"\b(?:python[23]?|perl|ruby|node)\s+<<"
)
_TRAILING_STDERR_MERGE_RE = re.compile(r"\s+2>&1\s*$")
_WINDOWS_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_GIT_READ_SUBCOMMANDS = {"show", "grep", "ls-files", "diff"}
_FORBIDDEN_GIT_OPTIONS = {"-C", "--git-dir", "--work-tree"}
_SAFE_GIT_SHOW_OPTIONS = {
    "--stat",
    "--name-only",
    "--patch",
    "-p",
    "--oneline",
}
_SAFE_GIT_GREP_OPTIONS = {
    "-i",
    "-n",
    "-E",
    "-F",
    "-v",
    "--ignore-case",
    "--line-number",
    "--extended-regexp",
    "--fixed-strings",
    "--invert-match",
}
_SAFE_GIT_LS_FILES_OPTIONS = {
    "-z",
    "-m",
    "-d",
    "-o",
    "-s",
    "--others",
    "--cached",
    "--deleted",
    "--modified",
    "--stage",
}
_SAFE_GIT_DIFF_OPTIONS = {
    "--stat",
    "--name-only",
    "--name-status",
    "--patch",
    "-p",
    "--cached",
    "--staged",
}
_SAFE_GREP_SHORT_FLAGS = set("inEFv")
_SAFE_GREP_COUNT_FLAGS = {"-A", "-B", "-C"}
_FORBIDDEN_GREP_OPTIONS = {
    "-r",
    "-R",
    "-f",
    "--recursive",
    "--dereference-recursive",
    "--file",
    "--include",
    "--exclude",
    "--exclude-dir",
}
_SAFE_FINDSTR_SWITCHES = {"/i", "/n", "/r"}


def _is_allowed_git_read_command(command: str, ctx: "HookContext") -> bool:
    stripped = _strip_trailing_stderr_merge(command.strip())
    if stripped is None or not stripped:
        return False
    if _has_forbidden_shell_syntax(stripped):
        return False

    pipeline_parts = _split_unquoted(stripped, "|")
    if pipeline_parts is None or len(pipeline_parts) > 2:
        return False

    git_part = pipeline_parts[0].strip()
    filter_part = pipeline_parts[1].strip() if len(pipeline_parts) == 2 else None
    if filter_part and not _is_safe_filter_command(filter_part):
        return False

    git_part = _strip_optional_safe_cd(git_part, ctx)
    if git_part is None:
        return False

    words = _parse_simple_shell_words(git_part)
    if words is None:
        return False
    return _is_safe_git_read_words(words)


def _strip_trailing_stderr_merge(command: str) -> str | None:
    match = _TRAILING_STDERR_MERGE_RE.search(command)
    if match is None:
        return command
    if _quote_state(command[: match.start()]) is not None:
        return None
    return command[: match.start()].rstrip()


def _strip_optional_safe_cd(command: str, ctx: "HookContext") -> str | None:
    parts = _split_unquoted(command, "&&")
    if parts is None or len(parts) > 2:
        return None
    if len(parts) == 1:
        return parts[0].strip()

    cd_words = _parse_simple_shell_words(parts[0].strip())
    if cd_words is None or not _is_safe_cd_words(cd_words, ctx):
        return None
    return parts[1].strip()


def _is_safe_cd_words(words: list[str], ctx: "HookContext") -> bool:
    if not words or words[0].lower() != "cd":
        return False
    if len(words) == 2:
        path = words[1]
    elif len(words) == 3 and words[1].lower() == "/d":
        path = words[2]
    else:
        return False

    sandbox_context = getattr(getattr(ctx, "agent", None), "_sandbox_context", None)
    if sandbox_context is None:
        return False
    try:
        sandbox_context.resolve_path(path)
    except Exception:
        return False
    return True


def _is_safe_git_read_words(words: list[str]) -> bool:
    if len(words) < 2 or words[0].lower() != "git":
        return False
    if _contains_forbidden_git_option(words):
        return False

    subcommand = words[1].lower()
    if subcommand not in _GIT_READ_SUBCOMMANDS:
        return False
    args = words[2:]
    if args.count("--") > 1:
        return False
    if not _all_obvious_paths_are_safe(args):
        return False

    if subcommand == "show":
        return _is_safe_git_show_args(args)
    if subcommand == "grep":
        return _is_safe_git_grep_args(args)
    if subcommand == "ls-files":
        return _is_safe_git_ls_files_args(args)
    if subcommand == "diff":
        return _is_safe_git_diff_args(args)
    return False


def _contains_forbidden_git_option(words: list[str]) -> bool:
    for word in words:
        if word in _FORBIDDEN_GIT_OPTIONS:
            return True
        if word.startswith("-C") and word != "-":
            return True
        if word.startswith("--git-dir") or word.startswith("--work-tree"):
            return True
    return False


def _is_safe_git_show_args(args: list[str]) -> bool:
    pathspecs = _pathspecs_after_separator(args)
    if pathspecs is not None and not all(_is_safe_git_pathspec(item) for item in pathspecs):
        return False

    for arg in _args_before_separator(args):
        if arg in _SAFE_GIT_SHOW_OPTIONS:
            continue
        if arg.startswith("-"):
            return False
        if ":" in arg and _looks_like_git_show_object_path(arg):
            _, path = arg.split(":", 1)
            if not _is_safe_git_pathspec(path):
                return False
    return True


def _is_safe_git_grep_args(args: list[str]) -> bool:
    pathspecs = _pathspecs_after_separator(args)
    if pathspecs is not None and not all(_is_safe_git_pathspec(item) for item in pathspecs):
        return False

    value_count = 0
    index = 0
    before = _args_before_separator(args)
    while index < len(before):
        arg = before[index]
        if _is_safe_git_grep_option(arg):
            index += 1
            continue
        if arg.startswith("-"):
            return False
        value_count += 1
        index += 1
    return value_count >= 1


def _is_safe_git_ls_files_args(args: list[str]) -> bool:
    before = _args_before_separator(args)
    after = _pathspecs_after_separator(args) or []
    for arg in before:
        if arg in _SAFE_GIT_LS_FILES_OPTIONS:
            continue
        if arg.startswith("-"):
            return False
        if not _is_safe_git_pathspec(arg):
            return False
    return all(_is_safe_git_pathspec(item) for item in after)


def _is_safe_git_diff_args(args: list[str]) -> bool:
    pathspecs = _pathspecs_after_separator(args)
    if pathspecs is not None and not all(_is_safe_git_pathspec(item) for item in pathspecs):
        return False

    for arg in _args_before_separator(args):
        if arg in _SAFE_GIT_DIFF_OPTIONS:
            continue
        if arg.startswith("-"):
            return False
        if _looks_like_unsafe_path(arg):
            return False
    return True


def _is_safe_git_grep_option(arg: str) -> bool:
    if arg in _SAFE_GIT_GREP_OPTIONS:
        return True
    if arg.startswith("-") and not arg.startswith("--"):
        return all(char in _SAFE_GREP_SHORT_FLAGS for char in arg[1:])
    return False


def _pathspecs_after_separator(args: list[str]) -> list[str] | None:
    if "--" not in args:
        return None
    index = args.index("--")
    if "--" in args[index + 1 :]:
        return None
    return args[index + 1 :]


def _args_before_separator(args: list[str]) -> list[str]:
    if "--" not in args:
        return args
    return args[: args.index("--")]


def _looks_like_git_show_object_path(arg: str) -> bool:
    if _WINDOWS_DRIVE_PATH_RE.match(arg):
        return True
    rev, path = arg.split(":", 1)
    return bool(rev and path)


def _all_obvious_paths_are_safe(args: list[str]) -> bool:
    return all(not _looks_like_unsafe_path(arg) for arg in args if arg != "--")


def _is_safe_git_pathspec(value: str) -> bool:
    return bool(value) and not _looks_like_unsafe_path(value) and "*" not in value


def _looks_like_unsafe_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("//"):
        return True
    if _WINDOWS_DRIVE_PATH_RE.match(value):
        return True
    return ".." in normalized.split("/")


def _is_safe_filter_command(command: str) -> bool:
    words = _parse_simple_shell_words(command)
    if words is None or not words:
        return False
    name = words[0].lower()
    if name == "grep":
        return _is_safe_grep_filter(words[1:])
    if name == "findstr":
        return _is_safe_findstr_filter(words[1:])
    return False


def _is_safe_grep_filter(args: list[str]) -> bool:
    pattern_count = 0
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in _FORBIDDEN_GREP_OPTIONS or any(
            arg.startswith(f"{option}=") for option in _FORBIDDEN_GREP_OPTIONS
        ):
            return False
        if arg in {"-i", "-n", "-E", "-F", "-v"}:
            index += 1
            continue
        if arg.startswith("-") and not arg.startswith("--"):
            flag_body = arg[1:]
            if flag_body in {"A", "B", "C"}:
                if index + 1 >= len(args) or not args[index + 1].isdigit():
                    return False
                index += 2
                continue
            if len(flag_body) > 1 and flag_body[0] in {"A", "B", "C"}:
                if not flag_body[1:].isdigit():
                    return False
                index += 1
                continue
            if all(char in _SAFE_GREP_SHORT_FLAGS for char in flag_body):
                index += 1
                continue
            return False
        if arg in _SAFE_GREP_COUNT_FLAGS:
            if index + 1 >= len(args) or not args[index + 1].isdigit():
                return False
            index += 2
            continue
        if arg.startswith("-"):
            return False
        if _looks_like_unsafe_path(arg):
            return False
        pattern_count += 1
        index += 1
    return pattern_count == 1


def _is_safe_findstr_filter(args: list[str]) -> bool:
    pattern_count = 0
    c_pattern_count = 0
    for arg in args:
        lower = arg.lower()
        if lower.startswith("/g:") or lower.startswith("/f:"):
            return False
        if lower in _SAFE_FINDSTR_SWITCHES:
            continue
        if lower.startswith("/c:"):
            c_pattern_count += 1
            continue
        if lower.startswith("/"):
            return False
        if _looks_like_unsafe_path(arg):
            return False
        pattern_count += 1
    return (c_pattern_count == 1 and pattern_count == 0) or (
        c_pattern_count == 0 and pattern_count == 1
    )


def _has_forbidden_shell_syntax(command: str) -> bool:
    quote: str | None = None
    index = 0
    pipe_count = 0
    and_count = 0
    while index < len(command):
        char = command[index]
        if quote:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == ";" or char in {"<", ">"} or char == "`":
            return True
        if char == "$" and index + 1 < len(command) and command[index + 1] == "(":
            return True
        if char == "|":
            if index + 1 < len(command) and command[index + 1] == "|":
                return True
            pipe_count += 1
        if char == "&":
            if index + 1 < len(command) and command[index + 1] == "&":
                and_count += 1
                index += 2
                continue
            return True
        index += 1
    return quote is not None or pipe_count > 1 or and_count > 1


def _split_unquoted(command: str, separator: str) -> list[str] | None:
    quote: str | None = None
    parts: list[str] = []
    start = 0
    index = 0
    while index < len(command):
        char = command[index]
        if quote:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if command.startswith(separator, index):
            parts.append(command[start:index])
            index += len(separator)
            start = index
            continue
        index += 1
    if quote is not None:
        return None
    parts.append(command[start:])
    return parts


def _parse_simple_shell_words(command: str) -> list[str] | None:
    words: list[str] = []
    current: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(command):
        char = command[index]
        if quote:
            if char == quote:
                quote = None
            else:
                current.append(char)
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char.isspace():
            if current:
                words.append("".join(current))
                current = []
            index += 1
            continue
        current.append(char)
        index += 1
    if quote is not None:
        return None
    if current:
        words.append("".join(current))
    return words


def _quote_state(text: str) -> str | None:
    quote: str | None = None
    for char in text:
        if quote:
            if char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
    return quote


def _evaluate_bash_file_task(command: str) -> tuple[PermissionReason, ...]:
    if _HIDDEN_EXECUTION_RE.search(command):
        return ()

    shell_task_match = _SHELL_TASK_LOG_PATH_RE.search(command)
    if shell_task_match and _BASH_FILE_READ_RE.search(command):
        task_id = shell_task_match.group(1)
        return (
            PermissionReason(
                source="bash_file_task",
                rule_id="shell_task_log_read",
                message="Bash command is polling or reading a background shell task log.",
                category="tool_boundary",
                severity="medium",
                guidance=(
                    f'Use `task_output(task_id="{task_id}", wait_for=..., timeout=...)` instead. '
                    "`task_output` handles waiting and returns the task status, return code, "
                    "log path, and output. Do not use `sleep` or `cat` to read this log."
                ),
            ),
        )
    if _BASH_FILE_DISCOVERY_RE.search(command):
        return (
            PermissionReason(
                source="bash_file_task",
                rule_id="file_discovery",
                message="Bash command is doing file discovery or directory listing.",
                category="tool_boundary",
                severity="medium",
                guidance=(
                    "Use `glob_search` to list files and folders, and `resolve_path` first if "
                    "the target path is fuzzy. Do not retry the same discovery step with `bash`, "
                    "because that bypasses sandbox and .tgagentignore protections."
                ),
            ),
        )
    if _BASH_TEXT_SEARCH_RE.search(command):
        return (
            PermissionReason(
                source="bash_file_task",
                rule_id="text_search",
                message="Bash command is doing text search inside files.",
                category="tool_boundary",
                severity="medium",
                guidance=(
                    "Use `grep` for content search, and `resolve_path` first when the search "
                    "scope path is ambiguous. Do not retry the same search step with `bash`, "
                    "because that bypasses sandbox and .tgagentignore protections."
                ),
            ),
        )
    if _BASH_FILE_READ_RE.search(command):
        return (
            PermissionReason(
                source="bash_file_task",
                rule_id="file_read",
                message="Bash command is trying to read file contents.",
                category="tool_boundary",
                severity="medium",
                guidance=(
                    "Use `read` for text files. If the path is unclear, call `resolve_path` "
                    "first. Do not retry the same read step with `bash`, because that bypasses "
                    "sandbox and .tgagentignore protections."
                ),
            ),
        )
    return ()


def _reasons_from_findings(
    findings: tuple[CommandSafetyFinding, ...],
) -> tuple[PermissionReason, ...]:
    return tuple(
        PermissionReason(
            source="command_safety",
            rule_id=finding.rule_id,
            message=finding.description,
            category=finding.category,
            severity=finding.severity,
        )
        for finding in findings
    )


def _highest_severity(
    findings: tuple[CommandSafetyFinding, ...],
) -> CommandSafetySeverity:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    if not findings:
        return "medium"
    return max((finding.severity for finding in findings), key=lambda item: order[item])
