"""Tools for Claude Code CLI."""

from tools.bash import bash
from tools.files import edit, read, write
from tools.resolve_path import resolve_path
from tools.sandbox import SandboxContext, get_sandbox_context, SecurityError
from tools.search import glob_search, grep
from tools.todos import done, todo_read, todo_write
from tools.task_output import task_output
from tools.task_status import task_status
from tools.task_cancel import task_cancel
from tools.xlsx import read_excel

from tools.run_subagent import run_subagent
from tools.run_parallel_subagents import run_parallel_subagents

# from tools.subagent_mp import (
#     spawn_subagent,
#     get_subagent_result,
#     check_subagent_status,
#     get_subagent_stats,
#     launch_parallel_subagents,
# )

ALL_TOOLS = [
    bash,
    resolve_path,
    read,
    read_excel,
    write,
    edit,
    glob_search,
    grep,
    todo_read,
    todo_write,
    done,
    task_output,
    task_status,
    task_cancel,
    run_subagent,
    run_parallel_subagents,
]

__all__ = [
    "ALL_TOOLS",
    "SandboxContext",
    "SecurityError",
    "get_sandbox_context",
    "bash",
    "resolve_path",
    "read",
    "read_excel",
    "write",
    "edit",
    "glob_search",
    "grep",
    "todo_read",
    "todo_write",
    "done",
    "task_output",
    "task_status",
    "task_cancel",
    "run_subagent",
    "run_parallel_subagents",
]
