"""Tools for Claude Code CLI."""

from tools.bash import bash
from tools.files import edit, read, write
from tools.sandbox import SandboxContext, get_sandbox_context, SecurityError
from tools.search import glob_search, grep
from tools.todos import done, todo_read, todo_write
# from tools.async_task import async_task
# from tools.task_status import task_status
# from tools.task_cancel import task_cancel
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
    read,
    write,
    edit,
    glob_search,
    grep,
    todo_read,
    todo_write,
    done,
    # async_task,
    # task_status,
    # task_cancel,
    run_subagent,
    run_parallel_subagents,
]

__all__ = [
    "ALL_TOOLS",
    "SandboxContext",
    "SecurityError",
    "get_sandbox_context",
    "bash",
    "read",
    "write",
    "edit",
    "glob_search",
    "grep",
    "todo_read",
    "todo_write",
    "done",
    # "async_task",
    # "task_status",
    # "task_cancel",
    "run_subagent",
    "run_parallel_subagents",
]
