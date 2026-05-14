"""Tools for Claude Code CLI."""

from tools.bash import bash
from tools.agent_tool import delegate, delegate_parallel
from tools.files import edit, read, write
from tools.message import message
from tools.resolve_path import resolve_path
from tools.sandbox import SandboxContext, SecurityError, get_sandbox_context
from tools.search import glob_search, grep
from tools.task_cancel import task_cancel
from tools.task_output import task_output
from tools.task_status import task_status
from tools.team_tool import (
    team_create,
    team_create_task,
    team_list_tasks,
    team_read_inbox,
    team_send_message,
    team_shutdown,
    team_snapshot,
    team_spawn_member,
    team_status,
    team_update_task,
)
from tools.todos import done, todo_read, todo_write
from tools.web import web_fetch
from tools.xlsx import read_excel

ALL_TOOLS = [
    bash,
    resolve_path,
    read,
    read_excel,
    # read_excel,
    write,
    edit,
    write,
    glob_search,
    grep,
    message,
    todo_read,
    todo_write,
    done,
    web_fetch,
    delegate,
    delegate_parallel,
    task_output,
    task_status,
    task_cancel,
    team_create,
    team_spawn_member,
    team_create_task,
    team_update_task,
    team_list_tasks,
    team_read_inbox,
    team_send_message,
    team_snapshot,
    team_status,
    team_shutdown,
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
    "message",
    "todo_read",
    "todo_write",
    "done",
    "web_fetch",
    "delegate",
    "delegate_parallel",
    "task_output",
    "task_status",
    "task_cancel",
    "team_create",
    "team_spawn_member",
    "team_create_task",
    "team_update_task",
    "team_list_tasks",
    "team_read_inbox",
    "team_send_message",
    "team_snapshot",
    "team_status",
    "team_shutdown",
]
