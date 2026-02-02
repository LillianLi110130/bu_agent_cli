"""Todo management tools."""

from bu_agent_sdk.tools import Depends, tool
from typing import Annotated

from tools.sandbox import SandboxContext, get_sandbox_context


# Session-scoped todo storage
_todos: dict[str, list[dict]] = {}


@tool("Read current todo list")
async def todo_read(
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Get the current todo list."""
    todos = _todos.get(ctx.session_id, [])
    if not todos:
        return "Todo list is empty"
    lines = []
    for i, t in enumerate(todos, 1):
        status = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[t["status"]]
        lines.append(f"{i}. {status} {t['content']}")
    return "\n".join(lines)


@tool("Update the todo list")
async def todo_write(
    todos: list[dict],
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Set the todo list. Each item needs: content, status, activeForm"""
    _todos[ctx.session_id] = todos
    stats = {
        "pending": sum(1 for t in todos if t.get("status") == "pending"),
        "in_progress": sum(1 for t in todos if t.get("status") == "in_progress"),
        "completed": sum(1 for t in todos if t.get("status") == "completed"),
    }
    return f"Updated todos: {stats['pending']} pending, {stats['in_progress']} in progress, {stats['completed']} completed"


@tool("Signal that the task is complete")
async def done(message: str) -> str:
    """Call this when the task is finished."""
    return f"TASK COMPLETE: {message}"
