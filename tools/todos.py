"""Session-local todo planning tool."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from agent_core.agent import TaskComplete
from agent_core.llm.messages import BaseMessage, ToolMessage
from agent_core.tools import Depends, tool
from tools.sandbox import SandboxContext, get_sandbox_context

TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]
VALID_STATUSES: tuple[str, ...] = ("pending", "in_progress", "completed", "cancelled")
ACTIVE_TODO_SNAPSHOT_HEADER = "[Active Todo List Preserved Across Context Compression]"
_STATUS_MARKERS = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "completed": "[x]",
    "cancelled": "[-]",
}


class TodoItemPatch(BaseModel):
    """Todo item or merge patch."""

    id: str | None = Field(
        default=None,
        description=(
            "Stable task id. Required for merge updates; replace mode auto-fills missing ids."
        ),
    )
    content: str | None = Field(
        default=None,
        description="Short task description. Required for useful new tasks.",
    )
    status: str | None = Field(
        default=None,
        description="One of: pending, in_progress, completed, cancelled.",
    )


class TodoParams(BaseModel):
    """Arguments for the session todo tool."""

    todos: list[TodoItemPatch] | None = Field(
        default=None,
        description=(
            "Optional task list. Omit this field to read the current list. "
            "When provided with merge=false, it replaces the full list. "
            "When provided with merge=true, items update existing tasks by id or append new tasks."
        ),
    )
    merge: bool = Field(
        default=False,
        description=(
            "False replaces the entire list with a fresh plan. "
            "True updates existing items by id and appends complete new items."
        ),
    )


class TodoStore:
    """In-memory todo state for one sandbox session."""

    def __init__(self) -> None:
        self._items: list[dict[str, str]] = []

    def read(self) -> list[dict[str, str]]:
        """Return a copy of the current todo items."""
        return [dict(item) for item in self._items]

    def clear(self) -> None:
        self._items.clear()

    def write(
        self,
        todos: list[TodoItemPatch | dict[str, Any]],
        *,
        merge: bool = False,
    ) -> dict[str, Any]:
        warnings: list[str] = []
        raw_items = [self._coerce_patch(item) for item in todos]
        if merge:
            self._merge(raw_items, warnings)
        else:
            self._replace(raw_items, warnings)
        self._enforce_single_in_progress(warnings)
        return self.result(warnings)

    def result(self, warnings: list[str] | None = None) -> dict[str, Any]:
        todos = self.read()
        summary = {status: 0 for status in VALID_STATUSES}
        for item in todos:
            summary[item["status"]] += 1
        return {
            "todos": todos,
            "summary": {
                "total": len(todos),
                "pending": summary["pending"],
                "in_progress": summary["in_progress"],
                "completed": summary["completed"],
                "cancelled": summary["cancelled"],
            },
            "warnings": list(warnings or []),
        }

    def has_incomplete(self) -> bool:
        return any(item["status"] in {"pending", "in_progress"} for item in self._items)

    def format_active_for_injection(self) -> str | None:
        active_lines = []
        for item in self._items:
            if item["status"] not in {"pending", "in_progress"}:
                continue
            marker = _STATUS_MARKERS[item["status"]]
            active_lines.append(f"- {marker} {item['id']}. {item['content']}")
        if not active_lines:
            return None
        return "\n".join([ACTIVE_TODO_SNAPSHOT_HEADER, *active_lines])

    def incomplete_prompt(self) -> str | None:
        if not self.has_incomplete():
            return None
        return (
            "There are unfinished todo items in the current task list.\n\n"
            "Before finishing, inspect the active todo list and either continue "
            "pending/in_progress work, mark completed work as completed, mark obsolete "
            "items as cancelled, or revise the todo list if the user's goal has changed.\n\n"
            "Do not provide a final answer until the todo list accurately reflects the "
            "task state."
        )

    def _replace(self, raw_items: list[dict[str, Any]], warnings: list[str]) -> None:
        normalized = [
            self._normalize_item(item, default_id=str(index + 1), warnings=warnings)
            for index, item in enumerate(raw_items)
        ]
        self._items = self._dedupe_by_id(normalized)

    def _merge(self, raw_items: list[dict[str, Any]], warnings: list[str]) -> None:
        patches = self._dedupe_by_id(raw_items, keep_raw=True)
        index_by_id = {item["id"]: index for index, item in enumerate(self._items)}
        for patch in patches:
            item_id = str(patch.get("id") or "").strip()
            if not item_id:
                warnings.append("Skipped merge item without id.")
                continue
            if item_id in index_by_id:
                current = dict(self._items[index_by_id[item_id]])
                if "content" in patch and patch.get("content") is not None:
                    content = str(patch.get("content") or "").strip()
                    current["content"] = content or "(no description)"
                    if not content:
                        warnings.append(f"Todo {item_id} had empty content; used fallback.")
                if "status" in patch and patch.get("status") is not None:
                    current["status"] = self._normalize_status(
                        patch.get("status"),
                        warnings=warnings,
                        item_id=item_id,
                    )
                self._items[index_by_id[item_id]] = current
                continue

            normalized = self._normalize_item(patch, default_id=item_id, warnings=warnings)
            self._items.append(normalized)
            index_by_id[normalized["id"]] = len(self._items) - 1

    def _normalize_item(
        self,
        item: dict[str, Any],
        *,
        default_id: str,
        warnings: list[str],
    ) -> dict[str, str]:
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            item_id = default_id
            warnings.append(f"Todo missing id; assigned id {item_id}.")

        content = str(item.get("content") or "").strip()
        if not content:
            content = "(no description)"
            warnings.append(f"Todo {item_id} had empty content; used fallback.")

        status = self._normalize_status(item.get("status"), warnings=warnings, item_id=item_id)
        return {"id": item_id, "content": content, "status": status}

    @staticmethod
    def _coerce_patch(item: TodoItemPatch | dict[str, Any]) -> dict[str, Any]:
        if isinstance(item, TodoItemPatch):
            return item.model_dump(exclude_unset=True)
        return dict(item)

    @staticmethod
    def _dedupe_by_id(
        items: list[dict[str, Any]],
        *,
        keep_raw: bool = False,
    ) -> list[dict[str, Any]]:
        last_index: dict[str, int] = {}
        for index, item in enumerate(items):
            item_id = str(item.get("id") or "").strip()
            if not item_id and keep_raw:
                item_id = f"__missing_{index}"
            last_index[item_id] = index
        return [items[index] for index in sorted(last_index.values())]

    @staticmethod
    def _normalize_status(value: Any, *, warnings: list[str], item_id: str) -> str:
        status = str(value or "pending").strip()
        if status in VALID_STATUSES:
            return status
        warnings.append(f"Todo {item_id} had invalid status {status!r}; used pending.")
        return "pending"

    def _enforce_single_in_progress(self, warnings: list[str]) -> None:
        in_progress_indexes = [
            index for index, item in enumerate(self._items) if item["status"] == "in_progress"
        ]
        if len(in_progress_indexes) <= 1:
            return
        keep_index = in_progress_indexes[-1]
        for index in in_progress_indexes:
            if index != keep_index:
                self._items[index]["status"] = "pending"
        warnings.append("Multiple in_progress todos were normalized to one.")


_stores: dict[str, TodoStore] = {}


def get_todo_store(session_id: str) -> TodoStore:
    store = _stores.get(session_id)
    if store is None:
        store = TodoStore()
        _stores[session_id] = store
    return store


def clear_todo_store(session_id: str) -> None:
    _stores.pop(session_id, None)


def format_active_todos_for_injection(session_id: str) -> str | None:
    return get_todo_store(session_id).format_active_for_injection()


def get_incomplete_todos_prompt(session_id: str) -> str | None:
    return get_todo_store(session_id).incomplete_prompt()


def hydrate_todo_store_from_messages(session_id: str, messages: list[BaseMessage]) -> bool:
    """Restore todo state from the latest persisted todo tool result."""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        if message.tool_name != "todo":
            continue
        text = message.text
        if '"todos"' not in text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        todos = payload.get("todos")
        if not isinstance(todos, list):
            continue
        store = get_todo_store(session_id)
        store.write(todos, merge=False)
        return True
    return False


@tool(
    "Manage your task list for the current session. Use this for complex tasks with "
    "3+ steps, multi-file coding work, debugging, refactoring, test repair, or when "
    "the user provides multiple tasks. Call with no `todos` argument to read the "
    "current list. Writing: provide a `todos` array to create or update tasks; "
    "`merge=false` replaces the entire list with a fresh plan; `merge=true` updates "
    "existing items by `id` and appends new complete items. Each item has exactly "
    "`id`, `content`, and `status`. Allowed statuses: pending, in_progress, completed, "
    "cancelled. List order represents priority. Keep at most one item in_progress. "
    "Mark tasks completed as soon as they are done. Mark obsolete tasks cancelled; "
    "add a revised task if needed. The tool always returns the full current list, "
    "summary counts, and warnings.",
    args_schema=TodoParams,
)
async def todo(
    params: TodoParams,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
) -> str:
    """Read or update the current session todo list."""
    store = get_todo_store(ctx.session_id)
    if params.todos is None:
        result = store.result()
    else:
        result = store.write(params.todos, merge=params.merge)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool("Signal that the task is complete")
async def done(message: str) -> str:
    """Call this when the task is finished. Raises TaskComplete to stop the agent."""
    raise TaskComplete(message)
