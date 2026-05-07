"""Shared task board for multi-process agent teams."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from agent_core.team.atomic_io import atomic_write_json, read_json
from agent_core.team.file_lock import FileLock
from agent_core.team.models import TeamTask, utc_now_iso


class TaskBoard:
    """JSON task board guarded by a filesystem lock."""

    ALLOWED_STATUSES = {"pending", "in_progress", "completed", "blocked"}

    def __init__(self, team_dir: Path):
        self.team_dir = team_dir
        self.tasks_path = team_dir / "tasks.json"
        self.locks_dir = team_dir / "locks"

    def create_task(
        self,
        *,
        title: str,
        description: str,
        assigned_to: str | None = None,
        depends_on: list[str] | None = None,
        write_scope: list[str] | None = None,
    ) -> TeamTask:
        task = TeamTask(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            title=title,
            description=description,
            assigned_to=assigned_to,
            depends_on=list(depends_on or []),
            write_scope=list(write_scope or []),
        )
        with self._lock("lead"):
            tasks = self.list_tasks()
            tasks.append(task)
            self._validate_dependencies(tasks, task.task_id, task.depends_on)
            self._write(tasks)
        return task

    def list_tasks(self) -> list[TeamTask]:
        payload = read_json(self.tasks_path, {"version": 1, "tasks": []})
        return [TeamTask.from_dict(item) for item in payload.get("tasks", [])]

    def claim_next(self, member_id: str) -> TeamTask | None:
        with self._lock(member_id):
            tasks = self.list_tasks()
            completed = {task.task_id for task in tasks if task.status == "completed"}
            for task in tasks:
                if task.status != "pending":
                    continue
                if task.assigned_to is not None and task.assigned_to != member_id:
                    continue
                if any(dep not in completed for dep in task.depends_on):
                    continue
                if not self._try_acquire_file_locks(task, member_id):
                    continue
                task.status = "in_progress"
                task.claimed_by = member_id
                task.claimed_at = utc_now_iso()
                task.updated_at = utc_now_iso()
                self._write(tasks)
                return task
        return None

    def complete_task(self, task_id: str, member_id: str, result: str) -> TeamTask:
        with self._lock(member_id):
            tasks = self.list_tasks()
            task = self._find(tasks, task_id)
            if task.claimed_by != member_id:
                raise ValueError(f"Task '{task_id}' is not claimed by {member_id}")
            task.status = "completed"
            task.result = result
            task.error = None
            task.updated_at = utc_now_iso()
            task.completed_at = utc_now_iso()
            self._write(tasks)
        self.release_file_locks(task, member_id)
        return task

    def block_task(self, task_id: str, member_id: str, error: str) -> TeamTask:
        with self._lock(member_id):
            tasks = self.list_tasks()
            task = self._find(tasks, task_id)
            if task.claimed_by not in (None, member_id):
                raise ValueError(f"Task '{task_id}' is claimed by {task.claimed_by}")
            task.status = "blocked"
            task.error = error
            task.updated_at = utc_now_iso()
            self._write(tasks)
        self.release_file_locks(task, member_id)
        return task

    def update_task(
        self,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        assigned_to: str | None = None,
        depends_on: list[str] | None = None,
        write_scope: list[str] | None = None,
        result: str | None = None,
        error: str | None = None,
    ) -> TeamTask:
        if status is not None and status not in self.ALLOWED_STATUSES:
            allowed = ", ".join(sorted(self.ALLOWED_STATUSES))
            raise ValueError(f"Invalid task status '{status}'. Allowed: {allowed}")

        with self._lock("lead"):
            tasks = self.list_tasks()
            task = self._find(tasks, task_id)
            old_status = task.status
            old_claimed_by = task.claimed_by
            old_write_scope = list(task.write_scope)

            if title is not None:
                task.title = title
            if description is not None:
                task.description = description
            if assigned_to is not None:
                task.assigned_to = assigned_to or None
            if depends_on is not None:
                next_depends_on = list(depends_on)
                self._validate_dependencies(tasks, task.task_id, next_depends_on)
                task.depends_on = next_depends_on
            if write_scope is not None:
                task.write_scope = list(write_scope)
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            if status is not None:
                task.status = status
                if status == "pending":
                    task.claimed_by = None
                    task.claimed_at = None
                    task.completed_at = None
                    task.result = None
                    task.error = None
                elif status == "completed":
                    task.completed_at = task.completed_at or utc_now_iso()
                    task.error = None
                elif status == "blocked":
                    task.completed_at = None

            task.updated_at = utc_now_iso()
            self._write(tasks)

        should_release = (
            old_claimed_by is not None
            and (
                task.status != "in_progress"
                or old_status != task.status
                or old_write_scope != task.write_scope
            )
        )
        if should_release:
            old_task = TeamTask(
                task_id=task.task_id,
                title=task.title,
                description=task.description,
                write_scope=old_write_scope,
            )
            self.release_file_locks(old_task, old_claimed_by)
        return task

    def _find(self, tasks: list[TeamTask], task_id: str) -> TeamTask:
        for task in tasks:
            if task.task_id == task_id:
                return task
        raise ValueError(f"Task not found: {task_id}")

    def _validate_dependencies(
        self,
        tasks: list[TeamTask],
        task_id: str,
        depends_on: list[str],
    ) -> None:
        existing_ids = {task.task_id for task in tasks}
        missing = [dep for dep in depends_on if dep not in existing_ids]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                "Invalid task dependencies: "
                f"{missing_text}. Create dependency tasks first and use returned task_id values."
            )
        if task_id in depends_on:
            raise ValueError(f"Task '{task_id}' cannot depend on itself")

        graph = {task.task_id: list(task.depends_on) for task in tasks}
        graph[task_id] = list(depends_on)
        if self._has_dependency_cycle(graph):
            raise ValueError(f"Task dependencies would create a cycle involving '{task_id}'")

    @staticmethod
    def _has_dependency_cycle(graph: dict[str, list[str]]) -> bool:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for dependency in graph.get(node, []):
                if dependency in graph and visit(dependency):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in graph)

    def _write(self, tasks: list[TeamTask]) -> None:
        atomic_write_json(
            self.tasks_path,
            {"version": 1, "tasks": [task.to_dict() for task in tasks]},
        )

    def _lock(self, owner: str) -> FileLock:
        return FileLock(self.locks_dir / "tasks.lock", owner=owner)

    def _try_acquire_file_locks(self, task: TeamTask, member_id: str) -> bool:
        acquired: list[FileLock] = []
        try:
            for item in task.write_scope:
                lock = FileLock(self._file_lock_path(item), owner=member_id)
                lock.acquire(timeout=0.1)
                acquired.append(lock)
        except Exception:
            for lock in acquired:
                lock.release()
            return False
        return True

    def release_file_locks(self, task: TeamTask, member_id: str) -> None:
        for item in task.write_scope:
            lock_path = self._file_lock_path(item)
            owner = read_json(lock_path / "owner.json", {})
            if isinstance(owner, dict) and owner.get("owner") == member_id:
                FileLock(lock_path, owner=member_id, acquired=True).release()

    def _file_lock_path(self, path: str) -> Path:
        digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
        return self.locks_dir / "files" / f"{digest}.lock"
