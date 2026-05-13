"""Read-only local HTTP server for the team dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from agent_core.team.atomic_io import read_json
from agent_core.team.models import TeamMessage
from agent_core.team.models import utc_now_iso


DEFAULT_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_DASHBOARD_PORT = 8787
STALE_AFTER_SECONDS = 300


@dataclass(slots=True)
class DashboardServerHandle:
    """Background dashboard server lifecycle handle."""

    server: ThreadingHTTPServer
    thread: threading.Thread
    host: str
    port: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def is_running(self) -> bool:
        return self.thread.is_alive()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def dashboard_html_path() -> Path:
    return Path(__file__).resolve().parent / "dashboard" / "team_dashboard.html"


def start_dashboard_server(
    *,
    runtime: Any,
    host: str = DEFAULT_DASHBOARD_HOST,
    port: int = DEFAULT_DASHBOARD_PORT,
) -> DashboardServerHandle:
    """Start the dashboard server in a background daemon thread."""
    handler = make_dashboard_handler(runtime=runtime)
    server = ThreadingHTTPServer((host, port), handler)
    actual_host, actual_port = server.server_address[:2]
    thread = threading.Thread(
        target=server.serve_forever,
        name="tg-agent-team-dashboard",
        daemon=True,
    )
    thread.start()
    return DashboardServerHandle(
        server=server,
        thread=thread,
        host=str(actual_host),
        port=int(actual_port),
    )


def run_dashboard_server(
    *,
    runtime: Any,
    host: str = DEFAULT_DASHBOARD_HOST,
    port: int = DEFAULT_DASHBOARD_PORT,
) -> None:
    """Run the dashboard server until interrupted."""
    handler = make_dashboard_handler(runtime=runtime)
    server = ThreadingHTTPServer((host, port), handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def make_dashboard_handler(*, runtime: Any) -> type[BaseHTTPRequestHandler]:
    class TeamDashboardHandler(BaseHTTPRequestHandler):
        server_version = "TgAgentTeamDashboard/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)
            try:
                if path in {"/", "/dashboard"}:
                    self._send_html(dashboard_html_path())
                elif path == "/api/active":
                    self._send_json({"active_team_id": runtime.get_active_team()})
                elif path == "/api/teams":
                    self._send_json(build_teams_payload(runtime))
                elif path.startswith("/api/teams/"):
                    self._handle_team_api(path, query)
                else:
                    self._send_json({"error": "not_found"}, status=404)
            except FileNotFoundError as exc:
                self._send_json({"error": "not_found", "message": str(exc)}, status=404)
            except Exception as exc:  # pragma: no cover - defensive server boundary
                self._send_json({"error": "internal_error", "message": str(exc)}, status=500)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _handle_team_api(self, path: str, query: dict[str, list[str]]) -> None:
            parts = [unquote(part) for part in path.split("/") if part]
            if len(parts) < 3:
                self._send_json({"error": "not_found"}, status=404)
                return
            team_id = parts[2]
            resource = parts[3] if len(parts) > 3 else "snapshot"
            if resource == "snapshot":
                self._send_json(build_snapshot_payload(runtime, team_id))
            elif resource == "events":
                limit = _int_query(query, "limit", 100)
                after = query.get("after", [None])[0]
                self._send_json(build_events_payload(runtime, team_id, limit=limit, after=after))
            elif resource == "tasks":
                self._send_json(build_tasks_payload(runtime, team_id, query=query))
            elif resource == "messages":
                self._send_json(build_messages_payload(runtime, team_id, query=query))
            elif resource == "chat":
                self._send_json(build_chat_payload(runtime, team_id))
            elif resource == "members" and len(parts) >= 5:
                self._send_json(build_member_payload(runtime, team_id, parts[4]))
            else:
                self._send_json({"error": "not_found"}, status=404)

        def _send_html(self, path: Path) -> None:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return TeamDashboardHandler


def build_teams_payload(runtime: Any) -> dict[str, Any]:
    active_team_id = runtime.get_active_team()
    teams = []
    for team in runtime.list_teams():
        try:
            status = runtime.status(team.team_id)
            messages = _peek_mailbox_messages(runtime, team.team_id, "lead")
        except FileNotFoundError:
            continue
        teams.append(
            {
                **team.to_dict(),
                "member_count": len(status["members"]),
                "task_count": len(status["tasks"]),
                "unread_lead_inbox_count": len(messages),
                "is_active": team.team_id == active_team_id,
            }
        )
    return {
        "active_team_id": active_team_id,
        "teams": teams,
        "updated_at": utc_now_iso(),
    }


def build_snapshot_payload(runtime: Any, team_id: str) -> dict[str, Any]:
    status = runtime.status(team_id)
    tasks = status["tasks"]
    members = _enrich_members(status["members"], tasks)
    lead_inbox = _peek_mailbox_messages(runtime, team_id, "lead")
    warnings = _build_warnings(members, tasks)
    return {
        "team": status["team"],
        "state": status["state"],
        "members": members,
        "tasks": tasks,
        "lead_inbox": lead_inbox,
        "warnings": warnings,
        "summary": {
            "members": _count_by(members, "display_status"),
            "tasks": _count_by(tasks, "status"),
            "unread_lead_inbox_count": len(lead_inbox),
        },
        "updated_at": utc_now_iso(),
    }


def build_events_payload(
    runtime: Any,
    team_id: str,
    *,
    limit: int = 100,
    after: str | None = None,
) -> dict[str, Any]:
    runtime.store.load_config(team_id)
    event_path = runtime.store.team_dir(team_id) / "events.jsonl"
    after_offset = _parse_event_offset(after)
    events: list[dict[str, Any]] = []
    if event_path.exists():
        for offset, line in enumerate(event_path.read_text(encoding="utf-8").splitlines()):
            if offset <= after_offset:
                continue
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = {"type": "malformed_event", "payload": {"line": line}}
            event.setdefault("event_id", f"evt_{offset}")
            event["offset"] = offset
            events.append(event)
    if limit > 0:
        events = events[-limit:]
    next_after = events[-1]["offset"] if events else after_offset
    events = list(reversed(events))
    return {
        "team_id": team_id,
        "events": events,
        "next_after": next_after,
        "updated_at": utc_now_iso(),
    }


def build_tasks_payload(
    runtime: Any,
    team_id: str,
    *,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    runtime.store.load_config(team_id)
    tasks = [task.to_dict() for task in runtime.list_tasks(team_id)]
    status_filter = query.get("status", [None])[0]
    assigned_filter = query.get("assigned_to", [None])[0]
    if status_filter:
        tasks = [task for task in tasks if task.get("status") == status_filter]
    if assigned_filter:
        tasks = [task for task in tasks if task.get("assigned_to") == assigned_filter]
    return {"team_id": team_id, "tasks": tasks, "updated_at": utc_now_iso()}


def build_messages_payload(
    runtime: Any,
    team_id: str,
    *,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    recipient = query.get("recipient", ["lead"])[0] or "lead"
    limit = _int_query(query, "limit", 50)
    messages = _peek_mailbox_messages(runtime, team_id, recipient, limit=limit)
    return {
        "team_id": team_id,
        "recipient": recipient,
        "messages": messages,
        "updated_at": utc_now_iso(),
    }


def build_chat_payload(runtime: Any, team_id: str) -> dict[str, Any]:
    status = runtime.status(team_id)
    messages = _read_chat_messages(runtime, team_id)
    participant_ids = ["lead"]
    for member in status["members"]:
        member_id = str(member.get("member_id") or "")
        if member_id and member_id not in participant_ids:
            participant_ids.append(member_id)
    for message in messages:
        for key in ("sender", "recipient"):
            value = str(message.get(key) or "")
            if value and value not in participant_ids:
                participant_ids.append(value)

    unread_by_recipient = _unread_counts_by_recipient(runtime, team_id, participant_ids)
    participants = []
    for participant_id in participant_ids:
        participants.append(
            {
                "id": participant_id,
                "label": participant_id,
                "message_count": _participant_message_count(messages, participant_id),
                "unread_count": unread_by_recipient.get(participant_id, 0),
            }
        )
    participants.append(
        {
            "id": "team-log",
            "label": "Team Log",
            "message_count": len(messages),
            "unread_count": sum(unread_by_recipient.values()),
        }
    )
    return {
        "team_id": team_id,
        "default_participant": "lead" if "lead" in participant_ids else (participant_ids[0] if participant_ids else "team-log"),
        "participants": participants,
        "messages": messages,
        "updated_at": utc_now_iso(),
    }


def build_member_payload(runtime: Any, team_id: str, member_id: str) -> dict[str, Any]:
    snapshot = build_snapshot_payload(runtime, team_id)
    member = next((item for item in snapshot["members"] if item.get("member_id") == member_id), None)
    if member is None:
        raise FileNotFoundError(f"Member not found: {member_id}")
    recent_events = [
        event
        for event in build_events_payload(runtime, team_id, limit=50)["events"]
        if event.get("actor") == member_id
        or event.get("member_id") == member_id
        or (event.get("payload") or {}).get("member_id") == member_id
    ]
    return {
        "team_id": team_id,
        "member": member,
        "heartbeat": member.get("heartbeat"),
        "current_task": member.get("current_task"),
        "recent_events": recent_events,
        "updated_at": utc_now_iso(),
    }


def _read_chat_messages(runtime: Any, team_id: str) -> list[dict[str, Any]]:
    event_path = runtime.store.team_dir(team_id) / "events.jsonl"
    messages: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not event_path.exists():
        return messages
    for offset, line in enumerate(event_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "message_sent":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        message_id = str(payload.get("message_id") or f"event_{offset}")
        if message_id in seen:
            continue
        seen.add(message_id)
        messages.append(
            {
                **payload,
                "message_id": message_id,
                "event_offset": offset,
                "created_at": payload.get("created_at") or event.get("created_at"),
            }
        )
    return messages


def _unread_counts_by_recipient(
    runtime: Any,
    team_id: str,
    participant_ids: list[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for participant_id in participant_ids:
        counts[participant_id] = len(_peek_mailbox_messages(runtime, team_id, participant_id))
    return counts


def _peek_mailbox_messages(
    runtime: Any,
    team_id: str,
    recipient: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read unread mailbox files without locking or acking them."""
    runtime.store.load_config(team_id)
    inbox = runtime.store.team_dir(team_id) / "mailboxes" / recipient / "inbox"
    if not inbox.exists():
        return []
    messages: list[dict[str, Any]] = []
    for path in sorted(inbox.glob("*.json")):
        if limit is not None and len(messages) >= limit:
            break
        payload = read_json(path, None)
        if payload is None:
            continue
        try:
            messages.append(TeamMessage.from_dict(payload).to_dict())
        except Exception:
            continue
    return messages


def _participant_message_count(messages: list[dict[str, Any]], participant_id: str) -> int:
    return sum(
        1
        for message in messages
        if message.get("sender") == participant_id or message.get("recipient") == participant_id
    )


def _enrich_members(members: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    task_by_id = {task["task_id"]: task for task in tasks}
    enriched = []
    for member in members:
        heartbeat = member.get("heartbeat") or {}
        raw_status = str(heartbeat.get("status") or member.get("status") or "running")
        heartbeat_at = heartbeat.get("updated_at") or member.get("last_heartbeat_at") or member.get("updated_at")
        age = _age_seconds(heartbeat_at, now=now)
        is_stale = age is not None and age > STALE_AFTER_SECONDS and raw_status not in {"stopped", "failed"}
        task_id = heartbeat.get("task_id")
        current_task = task_by_id.get(task_id) if task_id else None
        if current_task is None:
            current_task = next(
                (
                    task for task in tasks
                    if task.get("status") == "in_progress"
                    and (
                        task.get("claimed_by") == member.get("member_id")
                        or task.get("assigned_to") == member.get("member_id")
                    )
                ),
                None,
            )
        display_status = "stale" if is_stale else raw_status
        enriched.append(
            {
                **member,
                "display_status": display_status,
                "heartbeat_age_seconds": age,
                "current_task_id": current_task.get("task_id") if current_task else None,
                "current_task_title": current_task.get("title") if current_task else None,
                "current_task": current_task,
            }
        )
    return enriched


def _build_warnings(members: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for member in members:
        if member.get("display_status") == "stale":
            warnings.append(
                {
                    "type": "stale_heartbeat",
                    "member_id": member.get("member_id"),
                    "message": f"{member.get('member_id')} heartbeat is stale",
                }
            )
        elif member.get("display_status") == "failed":
            warnings.append(
                {
                    "type": "member_failed",
                    "member_id": member.get("member_id"),
                    "message": f"{member.get('member_id')} failed",
                }
            )
    for task in tasks:
        if task.get("status") == "blocked":
            warnings.append(
                {
                    "type": "task_blocked",
                    "task_id": task.get("task_id"),
                    "message": f"{task.get('task_id')} blocked: {task.get('error') or 'no error recorded'}",
                }
            )
    return warnings


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _age_seconds(timestamp: str | None, *, now: datetime) -> int | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0, int((now - parsed).total_seconds()))


def _parse_event_offset(value: str | None) -> int:
    if value is None:
        return -1
    if value.startswith("evt_"):
        value = value[4:]
    try:
        return int(value)
    except ValueError:
        return -1


def _int_query(query: dict[str, list[str]], key: str, default: int) -> int:
    try:
        return int(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return default
