"""Inspect the local IM bridge SQLite database without modifying it."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

DEFAULT_DB_RELATIVE_PATH = Path(".tg_agent") / "im_bridge" / "bridge.sqlite3"
TABLES = (
    "bridge_workers",
    "bridge_requests",
    "bridge_progress",
    "bridge_outbound_events",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only inspector for .tg_agent/im_bridge/bridge.sqlite3",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_RELATIVE_PATH,
        help="SQLite database path (default: .tg_agent/im_bridge/bridge.sqlite3)",
    )
    parser.add_argument(
        "--worker-no",
        default=None,
        help="Only show records for one terminal worker_no",
    )
    parser.add_argument(
        "--status",
        default=None,
        help="Only show requests and outbound events with this status",
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help="Show one request and its progress records",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum rows to print per section (default: 20)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of formatted tables",
    )
    return parser.parse_args()


def open_read_only(db_path: Path) -> sqlite3.Connection:
    resolved_path = db_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Bridge database not found: {resolved_path}")
    uri = f"file:{quote(resolved_path.as_posix(), safe='/:')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA busy_timeout = 2000")
    return conn


def inspect_database(conn: sqlite3.Connection, args: argparse.Namespace) -> dict[str, Any]:
    existing_tables = {
        str(row["name"])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    missing_tables = [table for table in TABLES if table not in existing_tables]
    if missing_tables:
        raise RuntimeError(f"Missing bridge tables: {', '.join(missing_tables)}")

    payload: dict[str, Any] = {
        "database": str(args.db.expanduser().resolve()),
        "summary": load_summary(conn),
        "workers": load_workers(conn, worker_no=args.worker_no, limit=args.limit),
        "requests": load_requests(
            conn,
            worker_no=args.worker_no,
            status=args.status,
            request_id=args.request_id,
            limit=args.limit,
        ),
        "progress": load_progress(conn, request_id=args.request_id, limit=args.limit),
        "outbound_events": load_outbound_events(
            conn,
            worker_no=args.worker_no,
            status=args.status,
            limit=args.limit,
        ),
    }
    return payload


def load_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    request_statuses = load_grouped_counts(conn, "bridge_requests", "status")
    progress_statuses = load_grouped_counts(conn, "bridge_progress", "status")
    outbound_statuses = load_grouped_counts(conn, "bridge_outbound_events", "status")
    worker_count = int(conn.execute("SELECT COUNT(*) FROM bridge_workers").fetchone()[0])
    return {
        "workers": worker_count,
        "requests": request_statuses,
        "progress": progress_statuses,
        "outbound_events": outbound_statuses,
    }


def load_grouped_counts(conn: sqlite3.Connection, table: str, column: str) -> dict[str, int]:
    rows = conn.execute(
        f"SELECT {column}, COUNT(*) AS count FROM {table} GROUP BY {column} ORDER BY {column}"
    ).fetchall()
    return {str(row[column]): int(row["count"]) for row in rows}


def load_workers(
    conn: sqlite3.Connection,
    *,
    worker_no: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM bridge_workers"
    params: list[Any] = []
    if worker_no:
        query += " WHERE worker_no = ?"
        params.append(worker_no)
    query += " ORDER BY last_seen_at DESC LIMIT ?"
    params.append(limit)
    return rows_to_dicts(conn.execute(query, params))


def load_requests(
    conn: sqlite3.Connection,
    *,
    worker_no: str | None,
    status: str | None,
    request_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM bridge_requests"
    clauses: list[str] = []
    params: list[Any] = []
    if worker_no:
        clauses.append("worker_no = ?")
        params.append(worker_no)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if request_id:
        clauses.append("request_id = ?")
        params.append(request_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY enqueue_at DESC, seq DESC LIMIT ?"
    params.append(limit)
    rows = rows_to_dicts(conn.execute(query, params))
    for row in rows:
        row["source_meta_json"] = parse_json_field(row.get("source_meta_json"))
    return rows


def load_progress(
    conn: sqlite3.Connection,
    *,
    request_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM bridge_progress"
    params: list[Any] = []
    if request_id:
        query += " WHERE request_id = ?"
        params.append(request_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    return rows_to_dicts(conn.execute(query, params))


def load_outbound_events(
    conn: sqlite3.Connection,
    *,
    worker_no: str | None,
    status: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM bridge_outbound_events"
    clauses: list[str] = []
    params: list[Any] = []
    if worker_no:
        clauses.append("worker_no = ?")
        params.append(worker_no)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC, seq DESC LIMIT ?"
    params.append(limit)
    return rows_to_dicts(conn.execute(query, params))


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def parse_json_field(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def print_report(payload: dict[str, Any]) -> None:
    print(f"Database: {payload['database']}")
    summary = payload["summary"]
    print("\nSummary")
    print(f"  workers: {summary['workers']}")
    print(f"  requests: {format_counts(summary['requests'])}")
    print(f"  progress: {format_counts(summary['progress'])}")
    print(f"  outbound_events: {format_counts(summary['outbound_events'])}")
    print_section(
        "Workers",
        payload["workers"],
        ("worker_no", "last_seen_at", "next_request_seq", "next_outbound_seq"),
    )
    print_section(
        "Requests",
        payload["requests"],
        (
            "request_id",
            "worker_no",
            "seq",
            "source",
            "status",
            "input_kind",
            "enqueue_at",
            "started_at",
            "lease_until",
            "finished_at",
            "content",
            "final_content",
            "error_code",
            "error_message",
            "source_meta_json",
        ),
    )
    print_section(
        "Progress",
        payload["progress"],
        ("progress_id", "request_id", "status", "created_at", "delivered_at", "content"),
    )
    print_section(
        "Outbound Events",
        payload["outbound_events"],
        (
            "event_id",
            "worker_no",
            "seq",
            "action",
            "status",
            "attempts",
            "created_at",
            "started_at",
            "lease_until",
            "delivered_at",
            "text",
            "file_path",
            "last_error",
        ),
    )


def print_section(title: str, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    print(f"\n{title} ({len(rows)})")
    if not rows:
        print("  <empty>")
        return
    for index, row in enumerate(rows, start=1):
        print(f"  [{index}]")
        for column in columns:
            value = row.get(column)
            if value is None or value == "":
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            print(f"    {column}: {value}")


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "<empty>"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    if args.limit <= 0:
        print("error: --limit must be greater than zero", file=sys.stderr)
        return 2
    try:
        with open_read_only(args.db) as conn:
            payload = inspect_database(conn, args)
    except (FileNotFoundError, RuntimeError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_report(payload)
    return 0


def configure_console_encoding() -> None:
    """Keep Windows consoles from crashing on emoji or mixed-language content."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
