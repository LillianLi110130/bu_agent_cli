---
name: gateway-scheduled-notifier
description: Generate Python scheduled task scripts that proactively notify the worker gateway when a timed job completes, fails, or needs to report final status. Use when the user asks for cron jobs, Windows Task Scheduler scripts, recurring Python jobs, scheduled automation, or task scripts that should call the gateway complete endpoint based on cli/worker/mock_server.py.
---

# Gateway Scheduled Notifier

## Overview

Create scheduled task scripts that separate job execution from gateway notification. Always send final job status through `POST /api/worker/complete`. For the bundled helper, keep the gateway URL in `scripts/send_gateway_complete.py`, pass worker ID, source, and timeout through CLI arguments, and read authorization from `~/.tg_agent/token.json`.

## Workflow

1. Identify the schedule target:
   - Generate only the Python task body if the user already has cron/Task Scheduler wiring.
   - Generate a complete script plus a cron line or Windows Task Scheduler command when requested.
2. Identify notification config:
   - Gateway URL: use the `GATEWAY_URL` constant in `scripts/send_gateway_complete.py`.
   - `--worker-id`: gateway worker identity for the scheduled task.
   - `--source`: optional, default to `scheduler` unless the user provides another source.
   - `--timeout-seconds`: optional HTTP timeout.
   - Authorization: read `authorization` from `~/.tg_agent/token.json` and send it as the `Authorization` request header.
3. Implement the task with this control flow:
   - Run the business task.
   - On success, call complete with `final_status="completed"` and a concise `final_content`.
   - On failure, call complete with `final_status="failed"`, `error_code`, and `error_message`, then preserve the original failure behavior.
   - If gateway notification fails, log it with `logging.exception`; do not hide the original task result or exception.
4. Keep task logic and notification logic separate. Put gateway notification in a small helper function or reuse `scripts/send_gateway_complete.py`.
5. Use `logging`, not `print`. Use f-strings for formatted strings.

## Gateway Contract

Read `references/gateway-complete-api.md` when exact request fields, local mock validation, or examples are needed.

Core endpoint:

```text
POST {GATEWAY_URL}/api/worker/complete
```

For `scripts/send_gateway_complete.py`, `{GATEWAY_URL}` comes from the script-level `GATEWAY_URL` constant.

Required payload fields:

- `worker_id`
- `final_content`

Optional payload fields:

- `source`
- `final_status`
- `error_code`
- `error_message`

## Script Pattern

For generated Python scripts, prefer this structure:

```python
def run_task() -> str:
    """Run the scheduled job and return a human-readable completion summary."""
    ...

def main() -> int:
    config = load_gateway_config()
    try:
        summary = run_task()
    except Exception as exc:
        notify_gateway_failed(config, exc)
        raise
    notify_gateway_completed(config, summary)
    return 0
```

Do not hardcode worker IDs, credentials, or schedules inside generated task bodies. When reusing `scripts/send_gateway_complete.py`, edit the script-level `GATEWAY_URL` constant only when the deployment gateway changes, pass runtime values as CLI arguments, and rely on `~/.tg_agent/token.json` for authorization.

## Reusable Helper

Use `scripts/send_gateway_complete.py` when the user needs a small standalone notifier. It can be called from a scheduled script after the job finishes, or copied into a generated script when a single-file artifact is preferred.

Example command:

```powershell
python skills/gateway-scheduled-notifier/scripts/send_gateway_complete.py --worker-id "nightly-report" --content "Nightly report finished"
```