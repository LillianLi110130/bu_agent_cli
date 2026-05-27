# Gateway Complete API

This contract is based on `cli/worker/mock_server.py`.

## Endpoint

```text
POST /api/worker/complete
```

Use it for final scheduled task notifications. Do not use it for incremental progress; use the gateway progress endpoint only when the user explicitly needs intermediate updates.

## Request Body

```json
{
  "worker_id": "nightly-report",
  "final_content": "Nightly report finished",
  "source": "scheduler",
  "final_status": "completed",
  "error_code": null,
  "error_message": null
}
```

Fields:
- `final_content`: required string. Human-readable final message sent to the gateway.
- `worker_id`: required string. Gateway worker identity for this scheduled task.
- `source`: optional string. Defaults to `im` in the mock gateway when omitted; scheduled jobs should usually send `scheduler`.
- `final_status`: optional string. Defaults to `completed`; use `failed` for task errors.
- `error_code`: optional string. Use stable codes such as `scheduled_task_failed` or domain-specific codes.
- `error_message`: optional string. Use a concise error summary, not a full secret-bearing traceback.

The gateway returns:

```json
{"ok": true}
```

## Generation Guidance

For scheduled scripts:

- Read gateway configuration from environment variables or existing config files.
- Log notification failures with `logging.exception`.
- Keep `final_content` short and useful.
- Preserve the original job failure after attempting failed-status notification.
- Avoid putting secrets, full tracebacks, or large payloads in `final_content` or `error_message`.
