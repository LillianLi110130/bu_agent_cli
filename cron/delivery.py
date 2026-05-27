"""Delivery port for scheduled job completion notifications."""

from __future__ import annotations

import logging

from cron.models import CronDeliveryResult, CronHostContext, CronJob

logger = logging.getLogger("cron.delivery")


class CronDeliveryPort:
    """Send remote cron output through the current worker gateway context."""

    async def complete(
        self,
        *,
        job: CronJob,
        run_id: str,
        final_content: str,
        context: CronHostContext,
    ) -> CronDeliveryResult:
        del run_id
        if context.worker_id is None or context.gateway_client is None:
            return CronDeliveryResult(
                ok=False,
                status="failed",
                error="remote delivery requires worker_id and gateway_client",
            )

        try:
            ok = await context.gateway_client.progress(
                worker_id=context.worker_id,
                content=final_content,
                source=job.source,
            )
        except Exception as exc:
            logger.exception(f"Cron remote delivery failed for job_id={job.id}: {exc}")
            return CronDeliveryResult(ok=False, status="failed", error=str(exc))
        if not ok:
            return CronDeliveryResult(
                ok=False,
                status="failed",
                error="gateway progress returned ok=false",
            )
        return CronDeliveryResult(ok=True, status="delivered")
