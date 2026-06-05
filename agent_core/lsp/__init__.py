"""LSP support for agent tools."""

from __future__ import annotations

import atexit
import asyncio
import logging
import weakref
from typing import Any

from agent_core.lsp.manager import LSPManager

logger = logging.getLogger("agent_core.lsp")

_registered_managers: weakref.WeakSet[LSPManager] = weakref.WeakSet()
_atexit_registered = False


def attach_lsp_manager(ctx: Any) -> None:
    """Attach an LSP manager to a sandbox context."""
    manager = LSPManager.from_settings(ctx.root_dir)
    ctx.lsp_manager = manager
    _register_lsp_manager(manager)


async def shutdown_lsp_manager(ctx: Any) -> None:
    """Shut down the LSP manager attached to a sandbox context."""
    manager = getattr(ctx, "lsp_manager", None)
    if manager is None:
        return
    await manager.shutdown_all()
    _registered_managers.discard(manager)


def _register_lsp_manager(manager: LSPManager) -> None:
    global _atexit_registered
    _registered_managers.add(manager)
    if not _atexit_registered:
        atexit.register(_atexit_shutdown_lsp_managers)
        _atexit_registered = True


def _atexit_shutdown_lsp_managers() -> None:
    managers = list(_registered_managers)
    _registered_managers.clear()
    for manager in managers:
        try:
            asyncio.run(manager.shutdown_all())
        except Exception as exc:
            logger.debug("LSP atexit async shutdown failed: %s", exc)
            manager.terminate_all_nowait()


__all__ = ["LSPManager", "attach_lsp_manager", "shutdown_lsp_manager"]
