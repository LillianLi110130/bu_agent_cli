"""MCP support for agent tools."""

from __future__ import annotations

import atexit
import asyncio
import logging
import weakref
from typing import Any

from agent_core.mcp.manager import MCPManager

logger = logging.getLogger("agent_core.mcp")

_registered_managers: weakref.WeakSet[MCPManager] = weakref.WeakSet()
_atexit_registered = False


def attach_mcp_manager(ctx: Any) -> None:
    """Attach an MCP manager to a sandbox context and start enabled servers."""
    manager = MCPManager.from_settings(ctx.root_dir)
    ctx.mcp_manager = manager
    _register_mcp_manager(manager)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("MCP manager attached without a running event loop; background start skipped")
        return
    manager.start_enabled_servers_background()


async def shutdown_mcp_manager(ctx: Any) -> None:
    """Shut down the MCP manager attached to a sandbox context."""
    manager = getattr(ctx, "mcp_manager", None)
    if manager is None:
        return
    await manager.shutdown_all()
    _registered_managers.discard(manager)


def _register_mcp_manager(manager: MCPManager) -> None:
    global _atexit_registered
    _registered_managers.add(manager)
    if not _atexit_registered:
        atexit.register(_atexit_shutdown_mcp_managers)
        _atexit_registered = True


def _atexit_shutdown_mcp_managers() -> None:
    managers = list(_registered_managers)
    _registered_managers.clear()
    for manager in managers:
        try:
            asyncio.run(manager.shutdown_all())
        except Exception as exc:
            logger.debug("MCP atexit async shutdown failed: %s", exc)
            manager.terminate_all_nowait()


__all__ = ["MCPManager", "attach_mcp_manager", "shutdown_mcp_manager"]
