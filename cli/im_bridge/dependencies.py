"""Dependency helpers for bridge-backed tools."""

from __future__ import annotations

from cli.im_bridge.store import SqliteBridgeStore


def get_bridge_store() -> SqliteBridgeStore:
    """Return the active bridge store from runtime dependency overrides."""
    raise RuntimeError("get_bridge_store() must be overridden via dependency_overrides")
