"""Dependency helpers for bridge-backed tools."""

from __future__ import annotations

from cli.im_bridge.store import FileBridgeStore


def get_bridge_store() -> FileBridgeStore:
    """Return the active file bridge store from runtime dependency overrides."""
    raise RuntimeError("get_bridge_store() must be overridden via dependency_overrides")
