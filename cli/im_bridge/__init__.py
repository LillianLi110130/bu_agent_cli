"""File-backed bridge primitives for local and remote message synchronization."""

from cli.im_bridge.models import BridgeRequest, BridgeResult
from cli.im_bridge.store import FileBridgeStore, resolve_session_binding_id

__all__ = ["BridgeRequest", "BridgeResult", "FileBridgeStore", "resolve_session_binding_id"]
