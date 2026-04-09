"""File-backed bridge primitives for local and remote message synchronization."""

from cli.im_bridge.dependencies import get_bridge_store
from cli.im_bridge.models import BridgeOutboundEvent, BridgeRequest, BridgeResult
from cli.im_bridge.store import FileBridgeStore, resolve_session_binding_id

__all__ = [
    "BridgeOutboundEvent",
    "BridgeRequest",
    "BridgeResult",
    "FileBridgeStore",
    "get_bridge_store",
    "resolve_session_binding_id",
]
