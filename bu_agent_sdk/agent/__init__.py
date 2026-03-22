"""
Agent module for running agentic loops with tool calling.
"""

from bu_agent_sdk.agent.compaction import (
    CompactionConfig,
    CompactionResult,
    CompactionService,
)
from bu_agent_sdk.agent.context import ContextManager
from bu_agent_sdk.agent.hooks import (
    AgentHook,
    AuditHook,
    FinishGuardHook,
    HookContext,
    HookDecision,
    HookManager,
    ToolPolicyHook,
)
from bu_agent_sdk.agent.model_routing_hook import ModelRoutingHook
from bu_agent_sdk.agent.events import (
    AgentEvent,
    FinalResponseEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from bu_agent_sdk.agent.registry import AgentRegistry, get_agent_registry
from bu_agent_sdk.agent.runtime_events import RuntimeEvent
from bu_agent_sdk.agent.runtime_state import AgentRunState
from bu_agent_sdk.agent.service import Agent, TaskComplete
from bu_agent_sdk.agent.config import AgentConfig, parse_agent_config

__all__ = [
    "Agent",
    "TaskComplete",
    # Events
    "AgentEvent",
    "FinalResponseEvent",
    "TextEvent",
    "ThinkingEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    # Runtime loop / hooks
    "RuntimeEvent",
    "AgentRunState",
    "AgentHook",
    "HookDecision",
    "HookContext",
    "HookManager",
    "FinishGuardHook",
    "ToolPolicyHook",
    "AuditHook",
    "ModelRoutingHook",
    # Compaction
    "CompactionConfig",
    "CompactionResult",
    "CompactionService",
    # Config and Registry
    "AgentConfig",
    "parse_agent_config",
    "AgentRegistry",
    "get_agent_registry",
    "ContextManager",
]
