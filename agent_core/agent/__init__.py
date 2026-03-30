"""
Agent module for running agentic loops with tool calling.
"""

from agent_core.agent.compaction import (
    CompactionConfig,
    CompactionResult,
    CompactionService,
)
from agent_core.agent.context import ContextManager
from agent_core.agent.hitl import (
    HumanApprovalDecision,
    HumanApprovalRequest,
    HumanInLoopConfig,
    HumanInLoopHandler,
    build_default_approval_policy,
)
from agent_core.agent.hooks import (
    AgentHook,
    AuditHook,
    BashFileTaskGuardHook,
    ExcelReadGuardHook,
    FinishGuardHook,
    HookContext,
    HookDecision,
    HookManager,
    HumanApprovalHook,
    ToolPolicyHook,
)
from agent_core.agent.model_routing_hook import ModelRoutingHook
from agent_core.agent.events import (
    AgentEvent,
    FinalResponseEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from agent_core.agent.registry import AgentRegistry, get_agent_registry
from agent_core.agent.runtime_events import RuntimeEvent
from agent_core.agent.runtime_state import AgentRunState
from agent_core.agent.service import Agent, TaskComplete
from agent_core.agent.config import AgentConfig, parse_agent_config

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
    "BashFileTaskGuardHook",
    "ExcelReadGuardHook",
    "ModelRoutingHook",
    "HumanApprovalRequest",
    "HumanApprovalDecision",
    "HumanInLoopConfig",
    "HumanInLoopHandler",
    "HumanApprovalHook",
    "build_default_approval_policy",
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
