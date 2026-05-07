"""Validation helpers for model-produced tool calls."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from agent_core.agent.tool_args import ToolArgumentsError, parse_tool_arguments_for_execution
from agent_core.llm.messages import ToolCall
from agent_core.tools.decorator import Tool

ToolCallValidationKind = Literal[
    "unknown_tool",
    "invalid_json",
    "missing_required",
    "unknown_argument",
    "invalid_value",
]


@dataclass(frozen=True)
class ToolCallValidationError:
    tool_call: ToolCall
    kind: ToolCallValidationKind
    message: str
    argument: str | None = None
    raw_arguments_preview: str | None = None


@dataclass
class WriteRecoveryState:
    """Compatibility placeholder for older write recovery plumbing."""

    @classmethod
    def empty(cls) -> "WriteRecoveryState":
        return cls()


def validate_tool_calls(
    tool_calls: list[ToolCall],
    tool_map: dict[str, Tool],
    *,
    write_recovery_state: WriteRecoveryState | None = None,
) -> list[ToolCallValidationError]:
    errors: list[ToolCallValidationError] = []
    for tool_call in tool_calls:
        errors.extend(
            validate_tool_call(
                tool_call,
                tool_map,
                write_recovery_state=write_recovery_state,
            )
        )
    return errors


def validate_tool_call(
    tool_call: ToolCall,
    tool_map: dict[str, Tool],
    *,
    write_recovery_state: WriteRecoveryState | None = None,
) -> list[ToolCallValidationError]:
    tool_name = tool_call.function.name
    tool = tool_map.get(tool_name)
    if tool is None:
        return [
            ToolCallValidationError(
                tool_call=tool_call,
                kind="unknown_tool",
                message=f"Unknown tool {tool_name!r}.",
            )
        ]

    raw_arguments = tool_call.function.arguments
    try:
        args = parse_tool_arguments_for_execution(raw_arguments)
    except ToolArgumentsError as exc:
        return [
            ToolCallValidationError(
                tool_call=tool_call,
                kind="invalid_json",
                message=f"Tool arguments are not a valid JSON object: {exc}",
                raw_arguments_preview=_preview(raw_arguments),
            )
        ]

    errors: list[ToolCallValidationError] = []
    errors.extend(_validate_required_arguments(tool_call, tool, args))
    errors.extend(_validate_unknown_arguments(tool_call, tool, args))
    errors.extend(_validate_enum_arguments(tool_call, tool, args))
    errors.extend(_validate_pydantic_schema(tool_call, tool, args))
    del write_recovery_state
    return errors


def remember_write_recovery_requirements(
    state: WriteRecoveryState,
    errors: list[ToolCallValidationError],
) -> None:
    del state, errors


def record_successful_write_recovery_chunk(
    state: WriteRecoveryState,
    tool_call: ToolCall,
) -> None:
    del state, tool_call


def build_invalid_tool_call_recovery_prompt(
    errors: list[ToolCallValidationError],
) -> str:
    lines = [
        (
            "The previous model response contained invalid tool call arguments "
            "and was rejected before execution."
        ),
        (
            "Do not repeat the same invalid call. Regenerate the tool call with "
            "valid JSON arguments."
        ),
        "",
        "Validation errors:",
    ]
    for error in errors:
        tool_name = error.tool_call.function.name
        call_id = error.tool_call.id
        argument = f" argument={error.argument!r}" if error.argument else ""
        lines.append(
            f"- tool={tool_name!r} id={call_id!r} kind={error.kind}{argument}: "
            f"{error.message}"
        )
        if error.raw_arguments_preview:
            lines.append(f"  Raw arguments preview: {error.raw_arguments_preview}")

    return "\n".join(lines)


def _validate_required_arguments(
    tool_call: ToolCall,
    tool: Tool,
    args: dict[str, Any],
) -> list[ToolCallValidationError]:
    missing = [name for name in _required_argument_names(tool) if name not in args]
    return [
        ToolCallValidationError(
            tool_call=tool_call,
            kind="missing_required",
            argument=name,
            message=f"Required argument {name!r} is missing.",
        )
        for name in missing
    ]


def _validate_unknown_arguments(
    tool_call: ToolCall,
    tool: Tool,
    args: dict[str, Any],
) -> list[ToolCallValidationError]:
    allowed = _allowed_argument_names(tool)
    if allowed is None:
        return []
    unknown = [name for name in args if name not in allowed]
    return [
        ToolCallValidationError(
            tool_call=tool_call,
            kind="unknown_argument",
            argument=name,
            message=f"Argument {name!r} is not accepted by tool {tool.name!r}.",
        )
        for name in unknown
    ]


def _validate_enum_arguments(
    tool_call: ToolCall,
    tool: Tool,
    args: dict[str, Any],
) -> list[ToolCallValidationError]:
    properties = tool.definition.parameters.get("properties")
    if not isinstance(properties, dict):
        return []

    errors: list[ToolCallValidationError] = []
    for name, value in args.items():
        prop = properties.get(name)
        if not isinstance(prop, dict) or "enum" not in prop:
            continue
        enum_values = prop.get("enum")
        if isinstance(enum_values, list) and value not in enum_values:
            errors.append(
                ToolCallValidationError(
                    tool_call=tool_call,
                    kind="invalid_value",
                    argument=name,
                    message=(
                        f"Argument {name!r} has invalid value {value!r}; "
                        f"expected one of {enum_values!r}."
                    ),
                )
            )
    return errors


def _validate_pydantic_schema(
    tool_call: ToolCall,
    tool: Tool,
    args: dict[str, Any],
) -> list[ToolCallValidationError]:
    if tool.args_schema is None:
        return []
    try:
        tool.args_schema(**args)
    except ValidationError as exc:
        return [
            ToolCallValidationError(
                tool_call=tool_call,
                kind="invalid_value",
                message=f"Arguments failed schema validation: {exc}",
            )
        ]
    return []


def _required_argument_names(tool: Tool) -> list[str]:
    if tool.args_schema is not None:
        return [
            name
            for name, field in tool.args_schema.model_fields.items()
            if field.is_required()
        ]

    required: list[str] = []
    signature = inspect.signature(tool.func)
    for name, param in signature.parameters.items():
        if name == "self" or name in tool._dependencies:
            continue
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return required


def _allowed_argument_names(tool: Tool) -> set[str] | None:
    if tool.args_schema is not None:
        return set(tool.args_schema.model_fields)

    signature = inspect.signature(tool.func)
    return {
        name
        for name in signature.parameters
        if name != "self" and name not in tool._dependencies
    }


def _preview(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
