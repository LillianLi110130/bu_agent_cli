"""Validation helpers for model-produced tool calls."""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from agent_core.agent.tool_args import ToolArgumentsError, parse_tool_arguments_for_execution
from agent_core.llm.messages import ToolCall
from agent_core.tools.decorator import Tool

WRITE_RECOVERY_CHUNK_MAX_CHARS = 1000

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
    chunk_required_paths: set[str]
    next_chunk_index_by_path: dict[str, int]
    require_chunking_for_next_write: bool = False

    @classmethod
    def empty(cls) -> "WriteRecoveryState":
        return cls(chunk_required_paths=set(), next_chunk_index_by_path={})


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
    if write_recovery_state is not None:
        errors.extend(
            _validate_write_recovery_constraints(
                tool_call,
                args,
                write_recovery_state,
            )
        )
    return errors


def remember_write_recovery_requirements(
    state: WriteRecoveryState,
    errors: list[ToolCallValidationError],
) -> None:
    for error in errors:
        if not _error_requires_write_chunking(error):
            continue
        file_path = _extract_write_file_path([error])
        if file_path:
            state.chunk_required_paths.add(file_path)
            state.next_chunk_index_by_path.setdefault(file_path, 0)
        else:
            state.require_chunking_for_next_write = True


def record_successful_write_recovery_chunk(
    state: WriteRecoveryState,
    tool_call: ToolCall,
) -> None:
    if tool_call.function.name != "write":
        return
    try:
        args = parse_tool_arguments_for_execution(tool_call.function.arguments)
    except ToolArgumentsError:
        return
    file_path = args.get("file_path")
    if not isinstance(file_path, str):
        return
    if file_path not in state.chunk_required_paths:
        return
    state.next_chunk_index_by_path[file_path] = (
        state.next_chunk_index_by_path.get(file_path, 0) + 1
    )


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

    if _should_add_write_chunking_hint(errors):
        file_path = _extract_write_file_path(errors)
        lines.extend(
            [
                "",
                (
                    "For write tool failures, a common cause is that the content "
                    "argument was too large and the upstream model truncated the tool call."
                ),
                "Retry by writing the file in smaller chunks:",
                '- First call write with mode="overwrite".',
                '- Then call write with mode="append" or mode="append_line" for subsequent chunks.',
                (
                    "Each following write.content for this target must be no longer than "
                    f"{WRITE_RECOVERY_CHUNK_MAX_CHARS} characters."
                ),
                (
                    "Do not retry a single write call containing the entire file content; "
                    "that call will be rejected before execution."
                ),
            ]
        )
        if file_path:
            lines.append(
                "- Preserve this file_path if it is still the intended target: "
                f"{file_path}"
            )

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


def _validate_write_recovery_constraints(
    tool_call: ToolCall,
    args: dict[str, Any],
    state: WriteRecoveryState,
) -> list[ToolCallValidationError]:
    if tool_call.function.name != "write":
        return []

    file_path = args.get("file_path")
    if not isinstance(file_path, str):
        return []

    requires_chunking = file_path in state.chunk_required_paths
    if not requires_chunking and state.require_chunking_for_next_write:
        requires_chunking = True
        state.chunk_required_paths.add(file_path)
        state.next_chunk_index_by_path.setdefault(file_path, 0)
        state.require_chunking_for_next_write = False

    if not requires_chunking:
        return []

    errors: list[ToolCallValidationError] = []
    content = args.get("content")
    if isinstance(content, str) and len(content) > WRITE_RECOVERY_CHUNK_MAX_CHARS:
        errors.append(
            ToolCallValidationError(
                tool_call=tool_call,
                kind="invalid_value",
                argument="content",
                message=(
                    f"write.content has {len(content)} characters, exceeding the "
                    f"recovery chunk limit of {WRITE_RECOVERY_CHUNK_MAX_CHARS}."
                ),
            )
        )

    chunk_index = state.next_chunk_index_by_path.get(file_path, 0)
    mode = args.get("mode", "overwrite")
    if chunk_index == 0 and mode != "overwrite":
        errors.append(
            ToolCallValidationError(
                tool_call=tool_call,
                kind="invalid_value",
                argument="mode",
                message='The first recovery write chunk must use mode="overwrite".',
            )
        )
    if chunk_index > 0 and mode not in {"append", "append_line"}:
        errors.append(
            ToolCallValidationError(
                tool_call=tool_call,
                kind="invalid_value",
                argument="mode",
                message='Subsequent recovery write chunks must use mode="append" or "append_line".',
            )
        )

    return errors


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


def _should_add_write_chunking_hint(errors: list[ToolCallValidationError]) -> bool:
    for error in errors:
        if _error_requires_write_chunking(error):
            return True
    return False


def _error_requires_write_chunking(error: ToolCallValidationError) -> bool:
    if error.tool_call.function.name != "write":
        return False
    if error.kind == "invalid_json":
        return True
    if error.kind == "missing_required" and error.argument == "content":
        return True
    if error.kind == "invalid_value" and error.argument in {"content", "mode"}:
        return True
    return False


def _extract_write_file_path(errors: list[ToolCallValidationError]) -> str | None:
    for error in errors:
        if error.tool_call.function.name != "write":
            continue
        raw_arguments = error.tool_call.function.arguments
        try:
            args = json.loads(raw_arguments)
        except json.JSONDecodeError:
            match = re.search(r'"file_path"\s*:\s*"(?P<path>(?:[^"\\]|\\.)*)"', raw_arguments)
            if not match:
                continue
            try:
                return json.loads(f'"{match.group("path")}"')
            except json.JSONDecodeError:
                return match.group("path")
        if isinstance(args, dict) and isinstance(args.get("file_path"), str):
            return args["file_path"]
    return None


def _preview(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
