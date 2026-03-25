"""Tool wrapper for deterministic path resolution."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from bu_agent_sdk.tools import Depends, tool

from tools.path_resolution import AmbiguousPathError, PathNotFoundError, resolve_target_path
from tools.sandbox import SandboxContext, get_sandbox_context

ResolveKind = Literal["file", "dir", "any"]


@tool("Resolve an approximate file or directory query to a real path in the workspace")
async def resolve_path(
    query: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox_context)],
    kind: ResolveKind = "any",
) -> str:
    """Resolve a user-provided path query before running other tools.

    Args:
        query: Approximate filename, relative path, or absolute path.
        kind: Restrict the result to a file, directory, or any path type.
    """
    try:
        path = resolve_target_path(query, ctx, kind=kind)
    except PathNotFoundError as e:
        return json.dumps(
            {
                "ok": False,
                "query": query,
                "kind": kind,
                "reason": "not_found",
                "message": str(e),
            },
            ensure_ascii=False,
            indent=2,
        )
    except AmbiguousPathError as e:
        return json.dumps(
            {
                "ok": False,
                "query": query,
                "kind": kind,
                "reason": "ambiguous",
                "message": str(e),
                "candidates": [str(candidate) for candidate in e.candidates],
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        return f"Security error: {e}"

    resolved_kind = "dir" if path.is_dir() else "file" if path.is_file() else "any"
    return json.dumps(
        {
            "ok": True,
            "query": query,
            "kind": resolved_kind,
            "resolved_path": str(path),
        },
        ensure_ascii=False,
        indent=2,
    )
