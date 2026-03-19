#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.runtime_helpers import (
    load_payload,
    read_prompt,
    render_prompt,
    require_devagent,
    run_devagent,
    spec_dir,
)


def main() -> int:
    payload = load_payload()
    plugin_root = Path(payload["plugin_root"])
    working_dir = Path(payload["working_dir"])
    args = payload.get("args", [])

    if not args:
        print("Usage: /frontend-workflow:requirement <spec_name>")
        return 0

    spec_name = args[0]
    current_spec_dir = spec_dir(working_dir, spec_name)
    input_dir = current_spec_dir / "input"
    artifacts_dir = current_spec_dir / "artifacts" / "frontend-workflow"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        return 0
    if not require_devagent():
        print("Error: devagent command not found")
        return 0

    prompt = render_prompt(
        read_prompt(plugin_root, "prompts/requirement.md"),
        spec_name=spec_name,
        input_dir=str(input_dir).replace('\\', '/'),
        requirement_file=str((artifacts_dir / '01_requirement.md')).replace('\\', '/'),
    )
    return run_devagent(working_dir, prompt)


if __name__ == "__main__":
    raise SystemExit(main())
