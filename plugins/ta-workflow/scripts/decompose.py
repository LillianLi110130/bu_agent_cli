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
    require_paths,
    run_devagent,
    spec_dir,
    sync_plugin_subagents,
)


def main() -> int:
    payload = load_payload()
    plugin_root = Path(payload["plugin_root"])
    working_dir = Path(payload["working_dir"])
    args = payload.get("args", [])

    if not args:
        print("Usage: /ta-workflow:decompose <spec_name>")
        return 0

    spec_name = args[0]
    current_spec_dir = spec_dir(working_dir, spec_name)
    artifacts_dir = current_spec_dir / "artifacts" / "ta-workflow"
    plan_dir = current_spec_dir / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)

    requirements_file = artifacts_dir / "01_requirements.md"
    design_file = artifacts_dir / "02_design.md"
    task_domains_file = artifacts_dir / "03_task_domains.md"
    missing = require_paths([requirements_file, design_file, task_domains_file])
    if missing:
        print("Error: missing prerequisite files:")
        for path in missing:
            print(f"  - {path}")
        return 0
    if not require_devagent():
        print("Error: devagent command not found")
        return 0

    sync_plugin_subagents(plugin_root, working_dir)

    prompt = render_prompt(
        read_prompt(plugin_root, "prompts/decompose.md"),
        requirements_file=str(requirements_file).replace('\\', '/'),
        design_file=str(design_file).replace('\\', '/'),
        task_domains_file=str(task_domains_file).replace('\\', '/'),
        plan_dir=str(plan_dir).replace('\\', '/'),
        plan_json=str((plan_dir / 'plan.json')).replace('\\', '/'),
    )
    return run_devagent(working_dir, prompt)


if __name__ == "__main__":
    raise SystemExit(main())
