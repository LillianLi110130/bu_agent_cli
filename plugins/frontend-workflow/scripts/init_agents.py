#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初始化脚本：复制 agent 配置到用户项目的 .devagent 目录
"""
import json
import shutil
import sys
from pathlib import Path

# 设置标准输出编码为 UTF-8（必须在最前面）
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )


def main() -> int:
    payload = json.load(sys.stdin)
    working_dir = Path(payload["working_dir"])
    plugin_root = Path(payload["plugin_root"])

    # 获取 spec_name 参数
    args = payload.get("args", [])
    spec_name = args[0] if args else None

    if not spec_name:
        print("Error: spec_name parameter is required")
        print("Usage: /frontend-workflow:init <spec_name>")
        return 0

    # 目标目录
    target_dir = working_dir / ".devagent" / "agents"
    target_dir.mkdir(parents=True, exist_ok=True)

    # 源 agent 文件
    agents_dir = plugin_root / "agents"
    agent_files = [
        "frontend-requirement-analyzer.md",
        "frontend-design-analyzer.md",
        "frontend-task-splitter.md",
    ]

    copied = []
    for agent_file in agent_files:
        src = agents_dir / agent_file
        if not src.exists():
            print(f"Warning: Agent file not found: {src}")
            continue
        dst = target_dir / agent_file
        shutil.copy2(src, dst)
        copied.append(agent_file)

    if copied:
        print(f"[OK] Copied {len(copied)} agent configs to {target_dir}")
        for f in copied:
            print(f"  - {f}")
    else:
        print("Warning: No agent files were copied")

    # 创建规格文档目录
    spec_dir = working_dir / "docs" / "spec" / spec_name
    spec_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created spec directory: {spec_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
