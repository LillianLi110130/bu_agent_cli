#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
需求分析脚本：通过 devagent 执行前端需求分析
"""
import json
import shutil
import subprocess
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
    args = payload.get("args", [])

    # 获取 spec_name 参数
    spec_name = args[0] if args else None

    if not spec_name:
        print("=" * 60)
        print("[Frontend Workflow] Requirement Analysis")
        print("=" * 60)
        print("")
        print("Please provide the spec_name parameter.")
        print("")
        print("Usage:")
        print("  /frontend-workflow:requirement <spec_name>")
        print("")
        print("Examples:")
        print("  /frontend-workflow:requirement my_spec")
        print("")
        print("=" * 60)
        return 0

    # 构建文档路径：docs/spec/<spec_name>/design.md
    doc_path = f"docs/spec/{spec_name}/design.md"
    doc_file = working_dir / doc_path

    # 检查文档是否存在
    if not doc_file.exists():
        print(f"Error: Design document not found: {doc_file}")
        print(f"Please ensure the file exists before running requirement analysis.")
        return 0

    # 检查 devagent 是否可用
    if not shutil.which("devagent"):
        print("Error: devagent command not found, please install devagent first")
        return 0

    # 构建完整的 query（在 prompt 中明确指定文件路径和使用的 subagent）
    query = f"""# 前端需求分析

你正在执行前端工作流的**第一阶段：需求分析**。

## 执行步骤

1. **读取原始需求文档**: 请使用 ReadFile 或其他文件读取工具读取以下文件：
   `{doc_path}`

2. **使用 subagent 执行分析**: 请使用 `frontend-requirement-analyzer` subagent 进行需求分析

3. **保存文档**: 将结果保存到 `DevAgentDoc/[迭代名]/[负责人]/01_需求分析.md`

## 输出格式规范

必须遵循以下结构：
- `# {{项目名称}}需求文档`
- `## 输入摘要`
- `## 术语表` (feature 场景)
- `## 需求` (WHEN...THEN...SHALL 格式)

## 注意事项

- 确认需求性质：[feature/refactor/fix]
- refactor/fix 场景禁止输出冗余信息
- 验收标准必须使用 WHEN...THEN...SHALL 格式

---

请先读取需求文档 `{doc_path}`，然后使用 `@frontend-requirement-analyzer` 执行分析。
"""

    # 执行 devagent 命令
    try:
        process = subprocess.Popen(
            "devagent --yolo",
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=str(working_dir),
        )

        stdout, stderr = process.communicate(input=query)

        if stdout:
            print(stdout)

        if stderr:
            print(f"Error: {stderr}", file=sys.stderr)

        return process.returncode

    except Exception as e:
        print(f"Error: Failed to execute devagent: {e}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
