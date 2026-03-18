#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术设计脚本：通过 devagent 执行前端技术设计（含源码调研）
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

# 设置标准输出编码为 UTF-8（必须在最前面）
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)


def main() -> int:
    payload = json.load(sys.stdin)
    working_dir = Path(payload["working_dir"])
    args_text = payload.get("args_text", "").strip()
    args = payload.get("args", [])

    # 获取需求分析文档路径
    doc_path = args_text if args_text else (" ".join(args) if args else "")

    if not doc_path:
        print("=" * 60)
        print("[Frontend Workflow] Technical Design")
        print("=" * 60)
        print("")
        print("Please provide the requirement analysis document path.")
        print("")
        print("Usage:")
        print("  /frontend-workflow:design <path-to-requirement-analysis-doc>")
        print("")
        print("Examples:")
        print("  /frontend-workflow:design DevAgentDoc/v1.0/zhangsan/01_需求分析.md")
        print("")
        print("=" * 60)
        return 0

    # 检查文档是否存在
    doc_file = working_dir / doc_path
    if not doc_file.exists():
        print(f"Error: Requirement analysis document not found: {doc_file}")
        return 1

    # 检查 devagent 是否可用
    if not shutil.which("devagent"):
        print("Error: devagent command not found, please install devagent first")
        return 1

    # 构建完整的 query（在 prompt 中明确指定文件路径和使用的 subagent）
    query = f"""# 前端技术设计

你正在执行前端工作流的**第二阶段：源码调研与技术设计**。

## 执行步骤

1. **读取需求分析文档**: 请使用 ReadFile 工具读取以下文件：
   `{doc_path}`

2. **源码调研**: 使用 Glob/Grep 工具读取 1-2 个现有相关文件（如 API 定义、类似组件）

3. **使用 subagent 执行设计**: 请使用 `@frontend-design-analyzer` subagent 进行技术设计

4. **保存文档**: 将结果保存到 `DevAgentDoc/[迭代名]/[负责人]/02_需求设计.md`

## 核心原则：源码锚定 (CRITICAL)

- **严禁凭空想象设计**
- 在文档开头列出参考的现有文件路径
- 技改需求只输出变更部分，使用 [新增]/[修改]/[删除] 标记

## 输出格式规范

必须遵循以下结构：
- `# {{项目名称}}设计文档`
- `## 概述`
- `## 架构` (必须包含 Mermaid 图表)
- `## 状态管理策略`
- `## 组件和接口设计` (TypeScript 伪代码)
- `## API 交互设计` (包含 API ID)

---

请先读取需求分析文档 `{doc_path}`，进行源码调研，然后使用 `@frontend-design-analyzer` 执行设计。
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
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
