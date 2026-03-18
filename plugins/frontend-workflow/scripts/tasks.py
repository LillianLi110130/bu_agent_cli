#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务拆分脚本：通过 devagent 执行前端任务拆分
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

    # 获取文档目录路径（如 "v1.0/zhangsan"）
    doc_dir = args_text if args_text else (" ".join(args) if args else "")

    if not doc_dir:
        print("=" * 60)
        print("[Frontend Workflow] Task Splitting")
        print("=" * 60)
        print("")
        print("Please provide the documents directory path.")
        print("")
        print("Usage:")
        print("  /frontend-workflow:tasks <iteration/owner>")
        print("")
        print("Examples:")
        print("  /frontend-workflow:tasks v1.0/zhangsan")
        print("")
        print("The script will look for:")
        print("  - DevAgentDoc/<iteration>/<owner>/01_需求分析.md")
        print("  - DevAgentDoc/<iteration>/<owner>/02_需求设计.md")
        print("")
        print("=" * 60)
        return 0

    # 构建文档路径
    base_dir = working_dir / "DevAgentDoc" / doc_dir
    requirement_doc = base_dir / "01_需求分析.md"
    design_doc = base_dir / "02_需求设计.md"

    # 检查文档是否存在
    if not requirement_doc.exists():
        print(f"Error: Requirement analysis document not found: {requirement_doc}")
        return 1

    if not design_doc.exists():
        print(f"Error: Design document not found: {design_doc}")
        return 1

    # 检查 devagent 是否可用
    if not shutil.which("devagent"):
        print("Error: devagent command not found, please install devagent first")
        return 1

    # 构建完整的 query（在 prompt 中明确指定文件路径和使用的 subagent）
    query = f"""# 前端任务拆分

你正在执行前端工作流的**第三阶段：任务拆分**。

## 执行步骤

1. **读取需求分析文档**: 请使用 ReadFile 工具读取以下文件：
   `DevAgentDoc/{doc_dir}/01_需求分析.md`

2. **读取技术设计文档**: 请使用 ReadFile 工具读取以下文件：
   `DevAgentDoc/{doc_dir}/02_需求设计.md`

3. **使用 subagent 执行任务拆分**: 请使用 `@frontend-task-splitter` subagent 进行任务拆分

4. **保存文档**: 将结果保存到 `DevAgentDoc/{doc_dir}/03_任务列表.md`

## 核心原则

- **拒绝废话**: 技改需求禁止生成初始化任务
- **任务精准化**: 描述必须包含具体文件修改点
- **扁平化输出**: 内部思考层级，输出扁平列表

## 任务格式规范

每个任务项必须严格遵循以下格式：
```
- [ ] 任务ID：描述 | status: pending | retries: 0 | 目标文件: [路径] | 关联需求: [ID]
```

## 依赖排序建议

1. 环境配置与初始化
2. 全局状态管理（Redux Store）
3. 通用工具函数/Hooks
4. 原子组件（按钮、输入框等）
5. 业务组件（组合组件）

---

请先读取需求分析和技术设计文档，然后使用 `@frontend-task-splitter` 执行任务拆分。
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
