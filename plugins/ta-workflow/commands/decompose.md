---
name: decompose
description: 基于 TA 中间产物生成最终 plan
usage: /ta-workflow:decompose <spec_name>
category: Workflow
mode: python
script: scripts/decompose.py
examples:
  - /ta-workflow:decompose my_spec
---

执行任务分解阶段，读取 `artifacts/ta-workflow/` 中间产物，生成 `plan/*.md` 和 `plan/plan.json`。
