---
name: tasks
description: 基于需求和设计生成 Ralph 计划
usage: /frontend-workflow:tasks <spec_name>
category: Frontend
mode: python
script: scripts/tasks.py
examples:
  - /frontend-workflow:tasks my_spec
---

读取 `artifacts/frontend-workflow/01_requirement.md` 和 `02_design.md`，直接生成 `plan/*.md` 与 `plan/plan.json`。
