---
name: design
description: 执行前端技术设计阶段
usage: /frontend-workflow:design <spec_name>
category: Frontend
mode: python
script: scripts/design.py
examples:
  - /frontend-workflow:design my_spec
---

读取 `artifacts/frontend-workflow/01_requirement.md`，生成 `artifacts/frontend-workflow/02_design.md`。
