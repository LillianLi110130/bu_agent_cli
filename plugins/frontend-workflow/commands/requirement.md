---
name: requirement
description: 执行前端需求分析阶段
usage: /frontend-workflow:requirement <spec_name>
category: Frontend
mode: python
script: scripts/requirement.py
examples:
  - /frontend-workflow:requirement my_spec
---

读取 `docs/spec/<spec_name>/input/` 原始材料，生成 `artifacts/frontend-workflow/01_requirement.md`。
