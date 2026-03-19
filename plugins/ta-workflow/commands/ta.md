---
name: ta
description: 生成 TA 阶段中间产物
usage: /ta-workflow:ta <spec_name>
category: Workflow
mode: python
script: scripts/ta.py
examples:
  - /ta-workflow:ta my_spec
---

执行 TA 阶段，读取 `docs/spec/<spec_name>/input/` 下的输入材料，生成 `artifacts/ta-workflow/` 中间文档。
