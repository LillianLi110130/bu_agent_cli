---
name: requirement
description: 执行前端需求分析阶段，读取规格文档目录下的 design.md
usage: /frontend-workflow:requirement <spec_name>
category: Frontend
mode: python
script: scripts/requirement.py
parameters:
  - name: spec_name
    description: 规格文档名称，将读取 docs/spec/<spec_name>/design.md
    required: true
examples:
  - /frontend-workflow:requirement my_spec
---

执行前端需求分析阶段，调用 devagent 处理规格文档目录下的 design.md。

**工作流程**:
1. 读取 `docs/spec/<spec_name>/design.md` 文件
2. 使用 `@frontend-requirement-analyzer` subagent 执行分析
3. 生成 `DevAgentDoc/[迭代名]/[负责人]/01_需求分析.md`

**输出格式**:
- `# {项目名称}需求文档`
- `## 输入摘要`
- `## 术语表` (feature 场景)
- `## 需求` (WHEN...THEN...SHALL 格式)

**示例**:
```
/frontend-workflow:requirement my_spec
```
