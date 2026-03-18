---
name: init
description: 初始化前端工作流：复制 agent 配置到项目 .devagent 目录，创建规格文档目录
usage: /frontend-workflow:init <spec_name>
category: Frontend
mode: python
script: scripts/init_agents.py
parameters:
  - name: spec_name
    description: 规格文档名称，将用于创建 docs/spec/<spec_name> 目录
    required: true
---

将前端工作流的三个 subagent 配置复制到当前项目的 `.devagent/agents/` 目录，并创建规格文档目录：

- `frontend-requirement-analyzer.md` - 需求分析 Agent
- `frontend-design-analyzer.md` - 技术设计 Agent
- `frontend-task-splitter.md` - 任务拆分 Agent

同时创建 `docs/spec/<spec_name>/` 目录用于存放规格文档。

**示例**:
```
/frontend-workflow:init my_spec
```
