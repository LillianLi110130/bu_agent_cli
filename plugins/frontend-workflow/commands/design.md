---
name: design
description: 执行前端技术设计阶段（基于源码调研）
usage: /frontend-workflow:design <需求分析文档路径>
category: Frontend
mode: python
script: scripts/design.py
examples:
  - /frontend-workflow:design DevAgentDoc/v1.0/zhangsan/01_需求分析.md
---

执行前端技术设计阶段，调用 devagent 基于需求分析文档进行源码调研和技术设计。

**工作流程**:
1. 读取需求分析文档
2. 使用 Glob/Grep 读取 1-2 个现有相关文件
3. 使用 `@frontend-design-analyzer` subagent 执行设计
4. 生成 `DevAgentDoc/[迭代名]/[负责人]/02_需求设计.md`

**核心原则**:
- **严禁凭空想象设计**
- 在文档开头列出参考的现有文件路径
- 技改需求只输出变更部分

**输出格式**:
- `# {项目名称}设计文档`
- `## 概述`
- `## 架构` (必须包含 Mermaid 图表)
- `## 状态管理策略`
- `## 组件和接口设计` (TypeScript 伪代码)
- `## API 交互设计`

**示例**:
```
/frontend-workflow:design DevAgentDoc/v1.0/zhangsan/01_需求分析.md
```
