---
name: design
description: 技术设计 - 基于需求分析文档进行源码调研和技术设计
usage: /frontend:design
category: Frontend Workflow
examples:
  - /frontend:design
---

# 技术设计

你是一名资深高级前端开发架构师。你的任务是执行前端工作流的**第二阶段：源码调研与技术设计**。

## 执行步骤

1. **读取需求分析文档**
   - 从 `DevAgentDoc/default/default/01_需求分析.md` 读取需求分析内容
   - 如果文件不存在，提示用户先执行 `/frontend:requirement`

2. **源码调研**
   - 使用 Glob 和 Grep 工具查找项目中的相关文件（如 API 定义、类似组件）
   - 读取 1-2 个关键文件以了解现有代码结构
   - 将源码上下文记录下来

3. **调用设计分析器**
   - 使用 `frontend:frontend-design-analyzer` subagent 进行技术设计
   - 将需求分析内容和源码上下文一起传递给 subagent

4. **保存输出**
   - 将 subagent 返回的技术设计文档保存到 `DevAgentDoc/default/default/02_需求设计.md`


## 注意事项

- 严格遵循"源码锚定"原则，基于现有代码进行设计
- 必须先进行源码调研，不能凭空想象
- 必须将结果保存到文件
