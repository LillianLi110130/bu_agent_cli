---
name: requirement
description: 需求分析 - 将原始需求转化为结构化的需求分析文档
usage: /frontend:requirement <path-to-requirement-doc>
category: Frontend Workflow
examples:
  - /frontend:requirement docs/requirement.md
  - /frontend:requirement docs/spec/user-auth.md
---

# 需求分析

你是一名资深高级前端开发架构师。你的任务是执行前端工作流的**第一阶段：需求分析**。

## 执行步骤

1. **读取原始需求文档**
   - 从用户输入中获取原始需求文档路径：`{{args}}`
   - 如果没有提供路径，提示用户：`请提供需求文档路径，例如：/frontend:requirement docs/requirement.md`
   - 使用 Read 工具读取该文档内容

2. **调用需求分析器**
   - 使用 `frontend:frontend-requirement-analyzer` subagent 进行需求分析
   - 将原始需求文档内容传递给 subagent

3. **保存输出**
   - 将 subagent 返回的需求分析报告保存到 `DevAgentDoc/default/default/01_需求分析.md`
   - 或根据项目命名约定保存

## 注意事项

- 确保 subagent 的输出严格遵循规范格式
- 必须将结果保存到文件，不能只在对话中输出
