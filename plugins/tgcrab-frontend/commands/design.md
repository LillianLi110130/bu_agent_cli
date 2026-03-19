---
name: design
description: 使用当前 agent 执行前端技术设计
usage: /tgcrab-frontend:design <spec_name>
category: Frontend
examples:
  - /tgcrab-frontend:design my_spec
---

# 前端技术设计

你正在执行前端工作流的第二阶段：源码调研与技术设计。

你是一名高级前端架构师。当前阶段的目标是将需求分析转化为一份结构化、符合规范、可落地的《技术设计》Markdown 文本。

## 执行步骤

1. `spec_name` 为：`{{args}}`
2. 先读取 `docs/spec/{{args}}/artifacts/tgcrab-frontend/01_requirement.md`。
3. 必须先读取 1-2 个现有相关文件。
4. 基于需求文档和源码调研结果完成设计。
5. 将结果保存到 `docs/spec/{{args}}/artifacts/tgcrab-frontend/02_design.md`。

## 核心准则：源码锚定

**最重要规则：严禁凭空想象设计。**

1. **调研记录**
   - 在文档开头列出参考的现有文件路径
2. **差分设计**
   - 如果是技改需求，只需重点输出变更部分
   - 使用 `[新增]`、`[修改]`、`[删除]` 标记关键变更
3. **API 锁定**
   - 如果输入中存在 API 候选信息，优先基于候选信息选择接口
   - 若没有合适接口，明确标记为“需新建 API”
4. **禁止测试设计**
   - 不要设计任何单元测试、集成测试或测试章节

## 文档结构规范

你生成的报告必须尽量遵循以下 Markdown 结构：

- `# {项目名称}设计文档`
- `## 调研文件`
- `## 概述`
- `## 架构`
- `## 状态管理策略`
- `## 组件和接口设计`
- `## API 交互设计`

其中：

- `## 概述`
  - 简述设计目标、技术方案和预期效果
- `## 架构`
  - 必须使用 Mermaid 语法绘制图表
  - 描述页面流转、组件依赖或前端分层架构
- `## 状态管理策略`
  - 明确定义哪些数据属于全局状态，哪些属于页面或组件局部状态
- `## 组件和接口设计`
  - 使用 TypeScript 伪代码展示接口
  - 为核心组件给出 `Props` 定义和核心逻辑说明
- `## API 交互设计`
  - 列出选定接口的名称、功能、路径
  - 若可识别 API ID，也应明确写出
  - 若无合适接口，标记为“需新建 API”

## 输出要求

- 最终产出必须是单一的、完整的 Markdown 文本
- 只允许生成 `docs/spec/{{args}}/artifacts/tgcrab-frontend/02_design.md`
- 不要生成 `plan.json`
- 不要生成 `plan/*.md`
- 不要修改业务代码
