---
name: tasks
description: 执行前端任务拆分阶段
usage: /frontend-workflow:tasks <迭代名/负责人>
category: Frontend
mode: python
script: scripts/tasks.py
examples:
  - /frontend-workflow:tasks v1.0/zhangsan
---

执行前端任务拆分阶段，调用 devagent 基于需求分析和技术设计文档进行任务拆分。

**工作流程**:
1. 读取需求分析和技术设计文档
2. 使用 `@frontend-task-splitter` subagent 执行拆分
3. 生成 `DevAgentDoc/[迭代名]/[负责人]/03_任务列表.md`

**核心原则**:
- **拒绝废话**: 技改需求禁止生成初始化任务
- **任务精准化**: 描述必须包含具体文件修改点
- **扁平化输出**: 内部思考层级，输出扁平列表

**任务格式**:
```
- [ ] 任务ID：描述 | status: pending | retries: 0 | 目标文件: [路径] | 关联需求: [ID]
```

**依赖排序建议**:
1. 环境配置与初始化
2. 全局状态管理（Redux Store）
3. 通用工具函数/Hooks
4. 原子组件（按钮、输入框等）
5. 业务组件（组合组件）

**示例**:
```
/frontend-workflow:tasks v1.0/zhangsan
```

脚本会自动查找以下文档：
- `DevAgentDoc/v1.0/zhangsan/01_需求分析.md`
- `DevAgentDoc/v1.0/zhangsan/02_需求设计.md`
