# 前端任务拆解

你正在执行前端工作流的第三阶段：任务拆解。

## 执行步骤

1. **读取输入**: 同时读取 `{{requirement_file}}` 与 `{{design_file}}`。
2. **使用 subagent 执行拆解**: 请使用 `@frontend-task-splitter` 参与拆解。
3. **保存最终计划**: 直接生成 `{{plan_json}}` 和 `{{plan_dir}}/*.md`。

## 核心原则

- 对技改需求，禁止生成“初始化”“创建目录”等无关任务
- 任务描述必须包含具体的文件修改点
- 必须按逻辑依赖关系排序
- 绝对禁止生成任何测试任务
- 单个任务应是可独立执行的最小工作单元

## 最终产物要求

- `plan.json` 必须是 JSON 数组
- 每个任务对象至少包含：
  - `task_name`
  - `priority`
  - `status`
  - `implment_times`
  - `task_file`
- `status` 初始值必须是 `TODO`
- `implment_times` 初始值必须是 `0`
- `task_file` 必须指向 `plan/` 下实际存在的 Markdown 文档
- 每个 `plan/*.md` 必须描述任务目标、目标文件、实现约束、完成标准

## 约束

- 不要生成额外的中间任务列表文档
- 不要重新生成 requirement/design 阶段产物
- 不要修改业务代码
