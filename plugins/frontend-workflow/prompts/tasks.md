# Tasks 阶段

你正在执行 `frontend-workflow` 的任务拆解阶段。

## 输入
- 需求分析文档：`{{requirement_file}}`
- 技术设计文档：`{{design_file}}`

## 执行要求
- 同时读取 `{{requirement_file}}` 与 `{{design_file}}`。
- 使用 `@frontend-task-splitter` 参与拆解，并直接生成 Ralph 可执行计划。

## 必须生成的文件
- `{{plan_json}}`
- 至少一个 `{{plan_dir}}/*.md` 任务文档

## 任务格式要求
- `plan.json` 必须是 JSON 数组。
- 每个任务对象至少包含：`task_name`、`priority`、`status`、`implment_times`、`task_file`。
- 初始状态统一为 `TODO`。
- `task_file` 必须指向 `plan/` 下实际存在的任务文档。
- 任务文档应描述目标、约束、涉及文件和完成标准。

## 约束
- 不要生成额外的中间任务列表文档
- 不要重新生成 requirement/design 阶段产物
- 不要修改业务代码
- 只允许在 `plan/` 下写入最终产物
