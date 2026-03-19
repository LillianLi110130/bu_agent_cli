# Decompose 阶段

你正在执行 `ta-workflow` 的第二阶段。请严格遵守以下约束：

## 输入
- `{{requirements_file}}`
- `{{design_file}}`
- `{{task_domains_file}}`

## 必须生成的文件
- `{{plan_json}}`
- 至少一个 `{{plan_dir}}/*.md` 任务文档

## 任务格式要求
- `plan.json` 必须是 JSON 数组。
- 每个任务对象至少包含：`task_name`、`priority`、`status`、`implment_times`、`task_file`。
- 初始状态统一为 `TODO`。
- `task_file` 必须指向 `plan/` 目录下实际存在的 Markdown 文档。
- 任务优先级使用整数，数值越小优先级越高。

## 约束
- 不要重新生成 TA 阶段产物
- 不要修改业务代码
- 只允许在 `plan/` 下写入最终产物
