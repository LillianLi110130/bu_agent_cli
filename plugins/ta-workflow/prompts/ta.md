# TA 阶段

你正在执行 `ta-workflow` 的第一阶段。请严格遵守以下约束：

## 输入
- 原始材料目录：`{{input_dir}}`
- spec 名称：`{{spec_name}}`

## 必须生成的文件
- `{{requirements_file}}`
- `{{design_file}}`
- `{{task_domains_file}}`

## 输出要求
- `01_requirements.md`：整理需求背景、目标、范围、验收标准。
- `02_design.md`：概述关键实现方向、涉及模块和主要风险。
- `03_task_domains.md`：给出后续任务分解所需的工作域和依赖关系。

## 约束
- 不要生成 `plan.json`
- 不要生成 `plan/*.md`
- 不要修改业务代码
- 只允许在 `artifacts/ta-workflow/` 下写入产物
