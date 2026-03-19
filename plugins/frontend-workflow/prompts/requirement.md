# Requirement 阶段

你正在执行 `frontend-workflow` 的需求分析阶段。

## 输入
- 原始材料目录：`{{input_dir}}`
- spec 名称：`{{spec_name}}`

## 执行要求
- 先读取 `{{input_dir}}` 下的原始材料。
- 使用 `@frontend-requirement-analyzer` 参与分析并组织文档内容。

## 必须生成的文件
- `{{requirement_file}}`

## 输出要求
- 产出结构化需求分析文档，包含背景、目标、范围、关键流程、验收标准。
- 对于技改或修复场景，只保留变更相关内容。

## 约束
- 不要生成设计文档
- 不要生成 `plan.json`
- 不要生成 `plan/*.md`
- 不要修改业务代码
- 只允许在 `artifacts/frontend-workflow/` 下写入产物
