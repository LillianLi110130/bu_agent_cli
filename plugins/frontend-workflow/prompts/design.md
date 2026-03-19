# Design 阶段

你正在执行 `frontend-workflow` 的技术设计阶段。

## 输入
- 需求分析文档：`{{requirement_file}}`
- 允许补充读取源码和 `{{input_dir}}` 中的原始材料

## 执行要求
- 先读取 `{{requirement_file}}`。
- 使用 `@frontend-design-analyzer` 参与设计，必要时补充读取源码做源码锚定。

## 必须生成的文件
- `{{design_file}}`

## 输出要求
- 明确涉及模块、组件设计、状态管理、接口和主要风险。
- 必须基于仓库现有实现进行源码锚定，不能凭空想象。

## 约束
- 不要生成 `plan.json`
- 不要生成 `plan/*.md`
- 不要修改业务代码
- 只允许在 `artifacts/frontend-workflow/` 下写入产物
