# Repository Guidelines

## 项目结构与模块组织
`claude_code.py` 是 CLI 入口。`cli/` 负责交互界面、slash 命令、`@` 技能命令和 Ralph 工作流。`tools/` 存放工具实现及沙箱、任务相关能力。`bu_agent_sdk/` 是核心 SDK，包含 agent 主循环、LLM 适配、skills、prompts、server 与 token 统计。`config/` 保存模型配置和预设。`tests/` 是主要 pytest 测试目录，仓库根目录下也保留了少量 `test_*.py` 风格的旧测试脚本。

## 构建、测试与开发命令
`python -m pip install -e ".[dev]"`：安装项目及开发依赖。
`python claude_code.py`：直接从源码启动 CLI。
`tg-agent`：运行 `pyproject.toml` 中声明的命令行入口。
`pytest`：执行全部测试。
`pytest tests/test_ralph_commands.py -k init_spec`：定向验证 Ralph 相关改动。
`ruff check .`：执行静态检查。
`black .`：统一代码格式。

## 代码风格与命名规范
项目目标版本为 Python 3.10。使用 4 空格缩进，单行不超过 100 列，与 Black 和 Ruff 配置保持一致。公共接口优先补充类型标注，函数职责应单一明确。模块、函数、变量使用 `snake_case`，类名使用 `PascalCase`。提示词、命令处理器和脚本文件名应直接反映用途，例如 `ralph_commands.py`、`frontend_developer.md`。

## 测试规范
异步逻辑使用 `pytest` 配合 `pytest-asyncio` 测试。新增测试优先放在 `tests/` 下，并与改动模块保持对应。测试文件命名为 `test_<feature>.py`，测试函数命名为 `test_<scenario>`。仓库当前没有强制覆盖率门禁，因此工具、slash 命令和 Ralph 流程相关改动应补充回归测试，不要只依赖手工验证。

## 提交与 Pull Request 规范
最近提交信息以简短祈使句为主，常见风格如 `修复ralph相关bug`、`集成ralph loop`、`新增TA拆解任务指令`。单个 commit 应聚焦一个明确改动。提交 PR 时请说明用户可见影响、列出已执行的验证命令、关联 issue 或 spec；只有在终端输出或交互流程发生变化时再附截图。

## 配置与安全建议
以 `.env.example` 为模板，本地敏感信息只放在未提交的 `.env` 中。修改 `config/model_presets.json` 时要特别谨慎，因为它会影响默认模型和 API 路由。不要提交 `.pytest_tmp/` 之类的临时测试产物。

## 方案规范
先从原始需求出发，不默认用户已经完全想清楚目标、约束和实现路径。 只有当需求存在关键歧义，且不同理解会导致明显不同方案或较高错误成本时，才先停下来澄清；否则基于最合理解释继续，并明确说明假设。 当需要给出修改或重构方案时，遵循以下原则：默认只围绕用户明确提出的目标设计方案，不擅自扩展业务目标，不引入替代业务路径。 优先给出满足目标的最小完整方案，而不是补丁式兼容方案；但如果“最短路径”与“非补丁”冲突，应优先选择不会引入结构性错误的最小正确方案。 不做与当前需求无关的兜底、降级或额外分支设计；但为保证逻辑闭合，允许加入必要的输入约束、状态检查和边界保护。 输出方案前，按输入、处理流程、状态变化、输出、上下游影响进行链路检查；对无法验证的部分必须明确标注假设和未验证前提，不得将推测表述为已确认事实。