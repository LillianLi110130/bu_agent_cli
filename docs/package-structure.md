# 目录与包结构说明

本文档用于说明 Crab CLI 当前代码仓库的主要目录结构、模块职责，以及常见改动应该去哪里找。

## 顶层结构

```text
.
├── tg_crab_main.py        # CLI 主入口，负责启动组装
├── pyproject.toml         # 包信息、依赖、命令入口、打包配置
├── uv.lock                # 依赖锁定文件
├── agent_core/            # Agent 核心运行时
├── cli/                   # 终端交互层
├── tools/                 # 提供给 Agent 调用的工具
├── skills/                # 内置技能
├── plugins/               # 插件包
├── config/                # 模型预设和运行配置
├── docs/                  # 文档、设计方案、使用手册
├── tests/                 # 测试用例
├── web/                   # Web 控制台前端
├── gateway/               # Gateway 服务工程
├── tg_mem/                # 长期记忆底层组件
├── template/              # 计划、日志、产物等运行模板
└── .devagent/             # 内置开发 agent 和命令定义
```

## 启动入口

```text
tg_crab_main.py
```

`tg_crab_main.py` 是主入口文件。`pyproject.toml` 中的命令入口指向这里：

```toml
crab = "tg_crab_main:cli_main"
```

主要职责：

- 解析命令行参数和环境变量。
- 加载模型配置、技能、记忆、插件、team、bridge 等运行配置。
- 创建主 `Agent`。
- 注册工具和 hooks。
- 创建 `TGAgentCLI`。
- 启动交互式 CLI 循环。
- 在特殊内部参数存在时，分发到 worker 或 team worker 模式。

可以把它理解成“启动装配层”：它不主要负责 UI 展示，也不主要负责 agent 推理逻辑，而是把各种组件组装起来。

## `cli/`

终端交互层，负责用户在命令行里看到和输入的大部分内容。

```text
cli/
├── app.py                 # TGAgentCLI 主类，负责终端渲染和交互流程
├── tui_app.py             # 固定输入框/TUI 模式
├── slash_commands.py      # slash 命令注册、解析、补全
├── at_commands.py         # @ 技能命令解析和补全
├── image_input.py         # 图片输入命令解析
├── interactive_input.py   # 输入提示和交互辅助
├── model_switch_service.py# /model 模型切换
├── session_runtime.py     # CLI 会话运行时绑定
├── session_store.py       # 会话持久化和恢复
├── resume_handler.py      # /resume 命令处理
├── settings_handler.py    # /settings 命令处理
├── skills_handler.py      # /skills 命令处理
├── memory_handler.py      # /memory 命令处理
├── plugins_handler.py     # 插件命令处理
├── agents_handler.py      # agent 配置命令处理
├── init_agent.py          # /init 辅助 agent 流程
├── ralph_*.py             # Ralph 工作流命令和进程管理
├── im_bridge/             # 本地/IM/Web 文件桥接队列
├── worker/                # IM worker 进程、鉴权、gateway client
└── team/                  # team 命令、dashboard、worker、inbox
```

重点文件：

- `cli/app.py`：最核心的 CLI UI 文件。欢迎面板、`/help`、Markdown 输出、工具调用展示、输入循环、桥接队列消费、agent 事件渲染都在这里。
- `cli/slash_commands.py`：定义 `/help`、`/model`、`/skills`、`/memory`、`/team` 等命令的元信息和补全。
- `cli/im_bridge/store.py`：用文件系统保存桥接请求、进度、结果、日志和外发事件。
- `cli/worker/runner.py`：运行 IM worker 循环，把 gateway 消息转成本地 bridge 请求。
- `cli/team/handler.py`：实现 `/team` 相关命令。

## `agent_core/`

Agent 核心运行时。这里是模型调用、上下文管理、工具调用、hooks、插件、记忆、team 等核心能力所在。

```text
agent_core/
├── agent/                 # Agent 循环、上下文、事件、hooks、压缩
├── bootstrap/             # 项目上下文和启动注入
├── llm/                   # LLM 抽象和 OpenAI-compatible 模型封装
├── memory/                # 记忆存储、memory review、memory 工具
├── plugin/                # 插件加载、管理、执行
├── prompts/               # 系统提示词、身份设定、远程 reset 提示词
├── runtime/               # 通用运行辅助
├── server/                # 服务端、Web console、Gateway 支持
├── skill/                 # 技能发现、解析、运行时、review
├── task/                  # 本地后台 agent task
├── team/                  # team 协议、状态、mailbox、task board
├── tokens/                # token 统计、价格、用量视图
├── tools/                 # 核心工具装饰器和依赖注入基础设施
├── runtime_paths.py       # 运行路径解析
└── version.py             # CLI 版本辅助
```

### `agent_core/agent/`

这是最核心的子包。

```text
agent_core/agent/
├── service.py             # Agent 类，query/query_stream 入口
├── runtime_loop.py        # Agent 循环执行机制
├── context.py             # 消息和上下文管理
├── events.py              # 流式事件类型
├── hooks.py               # Hook 系统和内置策略 hook
├── hitl.py                # Human-in-the-loop 审批模型
├── command_safety.py      # Bash 安全审批规则
├── tool_args.py           # 工具参数解析
├── tool_call_validation.py# 工具调用校验和恢复
├── model_routing_hook.py  # 模型路由 hook
├── budget.py              # 上下文预算评估
├── compaction/            # 上下文压缩服务和模型
├── registry.py            # Agent 注册表
└── config.py              # Agent 配置解析
```

主要职责：

- 维护对话上下文。
- 调用 LLM。
- 解析和校验工具调用。
- 执行工具。
- 向 CLI 流式输出事件，例如文本增量、思考增量、工具调用、最终回复。
- 运行安全、审批、模型路由、审计、压缩、子 agent 完成检查等 hooks。

## `tools/`

提供给 Agent 调用的工具集合。

```text
tools/
├── __init__.py            # 聚合 ALL_TOOLS
├── bash.py                # Shell 命令执行
├── files.py               # read/write/edit 文件工具
├── search.py              # glob_search 和 grep
├── resolve_path.py        # 路径解析
├── sandbox.py             # SandboxContext 和文件系统安全控制
├── todos.py               # todo_read、todo_write、done
├── skills.py              # skill_list、skill_view、skill_manage
├── message.py             # 消息/桥接工具
├── web.py                 # web_fetch
├── xlsx.py                # Excel 读取
├── agent_tool.py          # delegate 和 delegate_parallel
├── shell_tasks.py         # 后台 shell task 管理
├── task_status.py         # 后台任务状态工具
├── task_output.py         # 后台任务输出工具
├── task_cancel.py         # 后台任务取消工具
├── team_tool.py           # team 管理工具
└── text_encoding.py       # 文本编码辅助
```

`tools/__init__.py` 会导出 `ALL_TOOLS`，`tg_crab_main.py` 会把这些工具注册给主 `Agent`。

## `skills/`

内置技能库。

```text
skills/
├── using-superpowers/
├── brainstorming/
├── systematic-debugging/
├── test-driven-development/
├── writing-plans/
├── writing-skills/
├── requesting-code-review/
├── receiving-code-review/
├── subagent-driven-development/
├── dispatching-parallel-agents/
├── executing-plans/
├── finishing-a-development-branch/
├── calculator/
├── kplus-search/
├── llm-wiki/
├── web-access-main/
└── ...
```

每个技能通常以 `SKILL.md` 为入口。技能可以通过 `@` 命令调用，也可以通过 skill 工具查看和管理。

## `plugins/`

插件包目录。

```text
plugins/
├── awesome-subagents/
├── frontend-workflow/
├── review-kit/
├── ta-workflow/
├── tgcrab-frontend/
└── runtime_helpers.py
```

插件可以提供 agents、commands、skills 和工作流辅助能力。插件加载由 `agent_core/plugin` 负责。

## `config/`

运行配置目录。

```text
config/
├── model_config.py
├── model_presets.json
└── gateway_routes.server.example.json
```

主要用于模型预设、模型配置和 gateway 路由示例。

## `tg_mem/`

长期记忆底层组件。

```text
tg_mem/
├── configs/
├── embeddings/
├── graphs/
├── llms/
├── memory/
├── utils/
└── vector_stores/
```

用于支持记忆存储、embedding、图结构、向量库等长期记忆能力。

## `web/`

Web 控制台前端。

```text
web/
├── package.json
├── vite.config.ts
├── index.html
└── src/
    ├── App.tsx
    ├── main.tsx
    └── styles/
```

这是一个 Vite/React 前端工程，用于 Web console 或远程控制台相关能力。

## `gateway/`

Gateway 服务工程。

```text
gateway/
├── pom.xml
└── src/
```

看起来是 Java/Maven 工程，主要服务于远程 worker 或消息转发链路。

## `tests/`

Pytest 测试目录，覆盖范围比较广。

主要包括：

- CLI 默认参数和 UI 展示。
- Agent hooks 和运行时行为。
- 工具行为。
- Worker 鉴权和 gateway client。
- IM bridge 存储。
- Team runtime。
- 插件和技能。
- Memory review。
- Web console 和 gateway 集成。

## 运行链路

```text
用户启动 crab
  ↓
tg_crab_main.py
  ↓
解析参数、加载环境变量和配置、创建 Agent
  ↓
加载 tools、skills、memory、plugins、team、bridge
  ↓
创建 cli/app.py 中的 TGAgentCLI
  ↓
接收终端输入、slash 命令、@ 技能、图片输入或 bridge 消息
  ↓
调用 agent_core/agent 中的 Agent.query_stream
  ↓
LLM 调用和工具调用
  ↓
执行 tools/*
  ↓
事件流回到 TGAgentCLI
  ↓
使用 Rich 渲染到终端
```

## 常见修改点

```text
欢迎面板、Markdown 输出、工具调用展示
  -> cli/app.py

slash 命令列表和补全
  -> cli/slash_commands.py

@ 技能解析和补全
  -> cli/at_commands.py

模型切换逻辑
  -> cli/model_switch_service.py

Agent 主循环
  -> agent_core/agent/service.py
  -> agent_core/agent/runtime_loop.py

工具列表
  -> tools/__init__.py

单个工具行为
  -> tools/*.py

启动配置和依赖组装
  -> tg_crab_main.py

桥接队列
  -> cli/im_bridge/

远程 IM worker
  -> cli/worker/

Team 能力
  -> agent_core/team/
  -> cli/team/
  -> tools/team_tool.py

技能
  -> skills/*/SKILL.md

插件
  -> plugins/*/plugin.json
```
