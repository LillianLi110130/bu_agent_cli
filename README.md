# BU Agent CLI

一个面向编码场景的 Agent CLI，基于 `bu_agent_sdk` 实现，支持 OpenAI-compatible 模型、工具调用、子代理委派和上下文压缩。

## 核心能力

- 交互式 CLI：流式展示思考、工具调用和执行结果
- 工具调用：内置 `bash`、文件读写编辑、搜索、todo、子代理工具
- 子代理机制：通过 `task` 工具委派到专业代理（如 code_reviewer/frontend_developer）
- 上下文管理：自动 compaction，缓解长对话上下文溢出
- 模型切换：支持会话内通过 `/model` 命令切换预设模型（保留上下文）
- Token 统计：记录 token 使用，并可选计算成本

## 架构概览

- 入口层：`claude_code.py`
  - 负责组装 LLM、Agent、工具、Sandbox、系统提示词
- UI 层：`cli/`
  - `cli/app.py`：交互循环、事件渲染、slash 命令
  - `cli/slash_commands.py`：命令注册与补全
- Agent 核心：`bu_agent_sdk/agent/`
  - 主循环、工具调度、重试、完成判定、流式事件
- LLM 适配层：`bu_agent_sdk/llm/`
  - 当前主要实现为 `ChatOpenAI`（兼容 OpenAI API schema）
- 工具层：`tools/`
  - Bash、文件、搜索、todo、subagent
- 扩展层：
  - Skills：`bu_agent_sdk/skills/`
  - 子代理配置：`bu_agent_sdk/prompts/agents/*.md`

## 安装

```bash
pip install -e .
```

## 环境变量

可参考 `.env.example`：

- `OPENAI_API_KEY`：OpenAI-compatible API Key
- `LLM_MODEL`：默认模型（默认 `GLM-4.7`）
- `LLM_BASE_URL`：默认网关地址（默认 `https://open.bigmodel.cn/api/coding/paas/v4`）
- `ZHIPU_API_KEY`：可按你的预设配置使用

## 启动

```bash
# 方式 1：安装后的命令行入口
bu-agent

# 方式 2：直接运行脚本
python claude_code.py

# 指定模型
bu-agent --model gpt-4o

# 指定沙箱根目录
bu-agent --root-dir ./your-project
```

## 模型预设与会话内切换

预设文件：`config/model_presets.json`

示例：

```json
{
  "default": "glm",
  "presets": {
    "glm": {
      "model": "GLM-4.7",
      "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
      "api_key_env": "OPENAI_API_KEY"
    },
    "gpt4o": {
      "model": "gpt-4o",
      "base_url": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY"
    }
  }
}
```

运行中可使用：

```text
/model           # 打开编号选择器
/model list      # 查看预设
/model show      # 查看当前模型
/model <preset>  # 切换到某个预设
```

## Slash 命令

- `/help`
- `/exit` `/quit`
- `/pwd`
- `/clear` `/cls`
- `/model [show|list|<preset>]`
- `/ralph <init-spec|init-agent|decompose|dry-run|run|status|cancel> ...`
- `/skills`
- `/reset`
- `/history`（占位，暂未实现完整历史展示）

## Ralph Loop 使用指南

Ralph Loop 是一套面向需求拆解和批量实现的命令式工作流。它不走主 Agent 的工具调用链，而是通过 `/ralph` 这一组 Slash 命令显式触发。

### 功能概览

- 初始化 spec 目录骨架
- 初始化 `.devagent` 命令与 agent 配置
- 读取 `requirement/` 需求文档并调用 `devagent` 执行任务拆解
- 预览 `plan.json` 中的待执行任务
- 后台循环执行 `ralph_loop.py`
- 查询后台执行状态
- 取消后台执行

### 运行前提

- 工作区根目录中存在 `ralph_init.py`、`ralph_loop.py`、`template/`、`.devagent/`
- 环境变量中已配置好模型相关 API Key
- `devagent` 命令已安装并可通过 `PATH` 访问
- 当前工作目录就是 Ralph 要操作的项目根目录

### 目录约定

Ralph 默认围绕 `docs/spec/<spec_name>/` 这一套目录约定工作：

```text
docs/spec/<spec_name>/
├── requirement/     # 需求文档，需手动补充
├── plan/
│   └── plan.json    # 任务队列
├── implement/       # 实现规划与实现结果文档
└── logs/            # 任务执行日志
```

同时，工作区根目录下需要有：

```text
.devagent/
└── commands/
    └── ralph/
        ├── DECOMPOSE_TASK.md
        ├── decompose.toml
        └── implement.md
```

### 推荐使用流程

```text
/ralph init-spec my_spec
/ralph init-agent
# 手动将需求文档放入 docs/spec/my_spec/requirement/
/ralph decompose my_spec
/ralph dry-run my_spec
/ralph run my_spec --silent
/ralph status
```

### 命令详解

#### 1. 初始化 spec 目录

```text
/ralph init-spec <spec_name> [--target-dir <path>] [--force]
```

作用：

- 从 `template/` 复制出 spec 目录骨架
- 创建 `requirement/`、`plan/`、`implement/`、`logs/`
- 创建一个空的 `plan/plan.json`

参数说明：

- `<spec_name>`：spec 名称，最终目录默认是 `docs/spec/<spec_name>/`
- `--target-dir <path>`：指定目标根目录
  - 默认是当前工作目录
  - 当未指定时，最终会创建到 `./docs/spec/<spec_name>/`
  - 指定后，最终会创建到 `<path>/<spec_name>/`
- `--force`：目标目录非空时仍然尝试初始化

示例：

```text
/ralph init-spec payment_refactor
/ralph init-spec payment_refactor --target-dir D:/work/specs
/ralph init-spec payment_refactor --force
```

#### 2. 初始化 `.devagent`

```text
/ralph init-agent [--target-dir <path>]
```

作用：

- 将仓库中的 `.devagent/` 复制到目标工作区根目录
- 为后续 `decompose` 和 `run` 提供命令模板与 agent 定义

参数说明：

- `--target-dir <path>`：`.devagent` 的目标根目录
  - 默认是当前工作目录

示例：

```text
/ralph init-agent
/ralph init-agent --target-dir D:/work/project-a
```

#### 3. 需求拆解

```text
/ralph decompose <spec_name> [description...]
```

作用：

- 读取 `docs/spec/<spec_name>/requirement/` 下的需求文档
- 读取 `.devagent/commands/ralph/DECOMPOSE_TASK.md`
- 调用 `devagent --yolo`
- 生成 `plan/*.md` 和 `plan/plan.json`
- 输出拆解日志到 `docs/spec/<spec_name>/logs/decompose.log`

参数说明：

- `<spec_name>`：要拆解的 spec 名称
- `[description...]`：附加给 `devagent` 的补充说明，会拼接到拆解 prompt 末尾

前置条件：

- `docs/spec/<spec_name>/requirement/` 必须存在
- `requirement/` 目录不能是空的
- 工作区根目录必须存在 `.devagent/commands/ralph/DECOMPOSE_TASK.md`
- `devagent` 必须可执行

示例：

```text
/ralph decompose payment_refactor
/ralph decompose payment_refactor 先优先拆出数据库改造相关任务
```

#### 4. 预览任务执行计划

```text
/ralph dry-run [<spec_name>] [--plan-file <path>] [--log-dir <path>] [--max-retry <n>] [--delay <seconds>] [--enable-git] [--main-branch <name>] [--work-branch <name>] [--silent]
```

作用：

- 前台执行 `ralph_loop.py --dry-run`
- 读取 `plan.json`
- 展示待执行任务、优先级、状态和执行次数
- 不实际调用 `devagent`

参数说明：

- `<spec_name>`：按默认目录约定执行，对应 `docs/spec/<spec_name>/plan/plan.json`
- `--plan-file <path>`：直接指定 `plan.json` 的绝对路径或相对路径
- `--log-dir <path>`：覆盖默认日志目录
- `--max-retry <n>`：覆盖最大重试次数
- `--delay <seconds>`：覆盖循环轮次间隔
- `--enable-git`：启用 Ralph 的 Git 分支管理逻辑
- `--main-branch <name>`：指定主分支名称
- `--work-branch <name>`：指定工作分支名称
- `--silent`：静默模式，减少终端输出

说明：

- `<spec_name>` 和 `--plan-file` 二选一，建议优先使用 `<spec_name>`
- `dry-run` 会走和 `run` 相同的参数组装逻辑，只是自动追加 `--dry-run`

示例：

```text
/ralph dry-run payment_refactor
/ralph dry-run --plan-file D:/work/project/docs/spec/payment_refactor/plan/plan.json
/ralph dry-run payment_refactor --max-retry 5 --delay 2
```

#### 5. 后台运行 Ralph Loop

```text
/ralph run [<spec_name>] [--plan-file <path>] [--log-dir <path>] [--max-retry <n>] [--delay <seconds>] [--enable-git] [--main-branch <name>] [--work-branch <name>] [--silent]
```

作用：

- 后台启动 `ralph_loop.py`
- 由 `RalphProcessManager` 分配 `run_id`
- 持续执行 `plan.json` 中的 `TODO/FAILED` 任务
- 调用 `devagent --yolo`
- 更新 `plan.json` 和 `logs/`

参数说明：

- `<spec_name>`：默认模式，推荐使用
- `--plan-file <path>`：直接指定 `plan.json`
- `--log-dir <path>`：覆盖默认日志目录
- `--max-retry <n>`：单任务最大重试次数
- `--delay <seconds>`：每轮处理后的睡眠间隔
- `--enable-git`：启用 Git 分支工作流
- `--main-branch <name>`：Git 主分支名
- `--work-branch <name>`：Git 工作分支名
- `--silent`：静默模式，减少脚本终端输出

输出结果：

- 启动成功后会返回 `run_id`
- 同时会显示 `spec`、`plan`、`log_dir`

示例：

```text
/ralph run payment_refactor
/ralph run payment_refactor --silent
/ralph run payment_refactor --enable-git --main-branch main --work-branch devagent-work
/ralph run --plan-file D:/work/project/docs/spec/payment_refactor/plan/plan.json --log-dir D:/work/project/docs/spec/payment_refactor/logs
```

#### 6. 查询后台状态

```text
/ralph status [run_id]
```

作用：

- 不传 `run_id` 时，列出当前工作区中所有 Ralph 运行记录
- 传入 `run_id` 时，展示某次运行的详细状态
- 读取 `plan.json` 并统计 `DONE`、`FAILED`、`TODO`

状态数据来源：

- `.bu_agent/ralph/runs.json`
- `plan.json`

示例：

```text
/ralph status
/ralph status 8c1a2d3e4f5g
```

#### 7. 取消后台运行

```text
/ralph cancel <run_id>
```

作用：

- 终止指定的 Ralph 后台进程
- 更新运行状态为 `cancelled`

示例：

```text
/ralph cancel 8c1a2d3e4f5g
```

### 运行状态与日志

Ralph 会产生两类运行数据：

- 工作流状态文件：`.bu_agent/ralph/runs.json`
- 进程级日志：`.bu_agent/ralph/process_logs/`
- spec 级日志：`docs/spec/<spec_name>/logs/`

`/ralph status` 读取的就是这部分信息。

### 常见问题

#### 1. `decompose` 提示 requirement 目录为空

原因：

- `ralph_init` 只创建目录，不会自动生成需求文档

处理方式：

- 手动将需求说明写入 `docs/spec/<spec_name>/requirement/`
- 推荐使用 Markdown 文档

#### 2. `run` 或 `dry-run` 提示找不到 `devagent`

原因：

- `devagent` 不在系统 `PATH`

处理方式：

- 确认 `devagent` 已安装
- 确认当前终端可以直接执行 `devagent --yolo`

#### 3. `run` 提示找不到 `implement.md`

原因：

- 还没有执行 `/ralph init-agent`
- 或当前工作区根目录缺少 `.devagent/commands/ralph/implement.md`

#### 4. `status` 看不到运行记录

原因：

- 还没有执行过 `/ralph run`
- 或当前工作区下的 `.bu_agent/ralph/runs.json` 为空

## Skill 命令（@）

- `@<skill-name> <message>`：显式加载技能并执行任务
- `@` + `Tab`：补全可用技能
- `/skills`：按分类查看全部技能

示例：

```text
@brainstorming 设计一个新功能
@test-driven-development 为这个模块补测试
```

## 内置工具

- `bash`：执行 shell 命令
- `read`：读取文件（带行号）
- `write`：写入文件
- `edit`：按字符串替换编辑文件
- `glob_search`：按 glob 查找文件
- `grep`：正则搜索文件内容
- `todo_read` / `todo_write`：读写会话 todo
- `done`：显式完成任务
- `task`：调用子代理处理子任务

## 子代理

子代理定义在：`bu_agent_sdk/prompts/agents/*.md`

- 通过 frontmatter 指定 `mode/model/tools` 等配置
- `mode` 为 `subagent` 或 `all` 的代理可被 `task` 工具调用

## 项目结构（实际）

```text
bu_agent_cli/
├── claude_code.py
├── ralph_init.py
├── ralph_loop.py
├── .devagent/
├── template/
├── cli/
│   ├── app.py
│   ├── slash_commands.py
│   ├── ralph_commands.py
│   ├── ralph_service.py
│   ├── ralph_process_manager.py
│   └── ralph_models.py
├── tools/
│   ├── bash.py
│   ├── files.py
│   ├── search.py
│   ├── todos.py
│   ├── subagent.py
│   └── sandbox.py
├── config/
│   ├── model_config.py
│   └── model_presets.json
└── bu_agent_sdk/
    ├── agent/
    ├── llm/
    ├── tools/
    ├── skill/
    ├── skills/
    ├── prompts/
    └── tokens/
```

## 说明

- README 以当前仓库代码为准，若你新增工具或子代理，请同步更新本文档。
