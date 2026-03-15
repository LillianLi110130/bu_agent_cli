# BU Agent CLI

一个面向编码场景的 Agent CLI / Gateway 项目，基于 `bu_agent_sdk` 实现，支持 OpenAI-compatible 模型、工具调用、子代理委派、工作区指令注入，以及 Telegram 私聊网关接入。

## 核心能力

- **交互式 CLI**：本地终端对话，支持 slash 命令、`@skill` 调用和模型切换
- **工具调用**：内置 `bash`、文件读写编辑、搜索、todo、子代理执行等工具
- **子代理机制**：可将任务委派给专业子代理并同步等待结果，或并行执行多个子任务
- **工作区指令同步**：自动读取工作区 `AGENTS.md`，并同步到当前会话上下文
- **Gateway 会话隔离**：按 chat/session 维度维护独立运行时，避免不同会话串上下文
- **Telegram 集成**：支持 Telegram 私聊消息接入与回复
- **Heartbeat 机制**：定期检查 `HEARTBEAT.md`，在有任务时自动投递到网关会话
- **Token 统计**：支持 token 使用统计与成本计算能力

## 架构概览

- **入口层**
  - `claude_code.py`：本地 CLI 入口
  - `bu_agent_sdk/gateway/main.py`：IM gateway 入口
- **Bootstrap 层**
  - `bu_agent_sdk/bootstrap/agent_factory.py`：创建 Agent、LLM、SandboxContext
  - `bu_agent_sdk/bootstrap/session_bootstrap.py`：同步工作区 `AGENTS.md`
- **Agent 核心**
  - `bu_agent_sdk/agent/`：主循环、上下文、事件、子代理管理
- **LLM 适配层**
  - `bu_agent_sdk/llm/`：消息模型、schema、OpenAI-compatible 适配
- **Gateway 运行时层**
  - `bu_agent_sdk/runtime/`：按 session 管理 `AgentRuntime`
  - `bu_agent_sdk/gateway/`：消息分发、服务编排、配置解析
  - `bu_agent_sdk/channels/`：Telegram 等渠道接入
  - `bu_agent_sdk/bus/`：网关内部消息总线
  - `bu_agent_sdk/heartbeat/`：定时触发任务
- **工具与技能层**
  - `tools/`：bash、文件、搜索、todo、subagent 等工具
  - `bu_agent_sdk/skill/`：skill 解析与加载
  - `bu_agent_sdk/skills/`：内置 skills
  - `bu_agent_sdk/prompts/agents/`：子代理提示词

## 安装

建议使用 `uv` 管理依赖和虚拟环境：

```bash
uv sync
```

如果你需要开发依赖：

```bash
uv sync --dev
```

## 环境变量

可参考仓库根目录下的 `.env.example`：

- `OPENAI_API_KEY`：默认 OpenAI-compatible API Key
- `ZHIPU_API_KEY`：可选，供你的模型预设或其他接入使用
- `LLM_MODEL`：默认模型，默认值为 `GLM-4.7`
- `LLM_BASE_URL`：默认模型网关地址
- `TELEGRAM_BOT_TOKEN`：Telegram Bot Token
- `TELEGRAM_ALLOW_FROM`：允许访问 bot 的 Telegram 用户 ID 列表，逗号分隔；不配置则拒绝所有用户
- `TELEGRAM_PROXY`：Telegram 代理地址，可为空
- `GATEWAY_HEARTBEAT_ENABLED`：是否启用 heartbeat，默认 `true`
- `GATEWAY_HEARTBEAT_INTERVAL_SECONDS`：heartbeat 轮询间隔，默认 `1800`

示例：

```bash
cp .env.example .env
```

## 启动方式

### 1）CLI 模式

适合本地终端交互式使用：

```bash
# 默认使用当前目录作为工作区
uv run bu-agent

# 指定模型
uv run bu-agent --model gpt-4o

# 指定工作区根目录
uv run bu-agent --root-dir ./your-project

# 也可以直接运行入口脚本
uv run python claude_code.py
```

### 2）Gateway 模式

适合通过 Telegram、Zhaohu webhook 等 IM 渠道接入 Agent。

```bash
# 建议在目标工作区根目录执行
uv run bu-agent-gateway

# 指定工作区
uv run bu-agent-gateway --root-dir ./your-project

# 指定模型
uv run bu-agent-gateway --model gpt-4o

# 调整 heartbeat 间隔
uv run bu-agent-gateway --heartbeat-interval-seconds 600

# 禁用 heartbeat
uv run bu-agent-gateway --disable-heartbeat
```

#### Gateway 启动前置条件

1. 当前目录下准备好 `.env`
2. 至少启用一种 channel：
   - Telegram：配置 `TELEGRAM_BOT_TOKEN`
   - Zhaohu：配置 `ZHAOHU_ENABLED=true`
3. 如果启用 Telegram，还需要配置 `TELEGRAM_ALLOW_FROM`
   - 例如：`TELEGRAM_ALLOW_FROM=123456789,987654321`
   - 测试场景可使用 `TELEGRAM_ALLOW_FROM=*`
4. 如果启用 Zhaohu webhook channel skeleton，可按需配置：
   - `ZHAOHU_HOST`，默认 `0.0.0.0`
   - `ZHAOHU_PORT`，默认 `18080`
   - `ZHAOHU_WEBHOOK_PATH`，默认 `/webhook/zhaohu`
5. 建议在工作区根目录启动，这样可直接使用：
   - `AGENTS.md`：工作区规则，会在会话处理中自动同步到 Agent 上下文
   - `HEARTBEAT.md`：heartbeat 任务来源

> 注意：gateway 读取 `.env` 的位置是**当前启动目录**；`--root-dir` 控制的是 Agent 工作区，不会改变 `.env` 的加载位置。

#### Gateway 会话行为

- 当前实现的 Telegram channel 只处理 **private chat**
- 当前提供 `ZhaohuChannel` webhook skeleton：启动后会拉起 FastAPI server，提供 `GET /healthz`，并接受 `POST ZHAOHU_WEBHOOK_PATH` 的通用扁平 webhook payload 后投递到 bus；但鉴权、协议适配与 outbound 回传仍暂未实现
- 每个 chat/session 会创建独立的 `AgentRuntime`
- 同一个会话会复用上下文，不同会话之间不会串历史消息
- Telegram 内置命令：
  - `/start`
  - `/help`
  - `/new`：清空当前会话运行时，开始新会话

## 模型预设与会话内切换

预设文件位于：`config/model_presets.json`

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

当前 CLI 内置命令包括：

- `/help`
- `/exit` `/quit`
- `/pwd`
- `/clear`
- `/model [show|list|<preset>]`
- `/skills`
- `/reset`
- `/init`
- `/history`
- `/tasks`
- `/task <task_id>`
- `/task_cancel <task_id>`
- `/allow <path>`
- `/allowed`
- `/agents [list|show|create|edit|delete|import|export|validate|reload|templates]`

## Skill 命令（@）

- `@<skill-name> <message>`：显式加载 skill 并执行任务
- 输入 `@` 后按 `Tab`：补全可用 skill
- `/skills`：按分类查看全部 skill

示例：

```text
@brainstorming 设计一个新功能
@test-driven-development 为这个模块补测试
```

## 内置工具

当前默认注册的工具包括：

- `bash`：执行 shell 命令
- `read`：读取文件
- `write`：写入文件
- `edit`：按字符串替换文件内容
- `glob_search`：按 glob 查找文件
- `grep`：正则搜索文件内容
- `todo_read` / `todo_write`：读写会话 todo
- `done`：显式完成当前任务
- `run_subagent`：同步运行单个子代理并等待结果
- `run_parallel_subagents`：并行运行多个子代理并等待结果

## 子代理

子代理提示词位于：`bu_agent_sdk/prompts/agents/*.md`

- 通过 frontmatter 定义 `mode`、`model`、`tools` 等配置
- `mode` 为 `subagent` 或 `all` 的代理可被主代理调用
- CLI 中可结合工具或命令查看/管理代理配置

## 项目结构（当前仓库）

```text
bu_agent_cli/
├── claude_code.py
├── pyproject.toml
├── .env.example
├── README.md
├── cli/
│   ├── app.py
│   ├── agents_handler.py
│   ├── at_commands.py
│   ├── interactive_input.py
│   └── slash_commands.py
├── config/
│   ├── model_config.py
│   └── model_presets.json
├── tools/
│   ├── bash.py
│   ├── files.py
│   ├── run_parallel_subagents.py
│   ├── run_subagent.py
│   ├── sandbox.py
│   ├── search.py
│   ├── task_cancel.py
│   ├── task_status.py
│   └── todos.py
├── bu_agent_sdk/
│   ├── agent/
│   ├── bootstrap/
│   ├── bus/
│   ├── channels/
│   ├── gateway/
│   ├── heartbeat/
│   ├── llm/
│   ├── prompts/
│   ├── runtime/
│   ├── server/
│   ├── skill/
│   ├── skills/
│   ├── tokens/
│   └── tools/
├── docs/
└── tests/
```

## 开发与验证

```bash
# 运行测试
uv run pytest

# 格式化
uv run black .

# 静态检查
uv run ruff check .
```

## 说明

- README 以当前仓库代码为准
- 如果你新增了工具、skills、gateway channel 或运行入口，请同步更新本文档
