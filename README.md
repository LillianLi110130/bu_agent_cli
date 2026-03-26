# BU Agent CLI

一个面向编码场景的 Agent CLI / Gateway 项目，基于 `agent_core` 实现，支持 OpenAI-compatible 模型、工具调用、子代理委派、工作区指令注入，以及 Telegram 私聊网关接入。

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

- 入口层：`claude_code.py`
  - 负责组装 LLM、Agent、工具、Sandbox、系统提示词
- UI 层：`cli/`
  - `cli/app.py`：交互循环、事件渲染、slash 命令
  - `cli/slash_commands.py`：命令注册与补全
- Agent 核心：`agent_core/agent/`
  - 主循环、工具调度、重试、完成判定、流式事件
- LLM 适配层：`agent_core/llm/`
  - 当前主要实现为 `ChatOpenAI`（兼容 OpenAI API schema）
- 工具层：`tools/`
  - Bash、文件、搜索、todo、subagent
- 扩展层：
  - Skills：`agent_core/skills/`
  - 子代理配置：`agent_core/prompts/agents/*.md`
  - 规划插件：`plugins/`

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
- `/clear`
- `/model [show|list|<preset>]`
- `/ralph <init-spec|init-agent|dry-run|run|status|cancel> ...`
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

## Ralph 与规划插件

Ralph 现在只负责两件事：

- 初始化标准 spec 目录和最小 `.devagent`
- 执行 `plan/plan.json` 与 `plan/*.md` 中定义的任务

任务拆解不再由 Ralph Core 负责，而是交给插件完成。当前内置两类规划插件：

- `ta-workflow`
  - 两阶段：`/ta-workflow:ta`、`/ta-workflow:decompose`
- `frontend-workflow`
  - 三阶段：`/frontend-workflow:requirement`、`/frontend-workflow:design`、`/frontend-workflow:tasks`

推荐流程：

```text
/ralph init-spec my_spec
/ralph init-agent
# 将原始需求放入 docs/spec/my_spec/input/
/frontend-workflow:requirement my_spec
/frontend-workflow:design my_spec
/frontend-workflow:tasks my_spec
/ralph dry-run my_spec
/ralph run my_spec --silent
/ralph status
```

标准 spec 目录：

```text
docs/spec/<spec_name>/
├── input/
├── artifacts/
├── plan/
│   └── plan.json
├── implement/
└── logs/
```

说明：

- `input/`：原始需求输入
- `artifacts/`：插件拆解过程的中间产物
- `plan/`：Ralph 真正消费的最终计划
- `implement/`：执行阶段产物
- `logs/`：拆解和执行日志

最小 `.devagent` 结构：

```text
.devagent/
├── agents/
└── commands/
    └── ralph/
        └── implement.md
```

插件会在各自阶段命令执行前，自动把 `dev_subagents/*.md` 同步到工作区 `.devagent/agents/`。

详细说明见 [RALPH_LOOP_GUIDE.md](./RALPH_LOOP_GUIDE.md)。

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
- `read_excel`：读取 Excel 工作簿并预览 sheet 内容
- `write`：写入文件
- `edit`：按字符串替换编辑文件
- `glob_search`：按 glob 查找文件
- `grep`：正则搜索文件内容
- `todo_read` / `todo_write`：读写会话 todo
- `done`：显式完成任务
- `task`：调用子代理处理子任务

## 子代理

子代理定义在：`agent_core/prompts/agents/*.md`

- 通过 frontmatter 指定 `mode/model/tools` 等配置
- `mode` 为 `subagent` 或 `all` 的代理可被 `task` 工具调用

## 项目结构（实际）

```text
bu_agent_cli/
├── claude_code.py
├── cli/
│   ├── app.py
│   └── slash_commands.py
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
└── agent_core/
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
