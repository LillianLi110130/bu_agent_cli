# Crab CLI

Crab CLI 是一个面向编码和自动化任务的 Agent CLI。它基于 `agent_core` 构建，支持 OpenAI-compatible 模型、工具调用、技能系统、插件、子代理委派、长期记忆、上下文压缩、文件桥接和远程 worker。

## 核心能力

- **交互式 CLI**：在终端中直接对话，支持 Rich Markdown 输出、工具调用展示、slash 命令和 `@skill`。
- **模型预设与切换**：通过 `config/model_presets.json` 配置模型，运行中可用 `/model` 切换。
- **工具调用**：内置 shell、文件读写编辑、搜索、Excel 读取、todo、Web fetch、后台任务、team 管理等工具。
- **技能系统**：通过 `@ + Tab` 查看技能，通过 `@<skill-name> <message>` 显式调用技能。
- **插件系统**：支持内置插件和本地插件，插件可提供 commands、skills、agents 等扩展能力。
- **子代理与 team**：支持 `delegate`、`delegate_parallel` 和 `/team` 多进程 agent team。
- **长期记忆**：集成 memory review，可维护本地长期记忆。
- **上下文管理**：支持上下文预算显示和自动压缩。
- **本地/远程桥接**：通过文件队列连接 CLI、IM worker 和 Web/远程入口。

## 安装

推荐使用 `uv`：

```bash
uv sync
```

也可以使用 editable 安装：

```bash
pip install -e .
```

## 启动

安装后命令入口是 `crab`：

```bash
crab
```

也可以直接运行入口脚本：

```bash
uv run python tg_crab_main.py
```

常用启动参数：

```bash
# 指定模型
crab --model GLM-5.1

# 指定工作区根目录
crab --root-dir /path/to/workspace

# 关闭本地 bridge
crab --no-local-bridge

# 关闭 IM worker bridge
crab --no-im-enable

# 指定 IM gateway
crab --im-gateway-base-url http://127.0.0.1:8765
```

## 环境变量

Crab CLI 使用 OpenAI-compatible API 形式调用模型。

常用环境变量：

```bash
export OPENAI_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-chat"
```

如果使用项目默认 GLM/Gateway 配置，也可以只配置对应预设需要的 API key。

默认模型创建逻辑主要看：

- `LLM_MODEL`
- `LLM_BASE_URL`
- `OPENAI_API_KEY`
- `config/model_presets.json`

## 模型预设

模型预设文件：

```text
config/model_presets.json
```

当前预设结构大致如下：

```json
{
  "default": "GLM-5.1",
  "auto_vision_preset": "vision-default",
  "image_summary_preset": "vision-default",
  "presets": {
    "coding-default": {
      "provider": "gateway",
      "model": "coding-default",
      "base_url": "http://127.0.0.1:8000",
      "vision": false
    },
    "vision-default": {
      "provider": "gateway",
      "model": "vision-default",
      "base_url": "http://127.0.0.1:8000",
      "vision": true
    },
    "GLM-5.1": {
      "model": "GLM-5.1",
      "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
      "api_key_env": "OPENAI_API_KEY",
      "vision": false
    }
  }
}
```

运行中可使用：

```text
/model           # 打开编号选择器
/model list      # 查看模型预设
/model show      # 查看当前模型
/model <preset>  # 切换到指定预设
```

## 常用命令

基础命令：

```text
/help
/exit
/quit
/pwd
/clear
/history
```

模型与设置：

```text
/model [show|list|<preset>]
/approval [on|off|status]
/settings
```

会话：

```text
/reset
/new
/resume
/init
```

技能、记忆、插件、agent：

```text
/skills [list|reload|show <name>|review]
/memory [list|review]
/plugins [list|show|copy|reload|install|uninstall]
/agents [list|show|create|edit|delete|reload]
```

Subagent 的用户说明见 `docs/subagent-user-guide.md`。

文件系统和后台任务：

```text
/allow <path>
/allowed
/tasks
/task <task_id>
/task_cancel <task_id>
```

Team 和 Ralph：

```text
/team [auto|create|list|spawn|task|tasks|members|inbox|dashboard|send|status|stop|shutdown]
/ralph <init-spec|init-agent|dry-run|run|status|cancel> ...
```

## Skill 命令

Skill 可通过 `@` 调用：

```text
@brainstorming 设计一个新功能
@test-driven-development 为这个模块补测试
```

常见用法：

```text
@ + Tab                         # 查看和补全技能
@<skill-name> <message>          # 显式加载技能并执行任务
@"<path>"<message>               # 发送图片输入
@'<path>'<message>               # 发送图片输入
```

技能来源：

- 包内 `skills/`
- 用户级 `~/.tg_agent/skills/`
- 工作区 `<workspace_root>/skills/`

## 内置工具

主要工具包括：

```text
bash                  # 执行 shell 命令
resolve_path          # 解析路径
read                  # 读取文件
write                 # 写入文件
edit                  # 编辑文件
glob_search           # glob 文件查找
grep                  # 正则搜索
message               # 桥接消息工具
todo_read/todo_write  # 读写 todo
done                  # 显式完成任务
web_fetch             # 抓取网页
delegate              # 委派子代理
delegate_parallel     # 并行委派子代理
task_output           # 查看后台任务输出
task_status           # 查看后台任务状态
task_cancel           # 取消后台任务
team_*                # team 管理工具
```

完整工具列表见：

```text
tools/__init__.py
```

## Bridge 与 Worker

Crab CLI 默认启用本地 bridge 和 IM worker bridge：

```text
--local-bridge / --no-local-bridge
--im-enable / --no-im-enable
```

Bridge 会把本地终端输入、远程 IM/Web 请求和执行结果通过文件队列串起来。运行时相关信息可在 CLI 中输入 `/help` 查看，包括：

```text
桥接会话
工作日志
IM 工作进程
```

相关目录：

```text
cli/im_bridge/       # 文件桥接队列
cli/worker/          # worker 进程、鉴权和 gateway client
agent_core/server/   # 服务端和 web console 支持
web/                 # Web 控制台前端
gateway/             # Gateway 服务工程
```

## Ralph 与规划插件

Ralph 负责初始化标准 spec 目录，并执行 `plan/plan.json` 与 `plan/*.md` 中定义的任务。

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

## 目录结构

项目主要由以下几层组成：

```text
tg_crab_main.py   # CLI 启动入口，负责组装 Agent、工具、技能、插件、bridge
cli/              # 终端交互层
agent_core/       # Agent 核心运行时
tools/            # Agent 可调用工具
skills/           # 内置技能
plugins/          # 插件包
config/           # 模型预设和配置
tests/            # 测试
web/              # Web 控制台前端
gateway/          # Gateway 服务工程
tg_mem/           # 长期记忆底层组件
```

详细目录说明见：

[docs/package-structure.md](./docs/package-structure.md)

## 开发

运行测试：

```bash
uv run pytest
```

运行部分测试：

```bash
uv run pytest tests/test_cli_defaults.py -q
```

代码格式配置见：

```text
pyproject.toml
```

## 常见修改位置

```text
欢迎面板、Markdown 输出、工具调用展示
  -> cli/app.py

slash 命令列表和补全
  -> cli/slash_commands.py

模型切换
  -> cli/model_switch_service.py

Agent 主循环
  -> agent_core/agent/service.py
  -> agent_core/agent/runtime_loop.py

工具列表
  -> tools/__init__.py

单个工具实现
  -> tools/*.py

启动配置和依赖组装
  -> tg_crab_main.py

Bridge 队列
  -> cli/im_bridge/

远程 worker
  -> cli/worker/
```
