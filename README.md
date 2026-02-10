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
- `/reset`
- `/history`（占位，暂未实现完整历史展示）

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
