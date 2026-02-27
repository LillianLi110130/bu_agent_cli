# AGENTS.md

## 概述

**BU Agent CLI** 是一个面向编码场景的智能体命令行工具，基于 `bu_agent_sdk` 框架构建。它提供交互式编程助手体验，支持 OpenAI 兼容模型、工具调用、子代理委派和智能上下文管理。

### 核心特性

- **交互式 CLI**：流式展示思考过程、工具调用和执行结果
- **工具调用**：内置 bash、文件读写、搜索、todo 管理、子代理等工具
- **子代理机制**：通过 `task` 工具委派给专业代理（如 code_reviewer、frontend_developer）
- **上下文压缩**：自动压缩对话历史，缓解长上下文溢出问题
- **模型切换**：支持会话内通过 `/model` 命令切换预设模型（保留上下文）
- **Token 统计**：记录 token 使用量，并支持成本计算

## 项目结构

```
bu_agent_cli/
├── claude_code.py              # CLI 入口点
├── cli/                        # 命令行界面层
│   ├── app.py                  # 交互循环、事件渲染
│   ├── slash_commands.py       # Slash 命令系统
│   └── at_commands.py          # @ 技能命令系统
├── tools/                      # 工具实现
│   ├── bash.py                 # Shell 命令执行
│   ├── files.py                # 文件读写编辑
│   ├── search.py               # 文件搜索（glob/grep）
│   ├── todos.py                # Todo 管理
│   ├── async_task.py           # 异步子代理任务
│   ├── task_status.py          # 任务状态查询
│   ├── task_cancel.py          # 任务取消
│   └── sandbox.py              # 沙箱上下文管理
├── config/                     # 配置文件
│   ├── model_config.py         # 模型配置加载
│   └── model_presets.json      # 模型预设定义
├── bu_agent_sdk/              # 核心 SDK
│   ├── agent/                 # Agent 核心逻辑
│   │   ├── service.py         # Agent 主循环
│   │   ├── context.py         # 上下文管理
│   │   ├── events.py          # 事件类型定义
│   │   ├── config.py          # Agent 配置
│   │   ├── registry.py        # 子代理注册表
│   │   ├── subagent_manager.py # 子代理任务管理
│   │   └── compaction/       # 上下文压缩服务
│   ├── llm/                  # LLM 抽象层
│   │   ├── base.py            # 基础接口定义
│   │   ├── messages.py        # 消息类型
│   │   ├── views.py           # 响应视图
│   │   ├── schema.py          # Schema 优化
│   │   ├── exceptions.py      # 异常定义
│   │   └── openai/            # OpenAI 实现
│   │       ├── chat.py        # ChatOpenAI 实现
│   │       ├── like.py        # OpenAI 兼容实现
│   │       └── serializer.py # 消息序列化
│   ├── tools/                # 工具框架
│   │   ├── decorator.py       # @tool 装饰器
│   │   └── depends.py        # 依赖注入系统
│   ├── skill/                # 技能系统
│   │   ├── loader.py          # 技能加载器
│   │   ├── parser.py          # 技能解析器
│   │   └── types.py           # 技能类型定义
│   ├── server/               # HTTP 服务器
│   │   ├── app.py             # FastAPI 应用
│   │   ├── session.py         # 会话管理
│   │   ├── models.py          # API 数据模型
│   │   └── client.py          # Python 客户端
│   ├── tokens/               # Token 统计
│   │   ├── service.py         # Token 成本服务
│   │   ├── views.py           # 使用统计视图
│   │   └── custom_pricing.py  # 自定义定价
│   ├── prompts/              # 提示词模板
│   │   ├── system.md          # 系统提示词
│   │   └── agents/            # 子代理定义
│   │       ├── code_reviewer.md
│   │       ├── data_scientist.md
│   │       └── frontend_developer.md
│   └── skills/               # 技能库
│       ├── using-git-worktrees/
│       ├── test-driven-development/
│       ├── systematic-debugging/
│       ├── calculator/
│       ├── brainstorming/
│       ├── writing-plans/
│       └── ...（更多技能）
└── tests/                     # 测试文件
```

## 工作原理

### 1. 启动流程

```
用户运行 bu-agent
    ↓
claude_code.py 初始化
    ↓
加载环境变量 (.env)
    ↓
加载模型预设 (config/model_presets.json)
    ↓
创建 LLM 实例 (ChatOpenAI)
    ↓
初始化工具集合 (tools/)
    ↓
加载技能库 (bu_agent_sdk/skills/)
    ↓
创建 Agent 实例
    ↓
启动 CLI 交互循环 (cli/app.py)
```

### 2. Agent 主循环

Agent 采用 **工具调用循环** 模式：

```
用户输入消息
    ↓
Agent 接收消息 + 上下文
    ↓
调用 LLM 生成响应
    ↓
┌─────────────────────────┐
│ LLM 是否请求调用工具？   │
└─────────────────────────┘
    ↓ 是              ↓ 否
执行工具          返回文本响应
    ↓
收集工具结果
    ↓
更新上下文
    ↓
重复 LLM 调用
```

### 3. 事件流系统

Agent 通过流式事件提供细粒度可见性：

```python
async for event in agent.query_stream(message):
    match event:
        case TextEvent(content=text):
            # 文本内容
        case ToolCallEvent(tool=name, args=args):
            # 工具调用
        case ToolResultEvent(tool=name, result=result):
            # 工具结果
        case ThinkingEvent(content=text):
            # 思考过程
        case FinalResponseEvent(content=text):
            # 最终响应
```

### 4. 子代理系统

子代理通过 `async_task` 工具实现：

```python
async_task(
    subagent_name="code_reviewer",
    prompt="Review the auth module",
    label="Review auth"
)
```

子代理定义在 `bu_agent_sdk/prompts/agents/*.md`，通过 frontmatter 配置：

```yaml
---
description: 代码审查专家
mode: subagent
model: GLM-4.7
temperature: 0.1
tools:
  read: true
  write: false
  bash: false
---
系统提示词内容...
```

### 5. 上下文压缩

当 token 使用量接近模型上下文窗口时，自动触发压缩：

1. **检测**：监控 token 使用量
2. **生成摘要**：调用 LLM 总结对话历史
3. **替换上下文**：用摘要替换旧消息
4. **继续对话**：在压缩后的上下文中继续

### 6. 技能系统

技能通过 `@` 命令调用：

```
@brainstorming 设计一个新功能
@test-driven-development 为这个模块补测试
```

技能定义在 `bu_agent_sdk/skills/*/SKILL.md`，包含：

- YAML frontmatter（元数据）
- 技能描述和工作流程
- 最佳实践和规则

### 7. HTTP 服务器模式

通过 `bu_agent_sdk.server` 提供 REST API：

```python
from bu_agent_sdk.server import create_server

app = create_server(agent_factory=create_agent)
```

API 端点：
- `POST /agent/query` - 非流式查询
- `POST /agent/query-stream` - 步骤级别流式
- `POST /agent/query-stream-delta` - Token 级别流式
- `POST /sessions` - 创建会话
- `GET /sessions/{id}` - 获取会话信息

## 约束与假设

### 环境要求

- **Python 版本**：>= 3.10
- **API Key**：需要 OpenAI 兼容的 API Key（通过 `OPENAI_API_KEY` 环境变量）
- **网络访问**：需要访问 LLM API 端点

### 当前限制

1. **LLM 提供商**：主要支持 OpenAI 兼容 API
   - 已测试：智谱 GLM、OpenAI GPT-4
   - 通过 `base_url` 配置可支持其他兼容提供商

2. **工具执行环境**：
   - Shell 命令在用户的工作目录中执行
   - 文件操作受沙箱限制（`root_dir` 边界）
   - 默认超时：30 秒

3. **上下文压缩**：
   - 默认阈值：上下文窗口的 80%
   - 压缩使用相同的 LLM 进行摘要
   - 压缩不可逆（原始上下文丢失）

4. **子代理管理**：
   - 子代理在后台异步运行
   - 任务状态独立于主 Agent
   - 需要主动调用 `task_status` 查询结果

### 假设条件

1. **模型能力假设**：
   - LLM 支持工具调用（function calling）
   - LLM 遵循系统提示词的指令
   - LLM 能够理解工具描述并正确调用

2. **工具使用假设**：
   - 用户允许 Agent 执行 shell 命令
   - 文件操作不会破坏关键系统文件
   - 工具执行结果是可信的

3. **对话管理假设**：
   - 单轮对话不会无限循环（`max_iterations=200`）
   - 用户会及时反馈错误或不合理的结果
   - 上下文压缩不会丢失关键信息

### 安全考虑

1. **沙箱隔离**：
   - 文件操作限制在 `root_dir` 及其子目录
   - 支持额外的 `allowed_dirs` 白名单
   - 路径解析防止目录遍历攻击

2. **命令执行**：
   - bash 工具执行用户提供的命令字符串
   - 无额外的命令白名单/黑名单
   - 用户需自行评估命令风险

3. **API 密钥安全**：
   - API Key 通过环境变量传递
   - 不在日志中输出敏感信息
   - 建议：使用 `.env` 文件管理密钥

### 扩展建议

1. **添加新工具**：
   - 在 `tools/` 目录创建新工具文件
   - 使用 `@tool` 装饰器定义工具函数
   - 通过依赖注入获取 `SandboxContext`

2. **添加新子代理**：
   - 在 `bu_agent_sdk/prompts/agents/` 创建 `.md` 文件
   - 配置 frontmatter（mode, model, tools 等）
   - 编写专业的系统提示词

3. **添加新技能**：
   - 在 `bu_agent_sdk/skills/` 创建技能目录
   - 编写 `SKILL.md` 文件（含 frontmatter）
   - 遵循技能最佳实践（简洁、可测试）

4. **自定义 LLM 适配**：
   - 实现 `BaseChatModel` 协议
   - 处理工具调用序列化
   - 支持流式输出