# bu_agent_sdk.agent 代码结构文档

## 概述

`bu_agent_sdk.agent` 模块实现了一个简单的 Agentic 循环，支持原生工具调用（tool calling）和自动上下文压缩。该模块采用流式事件驱动的设计模式，提供对 Agent 执行过程的细粒度可见性。

## 目录结构

```
bu_agent_sdk/agent/
├── __init__.py           # 模块入口，导出公共 API
├── service.py            # 核心服务：Agent 类与 TaskComplete 异常
├── events.py             # 事件类型定义（流式输出）
└── compaction/           # 上下文压缩子模块
    ├── __init__.py       # 压缩模块入口
    ├── service.py        # 压缩服务实现
    └── models.py         # 压缩相关数据模型
```

## 模块依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                        agent/__init__.py                    │
│  - 导出 Agent, TaskComplete                                  │
│  - 导出所有事件类型                                          │
│  - 导出压缩相关类                                            │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│   service.py    │  │    events.py    │  │   compaction/       │
│                 │  │                 │  │                     │
│ - Agent 类      │  │ - TextEvent     │  │ - service.py        │
│ - TaskComplete  │  │ - ThinkingEvent │  │ - models.py         │
│                 │  │ - ToolCallEvent │  │                     │
│                 │  │ - ToolResultEvent│ │ - CompactionService │
│                 │  │ - FinalResponse  │ │ - CompactionConfig  │
│                 │  │ - Step*Event     │ │ - CompactionResult  │
└─────────────────┘  └─────────────────┘  └─────────────────────┘
```

---

## 1. service.py - 核心服务

### 类：Agent

Agentic 循环的核心实现类，负责管理工具调用和消息历史。

#### 属性

| 属性 | 类型 | 说明 |
|-----|------|-----|
| `llm` | `BaseChatModel` | 使用的语言模型 |
| `tools` | `list[Tool]` | 可用工具列表（通过 `@tool` 装饰器创建） |
| `system_prompt` | `str \| None` | 可选的系统提示词 |
| `max_iterations` | `int` | 最大迭代次数（默认 200） |
| `tool_choice` | `ToolChoice` | 工具选择策略（'auto', 'required', 'none'） |
| `compaction` | `CompactionConfig \| None` | 压缩配置 |
| `include_cost` | `bool` | 是否计算成本 |
| `dependency_overrides` | `dict \| None` | 工具依赖注入覆盖 |
| `ephemeral_storage_path` | `Path \| None` | 临时内容存储路径 |
| `require_done_tool` | `bool` | 是否需要显式调用 done 工具才结束 |
| `llm_max_retries` | `int` | LLM 调用最大重试次数 |

#### 核心方法

| 方法 | 说明 |
|-----|------|
| `query(message: str) -> str` | 发送消息并获取最终响应 |
| `query_stream(message) -> AsyncIterator[AgentEvent]` | 流式获取执行事件 |
| `clear_history()` | 清除消息历史 |
| `load_history(messages)` | 加载历史消息继续对话 |
| `get_usage() -> UsageSummary` | 获取使用统计 |

#### 执行流程

```
┌──────────────────────────────────────────────────────────────┐
│                      Agent.query() 流程                      │
├──────────────────────────────────────────────────────────────┤
│  1. 添加系统提示词（首次调用）                                │
│  2. 添加用户消息                                              │
│  3. 循环执行（最多 max_iterations 次）：                      │
│     a) 清理临时消息（_destroy_ephemeral_messages）            │
│     b) 调用 LLM（_invoke_llm，带重试机制）                    │
│     c) 检查是否需要压缩上下文（_check_and_compact）           │
│     d) 如果没有工具调用：                                     │
│        - 检查未完成的 todos（可选）                           │
│        - 返回最终响应                                        │
│     e) 如果有工具调用：                                       │
│        - 执行每个工具（_execute_tool_call）                   │
│        - 添加工具结果到历史                                  │
│        - 继续循环                                            │
└──────────────────────────────────────────────────────────────┘
```

#### 重试机制

Agent 实现了指数退避重试策略，与 browser-use 保持一致：

- **可重试的状态码**：429, 500, 502, 503, 504
- **基础延迟**：1.0 秒
- **最大延迟**：60.0 秒
- **抖动**：10%（随机）

### 异常：TaskComplete

用于显式标记任务完成的异常类。

```python
class TaskComplete(Exception):
    """工具可通过抛出此异常来结束 Agent 执行"""
    message: str  # 完成描述
```

---

## 2. events.py - 事件类型

所有事件都使用 `@dataclass` 装饰器，并实现 `__str__` 方法用于友好的日志输出。

### 事件类型

| 事件 | 说明 |
|-----|------|
| `TextEvent` | 助手产生文本内容时发出 |
| `ThinkingEvent` | 模型产生推理内容时发出 |
| `ToolCallEvent` | 工具被调用时发出 |
| `ToolResultEvent` | 工具返回结果时发出 |
| `FinalResponseEvent` | Agent 产生最终响应时发出 |
| `MessageStartEvent` | 新消息开始时发出 |
| `MessageCompleteEvent` | 消息完成时发出 |
| `StepStartEvent` | 逻辑步骤开始时发出 |
| `StepCompleteEvent` | 逻辑步骤完成时发出 |
| `HiddenUserMessageEvent` | Agent 注入隐藏用户消息时发出 |

### AgentEvent 联合类型

```python
AgentEvent = (
    TextEvent | ThinkingEvent | ToolCallEvent | ToolResultEvent |
    FinalResponseEvent | MessageStartEvent | MessageCompleteEvent |
    StepStartEvent | StepCompleteEvent | HiddenUserMessageEvent
)
```

---

## 3. compaction/ - 上下文压缩子模块

### models.py - 数据模型

#### CompactionConfig

压缩服务配置类。

| 属性 | 类型 | 默认值 | 说明 |
|-----|------|-------|-----|
| `enabled` | `bool` | `True` | 是否启用压缩 |
| `threshold_ratio` | `float` | `0.80` | 触发压缩的上下文比例 |
| `model` | `str \| None` | `None` | 生成摘要的模型 |
| `summary_prompt` | `str` | 默认提示词 | 摘要生成提示词 |

#### CompactionResult

压缩操作结果。

| 属性 | 说明 |
|-----|------|
| `compacted` | 是否执行了压缩 |
| `original_tokens` | 压缩前的 token 数 |
| `new_tokens` | 压缩后的 token 数 |
| `summary` | 生成的摘要文本 |

#### TokenUsage

Token 使用追踪。

```python
@property
def total_tokens(self) -> int:
    """计算总 token 数（与 Anthropic SDK 一致）"""
    return input_tokens + cache_creation_tokens + cache_read_tokens + output_tokens
```

### service.py - 压缩服务

#### CompactionService

管理对话上下文的服务类。

**核心方法：**

| 方法 | 说明 |
|-----|------|
| `update_usage(usage)` | 从响应更新 token 使用情况 |
| `should_compact(model) -> bool` | 检查是否应触发压缩 |
| `compact(messages, llm)` | 执行压缩操作 |
| `check_and_compact(messages, llm)` | 检查并在需要时压缩 |
| `reset()` | 重置服务状态 |

**压缩流程：**

```
┌─────────────────────────────────────────────────────────┐
│                    压缩触发条件                          │
├─────────────────────────────────────────────────────────┤
│  total_tokens >= context_window * threshold_ratio       │
│                                                         │
│  例：context_window=200000, ratio=0.80                  │
│      触发阈值 = 160000 tokens                           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    压缩执行步骤                          │
├─────────────────────────────────────────────────────────┤
│  1. 准备消息（移除最后的 tool_calls 避免API错误）         │
│  2. 添加摘要提示词                                       │
│  3. 调用 LLM 生成结构化摘要                              │
│  4. 提取 <summary> 标签内容                             │
│  5. 将整个历史替换为摘要消息                             │
└─────────────────────────────────────────────────────────┘
```

---

## 4. 使用示例

### 基本用法

```python
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools import tool
from bu_agent_sdk import Agent

@tool("Search the web")
async def search(query: str) -> str:
    return f"Results for {query}"

agent = Agent(
    llm=ChatOpenAI(model="gpt-4o"),
    tools=[search],
)

response = await agent.query("Find information about Python")
```

### 流式事件

```python
async for event in agent.query_stream("Schedule a meeting"):
    match event:
        case ToolCallEvent(tool=name, args=args):
            print(f"Calling {name}")
        case ToolResultEvent(tool=name, result=result):
            print(f"{name} returned: {result[:50]}")
        case FinalResponseEvent(content=text):
            print(f"Done: {text}")
```

### 自定义压缩配置

```python
from bu_agent_sdk.agent.compaction import CompactionConfig

agent = Agent(
    llm=ChatOpenAI(model="gpt-4o"),
    tools=[search],
    compaction=CompactionConfig(threshold_ratio=0.70),  # 70% 触发
    # 或禁用压缩:
    # compaction=CompactionConfig(enabled=False),
)
```

### 获取使用统计

```python
summary = await agent.get_usage()
print(f"Total tokens: {summary.total_tokens}")
print(f"Total cost: ${summary.total_cost:.4f}")
```

---

## 5. 关键设计特性

### 临时消息销毁（Ephemeral Messages）

工具可以指定其输出为"临时的"（ephemeral），这样可以按工具维度只保留最近的 N 条结果，节省上下文。

```python
@tool("browser", ephemeral=3)  # 只保留最近 3 条结果
async def browser_action(url: str) -> str:
    ...
```

### LLM 调用重试

实现了健壮的重试机制：
- 指数退避（1s → 2s → 4s → ... → 60s）
- 10% 抖动避免雷击
- 可配置的重试状态码

### 可观测性

通过 Laminar 集成支持分布式追踪：
- `@observe(name="agent_query")` 装饰 query 方法
- `@observe(name="agent_query_stream")` 装饰 query_stream 方法
- 工具执行自动创建子 span

---

## 导出 API

### 从 bu_agent_sdk.agent 导出

```python
# 核心类
Agent, TaskComplete

# 事件类型
AgentEvent, FinalResponseEvent, TextEvent, ThinkingEvent,
ToolCallEvent, ToolResultEvent

# 压缩相关
CompactionConfig, CompactionResult, CompactionService
```
