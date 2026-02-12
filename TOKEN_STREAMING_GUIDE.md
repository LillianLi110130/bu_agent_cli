# Token 级别流式输出使用指南

## 概述

新增的 `query_stream_delta` 方法提供了 **Token 级别的实时流式输出**，类似于 ChatGPT 的打字机效果。前端可以在 LLM 生成过程中逐 token 实时渲染文本。

### 与原有流式的区别

| 特性 | `/agent/query-stream` | `/agent/query-stream-delta` |
|------|----------------------|---------------------------|
| 流式粒度 | 步骤级别 | Token 级别 |
| 文本事件 | `TextEvent` (完整文本) | `TextDeltaEvent` (增量文本) |
| 适用场景 | 需要观察工具调用过程 | 需要打字机效果、实时渲染 |

---

## 架构说明

### 数据流

```
┌──────────────┐     HTTP SSE      ┌──────────────┐
│   前端 UI     │ ────────────────> │   FastAPI    │
│              │                    │   Server     │
│  buffer +=   │ <── text_delta ───│              │
│    delta     │                    │              │
└──────────────┘                    └──────┬───────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────┐
│                   SessionManager                      │
│  - 管理会话                                          │
│  - 调用 query_stream_delta()                          │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                    Agent                              │
│  - 调用 llm.astream()                                │
│  - 将 chunk 转换为 TextDeltaEvent                    │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  ChatOpenAI.llm                       │
│  - OpenAI API stream=True                           │
│  - 返回 ChatInvokeCompletionChunk                   │
└─────────────────────────────────────────────────────┘
```

### 新增事件类型

```python
@dataclass
class TextDeltaEvent:
    """流式文本增量事件 - 实时输出LLM生成的每个token"""
    delta: str  # 增量文本内容（可能为空字符串）
```

---

## API 使用

### 端点

```
POST /agent/query-stream-delta
```

### 请求格式

```json
{
  "message": "用户消息",
  "session_id": "会话ID（可选，不传则创建新会话）"
}
```

### 响应格式 (Server-Sent Events)

```
data: {"type": "text_delta", "delta": "你"}
data: {"type": "text_delta", "delta": "好"}
data: {"type": "text_delta", "delta": "，"}
data: {"type": "text_delta", "delta": "请"}
data: {"type": "text_delta", "delta": "问"}
...
data: {"type": "final", "content": "你好，请问..."}
data: {"type": "usage", "session_id": "...", "usage": {...}}
: done
```

### 事件类型

| type | 说明 | 字段 |
|------|------|------|
| `text_delta` | 增量文本 | `delta`: 增量内容 |
| `tool_call` | 工具调用 | `tool`, `args`, `tool_call_id` |
| `tool_result` | 工具结果 | `tool`, `result` |
| `final` | 最终响应 | `content`: 完整内容 |
| `usage` | 使用统计 | `total_tokens`, `total_cost` |
| `error` | 错误 | `error`: 错误信息 |

---

## Python 客户端示例

### 基础用法

```python
import asyncio
from bu_agent_sdk import AgentClient

async def main():
    async with AgentClient("http://localhost:8000") as client:
        await client.create_session()

        # Token 级别流式输出
        buffer = ""
        async for event in client.query_stream_delta("介绍一下 Python"):
            if event.type == "text_delta":
                buffer += event.delta
                print(event.delta, end="", flush=True)  # 实时打印
            elif event.type == "final":
                print(f"\n完成: {event.content}")

asyncio.run(main())
```

### 纯 HTTP 请求示例

```python
import asyncio
import json
import httpx

async def test_delta_stream():
    async with httpx.AsyncClient(timeout=60) as client:
        # 创建会话
        response = await client.post(
            "http://localhost:8000/sessions",
            json={}
        )
        session_id = response.json()["session_id"]

        # Token 流式查询
        async with client.stream(
            "POST",
            "http://localhost:8000/agent/query-stream-delta",
            json={"message": "你好", "session_id": session_id},
        ) as response:
            buffer = ""
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if data["type"] == "text_delta":
                        buffer += data["delta"]
                        print(data["delta"], end="", flush=True)
                    elif data["type"] == "final":
                        break

asyncio.run(test_delta_stream())
```

---

## 前端 JavaScript 示例

### Fetch API

```javascript
async function streamQuery(message, sessionId) {
    const response = await fetch('/agent/query-stream-delta', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message, session_id: sessionId})
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const {done, value} = await reader.read();
        if (done) break;

        const line = decoder.decode(value);
        const lines = line.split('\n');

        for (const l of lines) {
            if (l.startsWith('data: ')) {
                const data = JSON.parse(l.slice(6));

                if (data.type === 'text_delta') {
                    buffer += data.delta;
                    updateUI(buffer);  // 更新 UI
                } else if (data.type === 'final') {
                    console.log('完成:', data.content);
                }
            }
        }
    }
}
```

### React Hook 示例

```typescript
function useAgentStream() {
    const [response, setResponse] = useState("");
    const [isStreaming, setIsStreaming] = useState(false);

    const streamQuery = async (message: string, sessionId?: string) => {
        setIsStreaming(true);
        setResponse("");

        const res = await fetch("/agent/query-stream-delta", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({message, session_id: sessionId}),
        });

        const reader = res.body?.getReader();
        const decoder = new TextDecoder();

        try {
            while (true) {
                const {done, value} = await reader!.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split("\n");

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const event = JSON.parse(line.slice(6));

                        if (event.type === "text_delta") {
                            setResponse((prev) => prev + event.delta);
                        } else if (event.type === "final") {
                            setIsStreaming(false);
                        }
                    }
                }
            }
        } finally {
            setIsStreaming(false);
        }
    };

    return {response, isStreaming, streamQuery};
}

// 使用
function ChatComponent() {
    const {response, isStreaming, streamQuery} = useAgentStream();

    return (
        <div>
            <input
                type="text"
                onKeyDown={(e) => {
                    if (e.key === "Enter") {
                        streamQuery(e.currentTarget.value);
                    }
                }}
            />
            <div>{response}</div>
            {isStreaming && <span>...</span>}
        </div>
    );
}
```

---

## Vue 3 示例

```vue
<script setup lang="ts">
import { ref } from 'vue'

const message = ref('')
const response = ref('')
const isLoading = ref(false)

async function sendQuery() {
  isLoading.value = true
  response.value = ''

  const res = await fetch('/agent/query-stream-delta', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: message.value})
  })

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()

  while (true) {
    const {done, value} = await reader.read()
    if (done) break

    const chunk = decoder.decode(value)
    const lines = chunk.split('\n')

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const event = JSON.parse(line.slice(6))
        if (event.type === 'text_delta') {
          response.value += event.delta
        } else if (event.type === 'final') {
          isLoading.value = false
        }
      }
    }
  }
}
</script>

<template>
  <div>
    <input v-model="message" @keyup.enter="sendQuery" :disabled="isLoading" />
    <div>{{ response }}</div>
    <div v-if="isLoading">正在生成...</div>
  </div>
</template>
```

---

## 代码改动说明

### 1. LLM 层 (llm/)

#### llm/views.py - 新增 Chunk 类

```python
class ChatInvokeCompletionChunk(BaseModel):
    """流式响应的单个数据块"""
    delta: str = ""              # 增量文本
    tool_calls: list[ToolCall]   # 工具调用
    usage: ChatInvokeUsage       # Token统计
    stop_reason: str             # 停止原因
```

#### llm/base.py - 添加 astream 协议

```python
async def astream(
    self,
    messages: list[BaseMessage],
    tools: list[ToolDefinition] | None = None,
    ...
) -> AsyncIterator[ChatInvokeCompletionChunk]:
    """流式调用模型，逐token返回内容"""
    ...
```

#### llm/openai/chat.py - 实现流式调用

```python
async def astream(...):
    # 使用 OpenAI 的 stream=True
    stream = await self.get_client().chat.completions.create(
        model=self.model,
        messages=openai_messages,
        stream=True,
        ...
    )

    async for chunk in stream:
        yield ChatInvokeCompletionChunk(
            delta=chunk.choices[0].delta.content or "",
            ...
        )
```

### 2. Agent 层 (agent/)

#### agent/events.py - 新增事件

```python
@dataclass
class TextDeltaEvent:
    """流式文本增量事件"""
    delta: str  # 增量文本内容
```

#### agent/service.py - 新增方法

```python
async def query_stream_delta(...) -> AsyncIterator[AgentEvent]:
    """Token 级别的流式查询"""
    async for chunk in self.llm.astream(...):
        if chunk.delta:
            yield TextDeltaEvent(delta=chunk.delta)
        # 处理工具调用等...
```

### 3. Server 层 (server/)

#### server/models.py - API 模型

```python
class TextDeltaEvent(StreamEventType):
    type: Literal["text_delta"] = "text_delta"
    delta: str
```

#### server/app.py - 新端点

```python
@app.post("/agent/query-stream-delta", tags=["Agent"])
async def query_stream_delta(request: QueryRequest):
    """Token 级别流式查询"""
    async for event in session.query_stream_delta(...):
        yield _serialize_event(_agent_event_to_stream_event(event))
```

---

## 测试

### 运行测试脚本

```bash
# 启动服务器
conda run -n 314 python test_server.py

# 运行 Token 流式测试
conda run -n 314 python test_client_delta.py
```

### 预期输出

```
=== 创建会话 ===
Session ID: abc-123-def

=== Token 级别流式查询 ===
问题: 广州今天天气怎么样

回答: 根据我的查询，广[工]州[具]今天[调]的天气[用]...

[完成]
最终回答: 根据我的查询，广州今天的天气...
```

---

## 性能考虑

1. **网络延迟**：Token 流式对网络延迟更敏感，建议使用低延迟连接
2. **缓冲区设置**：确保服务器禁用缓冲（`X-Accel-Buffering: no`）
3. **超时设置**：流式请求建议设置较长超时（60s+）
4. **并发控制**：同一 session 的并发请求会被串行化处理

---

## 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 无增量输出 | LLM 不支持流式 | 检查模型是否支持 stream API |
| 卡在第一个 token | 缓冲问题 | 检查代理/网关的缓冲设置 |
| 事件类型错误 | 版本不匹配 | 更新客户端 SDK |
| 连接断开 | 超时 | 增加客户端超时时间 |
