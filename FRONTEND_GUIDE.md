# 前端与 Agent 交互指南

本文档描述前端如何通过 HTTP API 与 bu_agent_sdk Agent 进行交互，重点说明事件类型和数据格式。

---

## 概述

Agent 通过 **Server-Sent Events (SSE)** 实时推送事件流，前端需要：
1. 建立 SSE 连接
2. 解析事件流
3. 根据事件类型更新 UI

---

## API 端点

### Token 级别流式（推荐 - 打字机效果）

```
POST /agent/query-stream-delta
```

### 步骤级别流式（调试用）

```
POST /agent/query-stream
```

---

## 请求格式

### HTTP Headers

```
Content-Type: application/json
```

### Request Body

```json
{
  "message": "用户消息内容",
  "session_id": "会话ID，可选，不传则创建新会话"
}
```

---

## 事件类型详解

### 完整事件列表

| type | 名称 | 说明 | 适用场景 |
|------|------|------|----------|
| `text_delta` | 增量文本 | 逐 token 实时输出 | **推荐用于文本渲染** |
| `text` | 完整文本 | 一次性返回完整内容 | 非流式模式 |
| `tool_call` | 工具调用 | Agent 调用了工具 | 显示工具执行状态 |
| `tool_result` | 工具结果 | 工具执行完成 | 显示工具执行结果 |
| `step_start` | 步骤开始 | 逻辑步骤开始 | 显示进度 |
| `step_complete` | 步骤完成 | 逻辑步骤完成 | 更新进度状态 |
| `final` | 最终响应 | Agent 完成 | 结束对话 |
| `usage` | 使用统计 | Token 消耗 | 显示成本统计 |
| `error` | 错误 | 发生错误 | 错误提示 |

---

## 事件数据格式

### 1. text_delta - 增量文本事件

**用途**：逐 token 实时输出，实现打字机效果

```json
{
  "type": "text_delta",
  "delta": "你好",
  "timestamp": "2025-02-06T10:30:00.123Z"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `"text_delta"` |
| `delta` | string | **增量文本**，可能为空字符串 `""` |
| `timestamp` | string | 事件时间戳 |

**前端处理示例**：

```javascript
let buffer = "";  // 累积缓冲区

if (event.type === "text_delta") {
    buffer += event.delta;  // 追加增量
    updateUI(buffer);      // 实时更新 UI
}
```

**注意**：
- `delta` 可能是单个字符、半个汉字、或几个 token
- `delta` 可能为空字符串 `""`，需要处理
- 前端需要维护缓冲区累积完整文本

---

### 2. text - 完整文本事件

**用途**：一次性返回完整文本（步骤级别流式）

```json
{
  "type": "text",
  "content": "这是完整的回复内容",
  "timestamp": "2025-02-06T10:30:00.123Z"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `"text"` |
| `content` | string | 完整的文本内容 |
| `timestamp` | string | 事件时间戳 |

---

### 3. tool_call - 工具调用事件

**用途**：Agent 决定调用某个工具

```json
{
  "type": "tool_call",
  "tool": "get_weather",
  "args": {
    "location": "北京"
  },
  "tool_call_id": "call_abc123",
  "display_name": "",
  "timestamp": "2025-02-06T10:30:00.123Z"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `"tool_call"` |
| `tool` | string | 工具名称 |
| `args` | object | 工具参数（JSON 对象） |
| `tool_call_id` | string | 工具调用 ID，与 tool_result 对应 |
| `display_name` | string | 人类可读的工具描述 |
| `timestamp` | string | 事件时间戳 |

**前端处理建议**：
- 显示 "正在调用 XX 工具..."
- 显示工具参数（用户可审阅）
- 等待 tool_result 事件

---

### 4. tool_result - 工具结果事件

**用途**：工具执行完成，返回结果

```json
{
  "type": "tool_result",
  "tool": "get_weather",
  "result": "北京今天晴天，温度 25°C",
  "tool_call_id": "call_abc123",
  "is_error": false,
  "screenshot_base64": null,
  "timestamp": "2025-02-06T10:30:01.123Z"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `"tool_result"` |
| `tool` | string | 工具名称 |
| `result` | string | 工具执行结果 |
| `tool_call_id` | string | 对应的 tool_call ID |
| `is_error` | boolean | 是否执行错误 |
| `screenshot_base64` | string\| null | base64 编码的截图（浏览器工具） |
| `timestamp` | string | 事件时间戳 |

**前端处理建议**：
- `is_error=false`：显示工具执行成功
- `is_error=true`：显示错误信息
- 如果有 `screenshot_base64`，渲染图片

---

### 5. step_start - 步骤开始事件

**用途**：一个逻辑步骤开始（用于显示进度）

```json
{
  "type": "step_start",
  "step_id": "step_123",
  "title": "搜索天气信息",
  "step_number": 1,
  "timestamp": "2025-02-06T10:30:00.123Z"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `"step_start"` |
| `step_id` | string | 步骤唯一 ID |
| `title` | string | 步骤标题（人类可读） |
| `step_number` | number | 步骤序号（从 1 开始） |
| `timestamp` | string | 事件时间戳 |

---

### 6. step_complete - 步骤完成事件

**用途**：逻辑步骤完成

```json
{
  "type": "step_complete",
  "step_id": "step_123",
  "status": "completed",
  "duration_ms": 1234.56,
  "timestamp": "2025-02-06T10:30:01.123Z"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `"step_complete"` |
| `step_id` | string | 步骤 ID |
| `status` | string | 状态：`"completed"` 或 `"error"` |
| `duration_ms` | number | 耗时（毫秒） |
| `timestamp` | string | 事件时间戳 |

---

### 7. final - 最终响应事件

**用途**：Agent 完成，返回最终答案

```json
{
  "type": "final",
  "content": "根据查询结果，今天天气很好...",
  "timestamp": "2025-02-06T10:30:05.123Z"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `"final"` |
| `content` | string | 最终的完整回复 |
| `timestamp` | string | 事件时间戳 |

**注意**：这是流式响应的最后一个有效事件（除 `usage` 和 `: done` 外）

---

### 8. usage - 使用统计事件

**用途**：Token 消耗和成本统计

```json
{
  "type": "usage",
  "session_id": "sess_abc123",
  "usage": {
    "total_tokens": 1234,
    "total_prompt_tokens": 1000,
    "total_completion_tokens": 234,
    "total_cost": 0.0025,
    "by_model": {
      "gpt-4o": {
        "model": "gpt-4o",
        "prompt_tokens": 1000,
        "completion_tokens": 234,
        "total_tokens": 1234,
        "cost": 0.0025,
        "invocations": 5
      }
    }
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | 固定为 `"usage"` |
| `session_id` | string | 会话 ID |
| `usage.total_tokens` | number | 总 Token 数 |
| `usage.total_cost` | number | 总成本（美元） |
| `usage.by_model` | object | 按模型分组的统计 |

---

### 9. error - 错误事件

**用途**：发生错误

```json
{
  "type": "error",
  "error": "API connection timeout",
  "timestamp": "2025-02-06T10:30:00.123Z"
}
```

---

## SSE 流式响应格式

### 完整示例

```
: done
data: {"type": "text_delta", "delta": "你", "timestamp": "2025-02-06T10:30:00.123Z"}
data: {"type": "text_delta", "delta": "好", "timestamp": "2025-02-06T10:30:00.456Z"}
data: {"type": "text_delta", "delta": "，", "timestamp": "2025-02-06T10:30:00.789Z"}
data: {"type": "text_delta", "delta": "请", "timestamp": "2025-02-06T10:30:01.012Z"}
data: {"type": "text_delta", "delta": "问", "timestamp": "2025-02-06T10:30:01.345Z"}
data: {"type": "tool_call", "tool": "search", "args": {"query": "天气"}, ...}
data: {"type": "tool_result", "tool": "search", "result": "...", ...}
data: {"type": "text_delta", "delta": "根据", ...}
data: {"type": "final", "content": "你好，请问...", ...}
data: {"type": "usage", "session_id": "...", ...}
: done
```

### SSE 格式说明

- 每行以 `data: ` 开头，后面跟 JSON 数据
- 事件之间用 `\n\n` 分隔
- 结束信号为 `: done\n\n`

---

## 前端实现示例

### Vanilla JavaScript (Fetch API)

```javascript
async function queryAgent(message, sessionId, callbacks) {
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

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;

            const data = line.slice(6);  // 去掉 "data: " 前缀
            if (data.trim() === ': done') break;

            try {
                const event = JSON.parse(data);

                switch (event.type) {
                    case 'text_delta':
                        buffer += event.delta;
                        callbacks.onTextDelta?.(event.delta, buffer);
                        break;

                    case 'tool_call':
                        callbacks.onToolCall?.(event);
                        break;

                    case 'tool_result':
                        callbacks.onToolResult?.(event);
                        break;

                    case 'step_start':
                        callbacks.onStepStart?.(event);
                        break;

                    case 'step_complete':
                        callbacks.onStepComplete?.(event);
                        break;

                    case 'final':
                        callbacks.onFinal?.(event);
                        buffer = event.content;  // 更新为完整内容
                        break;

                    case 'usage':
                        callbacks.onUsage?.(event);
                        break;

                    case 'error':
                        callbacks.onError?.(event);
                        break;
                }
            } catch (e) {
                console.error('Failed to parse event:', line, e);
            }
        }
    }

    return buffer;
}

// 使用示例
queryAgent("广州天气怎么样", null, {
    onTextDelta: (delta, full) => {
        document.getElementById('output').textContent = full;
    },
    onToolCall: (event) => {
        console.log('调用工具:', event.tool);
    },
    onToolResult: (event) => {
        console.log('工具结果:', event.result);
    },
    onFinal: (event) => {
        console.log('完成:', event.content);
    },
    onUsage: (event) => {
        console.log('消耗:', event.usage.total_tokens);
    }
});
```

### React Hook

```typescript
import { useState, useCallback, useRef } from 'react';

interface AgentEvent {
    type: string;
    delta?: string;
    content?: string;
    tool?: string;
    args?: Record<string, any>;
    result?: string;
    tool_call_id?: string;
    is_error?: boolean;
    status?: string;
    step_id?: string;
    title?: string;
    step_number?: number;
    duration_ms?: number;
    usage?: {
        total_tokens: number;
        total_cost: number;
    };
    error?: string;
}

export function useAgentStream() {
    const [response, setResponse] = useState("");
    const [isStreaming, setIsStreaming] = useState(false);
    const [toolCalls, setToolCalls] = useState<AgentEvent[]>([]);
    const abortController = useRef<AbortController | null>(null);

    const streamQuery = useCallback(async (message: string, sessionId?: string) => {
        abortController.current?.abort();
        abortController.current = new AbortController();

        setIsStreaming(true);
        setResponse("");
        setToolCalls([]);

        let buffer = "";

        try {
            const res = await fetch("/agent/query-stream-delta", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({message, session_id: sessionId}),
                signal: abortController.current.signal,
            });

            const reader = res.body?.getReader();
            if (!reader) throw new Error("No reader");

            const decoder = new TextDecoder();

            while (true) {
                const {done, value} = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, {stream: true});
                const lines = chunk.split("\n");

                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;

                    const data = line.slice(6);
                    if (data.trim() === ": done") break;

                    try {
                        const event: AgentEvent = JSON.parse(data);

                        switch (event.type) {
                            case "text_delta":
                                buffer += event.delta || "";
                                setResponse(buffer);
                                break;

                            case "tool_call":
                                setToolCalls(prev => [...prev, event]);
                                break;

                            case "tool_result":
                                setToolCalls(prev => [...prev, event]);
                                break;

                            case "step_start":
                                setToolCalls(prev => [...prev, event]);
                                break;

                            case "step_complete":
                                setToolCalls(prev => [...prev, event]);
                                break;

                            case "final":
                                setResponse(event.content || "");
                                setIsStreaming(false);
                                break;

                            case "usage":
                                console.log("Usage:", event.usage);
                                break;

                            case "error":
                                console.error("Error:", event.error);
                                setIsStreaming(false);
                                break;
                        }
                    } catch (e) {
                        // 忽略解析错误
                    }
                }
            }
        } catch (error: any) {
            if (error.name !== "AbortError") {
                console.error("Stream error:", error);
            }
        } finally {
            setIsStreaming(false);
        }
    }, []);

    const abort = useCallback(() => {
        abortController.current?.abort();
        setIsStreaming(false);
    }, []);

    return { response, isStreaming, toolCalls, streamQuery, abort};
}
```

### Vue 3 Composition API

```vue
<script setup lang="ts">
import { ref } from 'vue'

const message = ref('')
const response = ref('')
const isStreaming = ref(false)
const events = ref<any[]>([])

async function sendQuery() {
    isStreaming.value = true
    response.value = ''
    events.value = []

    const res = await fetch('/agent/query-stream-delta', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: message.value})
    })

    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
        const {done, value} = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue

            const data = line.slice(6)
            if (data.trim() === ': done') break

            const event = JSON.parse(data)
            events.value.push(event)

            switch (event.type) {
                case 'text_delta':
                    buffer += event.delta
                    response.value = buffer
                    break
                case 'final':
                    isStreaming.value = false
                    break
                case 'error':
                    console.error(event.error)
                    isStreaming.value = false
                    break
            }
        }
    }
}
</script>

<template>
    <div>
        <input
            v-model="message"
            @keyup.enter="sendQuery"
            :disabled="isStreaming"
            placeholder="输入消息..."
        />
        <div>{{ response }}</div>
        <div v-if="isStreaming">正在生成...</div>
        <div>
            <div v-for="event in events" :key="event.timestamp">
                <pre>{{ event }}</pre>
            </div>
        </div>
    </div>
</template>
```

---

## 事件处理最佳实践

### 1. 文本渲染

```javascript
// 推荐：累积增量文本
let buffer = "";
if (event.type === "text_delta") {
    buffer += event.delta;
    // 使用 buffer 更新 UI
}

// 避免：直接渲染每个 delta（会导致重复）
if (event.type === "text_delta") {
    // ❌ 错误：会重复
    appendToUI(event.delta);
}
```

### 2. 工具调用状态

```javascript
const toolStatus = ref(new Map());

// 记录工具调用
if (event.type === "tool_call") {
    toolStatus.value.set(event.tool_call_id, "running");
    showToolIndicator(event.tool, event.args);
}

// 更新工具结果
if (event.type === "tool_result") {
    toolStatus.value.set(event.tool_call_id, event.is_error ? "error" : "success");
    showToolResult(event.tool, event.result);
}
```

### 3. 进度条

```javascript
const steps = ref([]);
const currentStep = ref(0);

if (event.type === "step_start") {
    steps.value.push(event);
}

if (event.type === "step_complete") {
    currentStep.value++;
}

// 显示进度
<progress value={currentStep} total={steps.length} />
```

### 4. 错误处理

```javascript
try {
    const event = JSON.parse(data);
    // 处理事件...
} catch (e) {
    console.error("Invalid event:", data);
    // 继续处理下一个事件，不要中断流
}
```

---

## 连接管理

### 建立连接

```javascript
const controller = new AbortController();

fetch("/agent/query-stream-delta", {
    signal: controller.signal,
    // ...
})
```

### 中断连接

```javascript
// 用户取消时
controller.abort();
```

### 超时处理

```javascript
const TIMEOUT = 60000; // 60秒

const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), TIMEOUT);

try {
    await fetch("/agent/query-stream-delta", {
        signal: controller.signal,
        // ...
    });
} finally {
    clearTimeout(timeoutId);
}
```

---

## 调试技巧

### 1. 打印所有事件

```javascript
async for (const line of response) {
    console.log('Raw line:', line);
    // ... 处理逻辑
}
```

### 2. 事件计数

```javascript
let eventCount = 0;
const eventTypes = {};

async function processStream() {
    // ... 流处理
    eventTypes[event.type] = (eventTypes[event.type] || 0) + 1;
    eventCount++;
}

// 结束时打印统计
console.log("Total events:", eventCount);
console.log("Event types:", eventTypes);
```

### 3. 时间测量

```javascript
const startTime = performance.now();

async for await (const event of streamEvents) {
    if (event.type === "final") {
        const duration = performance.now() - startTime;
        console.log(`Total time: ${duration.toFixed(0)}ms`);
        break;
    }
}
```

---

## 完整示例组件

### ChatInterface.vue

```vue
<script setup lang="ts">
import { ref, nextTick } from 'vue'

// 状态
const inputMessage = ref('')
const messages = ref<Array<{role: string, content: string}>>([])
const currentResponse = ref('')
const isStreaming = ref(false)
const sessionId = ref<string | null>(null)

// 事件处理器
const eventHandlers = {
    onTextDelta: (delta: string) => {
        currentResponse.value += delta
        scrollToBottom()
    },
    onToolCall: (event: any) => {
        addMessage('system', `调用工具: ${event.tool}`)
    },
    onToolResult: (event: any) => {
        addMessage('system', `工具结果: ${event.result.slice(0, 100)}...`)
    },
    onFinal: (event: any) => {
        addMessage('assistant', event.content)
        currentResponse.value = ''
        isStreaming.value = false
    },
}

// 发送消息
async function sendMessage() {
    if (!inputMessage.value.trim() || isStreaming.value) return

    const userMsg = inputMessage.value
    addMessage('user', userMsg)
    inputMessage.value = ''
    isStreaming.value = true

    try {
        // 创建会话（如果需要）
        if (!sessionId.value) {
            const res = await fetch('/sessions', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            })
            const data = await res.json()
            sessionId.value = data.session_id
        }

        // 发送流式请求
        await streamQuery(userMsg, sessionId.value, eventHandlers)
    } catch (error: any) {
        addMessage('system', `错误: ${error.message}`)
        isStreaming.value = false
    }
}

// 流式查询
async function streamQuery(message: string, sid: string, handlers: any) {
    const res = await fetch('/agent/query-stream-delta', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message, session_id: sid})
    })

    const reader = res.body!.getReader()
    const decoder = new TextDecoder()

    while (true) {
        const {done, value} = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, {stream: true})
        const lines = chunk.split('\n')

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue

            const data = line.slice(6)
            if (data.trim() === ': done') break

            try {
                const event = JSON.parse(data)

                if (event.type === 'text_delta') {
                    handlers.onTextDelta(event.delta)
                } else if (event.type === 'final') {
                    handlers.onFinal(event)
                } else if (event.type === 'tool_call') {
                    handlers.onToolCall(event)
                } else if (event.type === 'tool_result') {
                    handlers.onToolResult(event)
                } else if (event.type === 'error') {
                    handlers.onError?.(event)
                }
            } catch (e) {
                console.error('Parse error:', line)
            }
        }
    }
}

// 辅助函数
function addMessage(role: string, content: string) {
    messages.value.push({role, content})
    nextTick(scrollToBottom)
}

function scrollToBottom() {
    const container = document.querySelector('.chat-messages')
    if (container) {
        container.scrollTop = container.scrollHeight
    }
}
</script>

<template>
  <div class="chat-container">
    <!-- 消息列表 -->
    <div class="chat-messages">
      <div
        v-for="(msg, index) in messages"
        :key="index"
        :class="msg.role"
      >
        <div class="message-content">{{ msg.content }}</div>
      </div>

      <!-- 当前流式响应 -->
      <div v-if="currentResponse" class="assistant streaming">
        {{ currentResponse }}
      </div>
    </div>

    <!-- 输入区域 -->
    <div class="input-area">
      <input
        v-model="inputMessage"
        @keyup.enter="sendMessage"
        :disabled="isStreaming"
        placeholder="输入消息..."
      />
      <button
        @click="sendMessage"
        :disabled="isStreaming"
      >
        {{ isStreaming ? '生成中...' : '发送' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.chat-messages {
    max-height: 500px;
    overflow-y: auto;
    padding: 10px;
}

.message-content {
    padding: 8px 12px;
    border-radius: 8px;
    margin: 4px 0;
}

.user .message-content {
    background: #e3f2fd;
    margin-left: auto;
    max-width: 70%;
}

.assistant .message-content {
    background: #f5f5f5;
    margin-right: auto;
}

.streaming {
    opacity: 0.8;
}
</style>
```
