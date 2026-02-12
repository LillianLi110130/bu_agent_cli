# BU Agent SDK HTTP 服务器代码分析

## 目录结构

```
bu_agent_sdk/server/
├── __init__.py       # 模块导出
├── models.py         # API 请求/响应数据模型
├── session.py        # 会话管理器
├── app.py            # FastAPI 应用主文件
├── client.py         # Python HTTP 客户端
├── example_server.py # 示例服务器
└── README.md         # 使用文档
```

---

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                        客户端请求                            │
│                    (HTTP / SSE)                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI 应用                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ /health  │  |/sessions │  │/agent/   │  │/agent/   │   │
│  │          │  │          │  │query     │  │query-    │   │
│  │          │  │          │  │          │  │stream    │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │           │
└───────┼─────────────┼─────────────┼─────────────┼───────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SessionManager                            │
│  - 管理多个 Agent 实例                                        │
│  - 每个 session_id 对应一个独立的 Agent                       │
│  - 自动清理过期会话                                           │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                       Agent (SDK)                            │
│  - LLM 调用                                                  │
│  - Tool 执行                                                 │
│  - 对话历史管理                                              │
│  - Token 统计                                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心组件分析

### 2.1 数据模型 (models.py)

定义 API 的请求和响应格式：

| 模型 | 用途 |
|------|------|
| `QueryRequest` | 查询请求：`message`, `session_id`, `stream` |
| `QueryResponse` | 查询响应：`session_id`, `response`, `usage` |
| `TextEvent` | 文本事件 |
| `ToolCallEvent` | 工具调用事件 |
| `ToolResultEvent` | 工具结果事件 |
| `FinalResponseEvent` | 最终响应事件 |
| `SessionCreateResponse` | 会话创建响应 |
| `HealthResponse` | 健康检查响应 |

### 2.2 会话管理 (session.py)

**SessionManager 职责：**
- 创建和获取 Agent 会话
- 管理会话生命周期
- 自动清理过期会话
- 并发安全（使用 asyncio.Lock）

**关键方法：**
```python
async def get_or_create_session(session_id: str | None = None) -> AgentSession
async def delete_session(session_id: str) -> bool
async def cleanup_task(interval_seconds: int)
```

### 2.3 FastAPI 应用 (app.py)

#### 2.3.1 配置类

```python
class ServerConfig(BaseModel):
    session_timeout_minutes: int = 60      # 会话超时时间
    max_sessions: int = 1000                # 最大会话数
    cleanup_interval_seconds: int = 300     # 清理间隔
    enable_cleanup_task: bool = True        # 是否启用自动清理
```

#### 2.3.2 事件转换

将 SDK 内部事件转换为 API 事件格式：

```python
def _agent_event_to_stream_event(event) -> StreamEvent:
    # AgentTextEvent -> TextEvent
    # AgentToolCallEvent -> ToolCallEvent
    # AgentToolResultEvent -> ToolResultEvent
    # AgentFinalResponseEvent -> FinalResponseEvent
    # ...
```

#### 2.3.3 SSE 序列化

```python
def _serialize_event(event: StreamEvent) -> str:
    # 将 Pydantic 模型序列化为 SSE 格式
    # 输出: "data: {json}\n\n"
```

#### 2.3.4 生命周期管理

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    _session_manager = SessionManager(...)
    _cleanup_task = loop.create_task(...)
    yield
    # 关闭时
    _cleanup_task.cancel()
```

#### 2.3.5 端点列表

| 端点 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/sessions` | POST | 创建会话 |
| `/sessions` | GET | 列出所有会话 |
| `/sessions/{id}` | GET | 获取会话信息 |
| `/sessions/{id}` | DELETE | 删除会话 |
| `/sessions/{id}/clear` | POST | 清空会话历史 |
| `/agent/query` | POST | 非流式查询 |
| `/agent/query-stream` | POST | 流式查询 (SSE) |
| `/agent/usage/{id}` | GET | 获取使用统计 |

---

## 3. 流式查询核心流程

### 3.1 时序图

```
客户端                     FastAPI                  SessionManager              Agent
  │                          │                            │                        │
  │ POST /agent/query-stream │                            │                        │
  │─────────────────────────>│                            │                        │
  │                          │ get_or_create_session()    │                        │
  │                          │───────────────────────────>│                        │
  │                          │  AgentSession              │                        │
  │                          │<───────────────────────────│                        │
  │                          │                            │                        │
  │                          │ query_stream(message)      │                        │
  │                          │───────────────────────────────>                       │
  │                          │                            │                        │
  │<═════════════════════════ SSE 流 ═══════════════════════════════════════════>│
  │                          │                            │                        │
  │ data: {"type":"text",...}│                            │                        │
  │<═════════════════════════════════════════════════════════════════════════════│
  │                          │                            │                        │
  │ data: {"type":"tool_call",...}                        │                        │
  │<═════════════════════════════════════════════════════════════════════════════│
  │                          │                            │                        │
  │ data: {"type":"tool_result",...}                      │                        │
  │<═════════════════════════════════════════════════════════════════════════════│
  │                          │                            │                        │
  │ data: {"type":"final",...}                            │                        │
  │<═════════════════════════════════════════════════════════════════════════════│
  │                          │                            │                        │
  │ : done                    │                            │                        │
  │<═════════════════════════════════════════════════════════════════════════════│
```

### 3.2 代码流程

```python
async def query_stream(request: QueryRequest):
    # 1. 获取或创建会话
    session = await _session_manager.get_or_create_session(request.session_id)

    async def event_generator():
        # 2. 遍历 Agent 产生的事件
        async for event in session.query_stream(request.message):
            # 3. 转换为 API 事件格式
            stream_event = _agent_event_to_stream_event(event)
            # 4. 序列化为 SSE 格式
            yield _serialize_event(stream_event)

            # 5. 收到 final 事件后结束
            if isinstance(event, AgentFinalResponseEvent):
                yield "data: {'type': 'usage', ...}\n\n"
                yield ": done\n\n"
                return

    # 6. 返回 StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## 4. Server-Sent Events (SSE) 格式

### 4.1 标准格式

```
data: {"type":"text","content":"你好","timestamp":"2025-02-06..."}

data: {"type":"tool_call","tool":"add","args":{"a":1,"b":2}}

data: {"type":"final","content":"答案是3"}

: done
```

### 4.2 事件类型

| type | 说明 | content 字段 |
|------|------|-------------|
| `text` | 文本内容 | assistant 的回复文本 |
| `thinking` | 思考内容 | 模型的推理过程 |
| `tool_call` | 工具调用 | 工具名和参数 |
| `tool_result` | 工具结果 | 工具执行结果 |
| `step_start` | 步骤开始 | 步骤信息 |
| `step_complete` | 步骤完成 | 步骤状态和耗时 |
| `final` | 最终响应 | 完整的最终答案 |
| `usage` | 使用统计 | token 消耗和成本 |
| `error` | 错误 | 错误信息 |

---

## 5. 并发安全

### 5.1 会话级别的锁

```python
class AgentSession:
    def __init__(self, ...):
        self._lock = asyncio.Lock()

    async def query(self, message: str) -> str:
        async with self._lock:  # 同一会话的请求串行执行
            return await self.agent.query(message)
```

### 5.2 SessionManager 级别的锁

```python
class SessionManager:
    async def get_or_create_session(self, session_id: str | None = None):
        async with self._lock:  # 会话创建/获取是线程安全的
            # ...
```

---

## 6. 扩展点

### 6.1 添加认证

在 `app.py` 中添加中间件：

```python
from fastapi import Security
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/agent/query")
async def query(
    request: QueryRequest,
    credentials: Security = Security(security)
):
    # 验证 credentials
    ...
```

### 6.2 添加 Redis 持久化

```python
class RedisSessionManager(SessionManager):
    async def save_session(self, session_id: str):
        await redis.set(
            f"session:{session_id}",
            pickle.dumps(session.agent.messages)
        )
```

### 6.3 添加限流

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/agent/query")
@limiter.limit("10/minute")
async def query(request: QueryRequest, ...):
    ...
```

---

## 7. 部署建议

### 7.1 开发环境

```bash
uvicorn bu_agent_sdk.server.example_server:app --reload --port 8000
```

### 7.2 生产环境

```bash
uvicorn bu_agent_sdk.server.example_server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info
```

### 7.3 Docker

```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -e .
CMD ["uvicorn", "bu_agent_sdk.server.example_server:app", "--host", "0.0.0.0"]
```

### 7.4 Nginx 反向代理

```nginx
location /agent/query-stream {
    proxy_pass http://localhost:8000;
    proxy_set_header Connection '';
    proxy_http_version 1.1;
    chunked_transfer_encoding off;
    proxy_buffering off;
}
```

---

## 8. 监控和日志

### 8.1 关键日志点

- 会话创建/删除
- 查询开始/结束
- 错误和异常
- Token 消耗

### 8.2 建议的监控指标

- 活跃会话数
- 查询延迟 (P50, P95, P99)
- Token 消耗速率
- 错误率

---

## 9. 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 连接被拒绝 | 服务器未启动 | 检查 uvicorn 是否运行 |
| 422 错误 | 请求体格式错误 | 确保发送 `json={}` |
| 500 错误 | API Key 无效 | 检查 OPENAI_API_KEY |
| 流式响应卡住 | 客户端未正确处理 `:done` | 添加超时和 break 逻辑 |
| 会话丢失 | 超时被清理 | 增加 `session_timeout_minutes` |

---

## 10. 总结

该 HTTP 服务器设计简洁、职责清晰：

1. **models.py**: 数据契约
2. **session.py**: 会话生命周期
3. **app.py**: API 路由和响应
4. **client.py**: 客户端 SDK

核心优势：
- 完全异步，高并发支持
- SSE 流式响应，实时性好
- 会话隔离，状态管理清晰
- 易于扩展和部署
