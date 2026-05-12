# Web 远程对话 P1 接口设计

本文档描述的是 **下一步收敛后的接口方案**，先统一接口方向，暂不要求当前代码已经全部实现。

目标：

- 对 Web 侧尽量简化接口形态
- 对 worker 侧尽量不改原有协议
- 保持与现有 IM -> worker -> CLI 的模型一致

## 1. 总体原则

### 1.1 Web 对外接口按 `worker_id`

P1 不再把 Web SSE 订阅设计成按 `request_id` 建链。

原因：

- 一个 Web 页面只绑定一个 `worker_id`
- 一个 `worker_id` 同一时刻只有一条 Web 请求
- 页面没有多会话并发需求

因此：

- Web 对外接口优先按 `worker_id`
- `request_id` 不作为 Web 前端的核心接口参数

### 1.2 worker 协议保持轻量

worker 协议继续保持现有字段，不引入新的强依赖字段：

- `worker_id`
- `source`
- `content`
- `final_content`

P1 不要求：

- `request_id`
- `delivery_id`
- `session_key`

下沉到 Python worker 协议。

## 2. Web-BFF 接口

### 2.1 `GET /web-console/workers/{workerId}`

用途：

- 查询 worker 在线状态

响应示例：

```json
{
  "workerId": "worker-hk-01",
  "isOnline": true,
  "lastCompletedAt": "2026-05-12T10:00:00Z"
}
```

说明：

- `lastCompletedAt` 可选
- 仅用于页面状态展示

### 2.2 `POST /web-console/messages`

用途：

- Web 提交一条消息给指定 worker

请求示例：

```json
{
  "workerId": "worker-hk-01",
  "sessionId": "web-current",
  "content": "请帮我总结一下这个目录的功能"
}
```

响应示例：

```json
{
  "ok": true,
  "acceptedAt": "2026-05-12T10:01:00Z"
}
```

说明：

- `sessionId` 仍可保留，主要用于前端本地页面状态和日志字段
- Web 前端不依赖返回的 `requestId` 才能继续工作
- server 接收后即可把消息放入该 `worker_id` 对应的待处理队列

### 2.3 `GET /web-console/workers/{workerId}/events`

用途：

- Web 订阅当前 `worker_id` 的唯一 Web SSE

请求示例：

```http
GET /web-console/workers/worker-hk-01/events
Authorization: Bearer <token>
Accept: text/event-stream
```

响应头：

```http
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

说明：

- 该 SSE 只代表这台 worker 当前唯一的 Web 请求
- 一个 worker 同时最多允许一条 Web SSE
- 一个 worker 同时最多允许一条进行中的 Web 请求

### 2.4 不再提供 `stop` 接口

P1 收敛方案下，不再强依赖：

```text
POST /web-console/messages/{requestId}/stop
```

原因：

- 当前 Web 请求和当前 Web SSE 都已经唯一绑定到 `worker_id`
- 前端中止当前 `fetch + ReadableStream` 即可精准结束当前展示链路

语义说明：

- 断开 SSE = 停止 Web 当前页面接收
- 不等于取消本地 CLI 实际任务

## 3. worker 协议

worker 协议保持现有风格，不做 request 级增强。

### 3.1 `POST /api/worker/online`

请求：

```json
{
  "worker_id": "worker-hk-01"
}
```

响应：

```json
{
  "ok": true
}
```

### 3.2 `POST /api/worker/offline`

请求：

```json
{
  "worker_id": "worker-hk-01"
}
```

响应：

```json
{
  "ok": true
}
```

### 3.3 `POST /api/worker/poll`

请求：

```json
{
  "worker_id": "worker-hk-01"
}
```

响应：

```json
{
  "messages": [
    {
      "content": "hello",
      "worker_id": "worker-hk-01",
      "source": "web"
    }
  ]
}
```

说明：

- `source` 取值：
  - `im`
  - `web`

### 3.4 `GET /api/worker/stream`

请求：

```http
GET /api/worker/stream?worker_id=worker-hk-01
```

SSE 事件：

- `ready`
- `message`
- `heartbeat`
- `error`

`message` 事件示例：

```json
{
  "content": "hello",
  "source": "web"
}
```

### 3.5 `POST /api/worker/progress`

用途：

- worker 将本地 CLI 处理中产生的中间文本块回给 server

请求：

```json
{
  "worker_id": "worker-hk-01",
  "source": "web",
  "content": "正在整理目录结构，请稍候"
}
```

响应：

```json
{
  "ok": true
}
```

说明：

- `progress` 不得结束当前请求
- `progress` 只负责给 Web 推送中间文本

### 3.6 `POST /api/worker/complete`

用途：

- worker 将最终结果回给 server

请求：

```json
{
  "worker_id": "worker-hk-01",
  "source": "web",
  "final_content": "目录概览如下：..."
}
```

响应：

```json
{
  "ok": true
}
```

说明：

- `complete` 只用于最终结果
- `complete` 到达后，server 会向当前 `worker_id` 的 Web SSE 推送 `completed`
- 推送完成后，本次 Web 请求结束

## 4. Web SSE 事件格式

### 4.1 `submitted`

```json
{
  "type": "submitted",
  "workerId": "worker-hk-01",
  "ts": "2026-05-12T10:01:00Z"
}
```

### 4.2 `processing`

```json
{
  "type": "processing",
  "workerId": "worker-hk-01",
  "ts": "2026-05-12T10:01:01Z"
}
```

### 4.3 `progress`

```json
{
  "type": "progress",
  "workerId": "worker-hk-01",
  "content": "正在整理目录结构，请稍候",
  "ts": "2026-05-12T10:01:03Z"
}
```

### 4.4 `completed`

```json
{
  "type": "completed",
  "workerId": "worker-hk-01",
  "finalContent": "目录概览如下：...",
  "finishedAt": "2026-05-12T10:01:15Z"
}
```

### 4.5 `failed`

```json
{
  "type": "failed",
  "workerId": "worker-hk-01",
  "errorMessage": "worker timeout",
  "finishedAt": "2026-05-12T10:01:15Z"
}
```

## 5. 推荐事件流

成功链路：

```text
submitted
processing
progress
progress
completed
: done
```

失败链路：

```text
submitted
processing
progress
failed
: done
```

用户中止页面接收：

- 前端直接中止 SSE 读取
- server 通过连接关闭回调清理当前 `worker_id` 的 Web 订阅状态
- 不要求单独的 `stop` API

## 6. server 侧匹配原则

由于当前约束是：

- 一个 `worker_id` 同时最多一条 Web 请求
- 一个 `worker_id` 同时最多一条 Web SSE

因此 server 收到：

- `source=web` 的 `progress`
- `source=web` 的 `complete`

时，不需要复杂的 request 级定位，只需要把事件发给当前 `worker_id` 的 Web SSE 即可。

换句话说，下一步的目标实现是：

- 按 `worker_id` 定位当前 Web 订阅
- 而不是按 `request_id` 定位某条 SSE

## 7. 当前文档明确不采用的方案

P1 文档里明确不采用下面这些更重的设计：

- `/web-console/messages/{requestId}/events`
- `/web-console/messages/{requestId}/stop`
- Web 对外强依赖 `request_id`
- worker 协议下沉 `request_id`
- worker 协议下沉 `delivery_id`
- server 对外暴露复杂的 inflight request 模型

