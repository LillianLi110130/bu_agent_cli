# Worker SSE Phase 1 Server Guide

## 1. 文档目标

这份文档只描述当前第一阶段 SSE 方案下，服务端需要怎么做，以及相较原来的 HTTP 轮询协议新增了什么。

这里说的“当前服务端实现”，以仓库里的参考实现为准：

- [mock_server.py](/d:/llm_project/bu_agent_cli/cli/worker/mock_server.py)

这不是第二阶段的增强版协议说明。本文明确不引入：

- `client_instance_id`
- `delivery_id`
- `message_id`
- `Last-Event-ID`
- lease / redelivery

第一阶段只做一件事：把“下行取消息”从 `poll` 改成 SSE 推送，其他协议语义尽量保持不变。

## 2. 当前服务端怎么做

### 2.0 CLI 启动后发生了什么

从当前代码看，worker CLI 启动后的实际链路大致是这样：

1. 解析启动参数
2. 加载本地配置目录
3. 如果开启了认证，从本地读取持久化的 `Authorization`
4. 创建 `WorkerGatewayClient`
5. 创建 `WorkerRunner`
6. 初始化本地 `FileBridgeStore`
7. 调 `POST /api/worker/online`
8. 默认情况下，worker 会以 `--gateway-transport sse` 的模式建立 `GET /api/worker/stream`
9. 服务端通过 SSE 下发 `message`
10. worker 把消息写入本地 bridge，请本地 CLI 执行
11. worker 轮询等待本地结果文件出现
12. 拿到最终结果后调 `POST /api/worker/complete`
13. 退出时调 `POST /api/worker/offline`

对应代码入口主要在：

- [main.py](/d:/llm_project/bu_agent_cli/cli/worker/main.py)
- [runner.py](/d:/llm_project/bu_agent_cli/cli/worker/runner.py)
- [gateway_client.py](/d:/llm_project/bu_agent_cli/cli/worker/gateway_client.py)

如果展开一点看，可以分成 4 个阶段。

#### 阶段 1：启动与本地准备

`main.py` 里会先解析这些关键参数：

- `--worker-id`
- `--gateway-base-url`
- `--gateway-transport`
- `--config-dir`
- `--config-source-dir`
- `--root-dir`

然后它会做两件事：

- 根据配置目录加载 worker 鉴权配置
- 如果启用了认证，从本地持久化文件里取出 `Authorization`

接着创建：

- `WorkerGatewayClient`
- `WorkerRunner`

当前默认传输模式已经是：

- `sse`

也就是说，不显式传 `--gateway-transport` 时，worker 默认就会建立 SSE 下行流。

而 `WorkerRunner` 初始化时，会立刻创建并初始化本地 `FileBridgeStore`。

这一步的意思是：

- 远程消息虽然来自 gateway
- 但真正执行消息的，仍然是本地 CLI / bridge 链路
- worker 本身更像一个“远程网关适配器 + 本地执行协调器”

#### 阶段 2：向服务端报到

`runner.run_forever()` 启动后，第一件事不是收消息，而是先调：

- `POST /api/worker/online`

如果这一步失败，worker 会直接报错退出。

所以从服务端视角看，真正开始投递消息的前提是：

- CLI 已启动
- 本地 bridge 已初始化
- `online` 已成功

#### 阶段 3：建立下行通道

如果 transport 是 `poll`：

- worker 进入轮询循环
- 不断调 `POST /api/worker/poll`

如果 transport 是 `sse`：

- worker 进入流式循环
- 建立 `GET /api/worker/stream?worker_id=...`
- 持续消费 SSE 事件

当前 phase 1 的 SSE worker 只真正消费一种业务事件：

- `message`

这些事件会被这样处理：

- `ready`：收到但不作为业务消息处理
- `heartbeat`：收到但不作为业务消息处理
- `error`：连接结束后进入重连逻辑
- `message`：进入本地执行流程

换句话说，服务端新增 SSE 后，CLI 启动后的核心变化其实就一条：

- 原来是 worker 主动反复发 `poll`
- 现在是 worker 常驻一个 `stream`，等服务端主动推 `message`

#### 阶段 4：本地执行与结果回传

worker 收到一条 `message` 后，不会直接在网络层处理完，而是先写入本地 bridge：

- `enqueue_text(...)`
- `source="remote"`
- `source_meta={"worker_id": ...}`
- `remote_response_required=True`

然后 worker 会等待本地结果文件出现。

当前实现里，这一步是通过 `bridge_store.find_result(request_id)` 轮询完成的。也就是说：

- 远程 gateway 负责投递任务
- 本地 bridge 负责把任务交给 CLI 执行
- worker 再把 CLI 最终产出的结果回传给 gateway

当本地结果准备好后，worker 会调：

- `POST /api/worker/complete`

请求体仍然只有：

- `worker_id`
- `final_content`

最后，当 worker 被停止或退出时，会进入清理阶段，最终调用：

- `POST /api/worker/offline`

### 2.1 总体模型

第一阶段服务端仍然按 `worker_id` 维护 worker 状态和消息队列。

核心状态可以理解为 4 份：

- `online_workers`
  - 记录 worker 是否在线，以及最近一次活动时间
- `queued_messages`
  - 等待下发给某个 `worker_id` 的消息
- `inflight_messages`
  - 已经发给 worker、但尚未 `complete` 的消息
- `stream_versions`
  - 当前 `worker_id` 对应的活跃 SSE 连接版本号，用来实现“新连接顶掉旧连接”

### 2.2 服务端生命周期

当前参考实现里的服务端行为是：

1. worker 调 `POST /api/worker/online`
2. 服务端标记该 `worker_id` 在线
3. worker 建立 `GET /api/worker/stream?worker_id=...`
4. 服务端保持这条 SSE 长连接
5. 有新消息时，服务端把消息从 `queued` 转到 `inflight`
6. 然后通过 SSE 发一个 `message` 事件给 worker
7. worker 本地处理完成后，调 `POST /api/worker/complete`
8. 服务端按 `worker_id` 的 inflight FIFO 队列弹出最早一条，记录完成结果
9. worker 退出时调 `POST /api/worker/offline`

### 2.3 在线判定

当前服务端仍然保留 TTL 判活思路。

参考实现中：

- `online` 会把 worker 标为在线
- `stream` 活着期间，服务端会不断刷新 `last_seen_at`
- `offline` 会显式标记离线
- 如果超过 `worker_ttl_seconds` 没有活动，也会被视为离线

也就是说，第一阶段虽然把 `poll` 去掉了，但“活性刷新”没有消失，只是从“靠轮询请求刷新”变成了“靠 SSE 连接活动刷新”。

### 2.4 消息投递

当前第一阶段的消息投递模型非常克制，核心约束是：

- 单 `worker_id` 单活跃 SSE 连接
- 单 `worker_id` 串行处理
- `complete` 仍然只按 `worker_id` 归并

这意味着服务端现在不需要知道“这是同一个 worker 的第几次实例”，也不需要知道“这次 complete 对应哪一个 delivery”。

第一阶段的前提就是：一个 `worker_id` 同一时刻只允许一个稳定消费方。

### 2.5 顶号行为

当前参考实现已经支持“新连接顶旧连接”。

实现方式是：

- 每次 `worker_id` 建立新 SSE 连接时，服务端给它生成一个新的 `stream_version`
- 老连接在下一次循环里发现自己不是当前版本后，会收到：
  - `event: error`
  - `data: {"code":"replaced","message":"connection replaced by newer stream"}`
- 然后老连接退出

这个机制是第一阶段替代 `client_instance_id` 的简化做法。

## 3. 相较原轮询版，哪些接口保留了

第一阶段不是整套协议重做，而是在原接口上增量演进。

继续保留的接口有：

- `POST /api/worker/online`
- `POST /api/worker/complete`
- `POST /api/worker/offline`
- `POST /api/worker/poll`

说明：

- 前 3 个接口在 SSE 方案下仍然继续使用
- `poll` 在第一阶段不一定立刻删掉，建议保留一段时间用于兼容和回滚

也就是说，服务端改造后通常不是“替换协议”，而是“新增 SSE 下行通道，同时保留原轮询通道”。

## 4. 相较原轮询版，新增了什么

第一阶段真正新增的接口只有一个：

- `GET /api/worker/stream`

请求方式示例：

```http
GET /api/worker/stream?worker_id=80383648 HTTP/1.1
Authorization: Bearer xxx
Accept: text/event-stream
Cache-Control: no-cache
```

### 4.1 请求参数

当前第一阶段只需要：

- Query
  - `worker_id`

建议同时支持这些请求头：

- `Authorization`
- `Accept: text/event-stream`
- `Cache-Control: no-cache`

第一阶段不需要新增这些字段：

- `client_instance_id`
- `connection_id`
- `Last-Event-ID`

### 4.2 响应类型

服务端返回：

```http
Content-Type: text/event-stream
Cache-Control: no-cache
```

连接建立后，服务端持续输出 SSE frame。

## 5. SSE 事件新增了什么

相较原来 `poll` 的 JSON 响应，第一阶段新增的是 4 类 SSE 事件。

### 5.1 `ready`

作用：

- 告诉 worker，SSE 连接已经建立成功

示例：

```text
event: ready
data: {"worker_id":"80383648"}
```

### 5.2 `message`

作用：

- 下发一条待处理消息

示例：

```text
event: message
data: {"content":"hello from im"}
```

说明：

- 第一阶段 `message` 体里仍然只有 `content`
- 不新增 `delivery_id`
- 不新增 `message_id`

### 5.3 `heartbeat`

作用：

- 在没有新消息时维持连接活性
- 让 worker 和中间代理不至于长时间静默

示例：

```text
event: heartbeat
data: {"ts": 1760000000.0}
```

说明：

- 这个事件是第一阶段 SSE 相较轮询版新增的“传输层事件”
- 它不是业务消息
- worker 收到后通常只需要忽略或更新本地连接状态

### 5.4 `error`

作用：

- 表达当前 SSE 连接需要结束

当前第一阶段已用到的错误码：

- `replaced`

示例：

```text
event: error
data: {"code":"replaced","message":"connection replaced by newer stream"}
```

说明：

- 这是第一阶段为“单 worker_id 单活跃连接”服务的配套事件
- 方便旧连接感知自己被顶掉

## 6. 服务端内部行为相较原来需要怎么调整

### 6.1 下发链路从拉模式变成推模式

原来：

- worker 定时 `poll`
- 服务端在 `poll` 请求里返回 `messages`

现在：

- worker 建立 `stream`
- 服务端在后台等待消息
- 有消息时主动推一个 `message` 事件

### 6.2 队列模型仍然保留

虽然下发方式变了，但服务端内部依然建议保留：

- `queued`
- `inflight`
- `completed`

原因是：

- `complete` 语义没有变
- IM 入站与 worker 消费仍然是异步解耦的
- 未来第二阶段还要在这个基础上继续扩展幂等和重投递

### 6.3 enqueue 时要能唤醒 stream

轮询时代，服务端不需要主动唤醒任何连接。

SSE 时代，服务端在有新消息入队后，需要通知对应 `worker_id` 的 stream 循环，让它尽快把消息推下去。

参考实现里对应的是：

- `enqueue_message(...)` 后触发 `notify_stream(worker_id)`

### 6.4 online/offline 也要影响 stream

第一阶段里，`online` 和 `offline` 不只是状态写库动作，也会影响 SSE 循环。

参考行为：

- `online` 后可唤醒 stream 检查状态
- `offline` 后 stream 应尽快退出

### 6.5 poll 和 stream 可以共用同一套出队逻辑

当前参考实现里，`poll` 和 `stream` 都复用了同一个“从 `queued` 取一条并转成 `inflight`”的逻辑。

这很重要，因为它说明第一阶段新增 SSE 时，不一定要重写整套投递状态机。

更稳妥的做法是：

- 保留原有出队逻辑
- 只新增一种新的消费入口 `stream`

## 7. 服务端需要新增的最小能力清单

如果你们现在要把真实服务端从轮询升级到第一阶段 SSE，新增项最少是这些：

### 7.1 新增接口

- `GET /api/worker/stream`

### 7.2 新增响应能力

- 返回 `text/event-stream`
- 能持续输出 SSE frame
- 能输出 `ready`
- 能输出 `message`
- 能输出 `heartbeat`
- 在连接被替换时能输出 `error`

### 7.3 新增连接管理能力

- 按 `worker_id` 维护当前活跃 stream
- 新 stream 建立时替换旧 stream
- 连接空闲时定期发送 heartbeat

### 7.4 新增唤醒机制

- 当消息入队时，能唤醒对应 worker 的 stream 循环
- 当在线状态变化时，能让 stream 尽快感知

## 8. 第一阶段明确不需要新增什么

为了控制改造范围，第一阶段可以明确先不做这些：

- 不新增 `client_instance_id`
- 不新增 `delivery_id`
- 不新增 `message_id`
- 不做基于 `Last-Event-ID` 的断点续传
- 不做严格意义上的重复投递恢复
- 不做多 inflight 并发投递
- 不做复杂 lease / ack / retry 协议

这也是为什么第一阶段要坚持几个约束：

- 单 `worker_id` 单连接
- 单 `worker_id` 串行处理
- `complete` 继续按 inflight FIFO 归并

## 9. 给服务端同学的落地结论

如果只做第一阶段，服务端可以理解成：

- 保留原有 `online / poll / complete / offline`
- 新增一个 `GET /api/worker/stream`
- 把“消息下发”从“worker 主动 poll”变成“服务端通过 SSE 主动推 message”
- 不引入新的业务标识字段
- 通过“单 `worker_id` 单活跃连接 + inflight FIFO”维持现有协议可用性

所以，第一阶段服务端真正新增的内容并不多，主要是：

1. 一个新的 SSE 接口
2. 一套按 `worker_id` 管理活跃长连接的能力
3. 消息入队后主动唤醒并推送的能力
4. 心跳和连接替换处理

如果后面进入第二阶段，再继续引入：

- `client_instance_id`
- `delivery_id`
- 更严格的重连恢复和幂等语义
