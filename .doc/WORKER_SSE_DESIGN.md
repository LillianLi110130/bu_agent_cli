# Worker 远程通信从 HTTP 轮询切换到 SSE 的设计方案

## 1. 背景

当前 worker 与远程 gateway 的对接方式是典型的 HTTP 轮询：

1. worker 启动后调用 `POST /api/worker/online`
2. worker 持续调用 `POST /api/worker/poll`
3. gateway 有消息时返回 `messages`
4. worker 将消息写入本地 `FileBridgeStore`
5. 本地 CLI 处理完成后，worker 调用 `POST /api/worker/complete`
6. worker 退出时调用 `POST /api/worker/offline`

现状可以工作，但有几个明显问题：

- 空闲时仍然会持续发起轮询请求，链路有额外开销。
- 消息到达依赖下一次 `poll`，实时性取决于轮询节奏。
- `poll` 既承担取消息，又承担保活语义，职责耦合。
- 当前 `complete` 只带 `worker_id`，没有显式 `delivery_id`，在并发或重连场景下关联关系偏弱。
- 一旦后续希望支持更稳定的连接管理、去重、重放、限流，轮询协议扩展成本会越来越高。

SSE 适合这个场景，因为这里本质上是“远程向 worker 单向推送任务，worker 再通过独立 HTTP 接口回传结果”。下行改成 SSE，可以保留上行 `complete/offline` 这一套控制面，不必一次性推翻整个协议。

## 2. 设计目标

本方案的目标是：

- 将“下行取消息”从 HTTP 轮询切换为 SSE 长连接推送。
- 保留 worker 本地 `FileBridgeStore` 桥接模型，不改本地 CLI 执行链路。
- 保留上行 HTTP 控制面，避免一次性引入 WebSocket 双向协议。
- 明确 delivery 身份，补齐并发、重连、幂等和失败回传能力。
- 让在线状态与保活机制从 `poll` 中解耦出来。
- 允许分阶段灰度，支持新旧协议并存一段时间。

非目标：

- 本次不改本地 CLI 的执行模型。
- 本次不引入 WebSocket。
- 本次不要求把远程结果也改成流式返回给 IM；仍然保持“最终结果回传”。

## 3. 基于现状的约束

从现有实现看，下面这些边界最好保留：

- worker 启动入口仍然在 `cli/worker/main.py`
- 远程桥接主循环仍然由 `cli/worker/runner.py` 驱动
- gateway 客户端职责仍然集中在 `cli/worker/gateway_client.py`
- 本地落盘桥接仍然复用 `cli.im_bridge.FileBridgeStore`
- 鉴权仍然沿用现有 `Authorization` 头和本地 token 持久化机制

这意味着最稳妥的改法不是“把 worker 改成全新体系”，而是：

- 保留现有控制面：`online / heartbeat / complete / offline`
- 新增一个数据面：`GET /api/worker/stream`
- runner 从“主动 poll”改为“维护 SSE 连接并消费 delivery 事件”

## 4. 推荐方案总览

推荐采用“控制面 HTTP + 数据面 SSE”的混合协议。

### 4.1 控制面

继续使用普通 HTTP 请求处理状态和结果：

- `POST /api/worker/online`
- `POST /api/worker/heartbeat`
- `POST /api/worker/complete`
- `POST /api/worker/offline`

其中：

- `online` 负责注册 worker 会话能力
- `heartbeat` 负责保活、续租、附带 token 刷新
- `complete` 负责按 `delivery_id` 回传执行结果
- `offline` 负责显式下线

### 4.2 数据面

新增 SSE 长连接接口：

- `GET /api/worker/stream`

该接口只负责把待处理消息推给 worker，不负责接收 worker 的处理结果。

### 4.3 为什么不直接上 WebSocket

当前场景里，worker 与 gateway 的交互不是强双向会话，而是：

- gateway 下发任务
- worker 上行提交状态和最终结果

SSE + HTTP 足以覆盖这个模型，而且具备这些优势：

- 和现有 HTTP 网关更兼容，落地简单
- 代理、负载均衡、日志体系更容易复用
- 连接语义更清晰，数据面和控制面分离
- 比 WebSocket 更容易灰度到现有接口体系

### 4.4 两阶段落地建议

如果你希望先低风险把“轮询改 SSE”跑起来，再逐步补可靠性字段，完全可以拆成两个阶段。

#### 第一阶段：只替换下行传输，不引入新字段

这一阶段的目标是：

- 保持现有身份模型不变，只使用 `worker_id`
- 不引入 `client_instance_id`
- 不引入 `delivery_id`
- `complete` 仍然保持当前 `worker_id + final_content` 形式
- 先把 `poll` 改成 `GET /api/worker/stream`

第一阶段推荐约束：

- 服务端同一时刻只允许一个 `worker_id` 对应一个活跃 SSE 连接
- worker 侧保持与当前轮询版本一致的处理节奏，尽量单条投递、单条完成
- 服务端不要在第一阶段做多条并发推送，避免现有 `complete` 语义失真
- 断线重连采用“新连接顶掉旧连接”的简单策略
- 如无明确需要，第一阶段可以先不新增独立 `heartbeat` 接口，先依赖 SSE 长连接存活和超时回收

第一阶段的收益是：

- 改动面小
- 现有 worker 和网关模型容易迁移
- 能先验证 SSE 在网络、代理、部署环境中的可用性

第一阶段的限制也要明确接受：

- 不能优雅区分“同一 worker_id 的两个进程实例”
- 不适合多条消息并发执行
- 重连后的重复投递与幂等能力较弱
- `complete` 仍然缺少显式任务关联键

换句话说，第一阶段适合“先把链路从轮询改成长连接推送”，但不追求一次性解决并发、去重、lease、恢复这些问题。

#### 第二阶段：补齐可靠性字段和投递语义

第二阶段再引入增强能力：

- `client_instance_id`
- `delivery_id`
- `message_id`
- `heartbeat`
- `max_inflight`
- lease / redelivery / 幂等
- `complete.status=completed|failed`

这时协议才从“能用的 SSE 传输层”升级为“可靠的 SSE 投递层”。

## 5. 协议调整建议

## 5.1 保留并增强 `online`

下面这一节开始描述的是“完整增强版目标协议”，也就是更推荐的第二阶段终态。

如果先走第一阶段，可以把这里的增强字段视为后续扩展项：

- 第一阶段 `online` 仍然只传现有 `worker_id`
- 第一阶段 `stream` 只带 `worker_id`
- 第一阶段 `complete` 仍然只带 `worker_id + final_content`

### 请求

```json
{
  "worker_id": "user-123",
  "client_instance_id": "wk_01H...",
  "protocol": "sse",
  "max_inflight": 4,
  "worker_version": "0.1.0"
}
```

### 响应

```json
{
  "ok": true,
  "stream_path": "/api/worker/stream",
  "heartbeat_interval_seconds": 15,
  "session_ttl_seconds": 45,
  "max_inflight": 4
}
```

说明：

- `worker_id` 仍然表示稳定绑定身份，通常对应用户或固定 worker 标识。
- `client_instance_id` 是单次 worker 进程实例 ID，用于区分“同一个 worker_id 的不同进程”。
- `protocol` 用于灰度兼容，服务端可同时支持 `poll` 和 `sse`。
- `max_inflight` 用于显式限流，避免 SSE 一口气推太多任务。

## 5.2 新增 `GET /api/worker/stream`

建议使用：

- Method: `GET`
- Headers:
  - `Authorization`
  - `Accept: text/event-stream`
  - `Cache-Control: no-cache`
  - `Last-Event-ID`，可选
- Query:
  - `worker_id`
  - `client_instance_id`

连接建立成功后，服务端持续推送 SSE 事件。

### 建议事件类型

#### `ready`

表示连接建立成功。

```text
event: ready
data: {"connection_id":"conn_01H...","worker_id":"user-123","client_instance_id":"wk_01H..."}
```

#### `heartbeat`

用于保持链路活性，也可辅助客户端判断网关是否静默断开。

```text
event: heartbeat
data: {"ts":"2026-04-13T10:00:00Z"}
```

#### `delivery`

真正的任务投递事件。

```text
id: dly_01HXYZ
event: delivery
data: {"delivery_id":"dly_01HXYZ","message_id":"msg_01HABC","worker_id":"user-123","content":"hello from im","created_at":"2026-04-13T10:00:00Z"}
```

#### `drain`

可选事件，表示当前队列已清空，仅作为观测信号。

```text
event: drain
data: {"queued":0}
```

#### `error`

可选事件，用于表达业务级错误，例如重复连接、实例被顶替等。

```text
event: error
data: {"code":"duplicate_connection","message":"worker replaced by newer instance"}
```

服务端在发送 `error` 后可主动关闭连接。

## 5.3 新增 `heartbeat`

轮询模式下，`poll` 同时承担“拉取消息”和“刷新 last_seen”的作用。改成 SSE 后，这两件事要分开。

建议新增：

- `POST /api/worker/heartbeat`

### 请求

```json
{
  "worker_id": "user-123",
  "client_instance_id": "wk_01H...",
  "connection_id": "conn_01H..."
}
```

### 响应

```json
{
  "ok": true
}
```

用途：

- 刷新在线状态 TTL
- 刷新 worker session lease
- 给 gateway 一个稳定的 token 刷新响应机会
- 让服务端知道当前活动实例是谁

## 5.4 强烈建议升级 `complete`

这是本次协议升级里最重要的补强点之一。

当前 `complete` 只传：

```json
{
  "worker_id": "...",
  "final_content": "..."
}
```

这在串行场景勉强成立，但在以下场景会有风险：

- 同一 worker 并发处理多个请求
- SSE 重连导致重复投递
- 某个任务执行很慢，另一个任务先完成
- 老连接和新连接短时间并存

因此推荐把 `complete` 改成按 `delivery_id` 应答。

### 请求

```json
{
  "worker_id": "user-123",
  "client_instance_id": "wk_01H...",
  "delivery_id": "dly_01HXYZ",
  "status": "completed",
  "final_content": "done"
}
```

### 响应

```json
{
  "ok": true
}
```

如果希望顺手补齐失败语义，建议统一成：

```json
{
  "worker_id": "user-123",
  "client_instance_id": "wk_01H...",
  "delivery_id": "dly_01HXYZ",
  "status": "failed",
  "final_content": "",
  "error_code": "LOCAL_EXECUTION_ERROR",
  "error_message": "..."
}
```

这样可以避免“worker 本地处理失败后，远端 inflight 一直悬挂”的问题。

## 5.5 保留 `offline`

`offline` 仍然保留，用于：

- worker 优雅退出时显式下线
- 提前释放连接和投递 lease
- 减少 TTL 被动回收的等待时间

## 6. 服务端状态模型建议

为了让 SSE 可恢复、可去重、可并发，服务端要从“按 worker_id 的简化 FIFO”升级为“显式 delivery 模型”。

建议至少维护 3 类状态。

### 6.1 Worker Session

按 `worker_id + client_instance_id` 跟踪：

- `worker_id`
- `client_instance_id`
- `connection_id`
- `protocol`
- `status`
- `connected_at`
- `last_heartbeat_at`
- `last_stream_event_at`
- `max_inflight`

### 6.2 Delivery Queue

每条待处理消息都应有独立标识：

- `delivery_id`
- `message_id`
- `worker_id`
- `content`
- `status`
  - `queued`
  - `leased`
  - `completed`
  - `failed`
- `leased_to_client_instance_id`
- `leased_at`
- `lease_expires_at`
- `completed_at`

### 6.3 Completion Log

保留结果与审计信息：

- `delivery_id`
- `worker_id`
- `client_instance_id`
- `final_status`
- `final_content`
- `error_code`
- `error_message`
- `completed_at`

## 7. Worker 侧运行模型建议

## 7.1 `gateway_client` 分层

建议将 worker gateway client 拆成两个明确职责：

- 控制面 API
  - `online()`
  - `heartbeat()`
  - `complete()`
  - `offline()`
- 数据面 API
  - `stream_events()`

这样做的好处是：

- SSE 连接生命周期和普通 POST 重试逻辑更清晰
- 现有 Authorization 刷新逻辑更容易复用
- 后续即使保留 poll 模式，也能在同一个 client 中并存

## 7.2 `runner` 从 poll loop 改为 connection supervisor

当前 `runner` 的主逻辑是：

- online
- while true: poll
- 收到消息后 create_task 处理
- 退出时 offline

切到 SSE 后，建议改成：

1. `online`
2. 启动 `heartbeat` 后台任务
3. 建立 SSE 连接
4. `async for event in stream_events()`
5. 收到 `delivery` 后派发本地处理任务
6. 连接断开后按退避策略重连
7. 退出时关闭心跳并 `offline`

## 7.3 本地 bridge 保持不变

`FileBridgeStore` 的价值仍然成立，不需要因为远程协议改为 SSE 而推翻。

worker 收到 `delivery` 事件后仍然：

1. 把 `content` 写入本地 bridge
2. `source="remote"`
3. `source_meta` 中增加：
   - `worker_id`
   - `delivery_id`
   - `message_id`
   - `client_instance_id`
4. 标记 `remote_response_required=true`

本地 CLI 的处理链路可以保持原样。

## 7.4 Worker 侧去重

SSE 重连后，`delivery` 事件可能重复投递。worker 必须以 `delivery_id` 去重。

建议 runner 在内存中维护：

- `delivery_id -> request_id`
- `delivery_id -> task`
- `delivery_id -> latest_status`

处理策略：

- 如果同一 `delivery_id` 已在处理中，则忽略重复 `delivery`
- 如果同一 `delivery_id` 已有本地结果但 `complete` 可能没成功，则直接重试 `complete`
- 如果同一 `delivery_id` 已完成且远端已确认，则彻底忽略

这一步很关键，否则 SSE 重连会导致本地重复落盘、重复执行。

## 8. 鉴权与 token 刷新建议

当前 worker 客户端的一个重要语义是：

- 请求携带 `Authorization`
- 如果响应头返回新的 `Authorization`，则更新内存并持久化
- 如果请求失败且拿到了新 token，则重试一次

这套机制建议保留，但要扩展到 SSE 建连过程。

## 8.1 对控制面请求

继续沿用当前逻辑：

- `online`
- `heartbeat`
- `complete`
- `offline`

这些响应仍然允许返回新的 `Authorization` 头。

## 8.2 对 SSE 建连请求

建议也支持相同逻辑：

1. worker 发起 `GET /api/worker/stream`
2. 若服务端返回错误且响应头带新 `Authorization`
3. worker 更新 token
4. 建连重试一次

## 8.3 不建议依赖 SSE 事件体下发新 token

虽然技术上可以在 SSE `data` 中下发 token，但不建议作为主方案：

- 安全语义不如响应头自然
- 日志与链路抓包更难控
- 会让“控制面”和“数据面”职责混杂

更稳妥的方式是：

- token 刷新主要通过 `heartbeat` 和其他 HTTP 响应完成
- SSE 连接失效时按最新 token 重连

## 9. 重连、恢复与幂等

SSE 最大的设计重点不在“怎么连上”，而在“断了以后怎么恢复还不乱”。

## 9.1 连接重试

建议采用指数退避，示例：

- 第 1 次：1s
- 第 2 次：2s
- 第 3 次：5s
- 上限：15s 或 30s

在这些场景触发重连：

- 网络断开
- 服务端主动关闭连接
- 读取超时
- 5xx

在这些场景不无限重连，而是先刷新状态：

- 401/403
- `duplicate_connection`
- `worker_revoked`

## 9.2 `Last-Event-ID`

建议支持，但不要把它当作唯一恢复机制。

原因是：

- `Last-Event-ID` 适合处理“连接抖动后的短时续传”
- 但 worker 进程重启后，未必还保留完整上下文
- 真正可靠的恢复仍然要靠服务端 delivery lease 和 `delivery_id` 幂等

因此推荐：

- SSE 层支持 `Last-Event-ID`
- 投递层仍然以 `delivery_id` 为准做去重和补偿

## 9.3 Delivery Lease

服务端在推送 `delivery` 后，不应立刻把任务视为完成，而应进入 `leased` 状态。

只有在收到：

- `complete(status=completed)`
- `complete(status=failed)`

之后，才结束该 delivery。

如果连接断开或超时，则：

- 对未完成 delivery 进行 lease 过期回收
- 重新回到 `queued`
- 等待新的 SSE 连接重新投递

## 10. 背压与并发控制

轮询协议天然有一个“慢速阀门”：worker 只有发起 `poll`，服务端才会给下一条。

切成 SSE 后，如果不加控制，服务端可能会瞬间把大量消息全部推给 worker。

因此建议显式增加背压机制。

## 10.1 推荐方案

worker 在 `online` 时声明：

- `max_inflight`

服务端只在下面条件成立时继续推送：

- 当前 `leased` 且未完成的 delivery 数量 `< max_inflight`

这样可以避免：

- 本地 bridge 短时间堆积太多任务
- 进程重启时遗留大量未完成任务
- 某个 worker 被瞬时流量压垮

## 10.2 完成顺序不再假设 FIFO

改为 SSE 之后，甚至在现有轮询并发场景下，也不应该再假设“谁先投递谁先完成”。

因此服务端关联完成结果时必须依赖：

- `delivery_id`

而不能再依赖：

- “当前 worker_id 的 inflight FIFO 队头”

这是本方案里必须落地的协议修正。

## 11. 失败处理建议

当前 `runner._process_message()` 在本地等待结果异常时会直接返回，这意味着远端任务可能一直悬挂。

虽然本次是设计方案，不落代码，但协议层建议顺手补齐：

- `complete(status=failed, error_code, error_message)`

常见失败类别可以约定：

- `LOCAL_EXECUTION_ERROR`
- `LOCAL_TIMEOUT`
- `WORKER_SHUTTING_DOWN`
- `UNSUPPORTED_COMMAND`
- `AUTH_EXPIRED`

这样服务端可以决定：

- 直接回失败给 IM
- 标记为待人工处理
- 重新入队重试

## 12. 推荐时序

## 12.1 正常链路

```text
worker -> gateway: POST /api/worker/online
gateway -> worker: ok + stream_path + heartbeat interval

worker -> gateway: GET /api/worker/stream
gateway -> worker: SSE ready

IM -> gateway: inbound message
gateway -> worker: SSE delivery(delivery_id=d1)

worker -> local bridge: enqueue request
CLI -> local bridge: write result

worker -> gateway: POST /api/worker/complete(delivery_id=d1)
gateway -> worker: ok
gateway -> IM: final response
```

## 12.2 断线恢复

```text
worker <x> gateway: SSE disconnected
worker -> gateway: reconnect stream with Last-Event-ID
gateway: find leased but uncompleted deliveries
gateway -> worker: redeliver missing delivery
worker: by delivery_id dedupe or resume complete
```

## 13. 兼容与灰度发布建议

建议按你说的方式，拆成两个主阶段推进。

### Phase 1: 最小改造版 SSE

这一阶段只做“下行从 poll 改成 SSE”，尽量不引入新字段。

协议原则：

- 保持现有 `worker_id`
- 不引入 `client_instance_id`
- 不引入 `delivery_id`
- `complete` 继续使用现有结构
- `online / complete / offline` 请求体尽量不改

接口建议：

- 保留 `POST /api/worker/online`
- 新增 `GET /api/worker/stream?worker_id=...`
- 保留 `POST /api/worker/complete`
- 保留 `POST /api/worker/offline`
- `POST /api/worker/poll` 暂时保留作为回退通道

服务端行为建议：

- 一个 `worker_id` 只允许一个活跃 SSE 连接
- 新连接建立时，旧连接直接失效
- 一次只推一条消息，等待 `complete` 之后再推下一条
- 先不要做多 inflight
- 先不要做显式 lease 恢复

worker 行为建议：

- 保持现有单 worker_id 语义
- 收到 SSE 消息后仍走本地 `FileBridgeStore`
- 完成后仍按现有 `complete(worker_id, final_content)` 回传
- 断线后直接重连，不做复杂续传

第一阶段的目标不是把协议做完整，而是验证：

- SSE 在你的网关和部署环境里可不可用
- worker 主循环能否稳定从 poll 迁到 stream
- 现有 bridge 执行链路是否可以无感承接 SSE 下发

### Phase 2: 补齐可靠性和身份模型

第一阶段跑稳之后，再补下面这些字段和机制：

- `client_instance_id`
- `delivery_id`
- `message_id`
- `complete.status`
- `heartbeat`
- `max_inflight`
- `Last-Event-ID`
- lease / redelivery / 幂等

第二阶段的目标是解决第一阶段故意暂时不处理的问题：

- 同一 `worker_id` 多实例区分
- 多消息并发关联
- 重连后的去重与恢复
- 服务端 inflight 任务可回收
- 失败结果可显式上报

### 灰度顺序

两个主阶段内部，建议按这个顺序灰度：

1. 服务端同时支持 `poll` 和 `sse`
2. worker 增加配置开关，允许 `poll|sse` 切换
3. 先让少量 worker 使用第一阶段 SSE
4. 第一阶段稳定后，再上线第二阶段增强字段
5. 第二阶段稳定后，再考虑下线 `POST /api/worker/poll`

## 14. 测试建议

虽然这里不落代码，但后续实现时建议至少覆盖这些测试面。

### 14.1 客户端测试

- SSE 建连成功
- SSE 心跳事件可持续消费
- `delivery` 事件能正确写入本地 bridge
- 断线后自动重连
- 401 + 新 Authorization 时建连重试成功
- 重复 `delivery_id` 不会重复执行

### 14.2 服务端测试

- `online` 后建立流连接
- 同 worker_id 新实例上线时旧连接处理策略正确
- `delivery` 推送后进入 `leased`
- `complete(delivery_id)` 正确落账
- lease 超时后任务重新入队
- `max_inflight` 生效

### 14.3 集成测试

- 单条消息正常收发
- 多条消息并发处理并按 `delivery_id` 正确完成
- worker 处理中途断线，重连后不丢消息
- worker 本地执行失败时服务端正确感知

## 15. 风险与取舍

## 15.1 主要收益

- 降低空轮询开销
- 提升消息下发实时性
- 在线状态与消息下发职责更清晰
- 为后续可观测性、去重、重放、失败补偿打基础

## 15.2 主要风险

- SSE 长连接会暴露代理、LB、超时配置问题
- 如果没有 `delivery_id` 和 lease，重连后极易重复或丢消息
- 如果没有 `max_inflight`，服务端可能过度推送
- 如果不补失败回传，服务端会出现悬挂任务

## 15.3 核心取舍

这次最推荐的取舍不是“只把 poll 改成 SSE 就结束”，而是：

- 下行链路改 SSE
- 上行链路保留 HTTP
- 顺手补齐 `delivery_id + heartbeat + lease + 幂等`

因为真正复杂的地方不在传输格式，而在可靠投递语义。

## 16. 最终建议

如果你希望在现有代码结构上低风险演进，我现在更建议按两阶段来做：

1. 第一阶段先不引入 `client_instance_id`，也不引入 `delivery_id`
2. 第一阶段只把下行从 `poll` 改成 `GET /api/worker/stream`
3. 如果真实后端不维护任务状态，可以先保持“单 `worker_id` 单活跃 stream + worker 本地 pending 队列”的简化模型
4. `online / complete / offline` 先尽量复用现有字段和实现
5. 本地 `FileBridgeStore` 和 CLI 执行链路尽量不动
6. 第二阶段再补 `client_instance_id + delivery_id + heartbeat + lease + 幂等`
7. 第二阶段稳定后，再考虑彻底下线轮询

一句话总结：

最务实的改法是“先把 poll 换成 SSE，再视后端复杂度决定是否补强协议”。如果后端当前不维护任务状态，第一阶段可以先用简化模型跑起来；第二阶段再解决 delivery、实例身份、幂等和精确恢复。
