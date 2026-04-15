# Worker SSE 第一阶段方案

## 1. 目标

第一阶段的目标很明确：

- 将 worker 与远程 gateway 的下行通信从 HTTP 轮询改为 SSE
- 尽量复用现有协议字段和现有代码结构
- 不引入新的身份字段
- 不在这一阶段解决并发投递、强幂等、任务重放、多实例区分等增强问题

一句话概括：

第一阶段只解决“传输方式从 poll 变 stream”，不解决“协议可靠投递模型升级”。

## 2. 当前现状

当前 worker 侧链路是：

1. `POST /api/worker/online`
2. 持续 `POST /api/worker/poll`
3. gateway 返回 `messages`
4. worker 把消息写入本地 `FileBridgeStore`
5. 本地 CLI 处理完成后，worker 调用 `POST /api/worker/complete`
6. worker 退出时调用 `POST /api/worker/offline`

当前协议的关键特征：

- 只有 `worker_id`
- 没有 `client_instance_id`
- 没有 `delivery_id`
- `complete` 不显式关联某一条 delivery
- 现有模型本质上更偏串行、单 worker 实例

## 3. 第一阶段范围

第一阶段只做下面这些事：

- 保留 `worker_id` 作为唯一 worker 身份
- 保留 `online / complete / offline` 这几个现有接口
- 新增 `GET /api/worker/stream`
- worker 主循环从 `poll()` 改成消费 SSE 事件
- 收到消息后的本地处理链路仍然走 `FileBridgeStore`

第一阶段明确不做：

- 不新增 `client_instance_id`
- 不新增 `delivery_id`
- 不新增 `message_id`
- 不引入多 inflight 并发投递
- 不做严格 lease / redelivery
- 不做强幂等恢复
- 不把 `complete` 改成显式任务确认协议

## 4. 第一阶段预期效果

### 4.0 补充前提

结合当前真实后端的实现方式，第一阶段可以采用更简化的前提：

- 后端不维护任务状态
- 后端不区分 queued / inflight / completed
- 只要 worker 通过 `poll` 或 `stream` 收到消息，后端就视为已经下发
- 收到 `complete` 后，后端直接把结果发回 IM
- 后端不需要把 `complete` 精确关联到某一条任务

这意味着第一阶段的重点可以收敛为：

- 把下行从 `poll` 改成 SSE
- 保持单 `worker_id` 单活跃 SSE 连接
- worker 持续接收消息并落到本地 pending
- 本地 CLI 按自己的节奏串行执行

在这个前提下，第一阶段不需要强行引入：

- `delivery_id`
- `client_instance_id`
- 服务端 inflight / lease / ack 状态机

做完之后，可以达到：

- 消息下发变成实时推送，不再依赖下一次轮询
- 空闲时减少轮询请求
- worker 与 gateway 之间保持一个长连接
- 本地 bridge 和 CLI 执行模型基本不变
- 网关可以开始验证 SSE 在真实部署环境中的可用性

但也必须接受这些限制：

- 一个 `worker_id` 仍然只能按“单实例”理解
- 同一时刻只保留一个活跃 SSE 连接
- 后端不负责判断某条任务是否仍在执行
- 远程消息可以持续进入本地 pending，但本地执行节奏仍由 CLI 自己控制
- 断线恢复能力有限
- `complete` 关联关系仍然偏弱

## 5. 协议设计

## 5.1 保留现有接口

第一阶段继续保留：

- `POST /api/worker/online`
- `POST /api/worker/complete`
- `POST /api/worker/offline`
- `POST /api/worker/poll`

其中：

- `poll` 第一阶段先不删除，只作为兼容回退通道
- worker 默认可切到 SSE，但系统保留 poll 回退能力

## 5.2 新增 SSE 接口

新增：

- `GET /api/worker/stream`

建议参数：

- Query:
  - `worker_id`
- Headers:
  - `Authorization`
  - `Accept: text/event-stream`
  - `Cache-Control: no-cache`

第一阶段不要求：

- `Last-Event-ID`
- `client_instance_id`
- 显式 `connection_id`

## 5.3 SSE 事件模型

第一阶段建议只保留最小事件集合。

### `ready`

表示连接建立成功。

```text
event: ready
data: {"worker_id":"user-123"}
```

### `message`

表示下发一条待处理消息。

```text
event: message
data: {"content":"hello from im"}
```

### `heartbeat`

可选。第一阶段可以支持一个轻量心跳事件，但不强依赖独立 heartbeat 接口。

```text
event: heartbeat
data: {"ts":"2026-04-13T10:00:00Z"}
```

### `error`

可选。用于通知连接被替换或服务端准备关闭连接。

```text
event: error
data: {"code":"replaced","message":"connection replaced by newer stream"}
```

## 6. 服务端行为约束

为了在不引入新字段的前提下让第一阶段可控，服务端需要主动收紧行为。

### 6.1 单 worker_id 单活跃连接

同一时刻，一个 `worker_id` 只允许一个活跃 SSE 连接。

如果同一个 `worker_id` 再次建立新连接，建议：

- 新连接成功
- 旧连接被服务端主动关闭

这样可以避免因为没有 `client_instance_id` 而导致实例身份混乱。

### 6.2 单条推送、串行完成

第一阶段建议服务端采用最保守策略：

- 一次只向某个 `worker_id` 推送一条消息
- 在收到该 `worker_id` 的 `complete` 之前，不继续推下一条

这样做的原因很直接：

- 当前 `complete` 只有 `worker_id + final_content`
- 如果同一 worker 同时处理多条消息，服务端无法可靠知道 `complete` 对应哪一条

### 6.3 简单断线回收

第一阶段不做复杂 lease。

服务端可以采用简单规则：

- SSE 连接断开后，将当前 worker 状态标记为非活跃
- 未明确完成的消息重新回到待投递状态，或保持等待下一次连接再投递

这里不追求严格的一次且仅一次语义，只追求系统能继续工作。

## 7. Worker 侧行为约束

## 7.1 主循环调整

worker 主循环改为：

1. `online`
2. 建立 `GET /api/worker/stream`
3. `async for event in stream`
4. 收到 `message` 后写入 `FileBridgeStore`
5. 等本地结果
6. 调用现有 `complete`
7. 断线后重连
8. 退出时 `offline`

## 7.2 仍然保持串行语义

虽然技术上 SSE 可以持续推消息，但第一阶段 worker 应主动保持与现状一致的节奏：

- 不主动支持多条远程消息并发完成
- 收到一条，处理一条，完成一条

这样才能与现有 `complete` 语义匹配。

## 7.3 本地 bridge 不改模型

worker 收到 SSE `message` 后，继续沿用现在的桥接方式：

- `enqueue_text(...)`
- `source="remote"`
- `source_meta` 里仍然只放现有 `worker_id`
- `remote_response_required=True`

这意味着第一阶段基本不需要改本地 CLI 处理模型。

## 8. 鉴权策略

第一阶段继续沿用现有 `Authorization` 机制。

要求：

- SSE 建连时携带 `Authorization`
- `online / complete / offline` 继续支持响应头刷新 token
- SSE 建连失败时，如果响应头带新 token，worker 可以更新后重新建连

第一阶段不要求：

- 在 SSE 事件体中刷新 token
- 单独设计新的鉴权模型

## 9. 重连策略

第一阶段只采用简单重连策略，不做复杂续传。

建议：

- 断线后按固定间隔或简单退避重连
- 重连后重新订阅 `worker_id` 对应的 stream
- 服务端按当前队列状态重新决定是否推送消息

第一阶段不要求：

- `Last-Event-ID`
- 精确续传
- 去重恢复

## 10. 风险与边界

第一阶段最大的价值是低风险，但代价是能力边界比较明显。

### 10.1 已接受的边界

- 不保证并发消息关联正确
- 不保证强幂等
- 不保证多实例场景优雅处理
- 不保证断线后零重复、零丢失

### 10.2 适用场景

第一阶段适合：

- 一个 `worker_id` 基本只会启动一个 worker 进程
- 远程消息量不高
- 允许消息先进入本地 pending，再由 CLI 串行处理
- 当前最迫切需求是降低轮询开销、提高下发实时性

### 10.3 不适合场景

第一阶段不适合：

- 同一 `worker_id` 高并发消息处理
- 需要严格消息去重
- 需要精确恢复 inflight 任务
- 需要区分多个 worker 进程实例

## 11. 第一阶段实施清单

### 11.1 服务端

- 新增 `GET /api/worker/stream?worker_id=...`
- 为某个 `worker_id` 维护 SSE 长连接
- 新消息到达时向对应 stream 推送 `message`
- 同一 `worker_id` 新连接建立时关闭旧连接
- 保持 `online / complete / offline` 兼容
- 暂时保留 `poll`

### 11.2 Worker

- `gateway_client` 增加 SSE 建连与事件读取能力
- `runner` 从 poll loop 改成 stream consume loop
- 收到 `message` 后立即写入本地 bridge / pending
- `stream` 自己继续读取后续消息，不必等待前一条任务 `complete`
- 本地结果完成后继续调用现有 `complete`
- 增加断线重连

### 11.3 配置与灰度

- 增加 worker 运行模式开关：`poll` 或 `sse`
- 默认可以先灰度给小范围 worker
- 发生问题时可快速切回 `poll`

## 12. 验收标准

第一阶段完成后，建议至少满足这些验收标准：

### 功能验收

- worker 启动后可成功建立 SSE 连接
- 远程消息可通过 SSE 推送到 worker
- worker 可继续通过本地 bridge 触发 CLI 处理
- worker 在当前任务执行期间仍可继续接收新消息并落到 pending
- 最终结果仍可通过现有 `complete` 回传
- worker 退出时可正常 `offline`

### 稳定性验收

- SSE 空闲保持连接时无异常刷屏
- 连接断开后 worker 可自动重连
- 同一个 `worker_id` 重连时，新连接可替换旧连接
- 发生问题时 worker 可切回 `poll` 模式

### 边界验收

- 在第一阶段约束下，服务端不会对同一 worker 做多条并发推送
- worker 不会因为 SSE 改造破坏现有本地 bridge 行为

## 13. 第一阶段完成后的系统状态

第一阶段完成后，系统会处于这样一种状态：

- 传输层已经从轮询切换到长连接推送
- 协议身份仍然是简单 `worker_id`
- 处理模型仍然偏单实例、串行
- 系统已经可以享受 SSE 的实时性收益
- 但协议还没有升级成完整的可靠投递模型

这意味着第一阶段结束后，最自然的下一步就是第二阶段：

- 补 `client_instance_id`
- 补 `delivery_id`
- 补 `heartbeat`
- 补 lease / redelivery / 幂等
- 把协议从“能跑的 SSE”升级为“可靠的 SSE”

## 14. 结论

第一阶段是一个很适合先落地的版本，因为它只改变“消息怎么下来”，不急着改变“消息怎么被可靠地管理”。

它适合先快速达成两个结果：

- 验证 SSE 在你们环境中是否稳定可用
- 验证 worker 侧从 `poll` 迁移到 `stream` 的工程成本是否可控

如果第一阶段跑稳，第二阶段再去补字段和可靠性语义，会更稳，也更容易评审。
