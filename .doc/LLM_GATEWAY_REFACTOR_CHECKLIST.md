# LLM Gateway 改造清单与建议

## 目标

围绕 Gateway 建立一套可追踪的请求归属链路，明确区分以下两个核心概念：

- `user_id`：登录后获得的用户身份标识。
- `worker_no`：CLI 终端实例标识。一个用户可同时拥有多个 `worker_no`。

同时统一会话标识：

- `session_id`：CLI 本地与 Server/Gateway 远端共用的会话标识。

目标链路如下：

`user_id -> worker_no -> session_id -> model request / stream response / runtime trace`

## 现状判断

当前代码里已经存在一部分基础能力，但语义上还没有完全对齐：

- 现有 `worker_id` 更接近登录身份标识，语义上应理解为 `user_id`，不是终端实例 ID。
- CLI 本地已有 `session_id` / `conversation_session_id`，用于本地会话历史与恢复。
- Python server 侧已有 `session_id` 管理能力。
- LLM gateway 请求当前还没有显式传递 `worker_no`、统一后的 `session_id`。

因此，后续改造重点不是复用现有 `worker_id` 作为 `worker_no`，而是在现有登录身份之外新增终端实例标识，并让本地/远端共用同一个 `session_id`。

## 改造原则

### 1. 先纠正语义，再扩展协议

建议先统一概念：

- 现有登录接口返回的 `worker_id`，内部按 `user_id` 语义使用。
- 新增真正的 `worker_no`，表示终端实例。
- 新增或显式化 `session_no`，表示 Gateway / Server 侧会话。

### 2. 本地与远端统一使用一个 `session_id`

当前方案不再区分 `local_session_id` 和 `remote_session_no`。

统一规则：

- CLI 本地会话 ID = Gateway/Server 远端会话 ID = `session_id`
- 同一条会话内，所有 LLM 请求都必须携带同一个 `session_id`
- `/resume` 不需要专门接口，只要后续请求继续携带该 `session_id` 即可

旧历史会话暂不强制升级。由于后端按 `user_id` 维度隔离会话，历史本地 `session_id` 可以继续沿用。

### 3. 关键信息使用显式字段传递

不建议将 `worker_no`、`session_id` 临时塞进 `metadata`。

建议在请求模型和 SSE 协议中提供明确字段，便于：

- 网关日志检索
- 服务端校验
- 协议长期稳定
- 前后端联调

## 详细改造清单

## 一、字段语义与命名清理

### 目标

避免当前 `worker_id` 命名继续误导后续实现。

### 建议

- 文档层面统一：
  - `worker_id`（现状）按 `user_id` 语义理解
  - `worker_no` 表示终端实例
  - `session_no` 表示远端会话
- 代码层面逐步清理注释、日志和变量名。
- 第一阶段可以保留外部兼容字段，内部先用更准确的命名表达语义。

### 优先级

高。建议最先处理。

## 二、CLI 新增 `worker_no` 生命周期

### 目标

为每个 CLI 终端实例分配唯一标识，并在运行期间稳定使用。

### 建议

- CLI 启动时生成 `worker_no`。
- 将 `worker_no` 持久化到本地运行态目录，或明确为“每次进程启动生成一个新值”。
- 需要明确以下策略：
  - 同一进程重连时是否复用
  - 重新启动 CLI 后是否复用
  - 同一用户多开多个终端时如何避免冲突

### 推荐方案

第一版建议采用简单方案：

- `worker_no` 作为“单次 CLI 运行实例 ID”
- CLI 启动时生成
- 本次进程生命周期内保持不变
- 进程退出后失效

优点：

- 实现简单
- 多终端天然不冲突
- 能满足请求归属与轨迹关联的第一阶段需求

如果未来需要识别“同一个物理终端/安装实例”的长期身份，再引入第二层稳定标识。

## 三、Worker Gateway 协议补充 `worker_no`

### 目标

让 Gateway 在 worker 连接、收消息、发结果时都能识别是哪个终端实例。

### 建议

在当前 worker 协议中补充 `worker_no`：

- `/online`
- `/poll`
- `/stream`
- `/progress`
- `/complete`

### 语义建议

- 登录态/认证头：识别 `user_id`
- 请求 body/query：传递 `worker_no`

这样可以同时表达：

- 这个请求属于哪个用户
- 这个请求来自该用户的哪个终端实例

## 四、LLM 请求模型补充 `worker_no` / `session_id`

### 目标

让每一次模型请求都携带完整归属信息。

### 建议

在 LLM gateway 的请求模型中增加显式字段：

- `worker_no: str`
- `session_id: str | None`
- 可选：`user_id: str | None`

### 理由

- 满足文档“模型请求入参要求”
- 便于 Gateway 基于 `worker_no + session_id` 记录链路
- 避免将关键归属信息埋在不稳定的扩展字段里

## 五、`session_id` 创建与 SSE 回传

### 目标

支持首次请求自动建会话，并通过流式协议通知 CLI 当前会话号。

### 建议

当 CLI 首次请求未携带 `session_id` 时：

- Server/Gateway 创建新的 `session_id`
- 尽早通过 SSE 返回给 CLI
- CLI 本地保存该值

当 CLI 后续请求已携带 `session_id` 时：

- Server/Gateway 复用该会话
- 可选回送一次确认事件

### 推荐 SSE 事件

建议新增事件类型：

```json
{"type":"session","session_id":"xxx","is_new":true}
```

说明：

- `type = "session"`：固定事件名称，简单直接
- `session_id`：统一后的本地/远端会话标识
- `is_new`：是否为本次新建

### 备注

不建议通过普通文本消息、响应 header 或隐式字段传递 `session_id`，否则 CLI 处理会变复杂且易出错。

## 六、CLI 本地维护当前统一 `session_id`

### 目标

让 CLI 在运行过程中始终知道“当前会话 ID 是什么”。

### 建议

CLI 运行态只维护一个当前值：

- `session_id`

分工如下：

- 用于本地历史
- 用于 `/resume`
- 用于后续发往 Gateway 的模型请求

### 结论

不维护双字段，不维护本地/远端映射表。

## 七、`/new` 命令改造

### 目标

用户发起新会话时，本地与远端都进入新的会话上下文。

### 建议

CLI 执行 `/new` 时：

- 本地继续执行现有“切换新本地会话”的逻辑
- 同时调用 Gateway `/new`（或等价接口）
- 获取新的 `session_id`
- 更新当前 `session_id`

### 推荐行为

当前建议：

- `/new` 时由后端生成新的统一 `session_id`
- CLI 本地直接切换到这个新的 `session_id`

这样用户感知最一致，也最容易解释和调试。

## 八、`/resume <session_id>` 改造

### 目标

恢复本地历史后，让后续请求继续写入同一条远端会话。

### 建议

当前方案下，`/resume` 不需要单独的服务端接口。

处理方式：

1. CLI 根据本地 `session_id` 恢复本地历史
2. CLI 将当前会话切换为该 `session_id`
3. 后续所有 LLM 请求继续携带这个 `session_id`
4. 后端按同一个 `session_id` 继续写入历史

### 建议结论

- 不新增 `/resume` 专门接口
- 不维护本地/远端映射表
- `/resume` 的本质只是“切换当前会话 ID”

## 九、旧历史会话兼容策略

### 目标

兼容当前已经保存在本地、但并非由后端生成的历史 `session_id`。

### 当前结论

旧历史会话暂不需要升级。

原因：

- 旧历史会话的 `session_id` 由 CLI 本地生成
- 后端按 `user_id` 维度隔离会话，不是单纯依赖全局 `session_id`
- 因此当前可以接受旧本地 `session_id` 继续沿用

### 注意点

- 旧本地 `session_id` 的生成方式是短 UUID，理论上存在重复概率
- 但在当前“按用户隔离”的前提下，风险可接受
- 后续如后端改为全局唯一 `session_id` 约束，再评估是否需要统一迁移旧会话

## 十、服务端 Session 管理语义收敛

### 目标

让服务端已有 `session_id` 管理能力平滑承载新的 `session_no` 语义。

### 建议

当前服务端如果已经具备：

- 未传 session 时自动创建
- 传 session 时继续复用

那么可以直接将服务端内部 `session_id` 作为对外统一会话 ID。

### 好处

- 降低第一版改造面
- 避免重复实现一套新的 session 管理
- 先把对外协议补齐，再考虑内部命名整理

## 十一、SSE 协议扩展

### 目标

让 CLI 能稳定感知会话创建/复用结果。

### 建议

在现有 SSE 事件之外，新增：

- `session`

可选后续再增加：

- `trace_context`

第一阶段只建议加 `session`，避免协议过度设计。

## 十二、日志与可观测性改造

### 目标

让后续排查“某用户、某终端、某会话”的问题有清晰抓手。

### 建议

CLI、Gateway、Python server 三端日志统一带上：

- `user_id`
- `worker_no`
- `session_no`
- `request_id`
- `model`

### 效果

后续可以很方便地回答：

- 同一个用户是否多开终端
- 某个终端当前对应哪个会话
- 某次模型请求落在哪个会话下
- 某次异常响应来自哪个终端实例

## 十三、测试补充建议

### 必测场景

- 首次请求不带 `session_id` 时，服务端新建并返回 `session` 事件
- 后续请求带相同 `session_id` 时，服务端复用会话
- `/new` 后当前 `session_id` 发生切换
- 同一 `user_id` 下多个 `worker_no` 并存时不串线
- `/resume` 恢复本地会话后，后续请求继续沿用该 `session_id`
- 旧历史会话在 `/resume` 后可直接继续沿用原本地 `session_id`
- Gateway 日志或事件中能正确记录 `worker_no + session_id`

## 分期建议

## 第一期：最小闭环

目标：先打通 `worker_no + session_id` 的基本透传与回传。

建议内容：

- 明确 `worker_id` / `user_id` / `worker_no` 语义
- CLI 生成 `worker_no`
- LLM 请求显式携带 `worker_no`、`session_id`
- `/llm/query-stream` 在首次请求时返回 `session` SSE 事件
- CLI 保存当前统一 `session_id`

## 第二期：会话命令联动

目标：让 `/new`、`/resume` 与统一 `session_id` 真正接通。

建议内容：

- `/new` 联动后端新建统一 `session_id`
- `/resume` 切换本地当前 `session_id`
- 验证旧历史会话在按用户隔离前提下可以直接沿用旧本地 `session_id`

## 第三期：治理与收敛

目标：统一命名、日志、轨迹与兼容策略。

建议内容：

- 清理历史命名歧义
- 统一三端日志字段
- 持续验证旧短 `session_id` 的兼容性
- 如后端未来变更唯一性约束，再评估历史会话迁移

## 最终建议

### 建议一

不要把当前 `worker_id` 直接当成文档里的 `worker_no` 使用。

它们语义不同：

- 当前 `worker_id` 更像登录身份
- `worker_no` 是终端实例标识

### 建议二

当前方案下，本地与远端直接统一使用同一个 `session_id`。

### 建议三

`session_id` 一定要通过明确的 SSE 事件回传，不要隐式传递。

### 建议四

第一版优先做最小闭环，先把 `worker_no` 透传、`session_id` 回传、CLI 保存与 `/new` 切换机制跑通；`/resume` 不新增接口，只复用既有 `session_id`。

## 本地 `im_bridge` 文件系统改造边界

### 背景

当前本地 bridge 文件系统的目录大致是：

```text
.tg_agent/im_bridge/<binding_id>/
```

现状里 `<binding_id>` 实际偏向 `user_id`。这会导致同一个用户在同一个工作目录下多开多个终端时，多个终端可能共用同一个 bridge 队列，从而出现消息归属不清、请求互相抢占的问题。

本次改造的核心目标是：

- 同一个用户可以在同一工作目录下多开多个终端
- 每个终端都有自己的 bridge 队列
- 每个终端只能消费属于自己的消息

### 第一阶段结论

第一阶段只做多终端隔离，不做自动清理体系。

建议：

- 将 `binding_id` 改为使用 `worker_no`
- 目录结构保持一层：

```text
.tg_agent/im_bridge/<worker_no>/
```

- 不增加 `session_id` 目录层级
- 不增加 `user_id/worker_no` 双层目录
- `session_id` 作为请求归属字段透传，不参与 bridge 目录分层

原因：

- `im_bridge` 是本地运行态队列，不是历史会话存储
- 队列应该按终端实例隔离，而不是按会话隔离
- `/new`、`/resume` 会切换会话，但不应该导致本地 bridge 队列目录切换
- 多加 `session_id` 层级会放大目录数量，也会增加 pending / processing / progress / outbox 的迁移复杂度

### 第一阶段暂不做自动清理

当前阶段不实现以下机制：

- `meta.json`
- `last_seen_at`
- 定时心跳刷新
- 启动时 TTL 清理
- stale `processing` 自动失败化
- bridge 目录数量上限回收

原因：

- 本次改造的必要目标是多终端隔离，清理不是阻塞项
- 文件系统清理需要处理异常退出、处理中请求、损坏元数据、启动耗时等问题，容易扩大第一阶段改造面
- 后续如果将 bridge 存储改造成 SQLite，本地队列状态、超时处理和清理会更自然，届时统一设计更合适

### 当前文件系统目录的定位

`.tg_agent/im_bridge/` 应定义为可重建的运行态缓存目录。

删除它的影响：

- 不影响 `~/.tg_agent/sessions.db` 中的历史会话
- 不影响 `/resume` 的历史会话列表和上下文快照
- 会丢失当前未完成的 bridge 请求、进度、结果和 outbox 事件
- 没有正在执行的任务时，可以手动删除

### 后续 SQLite bridge v2 建议

后续可单独设计 `SQLiteBridgeStore`，用数据库表替代当前文件目录状态机。

建议表结构方向：

```text
bridge_requests(id, worker_no, session_id, source, content, status, created_at, claimed_at, finished_at, error_code)
bridge_progress(id, request_id, content, created_at, delivered_at)
bridge_outbox(id, worker_no, action, status, created_at, claimed_at, finished_at)
bridge_workers(worker_no, user_id, created_at, last_seen_at)
```

届时再统一处理：

- 超过 24 小时的 processing 请求标记失败
- 已完成/失败请求的保留周期
- progress / outbox 的过期清理
- worker_no 的 last_seen_at
- SQLite WAL、busy_timeout、多进程写入锁

### 阶段边界总结

第一阶段：

- `binding_id = worker_no`
- 保持现有文件队列结构
- 不做自动清理
- 明确 `im_bridge` 是可删除、可重建的运行态缓存

后续阶段：

- 评估并设计 SQLite bridge v2
- 在 SQLite v2 中统一解决清理、超时、异常退出恢复和运行态垃圾回收
