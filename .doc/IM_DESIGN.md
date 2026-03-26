# IM 同步详细方案设计

## 1. 背景

目标是让本地 CLI 在保留现有 agent 能力、slash 命令解析能力和会话上下文的前提下，对接一个云端入口
（web、IM、企业通讯软件等）。用户既可以在本地终端输入，也可以从云端发送消息，两边都进入同一个本地
agent 会话上下文串行执行。

本次设计基于以下已确认约束：

1. 本地终端输入和云端消息必须进入同一个 agent 会话上下文。
2. 本地 CLI 与额外 client 进程之间必须通过本地文件同步，不能直接依赖进程内队列作为主链路。
3. 云端侧第一阶段只支持“发送一条消息，等待最终回复”，不做流式中间态同步。
4. 云端侧第一阶段只支持可以单行执行完成的 slash 命令，不支持需要二次交互的命令。

## 2. 当前代码现状

### 2.1 CLI 现状

当前 CLI 主入口在 `claude_code.py`，启动后直接创建：

1. `Agent`
2. `SandboxContext`
3. `ClaudeCodeCLI`

然后进入 `ClaudeCodeCLI.run()` 的交互循环。当前循环特点是：

1. 使用 `prompt_toolkit` 阻塞等待本地用户输入。
2. 收到输入后在同一进程内直接判断：
   - 是否为图片输入
   - 是否为 `@skill`
   - 是否为 slash 命令
   - 否则进入 `_run_agent()`
3. slash 命令解析和 agent 执行都发生在 CLI 主进程内。

这意味着：如果不改主循环，云端消息即使到了本机，也无法在“用户还没按回车”的情况下被及时执行。

### 2.2 worker 原型现状

仓库中已经有 `cli/worker/*` 原型，当前模式是：

1. worker 长轮询服务端
2. 拉到消息后自己创建/持有 agent runtime
3. 直接执行 `agent.query()`
4. 调用服务端 complete 回传最终结果

这条链路不能直接满足本次需求，因为它会让 worker 自己持有独立 agent 上下文，而不是和本地 CLI 共用同一份
上下文。

### 2.3 现有 IM gateway 现状

仓库里也有 `agent_core.gateway.*` 的 in-process gateway 模式。这个模式由服务端进程自己持有 runtime，
与“本地 CLI + 本地附加 client 进程 + 本地文件同步”的目标也不同，因此本次不直接复用其 runtime 持有方式，
但可以复用其消息协议思路。

## 3. 设计目标

本方案只解决以下问题：

1. 本地 CLI 启动时，同时启动一个额外的本地 client/worker 进程。
2. worker 负责与云端服务通信，但不负责执行 agent。
3. CLI 主进程是唯一的执行者，唯一持有 agent、slash registry、skill registry 和会话上下文。
4. 本地终端输入和云端消息都先落本地文件，再由 CLI 主进程统一串行消费。
5. worker 只关心“收到远端请求 -> 写入本地文件 -> 等本地结果 -> 回服务端”。
6. 本地实例启动和退出时，需要向云端显式同步“已上线 / 已下线”状态。

## 4. 非目标

第一阶段明确不做：

1. 云端流式输出同步
2. 云端工具执行中间态同步
3. 云端审批确认、多轮补问、表单式交互
4. 需要二次本地 prompt 的 slash 命令
5. 一个本地 CLI 同时绑定多个独立远端会话
6. 文件监听作为正确性基础

## 5. 总体架构

### 5.1 进程划分

本地运行时拆成两个进程：

1. CLI 主进程
   - 持有唯一 `Agent`
   - 持有 slash/skill/plugin 运行时注册表
   - 负责执行本地输入和远端输入
   - 负责把执行结果写回本地同步文件

2. worker/client 进程
   - 长轮询云端服务
   - 把云端消息写入本地同步文件
   - 续租远端 delivery
   - 读取本地结果并调用服务端 complete

### 5.2 核心原则

核心原则是“执行权单点化，通信链路文件化”：

1. 任何需要改变 agent 会话上下文的输入，都只能由 CLI 主进程执行。
2. 任何跨进程交互，都必须先落文件。
3. 文件是事实来源，内存态只是缓存。

### 5.3 主链路

完整主链路如下：

1. 本地用户在终端输入一条消息
2. CLI 不直接执行，而是先将其写入本地请求目录
3. CLI 调度器从请求目录取出下一条待执行消息
4. CLI 执行 slash 或 agent
5. CLI 将结果写入本地结果目录
6. 如果消息来源是本地终端，则结果直接打印在本地控制台
7. 如果消息来源是云端，则 worker 读取结果并回传服务端

云端消息链路类似：

1. worker 长轮询拉到云端消息
2. worker 把消息写入本地请求目录
3. CLI 调度器取出并执行
4. CLI 写本地结果文件
5. worker 等到结果后 complete 到云端

## 6. 文件同步目录设计

### 6.1 目录位置

建议每个工作区在根目录下创建：

```text
.tg_agent/im_bridge/<session_binding_id>/
```

其中：

1. `.tg_agent` 已与仓库现有习惯一致，适合作为本地运行时目录。
2. `session_binding_id` 是对当前远端绑定会话的安全文件名表示，建议由 `session_key` 做 URL-safe 编码
   或 hash 后得到，避免路径中出现非法字符。

### 6.2 目录结构

建议目录结构如下：

```text
.tg_agent/im_bridge/<session_binding_id>/
  state/
    session.json
    sequence.json
    active_request.json
    presence.json
  locks/
    enqueue.lock/
  inbox/
    pending/
    processing/
  results/
    completed/
    failed/
  worker/
    lease/
    checkpoints/
  logs/
```

说明：

1. `inbox/pending/`：待执行请求，一条请求一个 JSON 文件。
2. `inbox/processing/`：CLI 已领取但尚未完成的请求文件。
3. `results/completed/`：执行成功结果文件。
4. `results/failed/`：执行失败或不支持结果文件。
5. `state/session.json`：当前桥接配置和版本信息。
6. `state/sequence.json`：全局递增序号，用于统一本地和远端消息顺序。
7. `state/active_request.json`：当前正在执行的请求，用于恢复与诊断。
8. `locks/enqueue.lock/`：跨进程序号分配与入队锁。
9. `state/presence.json`：本地 CLI 与 worker 的存活状态、本地就绪状态和最近心跳时间。

### 6.3 为什么不用单个大 JSON 文件

不建议使用一个总的 JSON 数组文件，原因如下：

1. 多进程同时写入时容易出现覆盖和损坏。
2. Windows 下追加写和部分覆盖处理麻烦。
3. 崩溃恢复时难以定位单条请求状态。
4. 单条文件天然支持原子写入：先写临时文件，再 `os.replace()`。

因此本方案采用“目录队列 + 单请求单文件”模型。

## 7. 请求与结果文件格式

### 7.1 请求文件格式

每条请求对应一个 `inbox/pending/<seq>.json` 文件，内容建议如下：

```json
{
  "version": 1,
  "request_id": "req_01H...",
  "seq": 42,
  "source": "local",
  "source_meta": {
    "session_key": "im:demo",
    "delivery_id": null,
    "worker_id": null,
    "sender_id": "local-user"
  },
  "content": "/model show",
  "content_type": "text",
  "enqueue_time": "2026-03-25T10:00:00Z",
  "remote_response_required": false,
  "status": "pending"
}
```

字段建议：

1. `request_id`
   - 全局唯一请求 ID
   - 本地输入由 CLI 生成
   - 远端输入建议稳定绑定 `delivery_id`，避免重复投递后重复执行

2. `seq`
   - 全局递增顺序号
   - 决定统一消费顺序

3. `source`
   - `local` 或 `remote`

4. `source_meta`
   - 远端请求需要保留 `session_key`、`delivery_id`、`worker_id`
   - 本地方向可记录操作者信息和终端 ID

5. `remote_response_required`
   - 本地输入为 `false`
   - 远端输入为 `true`

6. `status`
   - 文件初始写入时为 `pending`
   - CLI claim 后在 `processing` 目录对应文件里更新为 `running`

### 7.2 结果文件格式

每条请求完成后生成 `results/completed/<request_id>.json` 或 `results/failed/<request_id>.json`：

```json
{
  "version": 1,
  "request_id": "req_01H...",
  "seq": 42,
  "source": "remote",
  "final_status": "completed",
  "final_content": "当前模型是 xxx",
  "error_code": null,
  "error_message": null,
  "started_at": "2026-03-25T10:00:05Z",
  "finished_at": "2026-03-25T10:00:12Z"
}
```

失败时：

```json
{
  "version": 1,
  "request_id": "req_01H...",
  "seq": 42,
  "source": "remote",
  "final_status": "failed",
  "final_content": "该命令当前不支持从 IM 端执行：/model",
  "error_code": "UNSUPPORTED_REMOTE_COMMAND",
  "error_message": "interactive slash command is not allowed for remote source",
  "started_at": "2026-03-25T10:00:05Z",
  "finished_at": "2026-03-25T10:00:06Z"
}
```

### 7.3 状态流转

请求状态流转如下：

```text
pending -> running -> completed
pending -> running -> failed
```

文件目录流转如下：

```text
inbox/pending/<seq>.json
  -> inbox/processing/<seq>.json
  -> results/completed/<request_id>.json
or
  -> results/failed/<request_id>.json
```

## 8. 入队顺序与并发控制

### 8.1 为什么必须有统一序号

因为本地输入和云端输入必须进入同一个 agent 上下文，所以必须明确“谁先执行”。如果只按文件时间排序，
在跨进程并发写入时会存在时间分辨率和乱序问题。

因此必须引入统一 `seq`。

### 8.2 序号分配机制

建议通过文件锁目录实现简单的跨进程序号分配：

1. 写入方尝试原子创建 `locks/enqueue.lock/`
2. 成功后读取 `state/sequence.json`
3. 分配 `next_seq`
4. 写请求临时文件到 `inbox/pending/<seq>.json.tmp`
5. `os.replace()` 为正式文件
6. 更新 `state/sequence.json`
7. 删除 `locks/enqueue.lock/`

说明：

1. 这里只有入队时需要跨进程互斥。
2. 消费端只有 CLI 一个，不需要多消费者竞争设计。

### 8.3 为什么不把文件监听作为主机制

本方案建议用“短周期轮询 + 文件状态判断”作为正确性基础，不依赖文件系统 watch：

1. Windows 对文件变化事件和 rename 时机的表现更容易踩边界。
2. watch 更适合做性能优化，不适合做事实来源。
3. 轮询频率控制在 200ms 到 500ms 已足够满足首版体验。

后续可在不改变协议的情况下加 watch 作为唤醒优化。

## 9. CLI 主进程设计

### 9.1 核心职责

CLI 主进程新增两层职责：

1. 本地输入采集
2. 文件队列调度执行

但真正执行 slash/agent 的主体仍然是现有 `ClaudeCodeCLI`。

### 9.2 推荐重构方向

当前 `ClaudeCodeCLI.run()` 把“读终端输入”和“执行输入”耦合在一起。建议拆成两部分：

1. `read_local_input_once()`
   - 只负责读终端的一次输入
   - 不直接执行

2. `execute_request(request)`
   - 接收统一请求对象
   - 根据内容决定走 slash、`@skill`、agent 普通消息等分支

这样本地输入与云端输入就都可以复用同一套执行逻辑。

### 9.3 主循环形态

CLI 主循环建议改为三层：

1. 启动时创建 bridge store
2. 启动 worker 子进程
3. 进入统一调度循环

统一调度循环逻辑：

1. 如果当前没有运行中的请求，优先检查 `inbox/pending/` 是否存在待处理请求
2. 若有，则 claim 最小 `seq` 的请求并执行
3. 若没有，则等待本地终端输入
4. 本地终端输入拿到后先入队，再回到步骤 1

### 9.4 本地 prompt 阻塞问题

这里有一个关键问题：当前 prompt 是阻塞等待的。如果云端消息到来时 CLI 正停在输入框，会导致远端请求无法及时执行。

因此第一阶段建议把主循环改成“可中断的一次性 prompt”模式，而不是永久阻塞的 while prompt：

1. 每轮只发起一次 `prompt_async()`
2. 同时轮询 bridge queue
3. 如果远端请求先到：
   - 取消当前 prompt
   - 如能拿到当前 buffer，则保存草稿
   - 先执行远端请求
   - 完成后恢复本地 prompt，并尽量恢复草稿
4. 如果本地用户先提交：
   - 先把输入入队
   - 再统一走队列消费链路

这一步是本次改造的关键点，否则“同一个执行器同时支持本地和远端注入”无法成立。

### 9.5 本地执行结果展示

对于 `source=local` 的请求：

1. CLI 正常在终端展示 slash 输出或 agent 最终结果
2. 同时仍然写结果文件，便于统一审计和恢复

对于 `source=remote` 的请求：

1. CLI 可在本地打印一条简短提示，例如“收到远端请求 seq=42”
2. 不需要在终端完整重放一遍远端最终回复
3. 但建议保留 debug 日志

## 10. worker/client 进程设计

### 10.1 worker 的新职责

worker 不再持有 agent runtime，它只负责：

1. 轮询云端服务
2. 将远端消息入队到本地 bridge
3. 持续 renew 远端 lease
4. 等待本地结果文件
5. complete 服务端

### 10.2 与现有 worker 原型的关系

现有 `cli/worker/runner.py` 里最核心的变化是：

从：

1. poll
2. `agent.query()`
3. complete

改成：

1. poll
2. enqueue 到本地 bridge
3. 启动 renew loop
4. wait result file
5. complete

也就是说，worker 保留现有 poll / renew / complete 协议，但把“执行”改成“桥接”。

### 10.3 worker 处理时序

worker 处理一条远端消息的时序如下：

1. `poll(session_key, worker_id)` 拉到一条消息
2. 基于 `delivery_id` 生成稳定 `request_id`
3. 检查本地是否已有对应结果文件
   - 若已有 completed 结果，可直接走 complete，防止重复投递导致重复执行
4. 若不存在结果，则写入本地 `inbox/pending/`
5. 启动 renew loop
6. 轮询等待 `results/completed/<request_id>.json` 或 `results/failed/<request_id>.json`
7. 读取 `final_content`
8. 调用 `complete(...)`
9. 停止 renew loop

### 10.4 远端只支持最终结果

第一阶段 worker 不同步中间态，所以它只需要关心结果文件是否出现，不需要订阅 CLI 的执行事件流。

### 10.5 worker 等待策略

worker 等待本地结果建议采用：

1. 500ms 轮询结果目录
2. 每轮同时检查停止信号
3. renew loop 独立协程持续执行

这样实现简单且与文件事实来源一致。

### 10.6 上线 / 下线通知职责

上线 / 下线通知建议由 worker 负责发给云端，因为它已经是唯一与服务端通信的进程。

但这里有一个关键约束：

1. 不能只要 worker 活着，就向云端宣称“本地在线”
2. 云端真正关心的是“CLI 是否可执行请求”

因此建议把在线状态拆成两层：

1. 本地事实层
   - CLI 周期性刷新 `state/presence.json`
   - 表示“执行器仍然活着且 ready”

2. 云端同步层
   - worker 读取 `presence.json`
   - 只有当 CLI 心跳新鲜时，才向云端维持在线

这样可以避免出现“worker 还活着，但 CLI 已经死掉，云端却仍然显示在线”的错误状态。

## 11. slash 命令支持策略

### 11.1 为什么需要显式区分

当前很多 slash 命令虽然入口是单行，但内部会继续调用本地 prompt，例如选择、确认、编辑、多行输入等。这类命令
不能直接暴露给远端，否则 worker 只能卡住等待本地交互。

### 11.2 建议方案

第一阶段对远端 slash 命令采用显式 allowlist，而不是“自动认为所有 slash 都可以”。

建议增加一个判定层，例如：

1. `is_remote_safe_slash(parsed_command) -> bool`
2. 或者在 slash 注册元数据里增加 `remote_mode`

其中：

1. `remote_mode=single_turn`
   - 可远端执行
2. `remote_mode=local_only`
   - 只能本地执行
3. `remote_mode=unsupported`
   - 默认值，未明确标注前不允许远端执行

### 11.3 第一阶段建议支持的远端 slash

建议先只开放确定单行闭环的命令，例如：

1. `/help`
2. `/pwd`
3. `/reset`
4. `/model show`
5. `/model list`
6. `/model <preset>`
7. `/approval status`
8. `/approval on`
9. `/approval off`

### 11.4 第一阶段建议拒绝的远端 slash

建议明确拒绝：

1. `/model` 无参数模式，因为它会进入交互式选择
2. `/agents` 相关交互命令
3. `/plugins` 相关交互命令
4. 任何会弹确认、选择、多行输入、文件编辑的 slash 命令

### 11.5 远端不支持命令的返回方式

若远端发来不支持命令，应直接生成失败结果文件，内容给出清晰提示，例如：

```text
该命令当前仅支持在本地终端执行：/agents
原因：该命令需要进一步交互输入，IM 首版暂未支持。
```

## 12. 请求执行逻辑

### 12.1 统一入口

无论请求来自本地还是远端，都走统一执行函数：

1. 读取请求内容
2. 判定是否图片输入
3. 判定是否 `@skill`
4. 判定是否 slash
5. 否则走普通 agent query

### 12.2 远端执行差异

若 `source=remote`，需要多做两件事：

1. 执行前做远端能力校验
2. 执行结果不要依赖终端交互态

例如：

1. `source=local` 可以进入交互式 `/model`
2. `source=remote` 遇到该命令则直接失败返回

### 12.3 同一上下文串行保证

由于只有 CLI 主进程消费 `inbox/pending/`，且每次只 claim 一条请求执行，因此天然保证：

1. 任意时刻只会有一个请求修改 agent 上下文
2. 本地和远端不会并发写 history
3. 会话上下文顺序与 `seq` 一致

## 13. 启动与生命周期设计

### 13.1 CLI 启动参数

建议给 CLI 增加与 IM 桥接相关的参数：

1. `--im-session-key`
2. `--im-worker-id`
3. `--im-gateway-base-url`
4. `--im-enable`
5. `--im-bridge-dir`（可选，默认在工作区 `.tg_agent/im_bridge/...`）

只有 `--im-enable` 且必要参数齐全时，才启动 worker 子进程。

### 13.2 启动流程

启动流程建议如下：

1. CLI 解析参数
2. 初始化 bridge 目录和 `session.json`
3. 创建 agent/runtime registries
4. 启动 worker 子进程
5. 进入统一调度循环

### 13.3 退出流程

退出时：

1. CLI 通知 worker 停止
2. 等待 worker 退出
3. 若有正在处理的请求，写入恢复信息

如果无法优雅停止，可强制结束 worker，但不能删除 bridge 文件，以免远端任务状态丢失。

### 13.4 本地 Presence 文件设计

建议新增 `state/presence.json`，作为本地上线状态事实来源，内容示例：

```json
{
  "version": 1,
  "session_key": "im:demo",
  "client_instance_id": "cli_01H...",
  "cli_pid": 12345,
  "worker_pid": 23456,
  "status": "ready",
  "started_at": "2026-03-25T10:00:00Z",
  "last_cli_heartbeat_at": "2026-03-25T10:00:20Z",
  "last_worker_heartbeat_at": "2026-03-25T10:00:20Z",
  "shutdown_requested": false
}
```

字段说明：

1. `client_instance_id`
   - 表示这一次本地启动实例的唯一 ID
   - 用于和云端在线实例一一对应

2. `status`
   - `starting`
   - `ready`
   - `stopping`
   - `stopped`

3. `last_cli_heartbeat_at`
   - 由 CLI 周期性刷新
   - 表示执行器是否仍然健康

4. `last_worker_heartbeat_at`
   - 由 worker 周期性刷新
   - 用于本地排障和恢复判断

5. `shutdown_requested`
   - CLI 发起优雅退出时先写为 `true`
   - worker 读到后触发下线通知

### 13.5 云端上线 / 下线协议

建议在现有 worker 协议之外补三类接口：

1. `POST /api/worker/online`
2. `POST /api/worker/heartbeat`
3. `POST /api/worker/offline`

建议请求体至少包含：

1. `session_key`
2. `worker_id`
3. `client_instance_id`
4. `status`
5. `timestamp`
6. `capabilities`
   - 例如 `final_reply_only=true`
   - `single_turn_slash_only=true`

其中：

1. `online`
   - 表示本地实例已经 ready，可以接收任务
2. `heartbeat`
   - 表示本地实例仍然在线
3. `offline`
   - 表示本地实例主动下线

### 13.6 上线时机

建议不要在 CLI 进程刚启动时立刻通知云端在线，而是满足以下条件后再由 worker 发 `online`：

1. bridge 目录初始化完成
2. CLI 已创建 agent/runtime registries
3. `presence.json.status=ready`
4. worker 自己也已启动并拿到必要参数

原因是：

1. 如果太早宣称在线，云端可能马上派发消息
2. 但此时 CLI 还没进入可执行状态，首条消息可能卡住

所以“ready 后上线”比“进程创建即上线”更准确。

### 13.7 心跳与 TTL 策略

建议使用“显式上下线 + TTL 心跳兜底”的双保险方案：

1. worker 每 15 秒向云端发送一次 `heartbeat`
2. 云端将超过 45 秒未更新的实例标记为 `offline`
3. worker 发心跳前先检查 `presence.json.last_cli_heartbeat_at`
4. 若 CLI 心跳超过阈值未刷新，例如超过 20 到 30 秒：
   - worker 不再继续上报 `online`
   - 立即尝试发送一次 `offline`
   - 若发送失败，则依赖云端 TTL 自动下线

这样可以覆盖两类场景：

1. 正常退出
   - 立刻下线，用户体验好
2. 异常崩溃
   - 即使没来得及发 `offline`，也会被 TTL 清理

### 13.8 下线时机

建议以下情况发送 `offline`：

1. CLI 主动退出，且已写 `shutdown_requested=true`
2. worker 检测到 CLI 心跳超时
3. worker 自己收到停止信号，准备退出

其中第一种是主路径，第二三种是兜底。

### 13.9 云端状态语义

云端不要把“进程存在”直接等同于“可接单”，建议至少区分：

1. `online`
   - 最近心跳正常，且本地执行器健康
2. `degraded`
   - worker 还在，但 CLI 心跳已接近超时，可选状态
3. `offline`
   - 主动下线或 TTL 超时

如果首版不想引入 `degraded`，也至少要有：

1. `online`
2. `offline`

### 13.10 与消息投递的关系

云端投递策略应只向 `online` 实例派发消息。

一旦实例被标记为 `offline`：

1. 不再投递新消息
2. 已在途但未 complete 的 delivery 继续按 lease / timeout 机制处理
3. 若实例之后重新上线，视为新的 `client_instance_id`

### 13.11 重启场景

本地重启时建议视为“旧实例下线，新实例上线”，而不是沿用同一个实例标识。

也就是说：

1. 退出前尽量发旧实例 `offline`
2. 重启后生成新的 `client_instance_id`
3. 新实例完成 ready 后重新发 `online`

这样云端状态会更清晰，也便于排查“哪一次本地启动处理了哪批消息”。

## 14. 崩溃恢复与幂等

### 14.1 CLI 崩溃恢复

如果 CLI 在执行中崩溃，启动时应检查：

1. `state/active_request.json`
2. `inbox/processing/`

恢复策略建议：

1. 若存在 `processing` 中的请求且没有结果文件：
   - 将其移回 `pending`
   - 增加 `retry_count`
   - 重新执行
2. 若结果文件已存在但 `processing` 未清理：
   - 清理残留 processing 文件

### 14.2 worker 崩溃恢复

worker 崩溃后，服务端 delivery lease 最终会超时并重新投递。为避免重复执行：

1. 远端请求 `request_id` 必须稳定绑定 `delivery_id`
2. worker 收到重复消息时先看本地是否已有结果文件
3. 若已有结果文件，则直接 complete，不重新入队

### 14.3 本地重复入队保护

对于远端消息，建议建立简单去重规则：

1. 若同一个 `request_id` 已在 `pending`、`processing` 或 `results` 中存在
2. 则认为该消息已被接收，不重复创建请求文件

## 15. 失败处理

### 15.1 slash 不支持

直接生成失败结果文件，不进入 agent。

### 15.2 agent 执行异常

CLI 捕获异常后：

1. 写 `results/failed/<request_id>.json`
2. `final_content` 返回用户可读错误提示
3. `error_message` 记录内部异常信息

### 15.3 worker complete 失败

如果 worker 调用 complete 失败：

1. 结果文件保留
2. worker 记录失败日志
3. 下次同一 `delivery_id` 重新投递时可再次尝试 complete

### 15.4 上线 / 下线通知失败

如果 `online` / `heartbeat` / `offline` 调用失败：

1. 本地不影响消息桥接主功能
2. worker 记录失败日志
3. `online` 和 `heartbeat` 在下一轮继续重试
4. `offline` 若失败，则依赖云端 TTL 自动摘除在线状态

## 16. 模块拆分建议

建议新增一个独立的本地桥接模块，例如：

```text
cli/im_bridge/
  __init__.py
  models.py
  store.py
  scheduler.py
  launcher.py
  remote_policy.py
```

职责建议如下：

1. `models.py`
   - 请求、结果、状态数据结构

2. `store.py`
   - 文件路径管理
   - 锁目录管理
   - 原子写入
   - `enqueue() / claim_next() / complete() / fail() / find_result()`

3. `scheduler.py`
   - CLI 统一调度逻辑
   - 输入源与执行器协调

4. `launcher.py`
   - worker 子进程启动与关闭

5. `remote_policy.py`
   - 远端 slash allowlist / remote mode 校验

6. `presence.py`
   - 本地 `presence.json` 的读写
   - CLI / worker 心跳刷新
   - 在线状态判断

现有模块建议改造点：

1. `claude_code.py`
   - 新增 IM 相关启动参数
   - 启动 bridge 与 worker
   - 启动时写入 `presence.json`

2. `cli/app.py`
   - 抽取“读输入”和“执行输入”
   - 增加统一请求执行入口

3. `cli/worker/runner.py`
   - 从“执行器”改成“桥接器”
   - 补充 online / heartbeat / offline 通知逻辑

## 17. 测试方案

### 17.1 单元测试

至少补以下单测：

1. `store.enqueue()` 在双进程模拟下能生成严格递增 `seq`
2. `claim_next()` 总是拿到最小 `seq`
3. 结果文件原子写入后能被正确读取
4. 重复 `delivery_id` 不会重复入队
5. 远端不支持 slash 会返回明确失败结果
6. `presence.json` 心跳超时后，worker 不再继续维持在线状态

### 17.2 集成测试

建议增加以下集成测试：

1. 本地输入两条消息，确认上下文连续
2. 远端输入两条消息，确认上下文连续
3. 本地一条 + 远端一条交错入队，确认按 `seq` 串行执行
4. 远端 `/reset` 后再发普通消息，确认会话已清空
5. worker 拉到远端请求后，能等到本地结果并 complete
6. CLI ready 后 worker 会向云端发 `online`
7. CLI 正常退出时 worker 会向云端发 `offline`

### 17.3 崩溃恢复测试

建议补以下恢复测试：

1. 请求进入 `processing` 后模拟 CLI 崩溃，重启后能恢复执行
2. worker 在结果写出前崩溃，重启后能继续 complete
3. 服务端重复投递同一 `delivery_id` 时不会重复执行 agent
4. CLI 异常退出未发 `offline` 时，云端会在 TTL 后摘除在线状态

## 18. 分阶段落地建议

建议按最小可闭环顺序落地：

### Phase 1：本地文件桥

1. 完成 `im_bridge.store`
2. CLI 本地输入不再直接执行，而是先入队再消费
3. 先不启动 worker，只验证“本地输入走文件队列”能跑通

### Phase 2：worker 桥接

1. worker 改为 poll -> enqueue -> wait result -> complete
2. CLI 启动时拉起 worker
3. 打通一条远端普通文本消息

### Phase 3：远端 slash 策略

1. 增加 remote allowlist
2. 打通 `/help`、`/pwd`、`/reset`、`/model show`
3. 对交互式 slash 返回明确错误

### Phase 4：恢复与稳定性

1. 增加 processing 恢复
2. 增加重复投递幂等
3. 补齐测试

## 19. 最终结论

这次改造最关键的决策有三个：

1. `Agent` 只能由 CLI 主进程持有，worker 绝不能直接执行 agent。
2. 文件同步要做成“目录队列 + 单请求单文件 + 原子 rename”，不要做单个大 JSON 文件。
3. CLI 主循环必须从“阻塞读输入并立即执行”改成“输入先入队，执行统一出队”，否则无法真正统一本地和远端请求。
4. 在线状态不能只看 worker 是否存活，必须以 CLI 本地心跳为准，并通过 worker 同步到云端。

在以上约束下，第一阶段可以稳定实现：

1. 本地和远端共用同一会话上下文
2. 云端单条消息拿到最终回复
3. 远端执行有限的单行 slash 命令
4. 具备基础的恢复与幂等能力

这是一条最小但结构正确的落地路径，后续如果要扩展流式输出、远端审批、多轮交互，仍然可以在这套桥接协议上继续演进，而不需要推翻主体结构。
