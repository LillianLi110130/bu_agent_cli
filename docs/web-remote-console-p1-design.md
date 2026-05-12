# Web 远程对话 P1 设计文档

本文档描述的是 **下一步收敛后的 Web 对话方案**，用于统一后续实现方向。

说明：
- 本文档先更新设计，不代表当前代码已经全部切到本文档形态。
- 目标是让 Web 端尽量贴近现有 IM -> 本地 CLI 的投递模型。
- 目标是做最小可用闭环，不在 P1 引入复杂的请求级状态机。

## 1. 目标

P1 的目标很明确：

- Web 页面只负责给用户展示一个当前对话窗口
- Python server 只负责把 Web 消息中转给本地 worker
- 本地 worker 继续沿用现有 CLI / bridge 逻辑处理消息
- worker 把 `progress` / `complete` 回给 Python server
- Python server 再把事件通过 SSE 推给 Web

这条链路的目标形态是：

```text
Web
  -> Python server
  -> local worker
  -> local CLI / bridge
  -> worker progress / complete
  -> Python server SSE
  -> Web
```

## 2. 基本约束

P1 明确采用下面这组强约束：

- 一个 Web 页面只绑定一个 `worker_id`
- 一个 `worker_id` 同一时刻只允许一条正在进行中的 Web 请求
- Web 页面不做多会话并发
- Web 页面不做历史会话列表
- Web 页面不做请求级重连恢复
- Web 页面只做“当前对话”的展示

在这组约束下，服务端不需要维护复杂的 request 级订阅关系，也不需要对外暴露 `request_id` 级别的 SSE。

## 3. 设计原则

### 3.1 与 IM 通路保持一致

Web 侧不直接调用本地 agent，也不在 Python server 内部起后台 agent 执行消息。

Python server 的职责仅是：

- 接收 Web 消息
- 投递给 worker
- 接收 worker 回传
- 转发给 Web

### 3.2 对外接口尽量轻

对外的 Web-BFF 接口尽量围绕 `worker_id`，而不是围绕 `request_id`。

原因：

- 当前 Web 只有一个当前会话框
- 同一时刻只有一个进行中的请求
- `request_id` 对 Web 页面不再是必需概念

### 3.3 worker 协议尽量不动

worker 侧继续沿用当前已有的协议：

- `POST /api/worker/online`
- `POST /api/worker/offline`
- `POST /api/worker/poll`
- `GET /api/worker/stream`
- `POST /api/worker/progress`
- `POST /api/worker/complete`

worker 继续只传：

- `worker_id`
- `source`
- `content` 或 `final_content`

P1 不要求把 `request_id` / `delivery_id` 下沉到 worker 协议。

## 4. 页面模型

P1 页面模型收敛成“单对话模式”：

- 不展示历史会话列表
- 不展示 mock 会话容器
- 不展示多标签会话
- 只展示一个当前聊天窗口

页面关注点只有三件事：

- 本地终端是否在线
- 当前消息是否在处理中
- 中间结果和最终结果是否回来了

## 5. 服务端模型

### 5.1 对外模型

对 Web 来说，服务端只需要维护每个 `worker_id` 的两类信息：

- 当前是否有活跃的 Web SSE 订阅
- 当前是否有一条进行中的 Web 请求

### 5.2 不再强调 request 级状态机

下一步收敛方案里，不再把 Web-BFF 设计成复杂的：

- pending request 列表
- inflight request 列表
- request 级 subscriber 列表
- `/web-console/messages/{requestId}/events`

而是收敛成：

- 一个 worker 对应一条当前 Web SSE
- 一个 worker 对应一条当前 Web 请求

### 5.3 server 收到 worker 回调后的行为

当 worker 回调时：

- `source=web` 的 `progress`
  - 直接推给当前 `worker_id` 的 Web SSE
- `source=web` 的 `complete`
  - 直接推给当前 `worker_id` 的 Web SSE
  - 推完后结束当前这条 Web 请求
- `source=im` 的回调
  - 不进入 Web SSE
  - 继续走 IM 侧自己的逻辑

## 6. SSE 设计

### 6.1 订阅维度

P1 的 SSE 不再按 `request_id` 订阅，改为按 `worker_id` 订阅：

```text
GET /web-console/workers/{workerId}/events
```

这样更符合当前页面形态：

- 页面只关心这台 worker 当前正在处理的那条 Web 请求
- 不需要显式感知 `request_id`

### 6.2 事件类型

P1 仍保留这些 SSE 事件：

- `submitted`
- `processing`
- `progress`
- `completed`
- `failed`

事件含义：

- `submitted`
  - Python server 已接受 Web 消息
- `processing`
  - worker 已取走当前消息
- `progress`
  - 本地 CLI 处理中间结果
- `completed`
  - 本地 CLI 最终结果
- `failed`
  - 当前请求失败

### 6.3 不再单独提供 stop 接口

在收敛方案下，不再把“停止当前请求”设计成一个单独的 Web-BFF API。

原因：

- 一个 worker 当前只有一条 Web SSE
- 一个 worker 当前只有一条 Web 请求
- Web 前端只要断开当前 SSE，就可以精准结束当前这条 Web 展示链路

这里要明确：

- 断开 SSE 只表示 Web 停止接收和展示
- 不等于本地 CLI 任务被取消
- 本地任务可能仍在继续执行

因此页面文案应该明确提示：

- 已停止当前页面接收
- 本地终端任务可能仍在继续执行

## 7. 中间结果设计

P1 保留“中间结果可回 Web”的能力。

本地 CLI 处理中产生的阶段性文本：

- 仍先写入本地 bridge progress 文件
- 再由 worker 调 `POST /api/worker/progress` 回 server
- 再由 server 通过 Web SSE 推给前端

这样可以做到：

- Web 页面看到“正在回复”的过程
- 不需要伪装成 token 级流式

P1 的 `progress` 语义是：

- 阶段性文本块
- 不是 token delta
- 不是完整事件重放协议

## 8. 与当前实现的主要差异

当前代码里仍然存在一些更重的实现痕迹，下一步应逐步收敛：

- 当前 SSE 仍有按 `request_id` 订阅的实现
- 当前 server 内部仍保留 request 记录和 inflight 概念
- 当前还存在单独的 stop 接口

本文档定义的是下一步收敛方向：

- 外部接口以 `worker_id` 为主
- Web 页面不再围绕 `request_id`
- 停止行为优先通过断开 SSE 实现

## 9. P1 范围内做什么

- Web 单当前对话页
- Python server -> worker -> CLI -> server -> Web 闭环
- `progress` 和 `complete` 都能展示到 Web
- 连接状态可见
- 处理中状态可见

## 10. P1 明确不做什么

- 多会话并发
- 历史会话持久化
- 多标签页精细隔离
- 请求级恢复
- token 级 delta 流式
- 取消本地 CLI 正在执行的任务
- 将 `request_id` / `delivery_id` 下沉到 worker 协议

