# Agent CLI 与 Java Gateway 会话管理改造初稿

## 背景

Agent 的运行轨迹需要由 Gateway 统一记录。为了让 Gateway 能够关联一次模型请求的来源，需要在请求链路中明确两个标识：

- worker_no：标识当前请求来自哪个 CLI worker。
- session_no：标识当前请求归属哪个会话。

后续 Gateway 记录模型请求、流式响应、运行轨迹、会话切换等信息时，都应基于这两个字段做归属。

## 一、worker_no 生成与传递

worker_no 由 Agent CLI 生成，并作为 CLI 上线请求的入参传给 Gateway。

初步约定：

1. worker_no 的生成规则由 CLI 侧设计。
2. CLI 需要考虑多 worker 场景下的防重问题。
3. CLI 调用 /online 接口时，需要携带 worker_no。
4. Gateway 接收到 worker_no 后，用于识别当前 worker，并关联后续请求轨迹。

worker_no 的具体生成规则暂不在本文展开。

## 二、session_no 生成、复用与切换

session_no 由 Server/Gateway 生成，用于标识一次会话。

### 1. CLI 首次调用 /stream

CLI 第一次调用 /stream 时，如果请求中没有携带 session_no，由 Server 生成新的 session_no。

Server 需要通过 SSE 事件将新的 session_no 返回给 CLI。CLI 收到后，应在本地保存当前会话对应的 session_no，后续模型请求都需要携带该值。

### 2. CLI 固定重连 /stream

CLI 会每 20 分钟固定重连 /stream。

如果 CLI 调用 /stream 时已经携带 session_no，说明这是已有会话的重连场景，Server 无需新建 session_no，应继续复用请求中携带的 session_no。

### 3. CLI 使用 /new 命令

当用户在 CLI 中使用 /new 命令时，CLI 需要调用 Gateway 的 /new 接口通知 Server 创建新会话。

Server 创建新的 session_no 后，下发给 CLI。CLI 更新本地当前会话标识，后续模型请求使用新的 session_no。

### 4. CLI 使用 /resume <session_id> 命令

当用户在 CLI 中使用 /resume <session_id> 时，初步考虑由 CLI 调用 Gateway 的 /resume 接口通知 Server 恢复会话。

这里存在一个待确认点：如果 Server 侧的 session_no 和 CLI 本地的 session_id 本身是一致的，可能不需要额外通知 Server，只需要 CLI 后续请求继续携带该 session_no 即可。

该部分暂不细化，后续根据 CLI 本地会话 ID 与 Server 会话 ID 的关系再确认。

## 三、模型请求入参要求

CLI 每次请求模型时，都需要携带：

- worker_no
- session_no

Gateway 通过这两个字段记录模型请求归属：

- 通过 worker_no 识别请求来自哪个 worker。
- 通过 session_no 识别请求属于哪个会话。
- 二者组合后，可用于还原 Agent 在某个 worker、某个会话下的完整运行轨迹。

## 四、待确认问题

1. CLI 本地的 session_id 和 Server 生成的 session_no 是否保持一致。
2. /resume <session_id> 是否必须通知 Server。
3. worker_no 的生成规则，以及多 worker 防重策略。
4. session_no 通过 SSE 返回时的事件名称和字段结构。