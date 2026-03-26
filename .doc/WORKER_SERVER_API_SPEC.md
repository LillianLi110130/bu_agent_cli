# Worker Server API Spec

## 1. 文档目标

本文档说明当前本地 CLI / worker 方案在接入真实服务端时，服务端需要实现的功能、接口、请求参数、响应参数和状态语义。

适用范围：

- [claude_code.py](/d:/llm_project/bu_agent_cli/claude_code.py) 主进程前置登录
- [auth.py](/d:/llm_project/bu_agent_cli/cli/worker/auth.py) OAuth2 / SSO 授权码登录
- [main.py](/d:/llm_project/bu_agent_cli/cli/worker/main.py) worker 进程
- [gateway_client.py](/d:/llm_project/bu_agent_cli/cli/worker/gateway_client.py) worker gateway 客户端
- [runner.py](/d:/llm_project/bu_agent_cli/cli/worker/runner.py) 本地 worker 轮询与回传

## 2. 总体架构

当前链路分成两块服务能力：

1. 认证服务
   - 负责 OAuth2 / SSO 授权码登录
   - 负责把 `code` 换成业务登录结果
   - 返回 `Authorization` 和 `body.userNo`

2. Worker Gateway 服务
   - 负责维护 worker 在线状态
   - 负责接收 IM 消息并按 `worker_id` 入队
   - 负责给本地 worker 提供 `poll`
   - 负责接收 worker 的最终结果 `complete`
   - 负责把最终结果回发给 IM

可以是同一个服务，也可以拆成两个服务：

- `auth_host`
- `server_host`
- `gateway_base_url`

其中：

- `auth_host`：浏览器打开的授权入口
- `server_host`：`code -> login` 的服务端地址
- `gateway_base_url`：worker 轮询消息和回传结果的服务端地址

## 3. 核心业务语义

### 3.1 worker_id

当前版本 `worker_id` 是唯一标识。

当启用认证时，`worker_id` 来自登录响应：

- `response.json()["body"]["userNo"]`

这意味着服务端需要保证：

- 登录成功后，能返回稳定唯一的 `body.userNo`
- 后续 worker 调用 gateway 接口时，用的就是这个值

### 3.2 Authorization

登录成功后，主进程会从登录响应 header 中读取：

- `Authorization`

该值会持久化到本地 `.tg_agent/token.json`，随后 worker 调 gateway 接口时会带在请求头中：

```http
Authorization: <token>
```

建议服务端做两件事：

1. 校验 `Authorization` 是否有效
2. 校验 token 对应身份是否与 `worker_id` 匹配

### 3.3 在线状态

worker 启动后会调用：

- `POST /api/worker/online`

运行中会持续调用：

- `POST /api/worker/poll`

退出时会调用：

- `POST /api/worker/offline`

服务端不应只依赖显式 `offline`，应再做一层 TTL 判活。

## 4. 服务端必须实现的功能

### 4.1 认证登录

服务端需要支持 `code -> 登录态` 的兑换，并返回：

- `Authorization`
- `body.userNo`

### 4.2 worker 在线状态管理

服务端需要按 `worker_id` 维护：

- 在线 / 离线
- 最近一次心跳时间
- 是否允许投递消息

### 4.3 按 worker_id 的消息队列

服务端需要维护一个按 `worker_id` 组织的待处理消息队列。

要求：

- 一个 `worker_id` 可以连续积压多条消息
- worker 在处理第 1 条消息时，第 2 条消息仍然可以继续入队
- `poll` 每次至少能返回 0 或 1 条消息

### 4.4 结果回传与 IM 分发

服务端收到 `complete` 后，需要：

1. 记录完成结果
2. 根据内部映射找到原始 IM 会话
3. 把 `final_content` 回发到 IM

## 5. 认证接口规范

## 5.1 授权入口

这个入口可以是你的服务端，也可以是外部 SSO 服务。

当前客户端会打开：

```text
GET {auth_host}?client_id=...&response_type=code&redirect_uri=http://127.0.0.1:8088/callback
```

请求参数：

- `client_id: string`
- `response_type: string`
  - 固定为 `code`
- `redirect_uri: string`
  - 当前固定为 `http://127.0.0.1:8088/callback`

服务端行为：

- 用户完成登录后，302 跳转到 `redirect_uri`
- 并在 query 中附带 `code`

成功跳转示例：

```text
http://127.0.0.1:8088/callback?code=abc123
```

失败跳转示例：

```text
http://127.0.0.1:8088/callback?error=access_denied
```

### 5.2 登录兑换接口

接口：

```http
GET {server_host}/user-privilege/login?code=<code>
```

请求参数：

- `code: string`

成功响应要求：

- HTTP 状态码：`200`
- Header 中必须有：
  - `Authorization`
- JSON body 中必须有：
  - `returnCode`
  - `body`
  - `body.userNo`

成功响应示例：

```http
HTTP/1.1 200 OK
Authorization: Bearer eyJ...
Content-Type: application/json
```

```json
{
  "body": {
    "defaulted": null,
    "loginTime": "2026-03-26 10:08:47",
    "nickName": null,
    "orgId": "990017",
    "orgName": "托管清算开发室",
    "orgPath": "100001/100003/990001/991165/991187/990017",
    "orgPathName": null,
    "staffId": "80383648",
    "staffName": "李亦梁",
    "status": "ACTE",
    "userNo": "80383648",
    "ystId": "383648"
  },
  "errorMsg": null,
  "returnCode": "SUC0000"
}
```

当前客户端判定成功的规则：

1. HTTP 200
2. Header 中存在 `Authorization`
3. `returnCode == "SUC0000"`
4. `body.userNo` 非空

失败响应建议：

```json
{
  "body": null,
  "errorMsg": "invalid code",
  "returnCode": "ERR0001"
}
```

## 6. Worker Gateway 接口规范

所有 gateway 接口建议都校验：

- `Authorization` header
- JSON body 中的 `worker_id`

建议统一 Content-Type：

```http
Content-Type: application/json
```

### 6.1 `POST /api/worker/online`

作用：

- 标记某个 `worker_id` 已上线

请求头：

- `Authorization: <token>` 可选但推荐必校验

请求体：

```json
{
  "worker_id": "80383648"
}
```

成功响应：

```json
{
  "ok": true
}
```

服务端行为建议：

- 标记该 `worker_id` 在线
- 更新 `last_seen_at`
- 如果该用户已有旧在线实例，按你的业务策略覆盖或拒绝

### 6.2 `POST /api/worker/poll`

作用：

- worker 长轮询下一条待处理消息

请求头：

- `Authorization: <token>` 可选但推荐必校验

请求体：

```json
{
  "worker_id": "80383648"
}
```

成功响应，无消息时：

```json
{
  "messages": []
}
```

成功响应，有消息时：

```json
{
  "messages": [
    {
      "content": "请帮我总结今天的工作"
    }
  ]
}
```

当前客户端约束：

- 只要求 `messages[].content`
- 其他字段当前不会消费

服务端行为建议：

- 每次 poll 最多返回 1 条消息
- 返回消息时，把该消息从 queued 状态转到 inflight 状态
- 更新 `worker_id` 的 `last_seen_at`

### 6.3 `POST /api/worker/complete`

作用：

- worker 回传某次处理的最终结果

请求头：

- `Authorization: <token>` 可选但推荐必校验

请求体：

```json
{
  "worker_id": "80383648",
  "final_content": "这是最终回复内容"
}
```

成功响应：

```json
{
  "ok": true
}
```

服务端行为建议：

1. 找到该 `worker_id` 当前最早一条 inflight 消息
2. 将其标记为 completed
3. 记录：
   - `worker_id`
   - 原始输入内容
   - `final_content`
   - 完成时间
4. 把 `final_content` 发回 IM

注意：

- 当前客户端没有传 message_id / delivery_id
- 所以服务端必须自己维护“按 worker_id 的 inflight FIFO 队列”
- `complete` 时默认完成该 worker 当前最早的一条 inflight 消息

### 6.4 `POST /api/worker/offline`

作用：

- 标记某个 `worker_id` 主动下线

请求头：

- `Authorization: <token>` 可选但推荐必校验

请求体：

```json
{
  "worker_id": "80383648"
}
```

成功响应：

```json
{
  "ok": true
}
```

服务端行为建议：

- 标记离线
- 更新 `last_seen_at`
- 停止向该 `worker_id` 投递新消息

## 7. 服务端内部推荐数据模型

### 7.1 在线 worker 表

建议字段：

```json
{
  "worker_id": "80383648",
  "online": true,
  "last_seen_at": "2026-03-26T10:20:00Z"
}
```

### 7.2 待处理消息表

建议字段：

```json
{
  "id": "msg-001",
  "worker_id": "80383648",
  "content": "hello from im",
  "status": "queued",
  "created_at": "2026-03-26T10:21:00Z"
}
```

### 7.3 处理中消息表

建议字段：

```json
{
  "id": "msg-001",
  "worker_id": "80383648",
  "content": "hello from im",
  "status": "inflight",
  "polled_at": "2026-03-26T10:21:05Z"
}
```

### 7.4 完成结果表

建议字段：

```json
{
  "id": "msg-001",
  "worker_id": "80383648",
  "input_content": "hello from im",
  "final_content": "处理完成",
  "completed_at": "2026-03-26T10:21:20Z"
}
```

## 8. IM 侧与服务端的交互

当前本地 worker 并不直接和 IM 交互。

正确链路是：

1. IM -> 你的 server
2. server -> 入 `worker_id` 消息队列
3. worker `poll`
4. worker 本地执行
5. worker `complete`
6. server -> 回发 IM

因此你的真实服务端需要自己实现“IM 接入层”。

这个 IM 接入层至少要完成：

1. 从 IM 入站消息里确定目标 `worker_id`
2. 把消息写入该 `worker_id` 的 queued 队列
3. 当收到 `complete` 时，把结果回发给对应 IM 会话

## 9. 服务端推荐校验规则

### 9.1 认证校验

建议 gateway 每个接口都校验：

- `Authorization` 存在且有效
- token 对应的用户身份与 `worker_id` 一致

例如：

- token 对应 `userNo = 80383648`
- 请求体 `worker_id = 80383648`
- 才允许通过

### 9.2 在线校验

当 IM 侧要给某个 `worker_id` 投递消息时，建议先检查：

- 该 worker 是否在线
- 是否超过 TTL

如果离线，建议明确返回错误，不要静默吞消息。

### 9.3 TTL 策略

当前 mock server 的 TTL 是 30 秒。

真实服务端建议也做 TTL 判活：

- worker 长时间没有新的 `poll`
- 即判定为离线

这样即使本地进程崩溃、机器断网、`offline` 没发出来，服务端也能自动摘掉在线状态。

## 10. 推荐错误语义

### 10.1 登录接口错误

认证失败时建议：

- HTTP 仍可为 `200`
- 但 `returnCode != "SUC0000"`
- `errorMsg` 给出业务错误信息

### 10.2 gateway 接口错误

建议使用：

- `401`：Authorization 无效
- `403`：worker_id 与 token 身份不匹配
- `404`：worker_id 不存在
- `409`：worker 不在线，不能投递
- `422`：请求参数不合法
- `500`：服务端内部错误

## 11. 最小可用实现清单

如果你现在要先做一版最小真实 server，至少需要：

### 11.1 认证服务

1. 支持浏览器授权
2. 支持 `GET /user-privilege/login?code=...`
3. 返回：
   - `Authorization`
   - `returnCode`
   - `body.userNo`

### 11.2 Gateway 服务

1. `POST /api/worker/online`
2. `POST /api/worker/poll`
3. `POST /api/worker/complete`
4. `POST /api/worker/offline`

### 11.3 内部业务能力

1. `worker_id` 在线状态表
2. `worker_id` 消息队列
3. inflight FIFO 队列
4. 结果记录
5. IM 回发逻辑

## 12. 当前代码与服务端契约的对应关系

认证契约消费位置：

- [auth.py](/d:/llm_project/bu_agent_cli/cli/worker/auth.py)

gateway 契约消费位置：

- [gateway_client.py](/d:/llm_project/bu_agent_cli/cli/worker/gateway_client.py)
- [runner.py](/d:/llm_project/bu_agent_cli/cli/worker/runner.py)

mock 参考实现：

- [mock_auth_server.py](/d:/llm_project/bu_agent_cli/cli/worker/mock_auth_server.py)
- [mock_server.py](/d:/llm_project/bu_agent_cli/cli/worker/mock_server.py)

## 13. 后续扩展点

如果后续要支持更复杂的服务端能力，可以在当前协议上继续扩展：

- `poll` 返回 message_id / delivery_id
- `complete` 带回 request correlation id
- 流式输出
- 审批 / 交互式补问
- 多会话绑定
- 多 worker 实例竞争与抢占

但当前这版实现不依赖这些字段，按本文档的最小协议就能跑通。
