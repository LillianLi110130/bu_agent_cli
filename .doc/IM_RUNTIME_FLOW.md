# IM Runtime Flow

## 1. 当前实现概览

当前这套 IM 桥接实现采用“单本地执行器 + 单后台 worker”的模型：

- 主进程：`python claude_code.py`
- 后台子进程：`python -m cli.worker.main`

两者职责分离：

- CLI 主进程负责真正执行用户输入、slash 命令、skill 命令和普通对话
- worker 子进程只负责和远端 server 通信，把远端消息搬运到本地文件桥，再把结果回传给 server

当前版本已经简化为单标识方案：

- 不再使用 `session_key`
- `worker_id` 是唯一的外部标识
- 本地 bridge 目录名也直接来自 `worker_id` 的安全清洗结果

## 2. 默认启动值

直接启动：

```powershell
python claude_code.py
```

默认值如下：

- `im_enable = True`
- `local_bridge = True`
- `im_worker_id = worker-<hostname>`
- `im_gateway_base_url = http://127.0.0.1:8765`

其中 `<hostname>` 取值顺序是：

1. `COMPUTERNAME`
2. `HOSTNAME`
3. `socket.gethostname()`

## 3. 手动指定启动参数

指定 `worker_id`：

```powershell
python claude_code.py --im-worker-id user-123
```

指定 `worker_id` 和 server：

```powershell
python claude_code.py --im-worker-id user-123 --im-gateway-base-url http://127.0.0.1:9000
```

关闭 IM 和本地 bridge：

```powershell
python claude_code.py --no-im-enable --no-local-bridge
```

## 4. 主进程与 worker 的关系

启动 `claude_code.py` 后，会发生两件事：

1. 主进程初始化 CLI、agent、slash/skill registry、本地 bridge store
2. 如果 `im_enable=True`，主进程再拉起一个后台 worker 子进程

因此当前是：

- 1 个 CLI 主进程
- 1 个 worker 子进程

不是两个都执行任务，而是：

- worker 负责收发远端消息
- CLI 是唯一执行器

## 5. 本地文件桥目录

bridge 根目录在工作区下：

```text
.tg_agent/im_bridge/<worker_binding_id>/
```

这里的 `<worker_binding_id>` 来自 `worker_id` 的文件名安全清洗。

例如：

- `worker_id = worker-laptop`
- 目录可能就是 `.tg_agent/im_bridge/worker-laptop/`

目录结构如下：

```text
.tg_agent/im_bridge/<worker_binding_id>/
  inbox/
    pending/
    processing/
  results/
    results.json
  state/
    sequence.json
  logs/
    worker.log
```

## 6. 任务是怎么排队和执行的

所有输入最终都进入同一个本地队列：

- 本地终端输入
- 远端 IM 消息

入队时会分配一个本地自增 `seq`，写到：

```text
inbox/pending/<20位seq>.json
```

执行规则很简单：

- CLI 只串行执行
- 每次从 `pending` 目录里取 `seq` 最小的一条
- 领取后移动到 `processing`
- 执行完成后写入 `results/results.json`

所以当前不存在“本地和远端并发跑多个任务”的情况。

## 7. 从启动到执行完成的完整链路

### 7.1 CLI 启动

1. 解析启动参数
2. 创建 agent 和 CLI
3. 创建本地 `FileBridgeStore`
4. 如果启用 IM，则启动 worker 子进程
5. CLI 进入交互循环

### 7.2 本地输入

1. 用户在终端输入内容
2. CLI 先把输入写入本地 bridge 队列
3. CLI 从队列取出这条请求
4. 执行 slash / skill / 普通对话
5. 把结果写入 `results/results.json`

### 7.3 远端消息

1. worker 向 server 发送 `online`
2. worker 持续调用 `poll`
3. server 返回一条远端消息
4. worker 把消息写进本地 `pending`
5. CLI 自动发现并执行这条消息
6. CLI 把结果写入 `results/results.json`
7. worker 轮询到该结果后，调用 `complete`
8. server 收到最终结果，再转发给 IM

## 8. 结果文件格式

当前结果不会每条消息落一个单独 JSON，而是统一写到：

```text
results/results.json
```

每条结果至少包含这些字段：

- `request_id`
- `seq`
- `source`
- `input_content`
- `input_kind`
- `final_content`
- `status`

其中：

- `input_content`：原始用户输入
- `input_kind`：输入类型，当前会区分 `text`、`slash`、`skill`、`image`
- `final_content`：最终返回给远端 server 的文本结果

## 9. 当前 server 需要提供的接口

worker 当前依赖 4 个接口：

### 9.1 `POST /api/worker/online`

请求体：

```json
{
  "worker_id": "user-123"
}
```

作用：

- 标记某个 `worker_id` 已上线

### 9.2 `POST /api/worker/poll`

请求体：

```json
{
  "worker_id": "user-123"
}
```

返回体示例：

```json
{
  "messages": [
    {
      "content": "hello from im"
    }
  ]
}
```

作用：

- 返回这个 `worker_id` 下一条待处理消息

### 9.3 `POST /api/worker/complete`

请求体：

```json
{
  "worker_id": "user-123",
  "final_content": "处理完成"
}
```

作用：

- worker 回传最终结果
- server 收到后再转发给 IM

### 9.4 `POST /api/worker/offline`

请求体：

```json
{
  "worker_id": "user-123"
}
```

作用：

- 标记某个 `worker_id` 已下线

## 10. IM 与 server 的交互

当前推荐 IM 只和 server 交互，不直接碰本地 CLI。

IM -> server 时，最小请求可以是：

```json
{
  "worker_id": "user-123",
  "content": "hello from im"
}
```

也就是说，IM 侧现在不需要再维护：

- `session_key`
- `message_id`
- `delivery_id`

在这套简化方案里，server 只需要做这些事：

1. 判断 `worker_id` 是否在线
2. 如果在线，把消息放入该 `worker_id` 的待投递队列
3. 等 worker 调用 `complete`
4. 把 `final_content` 发回对应 IM 会话

## 11. 上线 / 下线通知建议

当前代码路径是：

- worker 启动后调用 `online`
- worker 退出时调用 `offline`

但为了可靠性，server 不应只依赖显式 `offline`，还应维护在线 TTL。

推荐 server 做两类下线判断：

1. 显式下线：收到 `/api/worker/offline`
2. 超时下线：某个 `worker_id` 长时间没有新的 `poll`

这样后续你要把“上线 / 下线”同步到别的通讯软件时，就可以直接基于 server 的在线状态变化做通知。

## 12. 当前 mock server 协议

本地联调时，mock server 支持：

### 12.1 投递消息

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8765/mock/messages `
  -ContentType "application/json" `
  -Body '{"worker_id":"user-123","content":"hello from im"}'
```

### 12.2 查看在线 worker

```powershell
Invoke-RestMethod http://127.0.0.1:8765/mock/online
```

### 12.3 查看完成结果

```powershell
Invoke-RestMethod http://127.0.0.1:8765/mock/completions
```

## 13. 适用边界

当前这版设计只适用于：

- 一个 `worker_id` 对应一个在线本地 CLI
- 单会话
- 单串行执行
- 远端只要求“发一条消息，等最终回复”

如果后续要支持这些能力，再考虑重新扩展协议：

- 同一用户多个并发 CLI
- 一个用户多会话绑定
- 流式输出
- 交互式 slash
- 远端审批 / 补充提问
