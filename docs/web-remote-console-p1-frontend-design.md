# Web 远程对话 P1 前端设计

本文档描述的是下一步前端收敛方案，和主设计文档保持一致。

说明：

- 本文档先更新前端设计方向，不代表当前代码已经完全切换
- 前端页面以单当前对话为核心
- 前端不再围绕 `request_id` 建模

## 1. 技术栈

- React
- Ant Design
- `less.module`

## 2. 页面定位

前端页面只做一件事：

- 连接一台本地 `worker_id`
- 展示这台 worker 当前对话的过程和结果

页面不承担：

- 历史会话管理
- 多对话切换
- 多标签并发状态机

## 3. 页面结构

页面结构保持单窗口聊天形态：

- 顶部轻量连接状态
- 中间聊天消息区
- 底部固定输入区

中间聊天区自己滚动，页面整体不滚动。

## 4. 数据流

### 4.1 发送消息

前端发送消息：

```text
POST /web-console/messages
```

请求体：

- `workerId`
- `sessionId`
- `content`

发送成功后：

- 页面立即进入 `submitted`
- 然后等待当前 worker 的 SSE 事件

### 4.2 订阅事件

前端不再按 `request_id` 建立 SSE，改为：

```text
GET /web-console/workers/{workerId}/events
```

使用方式：

- `fetch() + ReadableStream`

原因：

- 需要带鉴权头
- 需要自定义中止逻辑

### 4.3 停止当前接收

前端不再依赖单独的 `stop` API。

用户点击“停止接收”时：

- 直接中止当前 SSE 的 `fetch`
- 页面提示：
  - 已停止当前页面接收
  - 本地任务可能仍在继续执行

## 5. 前端状态机

前端页面状态收敛成：

- `idle`
- `submitting`
- `submitted`
- `processing`
- `completed`
- `failed`

说明：

- `submitted`
  - server 已接受消息
- `processing`
  - worker 已取走消息
- `completed`
  - 最终结果已返回
- `failed`
  - 当前请求失败

## 6. SSE 事件到 UI 的映射

### 6.1 `submitted`

- 插入或更新一条系统提示
- 页面整体状态切到 `submitted`

### 6.2 `processing`

- 更新系统提示为“处理中”
- 页面整体状态切到 `processing`

### 6.3 `progress`

- 在当前 assistant 气泡中持续追加中间文本
- 不新开多个最终消息气泡

### 6.4 `completed`

- 将当前 assistant 气泡收敛为最终结果
- 页面整体状态切到 `completed`

### 6.5 `failed`

- 插入失败提示
- 页面整体状态切到 `failed`

## 7. 服务层设计

### 7.1 普通 HTTP

普通 HTTP 请求统一使用：

- `axios`

包括：

- `GET /web-console/workers/{workerId}`
- `POST /web-console/messages`

### 7.2 SSE

SSE 使用：

- `fetch() + ReadableStream`

包括：

- `GET /web-console/workers/{workerId}/events`

不使用 `EventSource`，因为：

- 需要带鉴权头
- 需要手动中止

## 8. 组件建议

### 8.1 页面容器

- `RemoteConsolePage`

职责：

- 维护当前 worker 状态
- 维护当前 SSE 生命周期
- 维护页面级请求状态

### 8.2 消息区

- `ConversationView`

职责：

- 展示 user / assistant / system / error 消息
- 处理 assistant 处理中间文本的持续更新

### 8.3 输入区

- `ComposerPanel`

职责：

- 输入消息
- 发送消息
- 中止当前 SSE 接收

## 9. 当前前端设计明确不做什么

- 左侧历史会话栏
- 按 `request_id` 的 SSE 管理
- 请求级恢复
- 多请求并发
- token 级 delta 渲染
- 通过 stop 接口取消本地 CLI 任务

