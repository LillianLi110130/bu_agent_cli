# LLM 网关化与流式解析下沉方案

## 1. 背景

当前 CLI 直接在本地持有模型厂商 `base_url` 和 `api_key`，并通过 `agent_core/llm` 中的 `ChatOpenAI` 直接访问上游模型接口。

这套方案能工作，但长期有几个明显问题：

- 厂商 API Key 落在本地，不利于治理、轮换、审计和限流
- 厂商协议适配、流式 chunk 解析、tool call 重组都耦合在本地 CLI 中
- 接入多模型、多厂商后，客户端复杂度会持续增长

同时，本项目还有一个明确约束：

- 本地访问公司内部网关链路时，单个同步请求不能超过 90s
- 因此 `CLI -> 网关` 这一段必须支持流式返回
- 不能把本地调用改成“网关收完后一次性同步返回”

所以本方案的目标不是“把所有能力都搬到远端，然后本地走同步接口”，而是：

- 保持 `CLI -> 网关` 为流式
- 把厂商协议适配与流式解析下沉到远端
- 本地继续保留 `Agent Runtime`、事件驱动 loop、工具执行和上下文治理

## 2. 当前真实链路

当前已经落地的代码形态，以及后续线上真实链路，应理解为三段：

`CLI -> Java 网关 -> Python LLM 服务 -> 上游模型`

各段职责如下：

- CLI：
  - 运行本地 `AgentRuntimeLoop`
  - 维护 context、tools、hooks、finish 判定
  - 通过 `ChatGateway` 请求远端 `/llm/query-stream`
- Java 网关：
  - 负责用户鉴权、登录态校验、用户体系接入
  - 对 `/llm/query-stream` 做透明转发
  - 必须保留并透传请求头里的 `Authorization`
  - 不负责改写 LLM 协议语义
- Python LLM 服务：
  - 负责 alias 路由
  - 负责上游模型调用
  - 负责流式 chunk 归一化和 tool call 重组
  - 通过 SSE 把标准化事件回推给 CLI

这意味着：

- Java 网关不是新的 LLM runtime
- Python 服务也不是托管完整 Agent 的远程执行器
- 真正迁走的只是 `LLM 交互层`

## 3. 当前代码边界

### 3.1 必须保留在本地的部分

这些属于本地 runtime 协议和状态机，不应迁走：

- `agent_core/llm/base.py`
  - `ToolDefinition`
  - `BaseChatModel`
- `agent_core/llm/messages.py`
  - `UserMessage`
  - `SystemMessage`
  - `AssistantMessage`
  - `ToolMessage`
  - `DeveloperMessage`
  - `BaseMessage`
- `agent_core/llm/views.py`
  - `ChatInvokeUsage`
  - `ChatInvokeCompletionChunk`
  - `ChatInvokeCompletion`
- `agent_core/llm/schema.py`
  - tool schema 生成与优化
- `agent_core/llm/exceptions.py`
  - `ModelProviderError`
  - `ModelRateLimitError`

本地还必须继续保留：

- `AgentRuntimeLoop`
- hooks
- context compaction
- tool result 回流
- finish guard

### 3.2 适合迁到远端的部分

这些本质上是厂商适配逻辑，适合放到 Python LLM 服务：

- `agent_core/llm/openai/chat.py`
  - provider 请求构造
  - tool schema 序列化
  - provider 响应解析
  - 流式 chunk 解析
  - tool call 分片重组
  - usage、finish_reason、错误语义归一化
- `agent_core/llm/openai/serializer.py`
  - `BaseMessage -> provider request`

## 4. 当前本地 Agent Loop 形态

本地主链已经是事件驱动 runtime loop，不是老式 prompt while loop。

关键位置：

- `agent_core/agent/service.py`
  - `query()`
  - `query_stream()`
- `agent_core/agent/runtime_loop.py`

这意味着当前最重要的能力不是“能调到模型”，而是：

- 事件驱动状态推进
- hook 治理接入
- context maintenance
- tool calling 回流

这些能力必须继续保留在本地。

## 5. 为什么不能直接复用 `/agent/query-stream`

现有 Python server 中已经有：

- `POST /agent/query-stream`
- `POST /agent/query-stream-delta`

但这两个接口的语义是：

- 远端创建或复用 `Agent`
- 远端执行 `agent.query_stream(...)`
- 远端把 `AgentEvent` 序列化为 SSE

这不符合本方案目标，因为我们不是要“把完整 Agent 搬到远端”，而是要“只迁移 LLM 交互层”。

所以本方案必须新增一个独立的纯 LLM 接口：

- `POST /llm/query-stream`

它的职责应该只有：

- 接收统一的 LLM 请求
- 调上游模型
- 归一化流式事件
- 回推标准化 SSE

它不应该：

- 创建 `AgentSession`
- 运行远端 `AgentRuntimeLoop`
- 执行工具
- 管理远端多轮任务状态

## 6. 流式 chunk 重组到底指什么

“流式 chunk 重组”不只是为了绕过 90s 超时。

更核心地说，是因为上游模型在 `stream=true` 时返回的是一串增量片段，而不是一个完整结果。远端需要把这些片段重组成可被本地 runtime 消费的语义对象。

当前主要有两类：

### 6.1 文本 delta 重组

- 每个 chunk 可能只带一小段 `delta.content`
- 需要拼成完整文本

### 6.2 tool call 分片重组

这是最值得下沉的部分。

很多 OpenAI-compatible 接口在流式返回工具调用时：

- `id` 可能只在部分 chunk 中出现
- `arguments` JSON 常常被拆成多段
- 不同 chunk 之间可能只靠 `index` 或隐式顺序关联

因此远端需要：

- 建立 `tool_calls_buffer`
- 根据 `id/index` 建 alias
- 拼接完整 `arguments`
- 在流结束后输出完整 `ToolCall`

这一层适合放到 Python LLM 服务，不应继续压在本地 CLI 上。

## 7. 认证与 token 方案

这里有一个关键约束，必须和现有在线接口保持一致：

- `/llm/query-stream`
- `online`
- `offline`
- 其他 SSE / gateway 接口

都应复用同一套登录态 token。

### 7.1 主路径

主路径不再是单独的 `CRAB_GATEWAY_API_KEY`。

当前推荐的真实行为是：

- 用户先走现有登录流程
- 登录结果写入 `~/.tg_agent/token.json`
- 其中保存的是完整的 `Authorization` 头值
- `ChatGateway` 请求 `/llm/query-stream` 时，直接复用这份 `Authorization`

也就是说，CLI 发给 Java 网关的请求头应与其他 gateway 接口一致，例如：

```http
Authorization: Bearer <shared-login-token>
```

### 7.2 Java 网关要求

Java 网关对 `/llm/query-stream` 的要求是：

- 校验用户登录态
- 原封不动转发请求体
- 保留并透传 `Authorization` 请求头
- 保持 SSE 流式透传

换句话说，Java 网关不应该重新发明一套 LLM 鉴权协议，也不应该把 `/llm/query-stream` 改成同步聚合接口。

### 7.3 兼容兜底

为了兼容本地联调和过渡期，当前代码仍保留两个 fallback：

- `LLM_GATEWAY_AUTHORIZATION`
- `CRAB_GATEWAY_API_KEY`

但它们只是兼容入口，不是目标形态。主路径仍然应该是共享登录态 `Authorization`。

## 8. 配置边界

### 8.1 本地 CLI 配置

本地 `config/model_presets.json` 只应该保留：

- alias
- 网关地址
- 是否视觉模型
- token limits

本地不应该保留：

- 上游真实模型名
- 上游 base_url
- 上游 key env

当前 gateway preset 应该长这样：

```json
{
  "default": "coding-default",
  "auto_vision_preset": "vision-default",
  "image_summary_preset": "vision-default",
  "presets": {
    "coding-default": {
      "provider": "gateway",
      "model": "coding-default",
      "base_url": "http://127.0.0.1:8000",
      "vision": false,
      "max_input_tokens": 200000,
      "max_output_tokens": 128000
    },
    "vision-default": {
      "provider": "gateway",
      "model": "vision-default",
      "base_url": "http://127.0.0.1:8000",
      "vision": true,
      "max_input_tokens": 128000,
      "max_output_tokens": 4096
    }
  }
}
```

注意：

- gateway preset 不再要求 `api_key_env`
- 本地默认也不再宣传“单独的 gateway key”

### 8.2 Python 服务端配置

真实的 alias 到上游路由，应该只存在于 Python 服务端私有配置中，例如：

- `config/gateway_routes.server.json`

它应包含：

- alias
- provider
- upstream_model
- base_url
- api_key_env

例如：

```json
{
  "routes": {
    "coding-default": {
      "provider": "openai",
      "upstream_model": "GLM-5.1",
      "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
      "api_key_env": "OPENAI_API_KEY"
    },
    "vision-default": {
      "provider": "openai",
      "upstream_model": "GLM-4.6V",
      "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
      "api_key_env": "OPENAI_API_KEY"
    }
  }
}
```

这个文件不应提交到仓库，也不应暴露给 CLI。

## 9. 当前代码实现对照

### 9.1 本地 `ChatGateway`

位置：

- `agent_core/llm/gateway/chat.py`

职责：

- 实现 `BaseChatModel`
- 向 `/llm/query-stream` 发起流式请求
- 读取 SSE
- 映射为 `ChatInvokeCompletionChunk`
- 在本地聚合出 `ChatInvokeCompletion`

认证行为：

- 优先读取 `token.json` 中的持久化 `Authorization`
- 若响应头返回新的 `Authorization`，则刷新并持久化
- 若没有登录态，则回退到显式提供的 token

### 9.2 provider-aware 工厂

位置：

- `agent_core/llm/factory.py`
- `config/model_config.py`

职责：

- 解析 `provider`
- `provider == "gateway"` 时创建 `ChatGateway`
- 否则创建 `ChatOpenAI`

### 9.3 主链路接入点

位置：

- `agent_core/bootstrap/agent_factory.py`
- `agent_core/runtime/runner.py`
- `cli/model_switch_service.py`

职责：

- 主 agent
- 子 agent
- 模型切换

都走统一的 provider-aware 创建逻辑。

### 9.4 Python 服务端纯 LLM 路由

位置：

- `agent_core/server/app.py`
- `agent_core/server/llm_gateway.py`
- `agent_core/server/models.py`
- `agent_core/server/route_config.py`

职责：

- 提供 `POST /llm/query-stream`
- 从 server-only 路由配置中解析 alias
- 调用上游模型
- 将上游流式结果标准化后回推给 CLI

## 10. `/llm/query-stream` 请求与响应

### 10.1 请求体

建议继续使用本地统一语义，而不是让 CLI 直接构造 provider 原始格式：

```json
{
  "model": "coding-default",
  "messages": [
    {
      "role": "system",
      "content": "You are a coding assistant."
    },
    {
      "role": "user",
      "content": "帮我定位一下这个报错"
    }
  ],
  "tools": [
    {
      "name": "read",
      "description": "Read file content",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string"
          }
        },
        "required": ["path"]
      },
      "strict": true
    }
  ],
  "tool_choice": "auto",
  "metadata": {
    "session_id": "sid-123",
    "user_id": "u-456",
    "trace_id": "trace-789"
  }
}
```

### 10.2 请求头

CLI 发给第一跳网关的请求头，应该带完整登录态 `Authorization`：

```http
Authorization: Bearer <shared-login-token>
Accept: text/event-stream
```

Java 网关必须保留这个头并继续转发到 Python 服务。

### 10.3 SSE 事件

建议 `/llm/query-stream` 保持简洁的 LLM 语义事件：

- `text`
- `thinking`
- `tool_call`
- `usage`
- `done`
- `error`

示例：

```text
data: {"type":"text","content":"我先看一下这个问题。"}

data: {"type":"thinking","content":"先读取配置再判断当前路由逻辑。"}

data: {"type":"tool_call","tool":"read","args":{"path":"config/model_presets.json"},"tool_call_id":"call_1"}

data: {"type":"usage","session_id":"sid-123","usage":{"total_tokens":1290}}

: done
```

关键约束：

- `tool_call` 事件必须由 Python 服务先完成 provider chunk 归并后再发
- Java 网关不应改写这些事件
- CLI 不再关心 provider 原始 `id/index/arguments` 的分片细节

## 11. `ainvoke()` 与同步语义

在 90s 限制下，不建议再保留独立的长耗时同步接口。

当前推荐做法是：

- `astream()` 走 `/llm/query-stream`
- `ainvoke_streaming()` 在本地消费流式后聚合
- `ainvoke()` 直接复用 `ainvoke_streaming()`

这样可以保证：

- 主链路走流式
- 图片摘要走流式
- compaction summary 走流式
- 不会因为旁路调用偷偷走同步接口而再次踩到 90s 超时

## 12. 风险与注意事项

### 12.1 不要把“同步聚合”误当成“解析下沉”

下面这种模式没有意义：

- provider stream
- 网关收完
- 网关一次性同步返回给 CLI

因为这样依然会撞上 `CLI -> 网关` 这段的 90s 限制。

### 12.2 Java 网关必须保持流式透传

Java 网关虽然不负责解析 LLM 协议，但必须做到：

- 支持 SSE
- 不做整体缓冲
- 不吞掉 `Authorization`
- 不把 `/llm/query-stream` 退化成普通同步代理

### 12.3 切模逻辑也要共享登录态

不仅主链路要复用 `Authorization`，以下场景也要统一：

- `/model` 切换
- 自动视觉切模
- 图片摘要模型选择

不能出现“主链路走共享 token，切模逻辑还要求单独 API key”的割裂状态。

### 12.4 仍需保留 fallback

迁移初期建议保留：

- `ChatOpenAI`
- provider 直连配置
- `LLM_GATEWAY_AUTHORIZATION`
- `CRAB_GATEWAY_API_KEY`

但这些只应作为兼容和排障兜底，不应再成为推荐主路径。

## 13. 推荐实施顺序

建议按下面顺序推进：

1. 本地保留 `AgentRuntimeLoop` 不动
2. Python 服务提供 `/llm/query-stream`
3. Java 网关支持 `Authorization` 校验与 SSE 透明转发
4. CLI 切到 `ChatGateway`
5. gateway preset 默认切到 alias
6. 共享登录态 `Authorization` 接管 `/llm/query-stream`
7. 再逐步收敛本地 provider 直连能力

## 14. 最终结论

本项目不适合：

- 把整个 `agent_core/llm` 从本地移除
- 把整个 `Agent` 主循环搬到远端
- 为 `/llm/query-stream` 单独再造一套和现有登录态割裂的 token 机制

更合理的目标形态是：

- 本地保留 Runtime 协议层
- 本地保留事件驱动 `AgentRuntimeLoop`
- Java 网关负责统一鉴权和透明转发
- Python 服务负责 alias 路由、上游调用和流式标准化
- `CLI -> Java 网关 -> Python 服务` 全链路保持流式
- `/llm/query-stream` 复用和其他 gateway 接口相同的 `Authorization`

一句话总结：

不是“把 Agent 整体搬到云端”，而是“把厂商耦合的 LLM 交互层搬到远端，把事件驱动 Runtime 留在本地，并复用现有统一登录态”。 
