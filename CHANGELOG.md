# CHANGELOG

本文件参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 风格编写。

## [Unreleased]

### Added
- 新增顶层 vendored 包 `tg_mem/`，裁剪集成 mem0 的 MySQL-only 和记忆提炼能力，用于当前 Agent server 的会话历史恢复与轮次记忆写入。
- 新增 `bu_agent_sdk.server.memory_service`，支持从环境变量构建 tg_mem 适配器、按 `session_id` 恢复 `user/assistant` 历史、每轮注入当前 `user_id` 的全部 memory（带 20,000 字符安全上限），并在每轮对话后增量写入记忆。
- 新增覆盖 server 记忆接入与 server client `user_id` 行为的测试用例。

### Changed
- `bu_agent_sdk.server` 现支持 `user_id` 显式绑定 session；新建或未绑定 session 的 query/query-stream/query-stream-delta 请求必须提供 `user_id`。
- `SessionManager` 与 `AgentSession` 现支持首次历史恢复、清空后重新恢复，以及在非流式/流式最终响应后自动持久化本轮对话。
- `bu_agent_sdk.server.client.AgentClient` 与 `SimpleAgentClient` 现支持 `user_id`，避免生成匿名随机 session 破坏记忆绑定语义。

## [Gateway] - 2026-03-14

### Added
- 新增 `bu_agent_sdk.bootstrap` 共享装配层，统一 CLI 与 Gateway 的 Agent 初始化与工作区指令注入能力。
- 新增 `bu_agent_sdk.bus` 消息总线，提供 `InboundMessage`、`OutboundMessage` 与异步队列抽象。
- 新增 `bu_agent_sdk.runtime` 会话运行时管理，支持按 session 复用 Agent 运行时、串行访问与过期清理。
- 新增 `bu_agent_sdk.channels` 通道层，包含通道基类、通道管理器以及 Telegram 私聊通道实现。
- 新增 `bu_agent_sdk.gateway` 网关层，包含配置解析、消息分发、后台服务与 `bu-agent-gateway` 启动入口。
- 新增 `bu_agent_sdk.heartbeat` 主动唤醒能力，支持读取 `HEARTBEAT.md` 并将任务投递到网关处理链路。
- 新增 Telegram 私聊接入能力，首版支持 polling、allowlist 校验、typing indicator 与长消息分段发送。
- 新增 Zhaohu webhook channel skeleton，支持通过 FastAPI 启动 web server、暴露健康检查路由，并接收通用 webhook payload 后投递到 bus。
- 新增覆盖 bootstrap、bus、runtime、gateway、telegram、heartbeat、integration 的测试用例。

### Changed
- `claude_code.py` 改为复用共享 bootstrap 工厂，避免 CLI 与 Gateway 各自维护独立装配逻辑。
- `cli/app.py` 中 `AGENTS.md` 注入逻辑改为复用共享 session bootstrap，收敛重复实现。
- 更新项目打包入口，新增 `bu-agent-gateway` console script。
- 更新依赖锁文件，纳入网关运行所需的 Telegram 依赖。

### Fixed
- 修复 `bu_agent_sdk.llm.views` 中前向类型注解导致的导入错误，避免相关模块在导入阶段失败。
