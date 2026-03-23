# CHANGELOG

本文件参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 风格编写。


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
