# MCP 能力设计方案

## 1. 背景

当前 agent 已经具备：

- 本地工具系统：`read`、`grep`、`edit`、`bash`、`lsp` 等
- 插件系统：本地插件可以提供 commands、skills、agents
- LSP client：通过 stdio 连接 language server，提供代码语义能力
- 用户级 settings：`~/.tg_agent/settings.json`

这些能力覆盖了本地文件、代码语义和插件扩展，但还没有实现 MCP（Model Context Protocol）。为了接入 CodeGraph 这类外部代码智能服务，需要让当前 agent 具备 MCP Host/Client 能力：

- 读取 MCP server 配置
- 启动和管理 MCP server 子进程
- 完成 MCP initialize
- 拉取 MCP tools
- 把 MCP tools 暴露给 agent 调用
- 在 CLI 中查看和管理 MCP 状态

本方案重点参考 Claude Code 的使用体验：**agent 会话启动时自动启用已配置的 MCP server，除非该 server 显式设置为 `disabled`**。

## 2. 设计目标

### 2.1 P0 目标

P0 聚焦 stdio MCP tools client，优先满足 CodeGraph 接入。

能力包括：

- 支持 Claude Code 风格的 `mcpServers` 配置
- 会话启动时自动连接所有未 disabled 的 MCP server
- 支持 stdio transport
- 支持 MCP `initialize`
- 支持 `tools/list`
- 支持 `tools/call`
- 支持统一 `mcp` agent tool 调用 MCP tools
- 为后续动态注册 `mcp__server__tool` 预留结构
- 支持 `/mcp` slash command 查看状态和工具列表
- agent 退出时关闭 MCP server 子进程

### 2.2 P1 目标

P1 在 P0 稳定后增强：

- 支持项目级 `.tg_agent/mcp.json`
- 项目级 MCP 配置首次使用需要用户确认
- 支持 `notifications/tools/list_changed`
- 支持 MCP server instructions 注入或摘要展示
- 支持 MCP tool 动态注册为 agent tools
- 支持 `/mcp stop|reconnect|disable|enable`
- 支持失败重试或手动恢复

### 2.3 非目标

第一阶段不做：

- 把当前 agent 暴露成 MCP server
- HTTP / SSE / Streamable HTTP transport
- MCP resources
- MCP prompts
- OAuth 认证
- MCP server 自动安装
- 对 MCP tool 做复杂语义包装

## 3. 配置设计

## 3.1 配置格式

采用 Claude Code 兼容的顶层 `mcpServers` 格式。

用户级配置文件：

```text
~/.tg_agent/settings.json
```

示例：

```json
{
  "mcpServers": {
    "codegraph": {
      "type": "stdio",
      "command": "codegraph",
      "args": ["serve", "--mcp"],
      "env": {}
    }
  }
}
```

支持扩展字段：

```json
{
  "mcpServers": {
    "codegraph": {
      "type": "stdio",
      "command": "codegraph",
      "args": ["serve", "--mcp"],
      "env": {},
      "disabled": false
    }
  }
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `type` | string | transport 类型。P0 只支持 `stdio` |
| `command` | string | MCP server 启动命令 |
| `args` | string[] | 启动参数 |
| `env` | object | 传给子进程的环境变量 |
| `disabled` | bool | 是否禁用该 server，默认 `false` |

## 3.2 不使用 autoStart

为了贴近 Claude Code，P0 不引入 `autoStart`。

规则：

```text
server 出现在 mcpServers 中
  且 disabled != true
    -> 会话启动时自动连接
```

如果用户不希望启动某个 server，应写：

```json
{
  "mcpServers": {
    "codegraph": {
      "type": "stdio",
      "command": "codegraph",
      "args": ["serve", "--mcp"],
      "disabled": true
    }
  }
}
```

## 3.3 配置优先级

P0 只读取用户级配置：

```text
~/.tg_agent/settings.json
```

P1 再支持项目级：

```text
<workspace>/.tg_agent/mcp.json
```

P1 合并规则：

```text
project .tg_agent/mcp.json > user ~/.tg_agent/settings.json
```

同名 server 整条覆盖，不做字段深合并。

示例：

```json
{
  "mcpServers": {
    "codegraph": {
      "type": "stdio",
      "command": "custom-codegraph",
      "args": ["serve", "--mcp"]
    }
  }
}
```

如果项目级和用户级都定义 `codegraph`，项目级整条覆盖用户级 `codegraph`。

## 4. 启动生命周期

## 4.1 会话启动时自动连接

目标行为：

```text
启动 CLI / 创建 agent context
  -> load_mcp_config()
  -> attach_mcp_manager(ctx)
  -> MCPManager.start_enabled_servers_background()
  -> 每个未 disabled server 后台启动
  -> initialize
  -> tools/list
  -> 注册 MCP tools
```

MCP server 启动失败不应阻断 agent 启动。

失败时：

- 记录 server 状态为 `failed`
- 保存错误信息
- `/mcp status` 可见
- 对应 tools 不注册，或注册为调用时报错的 placeholder

P0 建议：**只注册成功连接的 MCP tools**。

## 4.2 状态模型

每个 server 有一个状态：

```text
configured
starting
running
failed
disabled
stopped
```

含义：

| 状态 | 说明 |
| --- | --- |
| `configured` | 已读取配置，但尚未启动 |
| `starting` | 正在启动进程或 initialize |
| `running` | 已 initialize，并完成 tools/list |
| `failed` | 启动、initialize 或 tools/list 失败 |
| `disabled` | 配置中 `disabled: true` |
| `stopped` | 用户手动停止或正常关闭 |

## 4.3 退出清理

和 LSP 一样，MCP manager 应提供：

```python
async def shutdown_all() -> None
def terminate_all_nowait() -> None
```

接入点：

- CLI 主流程 `finally`
- `shutdown_mcp_manager(ctx)`
- `atexit` fallback

退出时应：

1. 对 running client 发送 MCP shutdown/exit 或尽力关闭
2. 终止 stdio 子进程
3. 清空 manager 中的 clients/spawning tasks

## 5. 模块设计

建议新增目录：

```text
agent_core/mcp/
  __init__.py
  types.py
  config.py
  protocol.py
  client.py
  manager.py
  registry.py
  formatter.py
```

## 5.1 `types.py`

核心类型：

```python
@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    type: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    disabled: bool = False


@dataclass(frozen=True)
class MCPConfig:
    servers: dict[str, MCPServerConfig]


@dataclass
class MCPTool:
    server_name: str
    name: str
    description: str
    input_schema: dict[str, object]


@dataclass
class MCPServerStatus:
    name: str
    state: str
    type: str
    command: str
    args: list[str]
    tool_count: int = 0
    error: str | None = None
```

## 5.2 `config.py`

职责：

- 读取 `~/.tg_agent/settings.json`
- 解析顶层 `mcpServers`
- 校验 server name
- 校验 `type`
- 规范化 `args`
- 规范化 `env`
- 过滤或标记 disabled server

P0 只支持：

```json
{
  "mcpServers": {
    "name": {
      "type": "stdio",
      "command": "...",
      "args": [],
      "env": {},
      "disabled": false
    }
  }
}
```

错误处理：

- `mcpServers` 不是 object：抛配置错误
- server entry 不是 object：抛配置错误
- `type` 不是 `stdio`：P0 标记 unsupported 或抛配置错误
- `command` 缺失：抛配置错误
- `args` 非 list：抛配置错误
- `env` 非 object：抛配置错误

## 5.3 `protocol.py`

MCP 使用 JSON-RPC 2.0。P0 stdio transport 建议按 newline-delimited JSON 处理，和 LSP 的 `Content-Length` framing 分开实现。

职责：

```python
def make_request(id, method, params=None) -> dict
def make_notification(method, params=None) -> dict
def make_response(id, result=None) -> dict
def make_error_response(id, code, message) -> dict
async def read_message(reader) -> dict | None
async def write_message(writer, message) -> None
```

注意：

- 不复用 LSP `protocol.py`
- LSP 是 Content-Length framing
- MCP stdio 是按 JSON-RPC message 在 stdio 上传输，P0 应严格按 MCP stdio 规范实现
- invalid message 不应导致整个 agent 崩溃，应记录错误并关闭该 MCP client

## 5.4 `client.py`

一个 `MCPClient` 管理一个 MCP server 子进程。

核心字段：

```python
class MCPClient:
    server_config: MCPServerConfig
    process: asyncio.subprocess.Process | None
    _pending: dict[int, Future]
    _request_id: int
    _read_task: Task | None
    _tools: dict[str, MCPTool]
    _instructions: str | None
    _server_info: dict | None
```

核心方法：

```python
async def start() -> None
async def initialize() -> None
async def list_tools() -> list[MCPTool]
async def call_tool(name: str, arguments: dict[str, object]) -> object
async def shutdown() -> None
def terminate_nowait() -> None
```

启动流程：

```text
start()
  -> shutil.which(command)
  -> asyncio.create_subprocess_exec(command, *args)
  -> start read loop
  -> initialize()
  -> tools/list
```

initialize 请求：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "...",
    "capabilities": {},
    "clientInfo": {
      "name": "bu-agent-cli",
      "version": "<version>"
    }
  }
}
```

initialize 返回中应保存：

- `serverInfo`
- `capabilities`
- `instructions`

随后发送 initialized notification：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

tools/list：

```json
{
  "method": "tools/list"
}
```

tools/call：

```json
{
  "method": "tools/call",
  "params": {
    "name": "codegraph_explore",
    "arguments": {}
  }
}
```

## 5.5 `manager.py`

`MCPManager` 管理多个 server。

核心方法：

```python
class MCPManager:
    @classmethod
    def from_settings(cls, workspace_root: Path) -> MCPManager

    def start_enabled_servers_background(self) -> None
    async def start_enabled_servers(self) -> None
    async def start_server(self, name: str) -> MCPClient
    async def stop_server(self, name: str) -> None
    async def restart_server(self, name: str) -> MCPClient

    def list_tools(self) -> list[MCPTool]
    async def call_tool(self, server: str, tool: str, arguments: dict) -> object

    def status(self) -> dict[str, object]
    async def shutdown_all(self) -> None
    def terminate_all_nowait(self) -> None
```

启动策略：

- 会话启动后调用 `start_enabled_servers_background()`
- 每个 server 一个 background task
- 同名 server 避免重复启动
- 启动失败保存到 `_errors`
- running client 缓存在 `_clients`

P0 不做 lazy start。因为目标是：

```text
开启 agent 就启用 MCP，除非 disabled
```

## 5.6 `registry.py`

负责把 MCP tools 转成 agent tools。

命名规则：

```text
mcp__<server_name>__<tool_name>
```

示例：

```text
mcp__codegraph__codegraph_explore
mcp__codegraph__codegraph_search
mcp__codegraph__codegraph_callers
```

注册时需要：

- server name 只允许 `[a-zA-Z0-9_-]`
- tool name 只允许 `[a-zA-Z0-9_-]`
- description 做长度限制
- input schema 转为当前 `tool` decorator 支持的 schema

如果动态注册太复杂，P0 可以先提供统一工具：

```python
mcp(server: str, tool: str, arguments: dict)
```

但为了贴近 Claude Code，最终应支持动态工具名。

建议实施顺序：

1. P0 先实现统一 `mcp` 工具，确保 CodeGraph 可调用
2. P1 实现动态注册 MCP tools

## 5.7 `formatter.py`

职责：

- 把 MCP result content 格式化为 JSON payload
- 限制输出长度
- 保留 structuredContent
- 保留 isError
- 对 text content 做截断

统一返回：

```json
{
  "ok": true,
  "server": "codegraph",
  "tool": "codegraph_explore",
  "result": {},
  "truncated": false
}
```

错误：

```json
{
  "ok": false,
  "server": "codegraph",
  "tool": "codegraph_explore",
  "error": "..."
}
```

## 6. CodeGraph 接入

## 6.1 推荐配置

用户在 `~/.tg_agent/settings.json` 中添加：

```json
{
  "mcpServers": {
    "codegraph": {
      "type": "stdio",
      "command": "codegraph",
      "args": ["serve", "--mcp"]
    }
  }
}
```

CodeGraph 使用前需要在项目里完成初始化和索引：

```bash
codegraph init -i
```

否则 server 可能能启动，但查询结果不完整。

## 6.2 预期暴露工具

CodeGraph MCP server 预期提供：

- `codegraph_explore`
- `codegraph_search`
- `codegraph_callers`
- `codegraph_callees`
- `codegraph_impact`
- `codegraph_node`
- `codegraph_files`
- `codegraph_status`

动态注册后 agent 看到：

```text
mcp__codegraph__codegraph_explore
mcp__codegraph__codegraph_search
mcp__codegraph__codegraph_callers
mcp__codegraph__codegraph_callees
mcp__codegraph__codegraph_impact
mcp__codegraph__codegraph_node
mcp__codegraph__codegraph_files
mcp__codegraph__codegraph_status
```

## 6.3 CodeGraph 与 read/grep/LSP 的关系

建议 system/tool guidance 中说明：

- CodeGraph 用于代码图谱探索、调用链、影响面、结构化代码理解
- LSP 用于精确 editor semantic 操作，例如 go to definition、references、hover、diagnostics
- `read` 用于查看具体文件内容
- `grep` 用于文本兜底、非代码内容搜索、注释和配置搜索

CodeGraph 不替代全部工具，但在以下任务应优先：

- “这个功能怎么工作”
- “这个函数被谁调用”
- “修改这个函数会影响哪些地方”
- “这个模块有哪些核心入口”
- “沿调用链解释流程”

## 7. Slash Command 设计

新增：

```text
/mcp
/mcp status
/mcp tools
/mcp tools <server>
/mcp stop <server>
/mcp reconnect <server>
/mcp disable <server>
/mcp enable <server>
```

P0 至少实现：

```text
/mcp
/mcp status
/mcp tools
/mcp tools <server>
```

输出示例：

```text
MCP Service
===========
  enabled servers: 1
  running clients: codegraph

Registered Servers
==================
  ✓ codegraph [running] stdio  8 tools
      command: codegraph serve --mcp
      instructions: available
```

失败示例：

```text
MCP Service
===========
  enabled servers: 1
  running clients: none

Registered Servers
==================
  × codegraph [failed] stdio
      command: codegraph serve --mcp
      error: MCP server command not found: codegraph
```

disabled 示例：

```text
Registered Servers
==================
  - codegraph [disabled] stdio
      command: codegraph serve --mcp
```

`/mcp tools codegraph` 示例：

```text
MCP Tools: codegraph
====================
  codegraph_explore
      Explore repository behavior, architecture, and flows.
  codegraph_search
      Search indexed symbols.
  codegraph_callers
      Find callers of a symbol.
```

## 8. Agent 工具暴露设计

## 8.1 统一 MCP 工具

P0 可以先暴露统一工具：

```python
mcp(
    server: str,
    tool: str,
    arguments: dict[str, object] | None = None,
)
```

描述：

```text
Call a configured MCP server tool. Use /mcp tools or mcp status to inspect available servers and tools.
```

调用示例：

```json
{
  "server": "codegraph",
  "tool": "codegraph_status",
  "arguments": {}
}
```

## 8.2 动态 MCP tools

P1 实现动态注册后，优先让 agent 使用：

```text
mcp__codegraph__codegraph_explore
```

而不是通用：

```text
mcp(server="codegraph", tool="codegraph_explore")
```

动态 tool 的 description 来自 MCP `tools/list`，但需要处理：

- 最大长度
- 去除控制字符
- 保留 server 前缀
- 避免和本地工具重名

## 9. 安全设计

MCP server 是外部代码执行入口，必须比普通本地工具更谨慎。

P0 安全策略：

- 只支持用户级配置
- 只支持 stdio
- 不读取项目 `.tg_agent/mcp.json`
- server command 必须来自用户显式配置
- server 启动失败不影响 agent
- tool 名统一加 `mcp__server__tool` 前缀
- MCP result 做长度限制
- MCP tool description 做长度限制
- env 不做 shell 展开
- 不支持远程 HTTP server

P1 项目级安全策略：

- `.tg_agent/mcp.json` 首次发现时需要用户批准
- 记录 approved/rejected 状态
- rejected server 不启动
- `/mcp status` 显示 pending/rejected

## 10. 日志与观测

建议 logger：

```python
logger = logging.getLogger("agent_core.mcp")
```

关键日志：

- server configured
- server starting
- initialize success/fail
- tools/list success/fail
- tools/list_changed received
- tools/call timeout/fail
- server clean EOF
- process exited

默认只用普通 logger，不单独写文件。

后续如果需要文件日志，可加入：

```json
{
  "mcp": {
    "logFile": "~/.tg_agent/logs/mcp.log"
  }
}
```

但 P0 不做。

## 11. 测试计划

## 11.1 配置测试

- 空 settings 时没有 MCP server
- 能解析 Claude Code 风格 `mcpServers`
- `disabled: true` server 不启动
- 缺失 command 报错
- 非 stdio type 报错或 unsupported
- env 值被规范化为 string

## 11.2 protocol 测试

- request 带 JSON-RPC 2.0
- notification 不带 id
- response 匹配 pending request
- invalid message 不导致全局崩溃

## 11.3 client 测试

用 fake MCP server 子进程或内存 stream 测：

- initialize
- initialized notification
- tools/list
- tools/call
- shutdown
- command not found
- server clean EOF
- request timeout

## 11.4 manager 测试

- start_enabled_servers 启动未 disabled server
- disabled server 跳过
- failed server 不阻断其他 server
- duplicate start 去重
- status 返回 configured/running/failed/disabled
- shutdown_all 关闭所有 client

## 11.5 slash command 测试

- registry 包含 `/mcp`
- `/mcp status` 输出 running/failed/disabled
- `/mcp tools` 输出工具列表
- `/mcp tools codegraph` 只输出 codegraph 工具

## 12. 实施计划

### Phase 1：MCP 配置与 manager 骨架

新增：

- `agent_core/mcp/types.py`
- `agent_core/mcp/config.py`
- `agent_core/mcp/manager.py`

完成：

- 读取用户级 `mcpServers`
- 状态模型
- disabled 处理
- `/mcp status` 可显示配置

### Phase 2：stdio MCP client

新增：

- `agent_core/mcp/protocol.py`
- `agent_core/mcp/client.py`

完成：

- 启动 stdio 子进程
- initialize
- initialized notification
- tools/list
- shutdown

### Phase 3：会话启动自动连接

新增：

- `attach_mcp_manager(ctx)`
- `shutdown_mcp_manager(ctx)`
- `atexit` fallback

接入：

- `tg_crab_main.py`
- `agent_core/bootstrap/agent_factory.py`
- `cli/team/factory.py`

行为：

- agent/context 创建后启动所有未 disabled MCP server
- 失败不阻断启动

### Phase 4：统一 MCP tool

新增：

- `tools/mcp.py`

暴露：

```python
mcp(server, tool, arguments)
```

完成：

- 调用 manager.call_tool
- 格式化 result
- 错误返回

### Phase 5：slash command

新增：

- `cli/mcp_handler.py`

注册：

- `/mcp`

支持：

- status
- tools
- tools `<server>`

### Phase 6：动态工具注册

新增：

- `agent_core/mcp/registry.py`

完成：

- 把 MCP tools 注册为 `mcp__server__tool`
- 动态刷新 tool registry
- description/schema 转换

## 13. Open Questions

1. P0 是否先只做统一 `mcp` 工具，还是直接做动态工具注册？
2. MCP server instructions 是否进入 system prompt？
3. 如果进入 system prompt，是全量注入还是摘要注入？
4. CodeGraph tools 是否需要额外 usage guidance？
5. 项目级 `.tg_agent/mcp.json` 的 approval 状态存放在哪里？
6. MCP tools 是否走现有 HITL approval？

## 14. 推荐决策

建议当前项目采用：

1. 配置格式兼容 Claude Code 顶层 `mcpServers`
2. 不设计 `autoStart`
3. 默认会话启动自动连接所有未 disabled server
4. P0 只支持用户级配置和 stdio
5. P0 优先让 CodeGraph 跑通
6. P0 先提供统一 `mcp` 工具
7. P1 再做动态 `mcp__server__tool` 注册
8. P1 再做项目级 `.tg_agent/mcp.json` 和 approval

一句话总结：

> 当前 agent 的 MCP 方案应实现为 Claude Code 风格的 MCP Host：读取 `mcpServers`，会话启动时自动连接所有未 disabled 的 stdio server，拉取 tools，并通过 `/mcp` 与 agent tools 暴露给模型使用；CodeGraph 是第一优先接入目标。
