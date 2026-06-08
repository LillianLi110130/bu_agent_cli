# LSP 能力设计方案

## 1. 背景

当前 agent 主要依赖文件工具和 shell 工具理解代码：

- `read` 读取文件片段
- `grep` / `glob_search` 查找文本和文件
- `bash` 执行测试、构建和诊断命令

这套方式适合通用文件操作，但在语义级代码理解上不够稳定。例如：

- 查找定义时容易被同名符号干扰
- 查找引用时需要模型自己组合搜索策略
- 诊断错误需要额外执行语言工具或测试命令
- 对 TypeScript、Python、Java 等语言的项目语义依赖没有统一入口

LSP（Language Server Protocol）正好提供了这些能力：

- diagnostics
- go to definition
- references
- hover
- document symbols
- workspace symbols

因此，本方案目标是在当前 agent 中增加 LSP client 能力，让 agent 可以把 LSP 当成一组工具使用。

## 2. 设计目标

本期只覆盖 P0 和 P1。

### 2.1 P0：只读语义能力

agent 可以调用 LSP 查询：

- 单文件或 workspace 诊断
- 符号定义位置
- 符号引用位置
- hover 信息
- 单文件符号
- workspace 符号
- LSP 服务状态

这些工具只读，不直接修改文件。

### 2.2 P1：文件状态同步

LSP 查询前自动同步当前文件内容，确保 language server 看到的是 agent 当前 workspace 中的真实文件状态。

同步能力包括：

- 首次查询文件时发送 `textDocument/didOpen`
- 文件内容变化后发送 `textDocument/didChange`
- 维护文件版本号
- 缓存并更新 `textDocument/publishDiagnostics`

### 2.3 非目标

本期不做：

- code action 自动应用
- rename 自动写入
- formatting 自动写入
- completion
- agent 自己作为 language server 被 IDE 连接
- 项目级 LSP 配置覆盖

写文件仍交给现有 `edit` / `write` 工具，LSP 只负责提供语义信息。

## 3. 关键设计决策

## 3.1 LSP 配置来源

LSP server 命令和安装位置通常是本机开发环境的一部分，例如：

- `pyright-langserver`
- `typescript-language-server`
- `jdtls`
- `rust-analyzer`

这些命令不适合写入项目配置，否则会把个人机器路径和安装习惯带入项目。

因此本方案把 LSP 配置放在现有 settings 中：

```text
~/.tg_agent/settings.json
```

当前项目已经有 settings 机制：

- `agent_core/runtime_paths.py`
  - `tg_agent_home()`
  - `user_settings_path()`
  - `load_user_settings()`
  - `save_user_settings()`

LSP 配置直接扩展 `settings.json` 的 `lsp` 字段。

### 3.1.1 配置优先级

只保留两层：

```text
内置默认 LSP 配置
  < ~/.tg_agent/settings.json 的 lsp 配置
```

本期不读取：

```text
<workspace>/.tg_agent/lsp.json
<workspace>/.tg_agent/settings.json
```

workspace 只参与：

- 作为 LSP initialize 的 root
- 用于根据 root markers 识别语言服务器 root
- 提供被同步和查询的文件内容

## 3.2 JSON-RPC 使用 2.0

LSP 基于 JSON-RPC 2.0，因此本方案严格使用 JSON-RPC 2.0。

所有 request 和 notification 必须带：

```json
{
  "jsonrpc": "2.0"
}
```

request 必须带 `id`：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "textDocument/definition",
  "params": {}
}
```

notification 不带 `id`：

```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/didOpen",
  "params": {}
}
```

错误响应按 JSON-RPC 2.0 处理：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Method not found"
  }
}
```

不做 JSON-RPC 1.0 兼容。

## 3.3 LSP transport 使用 stdio + Content-Length framing

LSP over stdio 不是一行一个 JSON，而是通过 header framing：

```text
Content-Length: <bytes>\r\n
\r\n
<json payload>
```

因此需要单独实现：

- encode message
- read headers
- parse `Content-Length`
- read exact body bytes
- decode JSON

不要复用普通 line-based JSON 解析。

## 3.4 LSP 作为 workspace 长驻服务

language server 启动成本较高，并且需要维护 workspace index 和打开文件状态。

因此 LSP 不能每次工具调用都冷启动，而应由 `LSPManager` 管理为长驻服务：

```text
SandboxContext
  -> LSPManager
       -> LSPClient(python)
       -> LSPClient(typescript)
       -> ...
```

每个 workspace session 内，相同 language server 复用同一个进程。

## 4. 配置格式

示例：

```json
{
  "default_workspace": "/Users/me/code/project",
  "lsp": {
    "enabled": true,
    "autoStart": true,
    "requestTimeoutSeconds": 10,
    "diagnosticsSettleMs": 300,
    "servers": {
      "python": {
        "command": "pyright-langserver",
        "args": ["--stdio"],
        "extensions": [".py"],
        "languageId": "python",
        "rootMarkers": ["pyproject.toml", "setup.py", "setup.cfg", ".git"]
      },
      "typescript": {
        "command": "typescript-language-server",
        "args": ["--stdio"],
        "extensions": [".ts", ".tsx", ".js", ".jsx"],
        "languageId": "typescript",
        "rootMarkers": ["tsconfig.json", "package.json", ".git"]
      },
      "java": {
        "command": "jdtls",
        "args": [],
        "extensions": [".java"],
        "languageId": "java",
        "rootMarkers": ["pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts", ".git"]
      }
    }
  }
}
```

### 4.1 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `enabled` | bool | 是否启用 LSP 工具 |
| `autoStart` | bool | 第一次调用工具时是否自动启动 server |
| `requestTimeoutSeconds` | number | request 默认超时时间 |
| `diagnosticsSettleMs` | number | 文件同步后等待 diagnostics 通知的短暂窗口 |
| `servers` | object | 语言服务器配置 |
| `command` | string | language server 启动命令 |
| `args` | string[] | 启动参数 |
| `extensions` | string[] | 文件后缀匹配 |
| `languageId` | string | LSP `TextDocumentItem.languageId` |
| `rootMarkers` | string[] | 从文件向上查找 server root 的标记文件 |

### 4.2 默认配置

内置默认配置只提供通用 server 声明，不保证用户机器已经安装对应命令。

建议默认：

```json
{
  "enabled": true,
  "autoStart": true,
  "requestTimeoutSeconds": 10,
  "diagnosticsSettleMs": 300,
  "servers": {
    "python": {
      "command": "pyright-langserver",
      "args": ["--stdio"],
      "extensions": [".py"],
      "languageId": "python",
      "rootMarkers": ["pyproject.toml", "setup.py", "setup.cfg", ".git"]
    },
    "typescript": {
      "command": "typescript-language-server",
      "args": ["--stdio"],
      "extensions": [".ts", ".tsx", ".js", ".jsx"],
      "languageId": "typescript",
      "rootMarkers": ["tsconfig.json", "package.json", ".git"]
    },
    "java": {
      "command": "jdtls",
      "args": [],
      "extensions": [".java"],
      "languageId": "java",
      "rootMarkers": ["pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts", ".git"]
    }
  }
}
```

默认 `enabled: true`，让 LSP 能力开箱可用。由于 client 是 lazy spawn，启动 agent 本身不会立即启动 language server；只有工具实际请求匹配文件时才会尝试启动。若用户机器未安装对应命令，manager 会记录失败并返回清晰错误。

## 5. 模块设计

新增目录：

```text
agent_core/lsp/
  __init__.py
  config.py
  protocol.py
  client.py
  manager.py
  formatter.py
  types.py

tools/lsp.py
```

## 5.1 `agent_core/lsp/config.py`

职责：

- 定义 LSP 配置 dataclass
- 提供内置默认配置
- 从 `settings.json` 读取 `lsp`
- 合并默认配置和用户配置
- 校验配置结构

建议类型：

```python
@dataclass(frozen=True)
class LSPServerConfig:
    name: str
    command: str
    args: list[str]
    extensions: list[str]
    language_id: str
    root_markers: list[str]


@dataclass(frozen=True)
class LSPConfig:
    enabled: bool
    auto_start: bool
    request_timeout_seconds: float
    diagnostics_settle_ms: int
    servers: dict[str, LSPServerConfig]
```

建议入口：

```python
def load_lsp_config() -> LSPConfig:
    settings = load_user_settings()
    raw_lsp = settings.get("lsp")
    return merge_with_defaults(raw_lsp)
```

校验原则：

- `lsp` 缺失时返回默认配置
- `servers` 中无效项跳过或报错，需要保持一致
- `command` 为空时该 server 无效
- `extensions` 必须是非空字符串列表
- 所有后缀规范化成以 `.` 开头的小写形式

建议对无效用户配置抛 `ValueError`，因为 settings JSON 本身是用户可修复配置。

## 5.2 `agent_core/lsp/protocol.py`

职责：

- JSON-RPC 2.0 message 构造
- LSP Content-Length framing
- request / notification / response 类型判断

建议函数：

```python
def make_request(request_id: int, method: str, params: dict | None) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params or {},
    }


def make_notification(method: str, params: dict | None) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
    }


def encode_message(message: dict) -> bytes:
    ...


async def read_message(reader: asyncio.StreamReader) -> dict:
    ...
```

framing 注意事项：

- `Content-Length` 是 body bytes 长度，不是字符数
- body 使用 UTF-8
- header 大小写建议兼容，但写出时使用 `Content-Length`
- 读取 body 要 exact read

## 5.3 `agent_core/lsp/client.py`

职责：

- 管理单个 language server 进程
- 发送 request / notification
- 匹配 response 到 pending future
- 接收 server notification
- 维护 diagnostics 缓存
- 维护已同步 document 状态

建议类型：

```python
@dataclass
class SyncedDocument:
    uri: str
    language_id: str
    version: int
    content_hash: str
```

核心状态：

```python
class LSPClient:
    process: asyncio.subprocess.Process | None
    request_id: int
    pending: dict[int, asyncio.Future]
    synced_documents: dict[Path, SyncedDocument]
    diagnostics_by_uri: dict[str, list[LSPDiagnostic]]
    read_task: asyncio.Task | None
```

核心方法：

```python
async def start(self) -> None
async def initialize(self) -> None
async def shutdown(self) -> None

async def request(self, method: str, params: dict, timeout: float | None = None) -> Any
async def notify(self, method: str, params: dict) -> None

async def ensure_document_synced(self, path: Path) -> None
async def diagnostics(self, path: Path | None = None) -> list[LSPDiagnostic]
async def definition(self, path: Path, line: int, character: int) -> Any
async def references(self, path: Path, line: int, character: int) -> Any
async def hover(self, path: Path, line: int, character: int) -> Any
async def document_symbols(self, path: Path) -> Any
async def workspace_symbols(self, query: str) -> Any
```

### 5.3.1 初始化

启动后发送：

```text
initialize
initialized
```

`initialize` 参数至少包括：

- `processId`
- `rootUri`
- `workspaceFolders`
- `capabilities`

capabilities 可以先保持最小集合，后续再扩展。

### 5.3.2 读循环

读循环持续读取 server message：

- 如果包含 `id` 且匹配 pending request，完成对应 future
- 如果是 `textDocument/publishDiagnostics`，更新 diagnostics cache
- 其他 notification 暂时忽略
- server request 可以先返回 `Method not found` 或忽略，后续再补

### 5.3.3 文件同步

`ensure_document_synced(path)` 流程：

```text
读取当前文件内容
计算 sha256
获取 uri 和 languageId

如果文件未同步：
  version = 1
  notify textDocument/didOpen
  记录 SyncedDocument

如果文件已同步且 hash 变化：
  version += 1
  notify textDocument/didChange
  更新 SyncedDocument

如果 hash 不变：
  不发送 didChange
```

`didOpen` 参数：

```json
{
  "textDocument": {
    "uri": "file:///...",
    "languageId": "python",
    "version": 1,
    "text": "..."
  }
}
```

`didChange` 参数：

```json
{
  "textDocument": {
    "uri": "file:///...",
    "version": 2
  },
  "contentChanges": [
    {
      "text": "..."
    }
  ]
}
```

本期使用 full document sync，不做 incremental sync。

## 5.4 `agent_core/lsp/manager.py`

职责：

- 按 workspace 管理 LSP clients
- 根据文件后缀选择 server
- 根据 root markers 解析 server root
- 懒启动 server
- 提供 shutdown all

建议方法：

```python
class LSPManager:
    @classmethod
    def from_settings(cls, workspace_root: Path) -> "LSPManager":
        ...

    async def for_file(self, path: Path) -> LSPClient:
        ...

    async def diagnostics(self, path: Path | None = None) -> list[LSPDiagnostic]:
        ...

    async def shutdown_all(self) -> None:
        ...
```

### 5.4.1 server 选择

根据文件后缀选择：

```text
Path("foo.py").suffix.lower() -> ".py" -> python server
```

如果多个 server 匹配同一个后缀，使用用户配置顺序中的第一个。

### 5.4.2 root 解析

从文件所在目录向上查找 `rootMarkers`：

```text
file dir
  -> parent
    -> parent
      -> workspace root
```

查到 marker 则用该目录作为 LSP root。

如果没有找到，回退到 `workspace_root`。

查找不能越过 `workspace_root`。

### 5.4.3 client cache key

建议使用：

```text
server name + resolved root path
```

这样同一个 workspace 中如果存在多个独立 package，可以为不同 root 启动不同 client。

## 5.5 `agent_core/lsp/formatter.py`

职责：

- 把 LSP 原始返回压缩成 agent 友好的 JSON 字符串
- 把 0-based line/character 转为 1-based
- 添加文件 preview
- 限制输出大小

统一返回结构：

```json
{
  "ok": true,
  "tool": "lsp_definition",
  "results": [
    {
      "file": "/abs/path/foo.py",
      "line": 42,
      "character": 8,
      "preview": "def target(...):"
    }
  ]
}
```

错误返回结构：

```json
{
  "ok": false,
  "tool": "lsp_definition",
  "error": "LSP server not found: pyright-langserver"
}
```

## 5.6 `tools/lsp.py`

职责：

- 暴露 agent 可调用工具
- 校验路径在 sandbox 内
- 做 1-based 到 0-based 转换
- 调用 `ctx.lsp_manager`
- 返回 formatter 结果

建议工具：

```python
lsp_status()
lsp_diagnostics(file_path: str | None = None)
lsp_definition(file_path: str, line: int, character: int)
lsp_references(file_path: str, line: int, character: int)
lsp_hover(file_path: str, line: int, character: int)
lsp_document_symbols(file_path: str)
lsp_workspace_symbols(query: str, file_path: str)
```

`workspace_symbols` 使用 `file_path` 选择对应的 language server 和 workspace root，因此不会因为仓库中存在多种语言而启动所有 server。

### 5.6.1 行列号约定

工具参数面向 agent 和用户，使用 1-based：

```text
line=1 表示第一行
character=1 表示第一列
```

内部调用 LSP 前转成 0-based：

```python
lsp_line = line - 1
lsp_character = character - 1
```

返回结果再转回 1-based。

## 6. 与现有代码的接入点

## 6.1 `SandboxContext`

在 `tools/sandbox.py` 的 `SandboxContext` 增加：

```python
lsp_manager: Any | None = None
```

这样 LSP 生命周期和当前 CLI session 绑定。

## 6.2 Agent 创建入口

当前代码里不只有一个 agent 创建入口。

交互式 CLI 主路径走：

```text
tg_crab_main.py:create_agent()
```

worker 默认路径会走：

```text
agent_core/bootstrap/agent_factory.py:create_agent()
```

team member agent 走：

```text
cli/team/factory.py:create_team_member_agent()
```

因此 LSP manager 的接入不能只放在 `agent_core/bootstrap/agent_factory.py`。应提供一个共享 helper，并在所有创建 `SandboxContext` 的 agent bootstrap 路径中调用。

建议新增：

```python
def attach_lsp_manager(ctx: SandboxContext) -> None:
    ctx.lsp_manager = LSPManager.from_settings(ctx.root_dir)
```

然后分别在以下位置调用：

```python
ctx = SandboxContext.create(root_dir)
attach_lsp_manager(ctx)
```

至少覆盖：

- `tg_crab_main.py:create_agent()`
- `agent_core/bootstrap/agent_factory.py:create_agent()`
- `cli/team/factory.py:create_team_member_agent()`

测试或 echo-only runtime 可以不强制接入 LSP，除非测试需要覆盖 LSP 工具。

交互式 CLI 主路径的实际接入点是：

```python
ctx = SandboxContext.create(root_dir)
attach_lsp_manager(ctx)
```

如果配置 `enabled: false`，仍可以创建 manager，但 manager 返回 disabled 状态。

## 6.3 工具注册

在 `tools/__init__.py` 中导入 LSP 工具，并加入 `ALL_TOOLS`。

建议只有在配置 enabled 时工具实际可用，但工具注册可以始终存在。

如果 disabled，工具返回：

```json
{
  "ok": false,
  "error": "LSP is disabled by settings. Set settings.lsp.enabled to true in ~/.tg_agent/settings.json."
}
```

这样系统提示和工具 schema 稳定，不需要动态增删工具。

## 7. 主要调用流程

## 7.1 查询 definition

```text
agent 调 lsp_definition(file_path, line, character)
  -> tools/lsp.py 解析 sandbox path
  -> ctx.lsp_manager.for_file(path)
  -> manager 选择 server config
  -> manager 解析 root
  -> manager 获取或启动 LSPClient
  -> client ensure_document_synced(path)
  -> client request textDocument/definition
  -> formatter 转换位置和 preview
  -> 返回 JSON
```

## 7.2 查询 diagnostics

```text
agent 调 lsp_diagnostics(file_path)
  -> tools/lsp.py 解析 path
  -> manager.for_file(path)
  -> client.ensure_document_synced(path)
  -> 等待 diagnosticsSettleMs
  -> 读取 diagnostics cache
  -> formatter 输出
```

如果 `file_path` 为 `None`：

- 返回当前已启动 clients 的 diagnostics cache
- 不主动扫描整个 workspace

这样避免一次工具调用触发大量文件打开。

## 8. 错误处理

### 8.1 LSP 未启用

返回：

```json
{
  "ok": false,
  "error": "LSP is disabled by settings. Set settings.lsp.enabled to true in ~/.tg_agent/settings.json."
}
```

### 8.2 没有匹配 server

返回：

```json
{
  "ok": false,
  "error": "No LSP server configured for extension .py"
}
```

### 8.3 server 命令不存在

启动前可以用 `shutil.which(command)` 检查。

返回：

```json
{
  "ok": false,
  "error": "LSP server command not found: pyright-langserver"
}
```

### 8.4 request 超时

返回：

```json
{
  "ok": false,
  "error": "LSP request timed out: textDocument/definition"
}
```

### 8.5 server 退出

如果 server 意外退出：

- 清理 pending futures
- 标记 client dead
- 下次请求可尝试重启一次
- 重启仍失败则返回错误

## 9. 安全边界

LSP 工具必须遵守现有 sandbox：

- `file_path` 必须通过 `SandboxContext` 或 `resolve_target_path` 解析
- 不允许读取 workspace 之外文件，除非该路径在 allowed dirs 中
- LSP 返回的位置如果在 sandbox 外，formatter 应标记但不读取 preview
- `workspace_symbols` 返回的位置也要过 allowed check

注意：language server 进程本身可能读取 workspace 内依赖和配置。这个行为与在 IDE 中启动 language server 类似，是启用 LSP 后接受的能力边界。

## 10. 测试计划

### 10.1 config tests

- `settings.json` 缺失时返回默认 enabled 配置
- settings 配置可以关闭 LSP
- settings 配置可以覆盖 server command / args
- extensions 自动规范化
- 非 object 的 `lsp` 抛错
- 非法 server 配置抛错或跳过，按实现选择保持一致

### 10.2 protocol tests

- request 包含 `jsonrpc: "2.0"`
- notification 不包含 `id`
- encode message 使用正确 `Content-Length`
- read message 能解析 header + body
- body length 按 bytes 而不是字符数计算

### 10.3 client tests

使用 fake stream 或 fake subprocess：

- request id 可以匹配 response
- notification 不创建 pending future
- `publishDiagnostics` 更新 cache
- `ensure_document_synced` 首次发送 `didOpen`
- 文件 hash 不变不发送 `didChange`
- 文件 hash 变化发送 `didChange` 且 version 递增

### 10.4 manager tests

- `.py` 选择 python server
- `.ts` 选择 typescript server
- `.java` 选择 java server
- 未配置后缀返回错误
- root marker 不能越过 workspace root
- 同一 server + root 复用 client

### 10.5 tool tests

- disabled 时返回清晰错误
- 路径逃逸被 sandbox 拒绝
- line/character 从 1-based 转 0-based
- formatter 返回 1-based
- command 不存在时返回清晰错误

### 10.6 smoke test

在本地安装对应 language server 后：

```text
pyright-langserver --stdio
typescript-language-server --stdio
jdtls
```

手动验证：

- Python 文件 diagnostics
- Python definition
- TypeScript 文件 diagnostics
- TypeScript references
- Java 文件 diagnostics
- Java references

## 11. 实现顺序

建议按以下顺序落地：

1. `agent_core/lsp/config.py`
2. `agent_core/lsp/protocol.py`
3. `agent_core/lsp/formatter.py`
4. `agent_core/lsp/client.py` 的 JSON-RPC request/response
5. `client.ensure_document_synced`
6. `agent_core/lsp/manager.py`
7. `tools/lsp.py`
8. `SandboxContext` 和各 agent 创建入口接入
9. `tools/__init__.py` 注册工具
10. 单元测试
11. 真实 language server smoke test

## 12. 后续扩展

P2 可以增加只返回 edits、不自动写入的工具：

- `lsp_code_actions`
- `lsp_rename_preview`
- `lsp_format_preview`

P3 可以增加运行状态与治理：

- `lsp_restart`
- `lsp_shutdown`
- `lsp_status` 展示 server pid、root、同步文件数、诊断数
- 多 workspace root 支持
- 更细的 server capabilities 展示

这些扩展不影响本期 P0/P1 的核心架构。
