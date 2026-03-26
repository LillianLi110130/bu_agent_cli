# Worker Auth SSO Design

## 1. 背景

当前 worker 认证逻辑位于 [auth.py](/d:/llm_project/bu_agent_cli/cli/worker/auth.py) 和 [main.py](/d:/llm_project/bu_agent_cli/cli/worker/main.py) 中，认证发生在 worker 子进程启动阶段。

这与目标存在偏差：

- 目标要求在 [claude_code.py](/d:/llm_project/bu_agent_cli/claude_code.py) 主进程入口阶段先完成登录
- 登录成功后才允许启动 worker 和 CLI
- 登录接口返回的 `user_id` 需要作为最终 `worker_id`
- 登录失败时应直接退出，不允许继续进入 CLI

## 2. 目标

实现一套 OAuth2 / SSO 授权码模式登录流程，满足以下要求：

1. 启动 `claude_code.py` 时先执行登录
2. 登录成功后再启动 worker 和 CLI
3. 从登录接口响应中提取 `user_id`
4. 用 `user_id` 覆盖默认 `worker_id`
5. 登录失败时直接报错退出
6. 本地回调端口固定使用 `8088`

## 3. 非目标

本次设计不处理以下能力：

- 多账户切换
- 无浏览器环境下的替代登录方式
- token 刷新
- 多 worker 并行登录
- server 端真实接口实现

## 4. 现状问题

### 4.1 登录时机过晚

当前认证是在 worker 子进程内执行，CLI 主界面可能已经准备启动。这会导致：

- 登录失败时主进程退出语义不清晰
- `worker_id` 无法在主进程早期被确定
- bridge 目录和 worker 标识在登录前后可能不一致

### 4.2 认证结果不完整

当前认证流程只拿 `Authorization`，没有把 `user_id` 作为一等输出返回给主进程。

### 4.3 回调端口冲突风险

当前默认回调地址是 `127.0.0.1:8765`，容易和本地 mock gateway 冲突。新方案固定改为：

- `http://127.0.0.1:8088/callback`

## 5. 目标流程

### 5.1 启动主链路

新的启动顺序如下：

1. `claude_code.py` 解析参数
2. 主进程加载认证配置
3. 若 `enable_auth = true`，主进程执行 SSO 登录
4. 登录成功后得到：
   - `authorization`
   - `user_id`
5. 主进程将 `user_id` 作为 `worker_id`
6. 主进程创建 bridge store
7. 主进程启动 worker 子进程
8. 主进程进入 CLI 交互循环

### 5.2 授权码登录流程

认证链路采用标准 OAuth2 / SSO 授权码模式：

1. 组装授权 URL
2. 本地启动 `127.0.0.1:8088/callback` HTTP server
3. 浏览器打开授权地址
4. 用户完成登录
5. SSO 回调本地 callback，携带 `code`
6. 主进程用 `code` 请求业务登录接口
7. 从响应中提取：
   - `Authorization`
   - `user_id`
8. 主进程保存认证结果并继续启动

## 6. 模块设计

### 6.1 auth.py 职责调整

[auth.py](/d:/llm_project/bu_agent_cli/cli/worker/auth.py) 从“worker 内部认证工具”调整为“主进程认证服务模块”。

建议保留：

- `load_auth_config()`
- 本地 callback server 启停逻辑

建议新增：

```python
@dataclass(slots=True)
class AuthBootstrapResult:
    authorization: str
    user_id: str
```

```python
async def authenticate_startup(
    config: WorkerAuthConfig,
    base_dir: Path | str | None = None,
    client: httpx.AsyncClient | None = None,
) -> AuthBootstrapResult:
    ...
```

### 6.2 claude_code.py 职责调整

[claude_code.py](/d:/llm_project/bu_agent_cli/claude_code.py) 需要新增“启动前认证”阶段。

建议在主流程中：

1. 读取认证配置
2. 若启用认证，则调用 `authenticate_startup(...)`
3. 将返回的 `user_id` 写入 `args.im_worker_id`
4. 再继续创建 bridge、启动 worker、进入 CLI

### 6.3 worker/main.py 职责调整

[main.py](/d:/llm_project/bu_agent_cli/cli/worker/main.py) 不再负责交互式登录。

新的 worker 只负责：

- 读取主进程准备好的认证结果
- 使用 `Authorization` 调远端接口
- 执行 poll / complete / offline

## 7. 配置设计

### 7.1 配置文件

继续使用 `tg_crab_worker.json`，建议字段如下：

```json
{
  "enable_auth": true,
  "auth_host": "https://sso.example.com/oauth/authorize",
  "server_host": "https://gateway.example.com",
  "client_id": "client-123",
  "redirect_url": "http://127.0.0.1:8088/callback"
}
```

### 7.2 端口策略

本方案已明确固定为：

- `127.0.0.1:8088`

对应 callback path：

- `/callback`

## 8. 认证结果持久化

建议把认证结果持久化到本地 token 文件，供 worker 复用。

建议结构：

```json
{
  "authorization": "Bearer ...",
  "user_id": "user-123"
}
```

建议落盘位置沿用当前实现：

- `.tg_agent/token.json`

## 9. 登录接口契约

### 9.1 授权地址

请求参数：

- `client_id`
- `response_type=code`
- `redirect_uri=http://127.0.0.1:8088/callback`

### 9.2 登录兑换接口

建议调用：

- `GET {server_host}/user-privilege/login?code=<code>`

期望返回：

- Header 中包含 `Authorization`
- Body 或 Header 中包含 `user_id`

## 10. 未确认项

当前仍有 1 个关键接口事实未确认：

- `user_id` 在登录响应中的具体位置

可能位置包括：

- 响应 JSON：`user_id`
- 响应 JSON：`data.user_id`
- 响应 Header：`X-User-Id`

正式落代码前，需要按真实接口定这一点。

## 11. 失败策略

登录相关任一步失败，都直接终止启动：

1. 认证配置不完整
2. 浏览器授权地址打开失败
3. callback 超时
4. callback 未返回 `code`
5. 登录接口返回非 2xx
6. 响应缺少 `Authorization`
7. 响应缺少 `user_id`

失败时行为：

- 不启动 worker
- 不进入 CLI
- 直接输出错误并退出

## 12. 建议实现步骤

### Phase 1

重构 [auth.py](/d:/llm_project/bu_agent_cli/cli/worker/auth.py)：

- 把回调端口固定为 `8088`
- 提供 `authenticate_startup()`
- 返回 `authorization + user_id`

### Phase 2

改造 [claude_code.py](/d:/llm_project/bu_agent_cli/claude_code.py)：

- 启动前先认证
- 成功后覆盖 `im_worker_id`
- 失败则退出

### Phase 3

改造 [main.py](/d:/llm_project/bu_agent_cli/cli/worker/main.py)：

- 去掉 worker 内浏览器登录
- 只读取主进程准备好的认证结果

### Phase 4

补测试：

- auth 成功
- auth 失败阻止启动
- `user_id -> worker_id` 传递正确
- callback 端口固定为 `8088`

## 13. Mock 落地策略

在没有真实接口的情况下，完全可以先落代码并联调。

建议增加一个独立 mock auth server，模拟以下接口：

1. 授权入口
   - 接收 `client_id / response_type / redirect_uri`
   - 返回 302 重定向到 callback，并附带 `code`

2. 登录接口
   - 接收 `code`
   - 返回：
     - `Authorization` header
     - `user_id`

这样可以先把以下链路全部打通：

- 主进程登录前置
- 本地 callback server
- code 兑换
- `user_id -> worker_id`
- token 持久化
- worker 复用授权结果

## 14. 结论

最小正确方案是：

- 把认证从 worker 子进程前移到主进程入口
- 固定本地 callback 端口为 `8088`
- 登录成功后用返回的 `user_id` 作为 `worker_id`
- 登录失败就阻止整个 CLI 启动

在真实接口未提供前，可以先用 mock auth server 落代码，后续再把“登录响应里如何提取 `user_id`”替换成真实接口规则即可。
