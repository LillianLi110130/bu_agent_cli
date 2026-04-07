# Bash 长驻命令优化方案

## 1. 目标重述

这份方案聚焦的核心问题不是“让用户手动查看后台任务”，而是：

- agent 执行 `npm run dev`、`pnpm dev`、`next dev`、`vite dev`、`webpack --watch` 这类长驻命令时不被卡死
- agent 能等待服务真正 ready
- ready 后 agent 能自动继续后续步骤

换句话说，目标不是把 shell 任务做成一个需要用户手动管理的产品能力，而是把它做成 agent 内部可用的执行能力。

## 2. 当前问题

当前 `tools/bash.py` 的执行模型适合短命令，不适合长驻命令：

- 通过 `subprocess.Popen(..., shell=True, stdout=PIPE, stderr=PIPE)` 启动
- 默认等待进程退出
- 退出后再 `communicate()` 收集输出

这会导致两个问题：

### 2.1 长驻命令天然不退出

例如：

- `npm run dev`
- `pnpm dev`
- `next dev`
- `vite dev`
- `webpack --watch`

这些命令本来就会持续运行，因此当前 `bash` 会一直等待，agent 无法进入后续步骤。

### 2.2 即使“放后台”也可能卡住

即便模型尝试用 shell 语法把命令放后台，仍可能卡住，原因是：

- 子进程继承了当前工具调用的 `stdout/stderr` pipe
- 父 shell 退出后，子进程仍持有 pipe
- `communicate()` 要等 pipe 真正关闭才会返回

所以问题的根因不是超时时间短，而是当前 `bash` 的执行假设是“命令最终会结束并回收输出”。

## 3. 方案方向

建议把 `bash` 从“同步命令执行器”升级成“支持长驻服务启动”的工具，但启动逻辑仍然保持在 `bash` 内，而不是单独造一个 `dev server tool`。

核心思路是：

- `bash` 仍负责启动命令
- 对长驻命令走“托管启动”分支
- 工具内部把进程与当前工具调用解耦
- 工具内部等待服务 ready
- ready 后直接把结果返回给 agent
- agent 继续执行下一步

这里的关键不是“后台”本身，而是“脱离当前阻塞式 pipe + 等 ready + 继续执行”。

## 4. 期望行为

### 4.1 短命令

短命令仍保持现状：

- 执行
- 等待退出
- 返回 stdout/stderr/returncode

适用示例：

- `git status`
- `pytest -q`
- `dir`
- `type file.txt`

### 4.2 长驻命令

长驻命令改成“启动并等待 ready”：

1. `bash` 启动进程
2. stdout/stderr 不再绑在当前工具调用的 pipe 上
3. 输出改写到日志文件或内部缓冲
4. 工具进入 ready 等待阶段
5. 一旦检测到服务 ready，就立即返回“服务已就绪”
6. agent 根据返回结果继续执行，例如访问页面、跑接口测试、继续构建

对 agent 而言，它看到的不是“任务句柄”，而是一个像这样的成功结果：

- 进程已经启动
- 服务已经 ready
- 访问地址或端口是多少
- 日志文件在哪里

这才是最适合 agent 自动串联后续步骤的返回形式。

## 5. 不建议的方向

### 5.1 仅增加 timeout

这只能延后卡住的时间，不能解决问题。

### 5.2 依赖模型自己拼接后台 shell 语法

例如让模型自己写：

- `start`
- `nohup`
- `&`
- `Start-Process`

这种方式不稳定、跨平台差异大，而且仍然容易继承 pipe 导致工具调用不返回。

### 5.3 把重点放在 `/tasks`

`/tasks`、`/task`、`/task_cancel` 最多只能作为调试或人工排障入口，不应该成为主流程。

你现在要解决的是：

- agent 自己能继续往下跑

而不是：

- 用户再手动接管这个长任务

## 6. 推荐设计

## 6.1 bash 增加“长驻服务模式”

建议 `bash` 内部支持两种模式：

- 普通命令模式
- 长驻服务模式

长驻服务模式的语义不是“单纯后台执行”，而是：

- `detach process`
- `capture logs`
- `wait until ready`
- `return ready metadata`

也就是“托管启动并等服务可用”。

## 6.2 主返回值应保持通用

参考 Claude Code 的实现，`bash` 本身仍应是通用 shell 工具，不建议把服务语义直接编码进输出 schema。

更合适的返回结构是：

- `ok`
- `command`
- `cwd`
- `returncode`
- `stdout`
- `stderr`
- `timed_out`
- 可选 `backgroundTaskId`
- 可选 `persistedOutputPath`

也就是说：

- `bash` 负责“执行”和“后台化”
- 后续是否等待某个日志信号、是否判断服务 ready，由通用的输出读取/观察能力承担

这样更贴近 Claude 的 `BashTool + TaskOutput/Monitor` 组合方式，也更不容易把 `bash` 做成针对 dev server 的特化工具。

## 6.3 工具内部需要一个 service handle

虽然对 agent 不一定暴露 `task_id`，但实现上仍然建议在内部维护一个 service handle，用于：

- 记录 pid
- 记录日志路径
- 记录 ready 状态
- 在当前会话后续步骤里复用这个服务
- 出错时取消或清理进程

也就是说：

- 对外不强调“任务系统”
- 对内仍建议有一层统一的托管对象

这层对象可以叫：

- `ManagedShellProcess`
- `ServiceHandle`
- `ShellServiceSession`

名字不重要，关键是它代表“这个进程已经脱离当前工具调用，但仍受会话管理”。

## 7. 执行模型

### 7.1 短命令执行模型

保持当前路径：

- `Popen`
- 等待退出
- `communicate`
- 返回结构化结果

### 7.2 长驻服务执行模型

需要和短命令彻底分流：

1. 启动 shell 进程
2. stdout/stderr 定向到日志文件，不再使用当前 `PIPE`
3. 创建托管句柄
4. 启动内部 watcher
5. watcher 持续检查 ready 条件
6. ready 后工具返回
7. 若进程提前退出，则返回失败
8. 若等待超时，则返回超时

这里最关键的一点是：

- 工具调用本身可以等待 ready
- 但不能再等待“命令退出”

## 8. ready 检测

这是整个方案的核心。

你要的不是“进程活着”，而是“服务已经能用了”。

### 8.1 支持的 ready 条件

建议支持三种，按优先级组合：

- 端口探测
- 日志关键字
- HTTP 健康检查

### 8.2 推荐检测顺序

建议按下面的顺序执行：

1. 检查进程是否仍存活
2. 如果已退出，直接判定失败
3. 如果配置了日志关键字，扫描最近日志
4. 如果配置了端口，检测端口是否监听
5. 如果配置了健康检查 URL，请求健康检查
6. 任一 ready 条件满足，则返回 ready

### 8.3 常见命令的默认 hint

可以给常见命令内置默认 hint：

- `vite dev`
  - 默认端口 `5173`
  - 日志关键字 `Local:`
- `next dev`
  - 默认端口 `3000`
  - 日志关键字 `Ready in`
- `npm run dev`
  - 默认不猜具体框架，但允许通过日志推断框架后补齐 hint

### 8.4 返回给 agent 的信息

参考 Claude 的实现，后台启动时 `bash` 更适合返回：

- `backgroundTaskId`
- `persistedOutputPath`

随后由通用输出读取能力读取日志，并等待：

- 某个关键日志出现
- 某个端口打开
- 某个健康检查通过

这样 agent 仍然可以继续：

- 访问页面
- 调用接口
- 执行 UI 测试
- 触发后续验证步骤

## 9. 日志与输出策略

### 9.1 不能继续依赖 PIPE

对于长驻服务，必须放弃当前的 `stdout=PIPE`、`stderr=PIPE` 模式。

否则会继续遇到：

- pipe 句柄不释放
- `communicate()` 不返回
- 长输出撑爆缓冲

### 9.2 建议改成日志落盘

长驻服务模式建议：

- stdout/stderr 写入日志文件
- 工具只保留最近少量预览
- 失败或超时时把最近日志片段返回给 agent

### 9.3 为什么 agent 仍需要预览输出

虽然主路径不依赖 `/tasks`，但 agent 在失败时仍需要看到足够上下文，比如：

- 端口占用
- 依赖缺失
- 构建报错
- 配置文件错误

因此建议返回：

- `stdout_preview`
- `stderr_preview`
- `log_path`

这样 agent 可以在失败时继续分析，而不是只得到一句“服务启动失败”。

## 10. agent 侧最佳体验

这是这份方案最关键的一部分。

期望 agent 的执行体验如下：

1. agent 调用 `bash("npm run dev", ...)`
2. `bash` 识别为长驻服务模式
3. `bash` 托管启动进程
4. `bash` 等到服务 ready
5. `bash` 返回 `backgroundTaskId` 和日志路径
6. agent 通过通用输出读取/观察能力等待服务 ready
7. agent 继续下一步，例如：
   - 访问 `http://127.0.0.1:3000`
   - 调用浏览器工具
   - 执行接口测试
   - 继续验证改动

这条链路里，用户不需要手动介入。

## 11. 会话内生命周期管理

虽然不需要让用户手动 `/tasks`，但会话内仍然建议保留服务句柄，原因有三个：

- 后续步骤可能还要继续访问这个服务
- agent 结束时可能需要清理进程
- 若后续再次启动同类服务，需要避免重复拉起

因此建议在会话上下文中记录：

- 当前有哪些托管服务
- 每个服务对应的 pid、cwd、命令、日志路径、ready 地址

这允许 agent 做更智能的判断：

- 服务已经在跑，则复用
- 服务没 ready，则继续等
- 服务已死掉，则重新启动

## 12. 与现有后台任务能力的关系

现有 `/tasks`、`/task`、`/task_cancel` 能力可以保留，但定位应该降级为：

- 调试入口
- 人工排障入口
- 未来扩展入口

而不是主流程。

也就是说：

- 主流程：agent 调 `bash`，`bash` 等 ready，agent 继续
- 辅助流程：必要时开发者再查看内部托管状态

## 13. 提示词与工具描述建议

需要明确告诉模型：

- 短命令按普通模式执行
- `dev`、`watch`、`serve`、明显常驻的 `start` 优先走长驻服务模式
- 长驻服务模式的目标是“等 ready 后继续”，不是“丢给用户自己管理”

建议在工具描述里强化以下语义：

- For long-running dev servers, start the service in managed mode and wait until it is ready before proceeding.

这样模型的行为会更稳定，不会总想自己拼后台语法。

## 14. 分阶段落地

### Phase 1：先解决卡死

目标：

- `npm run dev` 不再阻塞在 `communicate()`

范围：

- 长驻服务模式分流
- stdout/stderr 改为日志落盘
- 检测进程是否存活
- 超时能安全返回失败

这一阶段还不要求非常智能的框架识别。

### Phase 2：让 agent 能自动继续

目标：

- 服务 ready 后 agent 能直接继续后续步骤

范围：

- 加入 ready check
- 增加通用的后台输出读取/观察能力
- 让 agent 能在拿到 `backgroundTaskId` 后等待日志信号或其他通用 ready 条件
- 会话内记录 service handle

### Phase 3：增强稳定性

范围：

- 常见框架默认 hint
- 更好的失败摘要
- 会话结束自动清理
- 服务复用与重复启动检测

## 15. 测试建议

至少覆盖以下场景：

- 短命令仍走原有同步路径
- 长驻命令不会卡在工具调用上
- 长驻命令启动后，工具等待 ready 而不是等待退出
- 进程提前退出时返回失败
- 超时时返回 `timed_out`
- ready 成功时，agent 能通过通用观察能力确认服务可用
- 日志里有启动报错时，agent 能拿到预览信息
- Windows 下不会因为 pipe 继承而卡死
- 同一会话可复用已经启动的服务句柄

## 16. 对现有代码的影响点

建议重点改动：

- `tools/bash.py`
  - 增加长驻服务模式
  - 将短命令和长驻命令彻底分流
  - 返回通用后台任务字段，而不是服务特定字段

- 会话上下文或托管层
  - 保存 service handle
  - 提供存活检查、ready 状态、清理能力

- 提示词 / 工具描述
  - 让模型优先把 `npm run dev` 当成“托管启动并等 ready”的命令

现有 CLI 的 `/tasks` 相关能力不是本次方案的重点，可以后续按需复用，但不应成为主依赖。

## 17. 结论

`bu_agent_cli` 这次真正需要的，不是“给用户一个查看后台任务的入口”，而是：

- 让 agent 能拉起一个长驻服务
- 让 agent 不被 shell 工具卡死
- 让 agent 在服务 ready 后自动继续执行

最合理的实现方向是：

- 保留 `bash` 作为统一 shell 入口
- 对长驻命令引入通用后台化能力
- 脱离当前阻塞式 pipe 模型
- 通过通用输出读取/观察能力等待服务 ready
- 避免在 `bash` 输出中引入 `ready_url`、`ready_port` 这类服务特定字段

这样 `npm run dev` 对 agent 来说，就不再是“卡死的 shell 命令”，而是“一个启动后可继续编排后续步骤的服务准备动作”。
