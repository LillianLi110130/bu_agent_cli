# Crab 自定义协议拉起设计文档

## 1. 背景

当前 Web 端已经可以通过服务端与本地 CLI worker 建立联调链路：

- Web 提交消息到 Python relay server
- 本地 worker 从 server 拉取消息
- 本地 CLI 处理消息
- worker 再把进度和最终结果回传给 server
- Web 通过 SSE 接收结果

当前仍有一个体验问题：

- 当 Web 检测到本地终端离线时，用户需要手工打开命令行并输入 `crab`
- 或手工双击桌面快捷方式启动本地 CLI

为了降低使用门槛，需要支持：

- 在 Web 页面中检测到离线时，展示一个“启动本地 Crab”按钮
- 用户点击后，能够直接拉起本地已安装的 Crab CLI

前提约束：

- 该能力运行在公司内部机器上
- 浏览器前端不能直接执行本地命令
- 必须通过操作系统支持的外部协议拉起机制实现

---

## 2. 目标

目标是提供一条正式、稳定、可部署的拉起链路：

`Web -> crab://open -> 本地 protocol launcher -> 现有 crab 启动入口 -> CLI 界面启动 -> worker 按现有逻辑上线`

要求：

1. Web 不直接执行本地命令
2. 启动结果与用户手工输入 `crab` 或双击桌面快捷方式时保持一致
3. 启动后必须出现完整 CLI 界面
4. worker 的启动方式、认证方式、配置读取方式，都沿用现有逻辑
5. 协议处理器不能成为“任意命令执行器”

---

## 3. 非目标

本方案不解决以下问题：

1. 不负责取消本地 CLI 正在执行的任务
2. 不负责唤醒一个已经存在的 CLI 窗口
3. 不负责把 Web 请求直接注入指定 CLI 会话
4. 不提供任意本地命令执行能力
5. 不保证 Linux 所有发行版和所有桌面环境都完全一致

---

## 4. 现有启动链路分析

当前项目中，`crab` 的标准启动入口是：

- [pyproject.toml](/d:/llm_project/bu_agent_cli/pyproject.toml)
  - `crab = "tg_crab_main:cli_main"`

运行逻辑在：

- [tg_crab_main.py](/d:/llm_project/bu_agent_cli/tg_crab_main.py)

Windows portable 安装后，运行时会使用：

- `%USERPROFILE%\.tg_agent\.venv\Scripts\python.exe`
- `%USERPROFILE%\.tg_agent\bin\crab-entry.py`

Windows bundle 里的现有启动器是：

- `tg-agent-launcher.bat`

相关打包与部署逻辑在：

- [scripts/release/windows/build_windows_portable.ps1](/d:/llm_project/bu_agent_cli/scripts/release/windows/build_windows_portable.ps1)
- [scripts/release/windows/readme.md](/d:/llm_project/bu_agent_cli/scripts/release/windows/readme.md)

Linux portable 安装后，运行时会使用：

- `~/.tg_agent/.venv/bin/python`
- `~/.tg_agent/bin/crab-entry.py`

相关逻辑在：

- [scripts/release/linux/build_linux_portable.sh](/d:/llm_project/bu_agent_cli/scripts/release/linux/build_linux_portable.sh)
- [scripts/release/linux/README.md](/d:/llm_project/bu_agent_cli/scripts/release/linux/README.md)

结论：

当前项目已经有一条稳定的标准启动链路。  
因此 protocol launcher 不应该重新实现一套启动逻辑，而应该复用现有入口。

---

## 5. 核心设计原则

### 5.1 protocol launcher 只做转发

protocol launcher 的职责应该尽可能薄：

1. 接收 `crab://...`
2. 校验协议动作是否合法
3. 转发到现有 `crab` 启动入口

它不应该：

1. 自己决定 worker 如何启动
2. 自己拼装复杂的 Python 启动参数
3. 直接执行任意 shell 命令

### 5.2 启动行为必须与手工启动一致

用户点击 Web 按钮后，最终效果要与以下方式保持一致：

1. 命令行输入 `crab`
2. 双击桌面快捷方式
3. 双击现有 launcher

也就是说：

- 要启动完整 CLI 界面
- 要保留现有 `tg_crab_main.py` 启动逻辑
- 要按当前 `.env`、`tg_crab_worker.json`、认证配置决定 worker 是否上线

### 5.3 不把 protocol 设计成命令执行器

禁止把协议设计成如下形式：

```text
crab://exec?cmd=...
```

这是高风险设计，不适合公司内部机器。

协议动作必须是白名单，例如：

```text
crab://open
```

或：

```text
crab://launch
```

---

## 6. 推荐协议形态

第一版推荐使用：

```text
crab://open
```

原因：

1. 语义简单
2. 与“打开本地 Crab”按钮一一对应
3. 不引入额外 worker 参数
4. 更容易保证与手工启动一致

不推荐第一版就引入：

```text
crab://start-worker?workerId=...
```

因为这会让 protocol launcher 开始介入 worker 启动细节，偏离“与手工启动一致”的目标。

如果未来确实有需要，再考虑扩展新的白名单动作，例如：

```text
crab://open
crab://open-worker
```

但第一版不建议做。

---

## 7. Windows 方案

### 7.1 目标

在 Windows portable 安装完成后：

1. 自动注册 `crab://`
2. 浏览器点击 `crab://open` 时
3. 调起本地 protocol launcher
4. protocol launcher 再调用现有 `tg-agent-launcher.bat`

最终链路：

`Web -> crab://open -> Windows URL Protocol -> crab-protocol-launcher.bat -> tg-agent-launcher.bat -> crab-entry.py -> tg_crab_main.py`

### 7.2 注册位置

推荐注册到当前用户级别：

- `HKCU\Software\Classes\crab`

原因：

1. 不需要管理员权限
2. 更适合公司内部普通用户机器
3. 可以由现有 deploy 流程自动完成

### 7.3 注册动作放在哪一步

建议放入现有部署流程：

- `deploy.bat`
- `win_deploy.ps1`

这样用户只要按当前安装方式执行一次 deploy，即可完成协议注册。

### 7.4 protocol launcher 的位置

建议在安装目录下生成一个独立文件，例如：

```text
%USERPROFILE%\.tg_agent\bin\crab-protocol-launcher.bat
```

它作为协议入口脚本存在，不替代现有 `tg-agent-launcher.bat`。

### 7.5 protocol launcher 的行为

它只做以下事情：

1. 接收完整 URL，例如 `crab://open`
2. 校验协议动作是否为白名单动作
3. 对 `open` 动作调用现有 `tg-agent-launcher.bat`

它不做：

1. worker 参数覆盖
2. gateway 参数覆盖
3. 任意命令拼接

### 7.6 为什么必须调用现有 launcher

因为现有 launcher 已经封装了：

1. 安装路径定位
2. Python 运行时定位
3. `crab-entry.py` 启动方式
4. 当前工作目录行为
5. 与桌面快捷方式一致的启动效果

如果 protocol launcher 直接重写一遍 Python 启动命令，就会带来两套行为，后续维护会变复杂。

---

## 8. Linux 方案

### 8.1 目标

在带桌面环境的 Linux 机器上：

1. 注册 `x-scheme-handler/crab`
2. 浏览器点击 `crab://open`
3. 调起本地 protocol launcher
4. 再调用现有 Linux launcher

### 8.2 实现基础

Linux 通常依赖：

1. `.desktop` 文件
2. `xdg-mime`
3. 桌面环境对 `x-scheme-handler/*` 的支持

### 8.3 注册位置

建议注册到当前用户目录：

```text
~/.local/share/applications/
```

由现有 `deploy.sh` 自动生成：

1. `crab-protocol.desktop`
2. protocol launcher 脚本
3. `xdg-mime default crab-protocol.desktop x-scheme-handler/crab`

### 8.4 行为原则

Linux 侧与 Windows 一样，protocol launcher 也只做转发：

1. 解析 `crab://open`
2. 调用现有 `tg-agent-launcher.sh` 或等价入口

### 8.5 风险说明

Linux 兼容性不如 Windows 稳定，主要风险包括：

1. 某些机器没有桌面环境
2. 不同发行版对协议拉起处理不完全一致
3. 浏览器策略和桌面环境耦合更强

因此建议：

- Windows 先做正式方案
- Linux 先做兼容支持，不作为第一优先级主路径

---

## 9. Web 端交互设计

### 9.1 展示时机

当 Web 页面检测到当前 worker 对应本地终端离线时，显示：

- `启动本地 Crab`

### 9.2 点击行为

点击后执行：

```js
window.location.href = 'crab://open'
```

### 9.3 用户可见行为

浏览器通常会弹：

- 是否打开外部应用

用户确认后，本地 CLI 窗口打开。

### 9.4 文案建议

建议文案明确表达这是“打开本地 CLI”，而不是“只启动 worker”。

推荐文案：

- `启动本地 Crab`
- `打开本地终端`

不建议：

- `启动 worker`

因为当前需求要求启动完整 CLI 界面，而不是只启动后台 worker。

---

## 10. 用户操作成本

### 10.1 Windows

如果协议注册并入现有 `deploy`：

用户通常只需要：

1. 像现在一样执行一次 `deploy.bat`
2. 第一次从浏览器点击时确认“打开外部应用”

不需要：

1. 手工改注册表
2. 手工注册协议
3. 额外安装独立组件

### 10.2 Linux

如果协议注册并入现有 `deploy.sh`：

用户通常只需要：

1. 执行一次 `deploy.sh`
2. 第一次浏览器拉起时点确认

但 Linux 仍可能出现：

1. 需要重新登录桌面会话
2. 需要刷新桌面数据库
3. 不同桌面环境行为差异

---

## 11. 安全要求

正式实现必须满足以下安全要求：

1. 只允许白名单协议动作
2. 不允许 URL 携带任意本地命令
3. 不允许 protocol launcher 拼接执行任意 shell 命令
4. 所有动作只允许转发到现有已知 launcher

推荐白名单：

- `crab://open`

不允许：

- `crab://exec?...`
- `crab://run?...`
- `crab://shell?...`

---

## 12. 推荐实施顺序

### 阶段一：Windows 正式接入

1. 在现有 Windows deploy 中自动注册 `crab://`
2. 新增 `crab-protocol-launcher.bat`
3. protocol launcher 转发到现有 `tg-agent-launcher.bat`
4. Web 加“启动本地 Crab”按钮

### 阶段二：Linux 兼容接入

1. 在 Linux deploy 中注册 `x-scheme-handler/crab`
2. 新增 Linux protocol launcher
3. 复用现有 Linux launcher

### 阶段三：失败兜底体验

如果浏览器无法拉起本地程序，可在 Web 页面提供：

1. “请先安装或部署本地 Crab”提示
2. 手工命令提示，例如：
   - `crab`

---

## 13. 结论

在当前项目结构下，最合理的正式方案不是新增一套“专门启动 worker”的协议逻辑，而是：

1. 通过 `crab://open` 拉起本地程序
2. protocol launcher 只做白名单校验和转发
3. 真正启动逻辑完全复用现有 `crab` / `tg-agent-launcher` 启动链路

这样可以保证：

1. 行为与命令行输入 `crab` 一致
2. 行为与桌面快捷方式一致
3. 启动后一定出现 CLI 界面
4. worker 是否上线继续遵循现有配置与现有逻辑

这是当前最稳、最容易落地、也最容易维护的方案。
