# Crab Linux 自定义协议拉起设计文档

## 1. 背景

当前已经完成 Linux 机器上的最小协议验证：

- `x-scheme-handler/crabtest` 可以成功注册
- `xdg-open 'crabtest://...'` 可以拉起本地处理程序
- 浏览器点击协议链接可以触发本地程序执行

这说明在目标 Linux 桌面环境中，已经具备通过自定义协议拉起本地程序的基础能力。

接下来需要把验证方案收敛成正式的 `crab://` 拉起方案，并与当前 Linux portable 安装链路对齐。

---

## 2. 目标

在 Linux 桌面环境中实现如下体验：

1. Web 页面检测到本地终端离线
2. 页面展示“启动本地 Crab”按钮
3. 用户点击按钮
4. 浏览器打开 `crab://open`
5. Linux 桌面环境调用本地 protocol launcher
6. protocol launcher 再调用现有 Crab 启动入口
7. 本地完整 CLI 界面启动
8. worker 按现有逻辑上线

核心要求：

1. 启动效果要与用户手工输入 `crab` 一致
2. 必须启动完整 CLI 界面，而不是只启动后台 worker
3. 不新增一套独立的启动逻辑
4. 尽可能复用现有 Linux portable 的 deploy、launcher、entry shim

---

## 3. 非目标

本方案不负责：

1. 唤醒已存在的 CLI 窗口
2. 取消正在执行的本地任务
3. 只启动 worker 而不显示界面
4. 注入特定 Web 会话到某个现有 CLI 进程
5. 在无桌面环境的 Linux 服务器上提供一致体验

---

## 4. 现有 Linux 启动链路

根据当前打包与安装脚本，Linux portable 安装后已有如下结构：

- 安装根目录：
  - `~/.tg_agent`
- Python 虚拟环境：
  - `~/.tg_agent/.venv`
- 入口 shim：
  - `~/.tg_agent/bin/crab-entry.py`
- PATH 命令：
  - `~/.tg_agent/bin/crab`
- bundle 内 launcher：
  - `tg-agent-launcher.sh`

相关脚本：

- [scripts/release/linux/build_linux_portable.sh](/d:/llm_project/bu_agent_cli/scripts/release/linux/build_linux_portable.sh)
- [scripts/release/linux/README.md](/d:/llm_project/bu_agent_cli/scripts/release/linux/README.md)

也就是说，当前 Linux 上已经存在一条标准启动路径：

`crab -> crab-entry.py -> tg_crab_main.py`

因此 protocol launcher 不应该绕过这条链路，也不应该自行拼装新的 Python 启动命令。

---

## 5. 核心设计原则

### 5.1 protocol launcher 只做转发

protocol launcher 的职责应该非常薄：

1. 接收 `crab://...`
2. 校验动作是否合法
3. 调用现有 launcher 或现有 `crab` 启动入口

不负责：

1. 自定义 worker 启动逻辑
2. 重写 Python 命令行参数
3. 提供任意 shell 执行能力

### 5.2 启动行为与手工启动一致

用户点击 Web 按钮后，效果应该等价于：

```bash
crab
```

或者：

```bash
./tg-agent-launcher.sh
```

也就是说：

1. 启动完整 CLI
2. 保留现有工作目录行为
3. 保留现有认证行为
4. 保留现有 `.env` / `tg_crab_worker.json` 配置读取方式
5. 保留现有 worker 自动上线逻辑

### 5.3 不把协议做成命令执行器

必须禁止如下形态：

```text
crab://exec?cmd=...
```

正式方案只允许固定白名单动作，例如：

```text
crab://open
```

---

## 6. 推荐协议形态

第一版 Linux 正式方案推荐使用：

```text
crab://open
```

原因：

1. 简单
2. 与“打开本地 Crab”按钮语义一致
3. 不需要在协议层暴露 worker 细节
4. 更容易保持与手工启动一致

不建议第一版使用：

```text
crab://start-worker?workerId=...
```

因为这会让 protocol launcher 介入本地 worker 启动细节，偏离“与现有 CLI 启动一致”的设计目标。

---

## 7. Linux 实现方案

### 7.1 总体链路

正式链路应为：

`Web -> crab://open -> 浏览器 -> Linux 桌面环境 -> x-scheme-handler/crab -> crab-protocol.desktop -> crab-protocol-launcher.sh -> 现有 tg-agent-launcher.sh 或 crab -> tg_crab_main.py`

### 7.2 协议注册方式

建议采用标准用户级桌面协议注册方式：

1. 在 `~/.local/share/applications/` 下生成 `.desktop`
2. 注册：

```bash
xdg-mime default crab-protocol.desktop x-scheme-handler/crab
```

### 7.3 推荐新增文件

建议在 deploy 阶段生成两个文件。

文件 1：

```text
~/.local/share/applications/crab-protocol.desktop
```

文件 2：

```text
~/.tg_agent/bin/crab-protocol-launcher.sh
```

说明：

- `.desktop` 负责对接桌面环境协议
- `crab-protocol-launcher.sh` 负责协议动作白名单校验和转发

### 7.4 `.desktop` 文件职责

它只做一件事：

- 把 `crab://open` 交给 `crab-protocol-launcher.sh`

例如逻辑上会类似：

```ini
[Desktop Entry]
Name=Crab Protocol Handler
Exec=/bin/bash -lc '~/.tg_agent/bin/crab-protocol-launcher.sh "%u"'
Type=Application
Terminal=false
MimeType=x-scheme-handler/crab;
NoDisplay=true
```

### 7.5 protocol launcher 职责

`crab-protocol-launcher.sh` 只负责：

1. 接收完整 URL
2. 校验是否为 `crab://open`
3. 调用现有的 `crab` 启动入口

推荐行为：

1. 解析 URL
2. 如果不是 `open` 动作，直接退出
3. 调用：
   - 现有 `tg-agent-launcher.sh`
   - 或 `~/.tg_agent/bin/crab`

更推荐优先调用现有 launcher，而不是直接拼 Python 命令。

---

## 8. 为什么要复用现有 launcher

当前 Linux portable 方案已经封装了：

1. 安装路径定位
2. 虚拟环境路径定位
3. `crab-entry.py` 的执行方式
4. PATH 与 shell profile 行为
5. bundle 内启动方式

如果 protocol launcher 再自己重写一套：

```bash
~/.tg_agent/.venv/bin/python ~/.tg_agent/bin/crab-entry.py
```

虽然短期也能跑，但会带来几个问题：

1. 与现有 launcher 行为可能出现偏差
2. 未来 launcher 变更时，协议入口需要重复维护
3. 一旦加入额外环境初始化逻辑，协议入口容易落后

因此更稳的方案是：

- protocol launcher 调现有 launcher
- 现有 launcher 再调 `crab-entry.py`

---

## 9. deploy 集成方式

### 9.1 集成位置

建议将协议注册并入现有 Linux portable 安装流程，也就是：

- `deploy.sh`

因为用户当前已经需要执行 deploy 才能完成安装，这时顺手完成协议注册最自然。

### 9.2 deploy 需要新增的工作

deploy 阶段建议新增以下步骤：

1. 生成 `~/.tg_agent/bin/crab-protocol-launcher.sh`
2. 生成 `~/.local/share/applications/crab-protocol.desktop`
3. 执行：

```bash
xdg-mime default crab-protocol.desktop x-scheme-handler/crab
```

4. 可选执行：

```bash
update-desktop-database ~/.local/share/applications 2>/dev/null || true
```

### 9.3 用户额外操作成本

如果这部分集成到 deploy，用户通常不需要：

1. 手工创建 `.desktop`
2. 手工注册协议
3. 手工配置 handler

用户通常只需要：

1. 像现在一样执行一次 `deploy.sh`
2. 第一次在浏览器中点击协议链接时确认打开外部应用

---

## 10. Web 端交互设计

### 10.1 按钮展示时机

当页面检测到本地 worker 离线时，展示：

- `启动本地 Crab`

### 10.2 点击行为

点击后执行：

```js
window.location.href = 'crab://open'
```

### 10.3 用户可见效果

预期用户体验：

1. 浏览器弹出“是否打开外部应用”
2. 用户确认
3. 本地 CLI 窗口打开
4. 本地 worker 按现有配置上线
5. Web 页面随后变为在线

### 10.4 文案建议

建议明确表达这是“打开本地 CLI”，而不是“只启动 worker”。

推荐：

- `启动本地 Crab`
- `打开本地终端`

不推荐：

- `启动 worker`

---

## 11. Linux 特有风险

Linux 相比 Windows，需要特别注意以下问题：

### 11.1 仅适用于桌面环境

该方案依赖：

1. 浏览器
2. 桌面环境
3. `xdg-open`
4. `x-scheme-handler/*`

因此不适用于：

1. 纯 SSH 服务器
2. 无 GUI 环境
3. 极简容器环境

### 11.2 桌面环境差异

不同 Linux 桌面环境可能存在行为差异，例如：

1. GNOME
2. KDE
3. 发行版自定义桌面

这些差异可能体现在：

1. 第一次点击时的确认提示不同
2. `.desktop` 刷新时机不同
3. 默认浏览器与 `xdg-mime` 的联动表现不同

### 11.3 shell / working directory 差异

如果 protocol launcher 直接调用命令，需要注意：

1. 当前工作目录可能不是用户期望目录
2. GUI 拉起时环境变量可能比交互 shell 少

因此更建议通过现有 launcher 统一处理，而不是在 protocol launcher 里自行扩展复杂环境逻辑。

---

## 12. 安全要求

Linux 正式方案必须满足：

1. 只允许白名单动作
2. 不允许协议 URL 携带任意命令
3. 不允许 protocol launcher 成为本地任意执行入口
4. 不允许从 URL 中拼接 shell 语句

允许：

```text
crab://open
```

不允许：

```text
crab://exec?cmd=...
crab://shell?...
crab://run?script=...
```

---

## 13. 推荐实施顺序

### 阶段一：Linux deploy 接入

1. 在现有 deploy 中生成 protocol launcher
2. 在现有 deploy 中生成 `.desktop`
3. 自动注册 `x-scheme-handler/crab`

### 阶段二：Web 页面接入

1. 在离线状态下展示“启动本地 Crab”
2. 点击后跳转 `crab://open`

### 阶段三：失败兜底

如果协议拉起失败，可在 Web 页面提示：

1. 请先执行本地部署
2. 或手工运行：

```bash
crab
```

---

## 14. 结论

既然 Linux 上最小协议验证已经通过，那么正式方案完全可行。

最合理的 Linux 实施方式是：

1. 使用 `crab://open`
2. 通过 `.desktop + xdg-mime` 注册协议
3. 新增一个非常薄的 `crab-protocol-launcher.sh`
4. 它只做协议动作校验和转发
5. 真正的 CLI 启动逻辑继续复用现有 `crab` / `tg-agent-launcher.sh`

这样可以保证：

1. 启动行为与手工输入 `crab` 一致
2. 启动后出现完整 CLI 界面
3. worker 仍按现有逻辑上线
4. 整体实现简单，维护成本低

这是当前 Linux 场景下最稳妥的正式方案。
