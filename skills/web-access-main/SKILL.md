---
name: web-access
license: MIT
github: https://github.com/eze-is/web-access
description: 当任务需要真实浏览器会话、登录态、DOM 交互、文件上传，或普通 HTTP 抓取无法完成的页面导航时使用此技能。
metadata:
  author: BU Agent CLI
  version: "1.0.0"
---

# web-access Skill

## 目的

当任务需要真实 Chrome 会话，而不是普通文本抓取时，使用这个技能。

典型触发场景：

- 页面依赖登录态
- 内容由浏览器动态渲染
- 任务需要点击、滚动、填表或上传文件
- 任务依赖页面 DOM，而不是原始 HTML

在当前 CLI 中，这个技能**不依赖**内置 `WebSearch` 或 `WebFetch`。
当前支持的执行路径是：

`bash` -> `node` -> 本地 CDP proxy -> Chrome

## 环境假设

这个技能已经针对当前 BU Agent CLI 仓库做了适配。

- 对当前内置 `web-access` 技能，默认运行时目录是 `~/.tg_agent/skills/.builtin/web-access-main`
- 默认脚本目录是 `~/.tg_agent/skills/.builtin/web-access-main/scripts/`
- 默认参考文档目录是 `~/.tg_agent/skills/.builtin/web-access-main/references/`
- shell 命令通过 CLI 的 `bash` 工具执行
- 在 Windows 下优先使用 `curl.exe`，不要使用 `curl`

在使用任何命令模板之前，先确认 skill 目录。

不要假设当前工作目录就是仓库根目录，也不要把源码树里的固定路径写死到命令里。

默认按下面这个运行时位置理解这个内置 skill：

- `~/.tg_agent/skills/.builtin/web-access-main`

如果系统提示或 skills 列表里显示的这个 skill 的实际 `Path` 与上面不同，则一律以实际 `Path` 为准，再据此推导 skill 目录的绝对路径：

- `SKILL.md` 所在目录就是 skill 根目录
- 脚本目录是 `<skill_root>/scripts/`
- 参考目录是 `<skill_root>/references/`

这个 skill 是内置 skill。当前 CLI 会在启动时把它同步到 `~/.tg_agent/skills/.builtin/` 下，所以通常不需要去当前工作区或安装包目录里搜索它。

如果系统提示中给出的当前 `SKILL.md` 实际路径与默认位置不同，必须以该实际路径为准解析出 `<skill_root>`，然后在 shell 命令里使用这个绝对路径。

## 核心规则

1. 优先做 DOM 提取，不要默认依赖截图。
2. 每个任务第一次使用浏览器前，都先跑依赖检查。
3. 除非用户明确要求检查已有 tab，否则只操作自己创建的 tab。
4. 任务结束后关闭自己创建的 tab。
5. 除非必须让用户手动开启 Chrome remote debugging，否则不要把执行工作推给用户。
6. 失败时要说清楚是 Chrome 配置、proxy 启动、tab 定位、selector 命中，还是页面逻辑出了问题。

## 这个技能在当前 CLI 里能做什么、不能做什么

当前 CLI 里效果比较好的能力：

- 在真实且已登录的 Chrome 会话里打开页面
- 通过 `/eval` 检查和提取 DOM 内容
- 导航、点击、滚动、提交表单
- 通过 `/clickAt` 和 `/setFiles` 处理文件上传
- 在必要时把截图保存到本地

当前限制：

- 这个 CLI 还没有一等浏览器工具，不能把截图自动作为视觉上下文回灌给模型
- `/screenshot` 目前主要用于保存证据、辅助排障，或为后续显式图片步骤做准备

## 每次都先做的第一步

在开始浏览器操作前，先运行依赖检查：

```powershell
node "<skill_root>/scripts/check-deps.mjs"
```

这一步会检查：

- Node.js 22+
- Chrome remote debugging 是否已开启
- CDP proxy 是否可达，或能否被成功拉起

如果因为 Chrome 没准备好而失败，要明确告诉用户：

1. 打开 Chrome
2. 访问 `chrome://inspect/#remote-debugging`
3. 开启 `Allow remote debugging for this browser instance`
4. 如有需要，重启 Chrome

## 标准命令模板

尽量固定使用下面这些命令形式，不要随意发挥。

列出 tab：

```powershell
curl.exe -s http://127.0.0.1:3456/targets
```

创建一个新的后台 tab：

```powershell
curl.exe -s "http://127.0.0.1:3456/new?url=https://example.com"
```

获取页面信息：

```powershell
curl.exe -s "http://127.0.0.1:3456/info?target=ID"
```

提取 DOM 数据：

```powershell
curl.exe -s -X POST "http://127.0.0.1:3456/eval?target=ID" -d "document.title"
```

用 JS 点击：

```powershell
curl.exe -s -X POST "http://127.0.0.1:3456/click?target=ID" -d "button.submit"
```

用真实鼠标事件点击：

```powershell
curl.exe -s -X POST "http://127.0.0.1:3456/clickAt?target=ID" -d "button.upload"
```

给 file input 设置文件：

```powershell
curl.exe -s -X POST "http://127.0.0.1:3456/setFiles?target=ID" -d "{\"selector\":\"input[type=file]\",\"files\":[\"D:\\path\\to\\file.png\"]}"
```

滚动页面：

```powershell
curl.exe -s "http://127.0.0.1:3456/scroll?target=ID&direction=bottom"
```

保存截图：

```powershell
curl.exe -s "http://127.0.0.1:3456/screenshot?target=ID&file=D:\\tmp\\web-access-shot.png"
```

关闭自己的 tab：

```powershell
curl.exe -s "http://127.0.0.1:3456/close?target=ID"
```

## 推荐的 Agent 工作流

### 1. 先判断浏览器是否真的必要

只有当普通文件读取或简单 HTTP 获取不足以完成任务时，才进入浏览器链路。

适合用浏览器的场景：

- 登录态页面
- 动态信息流
- 按钮驱动的交互流程
- 文件上传流程
- 数据被前端渲染后才出现的网站

### 2. 检查就绪状态

运行 `check-deps.mjs`。

如果失败，就停止继续操作，并准确说明用户还需要完成什么设置。

### 3. 创建自己的 tab

优先使用 `/new?url=...`，不要随便接管用户已有 tab。

拿到 `/new` 返回的 `targetId` 之后，后续操作都围绕这个 `targetId` 进行。

### 4. 先看清页面，再决定动作

先用 `/info` 和 `/eval` 了解页面结构。

安全的初始读取例子：

- `document.title`
- `location.href`
- 链接、按钮、表单的简要结构
- 与当前任务直接相关的可见文本块

### 5. 优先使用 DOM 原生动作

优先级建议：

1. `/eval` 用于读取结构化数据
2. `/click` 用于一般可点击元素
3. `/clickAt` 用于 JS 点击无效或上传弹窗必须触发的场景
4. `/setFiles` 用于已确认 file input 后的上传
5. `/screenshot` 只在 DOM 提取不足时使用

### 6. 干净收尾

关闭自己创建的 tab。
不要影响用户原有 tab。

## 输出风格要求

使用这个技能时，要用简短、具体的进度说明来同步状态：

- 当前打开了什么页面
- proxy 是否已经就绪
- 当前在操作哪个 `targetId`
- 提取到了什么内容，或哪个交互动作成功了
- 为什么要切换到某个 fallback

不要输出像“浏览器失败了”这种模糊描述。
要说成这样：

- “Chrome remote debugging 还没有开启。”
- “proxy 已经起来了，但目标 tab 已经被关闭。”
- “当前 selector 没有匹配到任何 file input。”

## 失败处理

把失败分成以下几层来判断：

1. Chrome 配置层
2. proxy 启动层
3. tab 创建或定位层
4. selector 命中层
5. 页面逻辑或反自动化层

如果某一层失败，不要对同一个命令做很多次无意义重试。
要调整方法。

例如：

- JS 点击失败：尝试 `/clickAt`
- 上传按钮触发原生文件框：改用 `/setFiles`
- 可见文本里没内容：改用 `/eval` 深入查 DOM
- 看起来必须截图：保存截图，但要记住当前 CLI 还不能把它当成真正浏览器视觉上下文自动理解

## 什么时候加载参考文档

在这些情况下读取 `references/cdp-api.md`：

- 需要确认 proxy API 的具体形式
- 需要稳定的命令模板
- 需要错误处理提示

在这些情况下读取 `references/site-patterns/{domain}.md`：

- 目标域名有对应站点经验文件

## 总结

在当前 CLI 里，`web-access` 的含义是：

- 使用这个仓库内的本地 Node 脚本
- 通过 CDP proxy 与 Chrome 通信
- 优先做结构化 DOM 提取，而不是截图
- 使用适配 Windows 的命令形式
- 让浏览器操作尽量克制、明确、可清理
