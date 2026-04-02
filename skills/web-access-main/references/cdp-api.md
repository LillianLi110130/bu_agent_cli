# CDP Proxy API 参考

这份参考文档已经适配当前 BU Agent CLI 环境。

## 基础信息

- 基础地址：`http://127.0.0.1:3456`
- 默认 Proxy 脚本：`~/.tg_agent/skills/.builtin/web-access-main/scripts/cdp-proxy.mjs`
- 默认依赖检查脚本：`~/.tg_agent/skills/.builtin/web-access-main/scripts/check-deps.mjs`
- 如果当前 skill 的实际 `Path` 与默认位置不同，则以实际 `Path` 推导出的 `<skill_root>` 为准
- Windows 下 shell 命令统一使用：`curl.exe`

先运行依赖检查：

```powershell
node "<skill_root>/scripts/check-deps.mjs"
```

## 常用端点

### 健康检查

```powershell
curl.exe -s http://127.0.0.1:3456/health
```

### 列出 tab

```powershell
curl.exe -s http://127.0.0.1:3456/targets
```

预期结果：

- 返回一个打开 tab 的数组
- 每一项通常包含 `targetId`、`title`、`url`

### 创建新 tab

```powershell
curl.exe -s "http://127.0.0.1:3456/new?url=https://example.com"
```

预期结果：

- 返回包含新 `targetId` 的对象

### 获取页面信息

```powershell
curl.exe -s "http://127.0.0.1:3456/info?target=ID"
```

适合用于：

- 确认当前页面是否正确
- 在交互前检查标题和 URL

### 在已有 tab 中导航

```powershell
curl.exe -s "http://127.0.0.1:3456/navigate?target=ID&url=https://example.com"
```

### 后退

```powershell
curl.exe -s "http://127.0.0.1:3456/back?target=ID"
```

### 关闭 tab

```powershell
curl.exe -s "http://127.0.0.1:3456/close?target=ID"
```

## DOM 与交互

### 执行 JavaScript

```powershell
curl.exe -s -X POST "http://127.0.0.1:3456/eval?target=ID" -d "document.title"
```

`/eval` 适合做这些事：

- 读取 DOM 文本
- 枚举链接、按钮、表单
- 提取结构化页面数据
- 填写字段和提交表单

使用建议：

- 只返回可序列化值
- 优先返回小型对象或字符串
- 不要直接返回原始 DOM 节点

### 用 JavaScript 点击

```powershell
curl.exe -s -X POST "http://127.0.0.1:3456/click?target=ID" -d "button.submit"
```

适用场景：

- 元素是普通可点击控件
- 简单 `el.click()` 大概率就足够

### 用真实鼠标事件点击

```powershell
curl.exe -s -X POST "http://127.0.0.1:3456/clickAt?target=ID" -d "button.upload"
```

适用场景：

- JS 点击被页面拦截
- 上传流程要求真实用户手势
- 站点对自动化行为比较敏感

### 给 file input 设置文件

```powershell
curl.exe -s -X POST "http://127.0.0.1:3456/setFiles?target=ID" -d "{\"selector\":\"input[type=file]\",\"files\":[\"D:\\path\\to\\file.png\"]}"
```

适用场景：

- 已经确认正确的 file input 元素
- 希望绕过原生文件选择框

## 滚动与截图

### 滚动页面

```powershell
curl.exe -s "http://127.0.0.1:3456/scroll?target=ID&direction=bottom"
```

适用于：

- 懒加载内容
- 长信息流
- 提取图片 URL 前先触发加载

### 保存截图

```powershell
curl.exe -s "http://127.0.0.1:3456/screenshot?target=ID&file=D:\\tmp\\web-access-shot.png"
```

在当前 CLI 中，截图主要用于：

- 保存证据
- 帮助排查页面状态
- 为后续显式图片处理做准备

只要 DOM 能解决，就优先用 DOM，不要默认截图。

## 实际使用顺序

推荐顺序：

1. `check-deps.mjs`
2. `/targets` 或 `/new`
3. `/info`
4. `/eval`
5. `/click` 或 `/clickAt`
6. 需要上传时使用 `/setFiles`
7. 需要滚动时使用 `/scroll`
8. 只有 DOM 不够时才使用 `/screenshot`
9. `/close`

## 故障处理

### Chrome remote debugging 没开启

现象：

- proxy 还没可用，依赖检查就失败

处理：

- 提示用户在 `chrome://inspect/#remote-debugging` 中开启 `Allow remote debugging for this browser instance`

### target 已失效

现象：

- 旧的 `targetId` 在 `/info`、`/eval` 或 `/click` 中报错

处理：

- 重新执行 `/targets`
- 必要时新建一个 tab

### selector 没命中

现象：

- 点击或上传接口失败

处理：

- 先用 `/eval` 检查页面结构
- 确认元素是否在 shadow DOM、iframe 中，或是否是延迟渲染

### 页面逻辑拦住了简单操作

现象：

- `/click` 没有触发预期行为

处理：

- 尝试 `/clickAt`
- 然后根据页面当前状态重新评估流程

## 最小排查清单

- 依赖检查通过了吗？
- 在 Windows 下是否使用了 `curl.exe`？
- 当前操作的 `targetId` 对吗？
- 在写 selector 前，是否先检查过 DOM？
- 当前任务是不是能通过 DOM 提取解决，而不必依赖截图？
