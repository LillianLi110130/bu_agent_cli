# BU Agent CLI 的 web-access 技能

这个目录保存的是 `web-access` 技能在当前 BU Agent CLI 仓库中的第一阶段适配版本。

这一阶段的目标很明确：

- 保留原有 CDP proxy 脚本
- 把技能文档适配到当前 CLI 运行环境
- 统一 Windows 和 `bash` 下的命令模板
- 暂时不做更深的 runtime 改造

## 这个技能是干什么的

当任务需要真实浏览器会话，而不是普通文本抓取时，使用 `web-access`。

典型场景：

- 登录态页面
- 动态前端渲染内容
- 按钮驱动的交互流程
- 文件上传
- 在真实 Chrome 会话里检查 DOM

## 目录中包含什么

- [SKILL.md](./SKILL.md)：给 agent 看的行为约束
- [scripts/check-deps.mjs](./scripts/check-deps.mjs)：依赖检查与 proxy 启动脚本
- [scripts/cdp-proxy.mjs](./scripts/cdp-proxy.mjs)：对 Chrome DevTools Protocol 的 HTTP 封装
- [references/cdp-api.md](./references/cdp-api.md)：稳定命令参考

## 当前运行时假设

这个适配版本假设：

- agent 可以使用 CLI 的 `bash` 工具
- 本机可以直接运行 `node`
- 本机安装了 Chrome
- 用户可以手动开启 Chrome remote debugging
- 在 Windows 下 shell 命令统一使用 `curl.exe`

## 第一阶段包含什么

这一阶段已经实现：

- 使用仓库内路径，不再依赖 Claude 专属路径约定
- 统一为 Windows 友好的命令示例
- 去掉对内置 `WebSearch` / `WebFetch` 的依赖
- 把工作流明确收敛到 `bash + node + curl.exe + CDP`

这一阶段还没有实现：

- 专门的 Python 浏览器工具
- proxy 操作的 slash command
- agent runtime 内部自动把截图回灌成视觉上下文

## 推荐的手动验证项

加载技能后，建议验证这些流程：

1. 依赖检查和 proxy 启动
2. 创建后台 tab 并读取 DOM 文本
3. 点击按钮并滚动页面
4. 用 `/setFiles` 上传文件
5. 把截图保存到本地

## 说明

- 技能名仍然是 `web-access`
- 目录名仍然是 `web-access-main`
- 第一阶段没有改动 CDP proxy 脚本本体，只改了技能的使用契约和文档表达
