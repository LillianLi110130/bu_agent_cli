# 插件开发文档

本文面向本仓库 Agent 的开发者，说明如何开发、安装、更新、删除插件。

本文定义的公开插件模型包括：

- `plugin.json`：插件元数据与能力声明
- `commands/`：通过本仓库主 Agent 执行的 slash 命令
- `skills/`：可通过 `@plugin:skill` 使用的技能
- `agents/`：可注册给本仓库子代理体系使用的 agent 配置

本文同时说明内置插件与工作区插件的区别，以及 `/plugins` 相关命令的使用方式。

## 1. 插件体系概念

插件用于向 Agent CLI 扩展以下能力：

- Slash 命令
- Skills
- Agents

运行时有两类插件来源：

- 内置插件：位于程序内置插件目录，对所有工作区生效
- 工作区插件：位于当前工作区 `.tg_agent/plugins/`，只对当前工作区生效

同名插件的覆盖规则：

- 工作区插件优先级高于内置插件
- 如果工作区中存在同名插件，则优先加载工作区版本

这意味着：

- 你可以先把通用插件安装为内置插件
- 再通过复制到工作区的方式，在某个项目里做局部定制

## 2. `/plugins` 命令语义

插件相关操作通过 `/plugins` 管理。

常用命令：

- `/plugins list`
- `/plugins show <name>`
- `/plugins install <plugin_path>`
- `/plugins uninstall <plugin_name> --force`
- `/plugins copy <plugin_name>`
- `/plugins reload`

命令语义如下：

- `/plugins install <plugin_path>`
  - 将一个插件目录安装到内置插件目录
  - 安装后切换工作区仍可继续使用
- `/plugins uninstall <plugin_name> --force`
  - 只删除内置插件目录中的插件
  - 不删除工作区插件
- `/plugins copy <plugin_name>`
  - 将某个内置插件复制到当前工作区 `.tg_agent/plugins/<name>`
  - 适合做项目级覆盖和定制
- `/plugins reload`
  - 重新发现并加载插件
  - 会重建系统提示词，并重置当前会话上下文

工作区插件删除方式：

- 手动删除当前工作区 `.tg_agent/plugins/<name>`
- 然后执行 `/plugins reload`

## 3. 插件目录结构

一个公开支持的插件目录结构如下：

```text
my-plugin/
├── plugin.json
├── commands/
│   └── review.md
├── skills/
│   └── code-review/
│       └── SKILL.md
└── agents/
    └── reviewer.md
```

目录说明：

- `plugin.json`
  - 必需
  - 定义插件名称、版本、描述、最低 CLI 版本、能力开关
- `commands/`
  - 可选
  - 存放插件提供的 slash 命令定义，文件格式为 Markdown
- `skills/`
  - 可选
  - 存放插件提供的 skill，文件格式为 `SKILL.md`
- `agents/`
  - 可选
  - 存放插件提供的 agent 配置，文件格式为 Markdown

## 4. `plugin.json` 规范

每个插件根目录下都必须包含 `plugin.json`。

最小示例：

```json
{
  "schema_version": 1,
  "name": "review-kit",
  "version": "0.1.0",
  "description": "Built-in review helpers",
  "min_cli_version": "0.1.0",
  "capabilities": {
    "commands": true,
    "skills": true,
    "agents": true
  }
}
```

字段说明：

- `schema_version`
  - 当前固定为 `1`
- `name`
  - 插件名
  - 必须使用 kebab-case，例如 `review-kit`
- `version`
  - 插件版本
  - 必填
- `description`
  - 插件描述
  - 必填
- `min_cli_version`
  - 可选
  - 声明该插件要求的最小 CLI 版本
- `capabilities`
  - 可选
  - 声明插件是否提供 `commands`、`skills`、`agents`

校验规则：

- `schema_version` 不是 `1` 会加载失败
- `name` 不符合 kebab-case 会加载失败
- 缺少 `version` 或 `description` 会加载失败
- `min_cli_version` 高于当前 CLI 版本时会加载失败

能力开关说明：

- 若省略 `capabilities`，系统会根据对应目录是否存在来判断是否加载
- 若显式写为 `false`，即使目录存在也不会加载该类资源

## 5. `commands` 开发规范

插件命令存放在 `commands/*.md` 中。

命令文件由两部分组成：

- YAML frontmatter：命令元数据
- Markdown 正文：传给主 Agent 的提示内容

示例：

```md
---
name: review
description: Review the current code or a specific target.
usage: /review-kit:review [target]
category: Review
examples:
  - /review-kit:review
  - /review-kit:review auth.py
  - /review-kit:review payment flow
---

# Review Command

You are performing a focused code review.

Review priorities:

- correctness bugs
- regression risks
- missing or weak tests
- unsafe assumptions

Expected output:

- list findings first
- keep the summary brief
- if no findings are discovered, say so explicitly

Target or focus:
{{args}}
```

frontmatter 字段：

- `name`
  - 命令名
  - 最终命令会注册为 `/<plugin-name>:<name>`
- `description`
  - 命令说明
- `usage`
  - 使用方式
- `category`
  - 命令分类
- `examples`
  - 使用示例列表

命令名称规则：

- 必须以字母或数字开头
- 后续允许字母、数字、下划线、连字符

参数注入规则：

- 正文中若包含 `{{args}}`，执行时会将 slash 命令参数原样替换到该位置
- 若正文中不包含 `{{args}}`，但用户传入了参数，系统会把参数追加到正文后部

执行方式：

- 用户输入 `/review-kit:review auth.py`
- CLI 解析出插件命令 `review-kit:review`
- 将 Markdown 正文与参数合成为最终 prompt
- 把该 prompt 交给本仓库主 Agent 继续执行

适用场景：

- 固定分析流程
- 结构化代码审查
- 规范化需求澄清
- 某类任务的专用提示模板

## 6. `skills` 开发规范

插件 skill 存放在 `skills/<skill-name>/SKILL.md` 中。

示例：

```md
---
name: code-review
description: Focus the next request on correctness, regression risk, and missing tests.
category: Review
---

# Code Review

Use this skill when the user wants a focused code review rather than implementation.

Review priorities:

- correctness bugs and behavioral regressions
- unsafe assumptions and edge cases
- missing or weak tests
- configuration and migration risk

Response style:

- lead with findings
- include concrete file or code references when possible
- keep summaries short unless the user asks for a full walkthrough
```

注册行为：

- `skills/<skill-name>/SKILL.md` 会被注册为 `@<plugin-name>:<skill-name>`
- 例如插件名为 `review-kit`、skill 名为 `code-review`
- 则最终使用方式为 `@review-kit:code-review`

frontmatter 字段：

- `name`
- `description`
- `category`

建议：

- skill 适合承载可复用的高质量提示模板
- skill 内容应尽量聚焦一个任务场景
- `name` 保持简短稳定，便于补全和复用

## 7. `agents` 开发规范

插件 agent 存放在 `agents/*.md` 中。

示例：

```md
---
description: Focused code reviewer for correctness, regression risk, and test coverage gaps.
mode: subagent
model: GLM-4.7
temperature: 0.1
tools:
  read: true
  grep: true
  glob_search: true
  bash: false
  write: false
  edit: false
  todo_read: false
  todo_write: false
---

You are a focused code reviewer.

Primary goals:

- identify correctness bugs
- identify regression risk
- identify missing tests

Output rules:

- present findings first, ordered by severity
- cite concrete files or symbols when possible
- keep explanations concise
- if there are no findings, say that explicitly and mention residual risk
```

注册行为：

- agent 最终名称由文件名决定，而不是 frontmatter 中的 `name`
- 例如文件为 `agents/reviewer.md`
- 插件名为 `review-kit`
- 则最终注册名为 `review-kit:reviewer`

支持的常见字段：

- `description`
- `mode`
- `model`
- `temperature`
- `tools`

建议：

- agent 的职责应单一明确
- 默认只开放完成该职责所需的最小工具集合
- 说明输出格式和边界，避免行为发散

## 8. 开发流程

推荐开发流程如下：

1. 在本地创建插件目录，例如 `my-plugin/`
2. 编写 `plugin.json`
3. 在 `commands/` 中添加至少一个命令
4. 如有需要，再补充 `skills/` 和 `agents/`
5. 执行 `/plugins install <plugin_path>` 安装到内置插件目录
6. 执行 `/plugins reload`
7. 执行 `/plugins show <plugin-name>` 检查状态
8. 实际调用命令或 skill 验证行为

建议先做最小可运行版本：

- 先只写一个 `commands/*.md`
- 等命令行为稳定后，再增加 skills 和 agents

## 9. 调试与验证

调试插件时建议按以下顺序检查：

1. `/plugins list`
   - 确认插件是否被发现
2. `/plugins show <plugin-name>`
   - 查看插件来源、状态、路径、错误信息、已注册资源
3. `/plugins reload`
   - 在修改后重新加载插件
4. 实际执行插件命令
   - 确认 prompt 内容和结果是否符合预期

需要注意：

- `/plugins reload` 会重建系统提示词并清空当前会话上下文
- 如果你修改了 plugin 目录但忘记 reload，变更不会立即生效

## 10. 安装、更新与删除

### 10.1 安装内置插件

假设你本地已经准备好了插件目录：

```text
D:\tmp\my-plugin
```

执行：

```text
/plugins install D:\tmp\my-plugin
/plugins reload
```

安装后该插件会进入内置插件目录，切换工作区后仍可使用。

### 10.2 更新内置插件

更新插件常见方式有两种：

- 直接修改内置插件目录中的内容，然后执行 `/plugins reload`
- 在外部目录修改完成后，重新安装或覆盖内置目录中的插件，再执行 `/plugins reload`

如果只是想针对当前项目做局部改动，建议不要直接改内置插件，而是先复制到工作区：

```text
/plugins copy review-kit
/plugins reload
```

### 10.3 删除内置插件

删除内置插件：

```text
/plugins uninstall review-kit --force
```

说明：

- 该命令只删除内置插件
- 不删除工作区插件
- 删除后建议执行 `/plugins reload`

### 10.4 删除工作区插件

工作区插件不通过 `/plugins uninstall` 删除。

删除方式：

1. 手动删除当前工作区 `.tg_agent/plugins/<plugin-name>`
2. 执行 `/plugins reload`

## 11. 覆盖规则与常见现象

工作区插件优先级高于内置插件，因此会出现下面这些现象。

### 11.1 内置插件改了但效果没变

可能原因：

- 当前工作区存在同名插件，覆盖了内置插件

排查方式：

- 执行 `/plugins show <plugin-name>`
- 查看 `Source` 是否为 `workspace`

### 11.2 删除了内置插件，但插件仍然可用

可能原因：

- 当前工作区中仍有同名插件

这是正常行为，因为工作区插件会继续覆盖并生效。

### 11.3 想在某个项目里定制插件，但不想影响其他项目

推荐方式：

- 使用 `/plugins copy <plugin-name>` 复制到当前工作区
- 修改工作区版本
- 执行 `/plugins reload`

## 12. 常见错误与排查

### 12.1 `plugin.json not found`

原因：

- 插件目录下缺少 `plugin.json`

处理：

- 检查插件根目录结构是否正确

### 12.2 `Plugin name must be kebab-case`

原因：

- `plugin.json` 中的 `name` 不符合 kebab-case

错误示例：

- `ReviewKit`
- `review_kit`

正确示例：

- `review-kit`

### 12.3 `Plugin version is required`

原因：

- `plugin.json` 中缺少 `version`

### 12.4 `Plugin description is required`

原因：

- `plugin.json` 中缺少 `description`

### 12.5 `Plugin requires CLI version >= ...`

原因：

- `min_cli_version` 高于当前 CLI 版本

处理：

- 升级 CLI，或降低插件要求的最小版本

### 12.6 插件状态为 `failed`

排查方式：

- 执行 `/plugins show <plugin-name>`
- 查看 `Error` 字段

### 12.7 命令加载了，但补全或使用方式不对

检查项：

- `commands/*.md` frontmatter 中的 `name`
- `usage`
- `examples`
- 最终调用命令是否为 `/<plugin-name>:<command-name>`

### 12.8 skill 或 agent 没有出现

检查项：

- `plugin.json` 中对应 `capabilities` 是否被禁用
- 目录结构是否正确
- `SKILL.md` 或 agent Markdown frontmatter 是否合法

## 13. 最佳实践

- 一个插件尽量聚焦一个领域，例如 review、frontend、api-design
- `plugin.json` 中的 `name` 一旦对外发布，尽量不要频繁修改
- 每个命令都应提供清晰的 `description`、`usage`、`examples`
- prompt 应直接约束目标、输出格式和边界，避免过于抽象
- 先做最小命令，再逐步增加更多能力
- 修改插件后及时执行 `/plugins reload`
- 如果需要项目级差异，优先使用工作区覆盖，而不是直接复制出多个内置版本

## 14. 示例参考

仓库中的示例插件：

- `plugins/review-kit`

可重点参考：

- `plugin.json`
- `commands/review.md`
- `skills/code-review/SKILL.md`
- `agents/reviewer.md`

它展示了一个完整的公开插件模型：

- 一个命令
- 一个 skill
- 一个 agent

适合作为新插件的起点模板。
