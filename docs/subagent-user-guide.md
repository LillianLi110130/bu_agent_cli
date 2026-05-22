# Subagent 使用说明

## 这是什么

Subagent 是 Crab CLI 在处理复杂任务时可以临时调用的“子智能体”。主 agent 仍然负责理解你的整体目标、决定是否委派、整合结果并对最终答复负责；subagent 只负责一段明确的子任务。

你可以把它理解成：

- 主 agent：项目负责人，决定怎么推进。
- subagent：被临时分派出去的专门执行者或复核者。

日常使用时，你不需要直接调用底层工具。通常只要用自然语言提出需求，主 agent 会在它判断确实有帮助时使用 subagent。

Subagent 的优势主要有：

- 并行处理：可以把复杂任务拆成多个方向同时推进，减少主流程等待时间。
- 上下文隔离：大量文件搜索、日志阅读、局部验证和中间推理会发生在 subagent 自己的上下文里，主 agent 通常只接收它返回的结论、证据和风险点。
- 保持主上下文清爽：主上下文可以更多保留用户目标、关键决策、最终汇总和对用户的解释，而不是塞满一次性检索结果。
- 专门角色处理专门问题：例如让 `explore` 探索代码、让 `debugger` 定位 bug、让 `frontend-developer` 实现前端、让 `verifier` 做独立验证。
- 第二视角：对于需要复核的任务，subagent 可以帮助主 agent 在汇总前发现遗漏和风险。

## 什么时候会用到

Subagent 适合这些情况：

- 任务可以拆成几个互不冲突的部分并行处理。
- 需要一个独立视角做代码探索、调试、实现、验证或规格检查。
- 需要使用某个专门角色，例如 `explore`、`debugger`、`frontend-developer`、`backend-developer`、`verifier`。
- 某个长任务可以放到后台运行，而主流程不需要立刻等待最终结果。

不适合这些情况：

- 一个简单问题主 agent 就能直接完成。
- 任务目标、范围和成功标准还没说清楚。
- 只是为了“更稳”而机械地多开一个 agent。
- 需要相关 skill 时，却想绕过 skill 直接委派。

## 可用 subagent 从哪里来

Crab 会从多个位置加载 agent 配置：

```text
agent_core/prompts/agents/              # 内置 agent
~/.tg_agent/agents/                     # 用户级 agent
<workspace>/.tg_agent/agents/           # 项目级 agent
plugins/<plugin>/agents 或类似目录       # 插件提供的 agent
```

查看当前可用 agent：

```text
/agents
/agents list
/agents show <name>
```

按来源筛选：

```text
/agents list workspace
/agents list user
/agents list builtin
/agents list plugin
```

## 内置 agent

Crab CLI 默认带有几个内置 agent。它们不依赖插件，通常启动后就能在 `/agents list builtin` 里看到。

`explore`

- 用来快速探索代码库。
- 适合找文件、找入口、梳理调用链、确认某个功能分布在哪些模块。
- 建议用于“先弄清楚在哪里改”的阶段。

示例：

```text
请让 explore 梳理登录流程相关代码，找出入口文件、关键服务和测试位置，不要修改文件。
```

`plan`

- 用来做实现方案设计。
- 适合在动手前分析影响范围、拆步骤、识别风险和验证方式。
- 建议用于较大改动、跨模块改动或不确定实现路径的任务。

示例：

```text
这个改动会影响 CLI 输入、agent registry 和 system prompt，请让 plan 先给出实施方案和风险点。
```

`general-purpose`

- 通用型 agent。
- 适合资料整理、复杂问题研究、没有明显专门角色的辅助任务。
- 如果没有合适的专用 agent，但任务又适合拆出去处理，可以考虑使用它。

示例：

```text
请启动一个后台 general-purpose subagent 梳理 docs 目录里的用户文档问题。
```

插件也可以提供额外 agent，例如 `awesome-subagents` 插件提供 `debugger`、`frontend-developer`、`backend-developer`、`verifier`。是否可用以你当前 `/agents list` 输出为准。

## Agent 配置文件长什么样

一个 agent 通常是一个 Markdown 文件，文件名一般和 agent 名称一致，例如：

```text
<workspace>/.tg_agent/agents/project-verifier.md
```

示例：

```md
---
name: project-verifier
description: 当前项目专用验证 agent，重点检查实现是否满足需求、测试证据是否充分。
model: inherit
tools:
  - read
  - grep
  - bash
disallowedTools:
  - write
  - edit
maxTurns: 8
skills:
  - test-driven-development
---

你是当前项目的验证 agent。你的任务是基于需求、改动和测试结果判断任务是否真正完成。

重点关注：

- 实现是否满足用户目标和任务文档。
- 是否有足够的测试、构建或运行证据。
- 是否存在明显的回归风险或未验证边界。

输出要求：

- 先给出 PASS、FAIL 或 NEEDS_MORE_EVIDENCE。
- 列出证据来源和仍然缺失的验证项。
- 不要批准没有证据支撑的完成声明。
```

常用字段：

- `name`：agent 名称，委派时使用。
- `description`：显示在可用 subagent 列表里，帮助主 agent 判断什么时候使用。
- `model`：指定模型；`inherit` 表示继承主 agent 当前模型。
- `tools`：允许该 agent 使用的工具列表。
- `disallowedTools`：禁用工具列表。
- `maxTurns`：该 agent 最大推理/工具循环次数。
- `skills`：启动该 agent 时附加加载的 skill。
- Markdown 正文：这个 agent 的系统提示词。

## 创建和维护 agent

交互式创建：

```text
/agents create project-verifier
```

创建时会依次询问：

- 描述
- 模型
- 允许工具
- 禁用工具
- 系统提示词草稿来源

系统提示词草稿可以选择：

- 默认模板
- 大模型生成

选择大模型生成时，可以按 `q` 取消生成并退回默认模板。最后都会进入可编辑的多行输入，你可以直接修改草稿，按 `Ctrl+Q` 结束编辑。

编辑已有 agent：

```text
/agents edit project-verifier
/agents edit project-verifier --editor
```

删除 agent：

```text
/agents delete project-verifier
```

重新加载 agent：

```text
/agents reload
```

注意：`/agents reload` 会重建主 agent 系统提示词，并重置当前会话上下文。命令执行前会要求确认。创建、编辑、删除 agent 后，如果希望当前主 agent 的系统提示词立刻看到 agent 注册变化，可以运行 `/agents reload`，或者重启 CLI。

## 主 agent 怎么使用 subagent

底层有两个委派工具：

- `delegate`：委派一个子任务。
- `delegate_parallel`：并行委派两个或更多前台子任务。

普通用户一般不需要直接写工具调用。你可以这样表达：

```text
请让 verifier 帮我独立验证这次改动是否满足需求，并检查测试证据是否充分。
```

或者：

```text
这个任务可以拆成前端和后端两部分并行做，请分别让合适的 agent 处理。
```

你也可以明确指定数量。比如你说“用 2 个 explore subagent”，主 agent 会按你的数量启动两个 `explore` 子任务，并把任务拆成两个互不重叠的方向。

示例：

```text
请用 2 个 explore subagent 并行梳理这个项目：一个负责 CLI 输入和命令处理，一个负责 agent_core 的 subagent 执行链路。最后由你汇总。
```

主 agent 会根据当前可用 agent、任务复杂度和上下文决定是否委派。

## 命名 subagent 和 fork subagent

Crab 里有两类子执行方式。

命名 subagent：

- 使用某个已有 agent 配置，例如 `debugger`、`frontend-developer`、`backend-developer`、`verifier`。
- 使用该配置里的系统提示词、工具限制、模型设置和 skills。
- 适合专门角色任务，比如 review、debug、frontend 实现、验证。

Fork subagent：

- 不指定 agent 名称时，会从当前主 agent 上下文 fork 一个子执行体。
- 继承主 agent 当前模型和上下文快照。
- 适合短期、局部、需要复用当前上下文的小任务。
- 不适合替代命名 agent，也不能继续二次委派。

## 前台和后台

你可以在自然语言里直接指定 subagent 的运行方式：

- 想让主 agent 等结果再继续，就写“前台执行”“等它完成后再继续”。
- 想让任务先跑起来，不阻塞当前对话，就写“后台执行”“启动后台 subagent”。

示例：

```text
请前台让 verifier 验证这次改动，等它完成后再继续汇总。
```

```text
请后台启动 general-purpose subagent 梳理 docs 目录里的用户文档问题。
```

前台 subagent：

- 主 agent 会等待它完成。
- 适合后续步骤立刻依赖结果的任务。
- 执行过程中 CLI 会显示类似这样的进度：

```text
[subagent] <task_id> | debugger | Debug failing tests | tools=2 | tokens=1.2k | 8s | Calling grep...
```

后台 subagent：

- 适合较长任务，且主流程暂时不需要立刻消费结果。
- 启动后会返回 task id。
- 完成后 CLI 会显示一条后台任务通知，并把完成信息注入到主 agent 后续可见的上下文里。它不是一个独立的主动提醒或定时通知功能。

```text
[bg done] verifier <task_id> finished ... Use /task <task_id> to inspect the result.
```

查看所有任务：

```text
/tasks
```

查看某个任务详情：

```text
/task <task_id>
```

取消任务：

```text
/task_cancel <task_id>
```

前台 subagent 运行时，如果你在固定输入界面按取消键，CLI 会尝试取消正在运行的前台 subagent。

## 委派时的好 prompt 长什么样

如果你希望主 agent 使用某个 subagent，最好把任务说清楚。

比较好的说法：

```text
请让 verifier 检查当前改动是否满足需求。重点看测试证据、构建结果和未验证风险，输出 PASS、FAIL 或 NEEDS_MORE_EVIDENCE。
```

更适合并行的说法：

```text
请把这个需求拆成前端实现和后端接口两部分并行处理。让 frontend-developer 只改 web/src，让 backend-developer 只改 gateway/src，最后由你汇总结果和风险。
```

不太好的说法：

```text
找个 agent 帮我看看。
```

原因是缺少目标、范围、关注重点和交付格式，主 agent 很难安全地委派。

## 常见场景

代码探索：

```text
请让 explore 快速梳理登录流程相关代码，找出入口文件、关键服务和测试位置，不要修改文件。
```

调试问题：

```text
这个报错涉及多处调用链，请让 debugger 帮我定位根因，重点看失败测试、错误日志和相关代码路径。
```

前后端并行实现：

```text
请并行处理这个需求：frontend-developer 负责 web/src 下的 UI 和交互，backend-developer 负责 gateway/src 下的接口和数据流，最后由你汇总改动和验证结果。
```

独立验证：

```text
请让 verifier 独立验证这次实现是否满足需求，重点检查测试证据、构建结果和剩余风险。
```

方案规划：

```text
这个改动会影响 CLI 输入、agent registry 和 system prompt，请让 plan 先给出实施方案和风险点。
```

后台长任务：

```text
请启动一个后台 general-purpose subagent 梳理 docs 目录里的用户文档问题。
```

## 注意事项

- Subagent 的输出不是绝对事实，主 agent 仍会整合和判断。
- 命名 subagent 只能调用当前注册列表里存在的 agent。
- 子 agent 不能继续委派其他 subagent，避免无限嵌套。
- 命名 subagent 会过滤 delegation 工具，也会按自己的 `tools` 和 `disallowedTools` 限制能力。
- `delegate_parallel` 只用于前台并行，不用于后台任务。
- 后台任务不要反复轮询等待；系统会在完成时通知。
- 修改 agent 配置后，当前主 agent 的系统提示词不一定立即感知注册变化。需要 `/agents reload` 或重启 CLI。

## 排查问题

找不到某个 agent：

```text
/agents list
/agents show <name>
```

如果刚创建或编辑过 agent，运行：

```text
/agents reload
```

后台任务没有结果：

```text
/tasks
/task <task_id>
```

需要停止任务：

```text
/task_cancel <task_id>
```

不确定当前 subagent 是否适合某个任务时，可以直接问：

```text
这个任务适合用 subagent 吗？如果适合，请说明会怎么拆分。
```
