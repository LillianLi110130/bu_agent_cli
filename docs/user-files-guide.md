# 用户文件使用手册

Crab CLI 支持通过若干用户文件配置项目规则、协作风格、长期记忆和自定义 agent。正确维护这些文件，可以让 Crab 在不同项目和会话中保持稳定、一致的工作方式。

| 类别 | 用途 | 对应文件 |
| --- | --- | --- |
| 项目规则 | 配置当前项目的工作约定 | `<workspace>/TGAGENTS.md` |
| 协作风格 | 配置 Crab 的语气和协作方式 | `~/.tg_agent/SOUL.md` |
| 身份职责 | 配置 Crab 的身份定位和职责边界 | `~/.tg_agent/IDENTITY.md` |
| 用户记忆 | 保存用户长期偏好 | `~/.tg_agent/memories/USER.md` |
| 工作记忆 | 保存项目经验、环境事实和工具注意事项 | `~/.tg_agent/memories/MEMORY.md` |
| 自定义 agent | 配置可复用的专业 agent | `~/.tg_agent/agents/<agent-name>.md` 或 `<workspace>/.tg_agent/agents/<agent-name>.md` |

## 快速选择

| 你想做的事 | 推荐文件 | 推荐方式 |
| --- | --- | --- |
| 让 Crab 记住某个项目的规则 | `<workspace>/TGAGENTS.md` | 直接告诉 Crab：“帮我写入 TGAGENTS.md” |
| 调整 Crab 的说话风格 | `~/.tg_agent/SOUL.md` | 手动编辑，或让 Crab 帮你生成 |
| 明确 Crab 的职责边界 | `~/.tg_agent/IDENTITY.md` | 手动编辑，或让 Crab 帮你生成 |
| 让 Crab 记住你的个人偏好 | `~/.tg_agent/memories/USER.md` | 直接对 Crab 说：“请记住……” |
| 让 Crab 记住项目经验或工具坑点 | `~/.tg_agent/memories/MEMORY.md` | 直接对 Crab 说：“请记住……” |
| 创建一个专门做某类任务的助手 | `~/.tg_agent/agents/<agent-name>.md` | 让 Crab 帮你创建 |

## 用户应该怎么修改这些文件

你有两种方式。

### 方式一：直接告诉 Crab

推荐优先使用这种方式，尤其是在不确定文件格式或文件位置时。

例如：

```text
请记住：我希望以后默认用中文回答。
```

```text
请帮我把当前项目的规则写进 TGAGENTS.md：提交前只提交相关文件，不要提交本地配置。
```

```text
请帮我创建一个代码审查 agent，重点关注 bug、回归风险和测试缺口。
```

### 方式二：自己打开文件编辑

如果你知道文件位置，也可以直接用编辑器打开修改。

适合自己编辑的文件：

```text
<workspace>/TGAGENTS.md
~/.tg_agent/SOUL.md
~/.tg_agent/IDENTITY.md
~/.tg_agent/agents/<agent-name>.md
<workspace>/.tg_agent/agents/<agent-name>.md
```

不建议手动编辑的文件：

```text
~/.tg_agent/memories/USER.md
~/.tg_agent/memories/MEMORY.md
```

原因是 memory 文件有内部分隔格式，更适合通过“请记住……”让 Crab 写入。

## 文件 1：项目规则文件 `TGAGENTS.md`

### 它是什么

`TGAGENTS.md` 是当前项目的“工作说明书”。

它告诉 Crab：

- 这个项目怎么运行。
- 这个项目怎么测试。
- 这个项目有什么禁忌。
- 哪些文件不要动。
- 提交代码时有什么规则。
- 当前团队希望 Crab 遵守什么约定。

### 放在哪里

放在项目根目录。

```text
<workspace>/TGAGENTS.md
```

例如你的项目在：

```text
/Users/you/my-project
```

那文件就是：

```text
/Users/you/my-project/TGAGENTS.md
```

### 适合写什么

适合写“这个项目里长期有效的规则”。

例如：

```md
# 项目规则

- 默认使用中文回答。
- 只修改和当前任务直接相关的文件。
- 不要提交本地运行配置、临时文件和 PPT 产物。
- 测试命令是 `uv run pytest`。
- 修改 CLI UI 后，需要注意 Windows、Linux、macOS 终端兼容性。
```

### 怎么让 Crab 帮你修改

你可以直接说：

```text
请帮我创建 TGAGENTS.md，写入当前项目规则：测试用 uv run pytest，提交时只提交相关文件。
```

或者：

```text
请帮我把 TGAGENTS.md 补充一条：不要提交本地运行配置和临时 PPT 文件。
```

### 会不会自动更新

不会。

Crab 不会自动把所有经验写进 `TGAGENTS.md`。如果你希望某条项目规则长期保留，需要明确告诉 Crab 写进去，或者自己编辑。

### 什么时候生效

通常当前会话下一轮输入就会同步。

如果你不确定，可以输入：

```text
/reset
```

或者重新启动 Crab CLI。

## 文件 2：性格设定 `SOUL.md`

### 它是什么

`SOUL.md` 是 Crab 的“性格说明”。

它决定 Crab 更像什么样的协作者，比如：

- 说话更简洁还是更详细。
- 更主动还是更保守。
- 更工程化还是更业务化。
- 遇到风险时要不要提醒。
- 用户焦虑时语气要不要更稳定。

### 放在哪里

```text
~/.tg_agent/SOUL.md
```

这是用户级文件。换一个项目，它仍然会影响 Crab。

### 适合写什么

例如：

```md
# SOUL.md

你是一个中文优先的工程协作者。

协作风格：

- 回答直接、清楚，不绕。
- 遇到代码任务时先看现有代码，再动手。
- 不擅自覆盖用户未提交改动。
- 对高风险操作要先提醒。
- 用户明显焦虑时，语气要稳一点。
```

### 怎么让 Crab 帮你修改

你可以说：

```text
请帮我创建 ~/.tg_agent/SOUL.md。我希望你的风格是中文、简洁、稳妥，有工程判断，不要废话。
```

或者：

```text
请把 ~/.tg_agent/SOUL.md 调整得更温和一点，但仍然保持直接。
```

### 会不会自动更新

不会。

`SOUL.md` 不会被自动沉淀。它需要用户主动修改。

### 什么时候生效

建议重启 Crab CLI 后生效。

## 文件 3：职责说明 `IDENTITY.md`

### 它是什么

`IDENTITY.md` 是 Crab 的“岗位说明书”。

如果说 `SOUL.md` 决定“怎么说话”，那么 `IDENTITY.md` 决定“它是谁、负责什么、不负责什么”。

### 放在哪里

```text
~/.tg_agent/IDENTITY.md
```

### 适合写什么

例如：

```md
# IDENTITY.md

你是用户的本地开发助手。

主要职责：

- 阅读和解释当前工作区代码。
- 根据用户请求修改代码。
- 运行必要测试并说明结果。
- 帮助用户处理 Git 工作流。
- 维护 README、用户手册和设计文档。

边界：

- 不擅自提交或推送，除非用户明确要求。
- 不擅自删除用户文件。
- 不把临时产物混进提交。
```

### 怎么让 Crab 帮你修改

可以说：

```text
请帮我创建 ~/.tg_agent/IDENTITY.md，定位为我的本地开发助手，主要负责代码修改、测试、Git 和文档维护。
```

### 会不会自动更新

不会。

### 什么时候生效

建议重启 Crab CLI 后生效。

## 文件 4：用户偏好记忆 `USER.md`

### 它是什么

`USER.md` 是 Crab 对“用户本人”的长期记忆。

它适合保存你的长期偏好，比如：

- 你喜欢中文回答。
- 你不喜欢太长的解释。
- 你提交代码时只想提交相关文件。
- 你希望先解释风险再执行高风险操作。

### 放在哪里

```text
~/.tg_agent/memories/USER.md
```

### 推荐怎么修改

推荐直接告诉 Crab：

```text
请记住：我希望以后默认用中文回答。
```

```text
请记住：我提交代码时通常只想提交当前任务相关文件，不要提交本地配置。
```

不推荐手动编辑这个文件，因为 memory 文件内部有分隔格式。

### 会不会自动更新

会。

Crab 有 memory review 机制，会在合适的时候把值得长期保存的信息沉淀进去。

但是最稳妥的方式仍然是明确说：

```text
请记住：...
```

### 什么时候生效

通常下一个 CLI 会话生效。

也就是说，当前会话中刚刚写入的记忆，更多是给未来会话使用。

### 示例内容

```text
用户偏好中文回答。
§
用户希望提交代码时只提交当前任务相关文件。
§
用户不希望把本地运行配置和临时产物提交到 Git。
```

这里的 `§` 是内部使用的分隔符。

## 文件 5：工作经验记忆 `MEMORY.md`

### 它是什么

`MEMORY.md` 是 Crab 对“环境、项目、工具经验”的长期记忆。

它适合保存：

- 项目的稳定事实。
- 常用命令。
- 某些工具的坑点。
- 某个模型的兼容问题。
- 以后可能反复遇到的经验。

### 放在哪里

```text
~/.tg_agent/memories/MEMORY.md
```

### 推荐怎么修改

推荐直接告诉 Crab：

```text
请记住：这个项目启动 CLI 用 uv run python tg_crab_main.py。
```

```text
请记住：DeepSeek 模型不要使用 thinking mode，否则可能要求回传 reasoning_content。
```

### 会不会自动更新

会。

Crab 会通过 memory review 自动判断是否有值得长期保存的项目事实或工具经验。

### 什么时候生效

通常下一个 CLI 会话生效。

### 示例内容

```text
该项目 CLI 入口是 crab。
§
DeepSeek 使用 OPENAI_API_KEY 和 LLM_BASE_URL=https://api.deepseek.com。
§
当前项目有一些本地运行文件长期未提交，不要自动 revert。
```

## 文件 6：专家助手 `<agent-name>.md`

### 它是什么

`<agent-name>.md` 是一个“专业助手配置文件”。

你可以用它创建不同类型的专家助手，例如：

- 代码审查助手。
- 前端开发助手。
- 测试修复助手。
- 项目探索助手。
- 文档整理助手。

### 放在哪里

用户级 agent，适合所有项目复用：

```text
~/.tg_agent/agents/<agent-name>.md
```

项目级 agent，只适合当前项目：

```text
<workspace>/.tg_agent/agents/<agent-name>.md
```

### 推荐怎么修改

推荐让 Crab 帮你创建。

例如：

```text
请帮我创建一个用户级 agent：code-reviewer，用来做代码审查，重点关注 bug、回归风险和测试缺口。
```

或者：

```text
请在当前项目的 .tg_agent/agents 里创建一个 frontend-developer.md，专门负责前端 UI 实现。
```

### 会不会自动更新

不会。

Agent 配置需要用户明确创建或修改。

### 什么时候生效

通常需要重新加载 agent 或重启 CLI。

可以尝试：

```text
/agents reload
```

如果不确定，重启 Crab CLI 最稳。

### 示例内容

```md
---
name: code-reviewer
description: Review code changes and identify bugs, regressions, and missing tests.
---

你是一个代码审查 agent。

审查重点：

- 优先指出 bug 和行为回归。
- 关注测试缺口。
- 不要把风格偏好当成严重问题。
- 输出要包含文件和行号。
```

## 怎么判断应该放在哪里

### 这是项目规则吗？

放：

```text
<workspace>/TGAGENTS.md
```

例子：

```text
这个项目测试命令是 uv run pytest。
这个项目不要提交 uv.lock。
这个项目 UI 输出要兼容多终端。
```

### 这是我的个人偏好吗？

放：

```text
~/.tg_agent/memories/USER.md
```

例子：

```text
我喜欢中文回答。
我希望回答短一点。
我提交时只想提交相关文件。
```

### 这是项目经验或工具坑点吗？

放：

```text
~/.tg_agent/memories/MEMORY.md
```

例子：

```text
DeepSeek 某些模型不兼容 thinking mode。
这个项目的 CLI 命令入口是 crab。
```

### 这是 Crab 的整体语气吗？

放：

```text
~/.tg_agent/SOUL.md
```

例子：

```text
回答要温和、直接、中文优先。
```

### 这是 Crab 的职责边界吗？

放：

```text
~/.tg_agent/IDENTITY.md
```

例子：

```text
你是我的本地开发助手，不要擅自提交或推送。
```

### 这是一个专业角色吗？

放：

```text
~/.tg_agent/agents/<agent-name>.md
<workspace>/.tg_agent/agents/<agent-name>.md
```

例子：

```text
code-reviewer
frontend-developer
test-fixer
docs-writer
```

## 不建议手动修改的文件

以下文件或目录属于系统内置资源，不建议直接修改：

```text
~/.tg_agent/skills/.builtin/
agent_core/prompts/
agent_core/prompts/agents/
```

原因：

- `.builtin` 是内置技能同步目录，可能被自动刷新。
- `agent_core/prompts/` 是包内默认提示词，升级后可能变化。
- 内置 agent 建议通过用户级或项目级 agent 覆盖，而不是直接修改源码。

## 推荐流程

如果你不确定该改哪个文件，直接告诉 Crab：

```text
我希望你以后记住：……
```

或者：

```text
请把这条规则写进当前项目的 TGAGENTS.md：……
```

或者：

```text
请帮我创建一个专门做……的 agent。
```

Crab 会根据内容选择更合适的文件。日常使用时，不需要记住所有路径。
