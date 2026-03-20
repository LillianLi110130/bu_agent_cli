# Ralph 使用文档

## 1. Ralph 是什么

Ralph 是当前仓库里的任务执行工作流。

它只负责两件事：

- 初始化标准 spec 目录和最小 `.devagent`
- 消费 `plan/plan.json`，按顺序执行 `plan/*.md` 中定义的任务

Ralph **不负责需求拆解**。需求分析、技术设计、任务拆解由插件完成；Ralph 只接收最终计划并执行。

当前仓库内置了 3 组可配合 Ralph 使用的拆解插件：

- `frontend-workflow`
- `ta-workflow`
- `tgcrab-frontend`

## 2. 30 秒快速开始

如果你只想先跑通一次，按下面步骤做：

```text
/ralph init-spec my_spec
/ralph init-agent
```

把原始需求材料放到：

```text
docs/spec/my_spec/input/
```

然后任选一种拆解插件生成计划。

方案 A：使用当前 agent 做前端拆解

```text
/tgcrab-frontend:requirement my_spec
/tgcrab-frontend:design my_spec
/tgcrab-frontend:tasks my_spec
```

方案 B：使用 `frontend-workflow`

```text
/frontend-workflow:requirement my_spec
/frontend-workflow:design my_spec
/frontend-workflow:tasks my_spec
```

方案 C：使用 `ta-workflow`

```text
/ta-workflow:ta my_spec
/ta-workflow:decompose my_spec
```

计划生成后执行：

```text
/ralph dry-run my_spec
/ralph run my_spec --silent
/ralph status
```

## 3. 前置条件

使用 Ralph 前请确认：

- 当前工作区可正常运行本仓库 CLI
- 工作区中已执行 `/ralph init-agent`
- 机器上存在 `devagent` 命令并已加入 `PATH`

说明：

- `tgcrab-frontend` 的拆解阶段使用当前仓库主 agent
- `frontend-workflow` 和 `ta-workflow` 的拆解阶段会调用 `devagent`
- Ralph 的执行阶段统一调用 `devagent`

所以无论你使用哪一种拆解插件，只要最终要执行 Ralph，都需要本机有 `devagent`。

## 4. 标准目录结构

Ralph 围绕 `docs/spec/<spec_name>/` 工作。

```text
docs/spec/<spec_name>/
├── input/
├── artifacts/
├── plan/
│   └── plan.json
├── implement/
└── logs/
```

目录含义：

- `input/`
  - 原始需求输入材料
- `artifacts/`
  - 拆解插件生成的中间文档
- `plan/`
  - Ralph 真正消费的最终计划
- `implement/`
  - 执行阶段的实现记录
- `logs/`
  - 拆解和执行日志

## 5. `/ralph` 命令

### 5.1 `/ralph init-spec`

```text
/ralph init-spec <spec_name> [--target-dir <path>] [--force]
```

作用：

- 初始化标准 spec 目录
- 创建空的 `plan/plan.json`
- 创建 `input/`、`artifacts/`、`plan/`、`implement/`、`logs/`

示例：

```text
/ralph init-spec my_spec
```

### 5.2 `/ralph init-agent`

```text
/ralph init-agent [--target-dir <path>]
```

作用：

- 初始化最小 `.devagent`
- 为 Ralph 执行阶段提供 `.devagent/commands/ralph/implement.md`

示例：

```text
/ralph init-agent
```

### 5.3 `/ralph dry-run`

```text
/ralph dry-run <spec_name>
```

作用：

- 读取 `plan/plan.json`
- 展示待执行任务顺序
- 不真正执行任务

建议在每次生成或修改计划后先跑一次。

### 5.4 `/ralph run`

```text
/ralph run <spec_name> [--silent]
```

作用：

- 后台启动 `ralph_loop.py`
- 按 `plan.json` 中的优先级顺序执行任务
- 更新任务状态和日志

常用示例：

```text
/ralph run my_spec --silent
```

### 5.5 `/ralph status`

```text
/ralph status [run_id]
```

作用：

- 查看 Ralph 后台运行状态
- 不带 `run_id` 时显示该工作区的运行列表和计划摘要
- 带 `run_id` 时显示单次运行详情

### 5.6 `/ralph cancel`

```text
/ralph cancel <run_id>
```

作用：

- 终止指定的后台执行

## 6. 如何选择拆解插件

### 6.1 `tgcrab-frontend`

适合场景：

- 希望直接使用当前 CLI 主 agent 完成需求分析、技术设计和任务拆解
- 希望 prompt 可直接在插件 Markdown 中修改

命令链：

```text
/tgcrab-frontend:requirement <spec_name>
/tgcrab-frontend:design <spec_name>
/tgcrab-frontend:tasks <spec_name>
```

特点：

- 纯 prompt 插件
- 任务拆解阶段不额外调用插件脚本
- 最终直接生成 `plan/plan.json` 和 `plan/*.md`

### 6.2 `frontend-workflow`

适合场景：

- 标准前端三阶段流程
- 希望把 requirement、design、tasks 分阶段产出和审核

命令链：

```text
/frontend-workflow:requirement <spec_name>
/frontend-workflow:design <spec_name>
/frontend-workflow:tasks <spec_name>
```

阶段输出：

- `requirement`
  - 输出 `artifacts/frontend-workflow/01_requirement.md`
- `design`
  - 输出 `artifacts/frontend-workflow/02_design.md`
- `tasks`
  - 输出 `plan/plan.json` 和 `plan/*.md`

### 6.3 `ta-workflow`

适合场景：

- 希望先做 TA 分析，再做最终原子任务拆解
- 需要较强的中间层任务域分析

命令链：

```text
/ta-workflow:ta <spec_name>
/ta-workflow:decompose <spec_name>
```

阶段输出：

- `ta`
  - 输出：
    - `artifacts/ta-workflow/01_requirements.md`
    - `artifacts/ta-workflow/02_design.md`
    - `artifacts/ta-workflow/03_task_domains.md`
- `decompose`
  - 输出 `plan/plan.json` 和 `plan/*.md`

## 7. 推荐使用流程

### 7.1 使用 `tgcrab-frontend`

```text
/ralph init-spec my_spec
/ralph init-agent
# 将原始需求材料放入 docs/spec/my_spec/input/
/tgcrab-frontend:requirement my_spec
/tgcrab-frontend:design my_spec
/tgcrab-frontend:tasks my_spec
/ralph dry-run my_spec
/ralph run my_spec --silent
/ralph status
```

### 7.2 使用 `frontend-workflow`

```text
/ralph init-spec my_spec
/ralph init-agent
# 将原始需求材料放入 docs/spec/my_spec/input/
/frontend-workflow:requirement my_spec
/frontend-workflow:design my_spec
/frontend-workflow:tasks my_spec
/ralph dry-run my_spec
/ralph run my_spec --silent
/ralph status
```

### 7.3 使用 `ta-workflow`

```text
/ralph init-spec my_spec
/ralph init-agent
# 将原始需求材料放入 docs/spec/my_spec/input/
/ta-workflow:ta my_spec
/ta-workflow:decompose my_spec
/ralph dry-run my_spec
/ralph run my_spec --silent
/ralph status
```

## 8. 如何调整拆解 prompt

如果你想调整某个内置拆解插件的 prompt，不要直接修改内置插件目录。推荐做法是先复制到当前工作区。

### 8.1 复制插件到工作区

例如复制 `tgcrab-frontend`：

```text
/plugins copy tgcrab-frontend
/plugins reload
```

复制后，工作区会生成：

```text
.tg_agent/plugins/tgcrab-frontend/
```

同名工作区插件会覆盖内置插件，因此后续调用会优先使用你修改后的版本。

### 8.2 不同插件应修改哪些文件

`tgcrab-frontend`

- 需求分析 prompt：
  - `.tg_agent/plugins/tgcrab-frontend/commands/requirement.md`
- 技术设计 prompt：
  - `.tg_agent/plugins/tgcrab-frontend/commands/design.md`
- 任务拆解 prompt：
  - `.tg_agent/plugins/tgcrab-frontend/commands/tasks.md`

`frontend-workflow`

- 需求分析 prompt：
  - `.tg_agent/plugins/frontend-workflow/prompts/requirement.md`
- 技术设计 prompt：
  - `.tg_agent/plugins/frontend-workflow/prompts/design.md`
- 任务拆解 prompt：
  - `.tg_agent/plugins/frontend-workflow/prompts/tasks.md`

`ta-workflow`

- TA 分析 prompt：
  - `.tg_agent/plugins/ta-workflow/prompts/ta.md`
- 最终拆解 prompt：
  - `.tg_agent/plugins/ta-workflow/prompts/decompose.md`

### 8.3 修改后怎么生效

修改完成后执行：

```text
/plugins reload
```

然后重新运行对应插件命令。

## 9. `plan.json` 最低契约

Ralph 最终只关心 `plan/plan.json` 和 `plan/*.md`。

`plan.json` 必须是一个 JSON 数组。每个任务至少包含：

```json
[
  {
    "task_name": "task_01",
    "priority": 0,
    "status": "TODO",
    "implment_times": 0,
    "task_file": "D:/project/docs/spec/my_spec/plan/01-task.md"
  }
]
```

关键要求：

- `task_name`
  - 唯一
- `priority`
  - 唯一整数
  - 数值越小越早执行
- `status`
  - 初始值必须是 `TODO`
- `implment_times`
  - 初始值必须是 `0`
- `task_file`
  - 必须指向真实存在的任务 Markdown
  - 推荐使用绝对路径

插件通常还会额外写入这些字段：

- `complete_time`
- `last_output`
- `log_file`
- `branch`

Ralph 会在执行过程中更新状态和日志字段。

## 10. 日志和运行状态怎么看

Ralph 运行相关的信息主要在下面几个位置：

- 运行状态：
  - `.tg_agent/ralph/runs.json`
- 后台进程输出：
  - `.tg_agent/ralph/process_logs/*.stdout.log`
  - `.tg_agent/ralph/process_logs/*.stderr.log`
- 任务执行日志：
  - `docs/spec/<spec_name>/logs/`
- 计划执行结果：
  - `docs/spec/<spec_name>/plan/plan.json`

排查建议：

1. 先执行 `/ralph status`
2. 再查看对应 run 的 `stdout_log` 和 `stderr_log`
3. 再打开 `plan.json` 看任务状态是否被更新
4. 最后查看 `logs/` 和具体 `task_file`

## 11. 常见问题

### 11.1 `devagent command not found`

原因：

- 系统中没有安装 `devagent`
- 或未加入 `PATH`

影响：

- Ralph 执行无法启动
- `frontend-workflow` / `ta-workflow` 的拆解阶段也无法运行

### 11.2 `Implement prompt not found`

原因：

- 没有执行 `/ralph init-agent`
- 或工作区 `.devagent/commands/ralph/implement.md` 缺失

处理：

```text
/ralph init-agent
```

### 11.3 `Plan file not found`

原因：

- 还没有运行拆解插件生成计划
- 或 `plan/plan.json` 被删除

处理：

- 先执行对应拆解插件命令
- 再执行 `/ralph dry-run` / `/ralph run`

### 11.4 插件生成了中间文档，但没有生成最终计划

说明：

- 你可能只执行了中间阶段，没有执行最终拆解阶段

例如：

- `frontend-workflow` 必须执行到 `tasks`
- `ta-workflow` 必须执行到 `decompose`
- `tgcrab-frontend` 必须执行到 `tasks`

### 11.5 修改了内置插件 prompt，但效果没变化

可能原因：

- 当前工作区里已经有同名工作区插件覆盖了内置插件

排查方式：

```text
/plugins show <plugin_name>
```

查看 `Source` 是否为 `workspace`。

### 11.6 删除了内置插件，但插件仍然可用

可能原因：

- 当前工作区中还有同名插件

这是正常行为，因为工作区插件优先级更高。

## 12. 边界说明

请把 Ralph 理解为“计划执行器”，而不是“拆解器”。

职责边界如下：

- 拆解插件负责：
  - 需求分析
  - 技术设计
  - 任务拆解
  - 生成 `plan/plan.json`
- Ralph 负责：
  - 初始化目录
  - 校验执行前置条件
  - 执行 `plan.json`
  - 维护运行状态和日志

你也可以完全跳过插件，手工准备：

- `plan/plan.json`
- `plan/*.md`
- `.devagent/commands/ralph/implement.md`

然后直接运行 Ralph。
