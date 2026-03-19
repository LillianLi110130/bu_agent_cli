# Ralph Loop 使用指南

## 1. 定位

Ralph Core 现在只负责两件事：

- 初始化标准 spec 目录与最小 `.devagent`
- 执行 `plan/plan.json` 和 `plan/*.md`

任务拆解不再内置在 Ralph Core 中，而是由插件完成。当前仓库提供两类规划插件：

- `ta-workflow`
  - `ta`
  - `decompose`
- `frontend-workflow`
  - `requirement`
  - `design`
  - `tasks`

## 2. 标准目录结构

Ralph 围绕 `docs/spec/<spec_name>/` 这套结构工作：

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
  - 各插件的中间拆解产物，供人工审核
- `plan/`
  - Ralph 真正消费的最终任务计划
- `implement/`
  - 执行阶段生成的实现记录
- `logs/`
  - 插件拆解日志和 Ralph 执行日志

## 3. 最小 `.devagent`

`/ralph init-agent` 会把仓库内置的最小 `.devagent` 复制到工作区：

```text
.devagent/
├── agents/
└── commands/
    └── ralph/
        └── implement.md
```

说明：

- `.devagent/agents/` 初始为空目录
- 插件会在各自阶段命令执行前，把自己的 `dev_subagents/*.md` 同步到工作区 `.devagent/agents/`
- `implement.md` 属于 Ralph Core，用于执行期统一约束
- `implement.md` 对 `code-reviewer`、`qa-validation-engineer` 是可选依赖：存在就可调用，不存在就跳过

## 4. Ralph Core 命令

### 4.1 `/ralph init-spec`

```text
/ralph init-spec <spec_name> [--target-dir <path>] [--force]
```

作用：

- 初始化 `docs/spec/<spec_name>/` 标准骨架
- 默认创建 `input/`、`artifacts/`、`plan/`、`implement/`、`logs/`
- 创建空的 `plan/plan.json`

### 4.2 `/ralph init-agent`

```text
/ralph init-agent [--target-dir <path>]
```

作用：

- 把最小 `.devagent` 复制到工作区
- 为后续 `/ralph run` 提供 `implement.md`

### 4.3 `/ralph dry-run`

```text
/ralph dry-run <spec_name>
```

作用：

- 读取 `plan/plan.json`
- 展示待执行任务顺序
- 不真正执行任务

### 4.4 `/ralph run`

```text
/ralph run <spec_name> [--silent]
```

作用：

- 后台启动 `ralph_loop.py`
- 按 `plan.json` 中的顺序执行任务
- 更新任务状态和日志

### 4.5 `/ralph status`

```text
/ralph status [run_id]
```

作用：

- 查看后台运行状态
- 不带 `run_id` 时显示最近的运行记录

### 4.6 `/ralph cancel`

```text
/ralph cancel <run_id>
```

作用：

- 终止指定后台运行任务

## 5. 规划插件

### 5.1 `ta-workflow`

命令：

```text
/ta-workflow:ta <spec_name>
/ta-workflow:decompose <spec_name>
```

阶段约定：

- `ta`
  - 输入：`docs/spec/<spec_name>/input/`
  - 输出：
    - `artifacts/ta-workflow/01_requirements.md`
    - `artifacts/ta-workflow/02_design.md`
    - `artifacts/ta-workflow/03_task_domains.md`
- `decompose`
  - 输入：上述三个中间产物
  - 输出：
    - `plan/*.md`
    - `plan/plan.json`

插件资产：

- `prompts/ta.md`
- `prompts/decompose.md`
- `dev_subagents/code-reviewer.md`
- `dev_subagents/qa-validation-engineer.md`

### 5.2 `frontend-workflow`

命令：

```text
/frontend-workflow:requirement <spec_name>
/frontend-workflow:design <spec_name>
/frontend-workflow:tasks <spec_name>
```

阶段约定：

- `requirement`
  - 输入：`docs/spec/<spec_name>/input/`
  - 输出：`artifacts/frontend-workflow/01_requirement.md`
- `design`
  - 输入：`artifacts/frontend-workflow/01_requirement.md`
  - 输出：`artifacts/frontend-workflow/02_design.md`
- `tasks`
  - 输入：
    - `artifacts/frontend-workflow/01_requirement.md`
    - `artifacts/frontend-workflow/02_design.md`
  - 输出：
    - `plan/*.md`
    - `plan/plan.json`

插件资产：

- `prompts/requirement.md`
- `prompts/design.md`
- `prompts/tasks.md`
- `dev_subagents/frontend-requirement-analyzer.md`
- `dev_subagents/frontend-design-analyzer.md`
- `dev_subagents/frontend-task-splitter.md`

## 6. 推荐使用流程

### 6.1 使用 `frontend-workflow`

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

### 6.2 使用 `ta-workflow`

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

### 6.3 手工维护计划

如果你已经自己准备好了 `plan/*.md` 和 `plan/plan.json`，可以直接跳过所有规划插件：

```text
/ralph init-spec my_spec
/ralph init-agent
# 手工编写 docs/spec/my_spec/plan/*.md
# 手工编写 docs/spec/my_spec/plan/plan.json
/ralph dry-run my_spec
/ralph run my_spec --silent
```

## 7. `plan.json` 最低格式要求

`plan.json` 必须是一个 JSON 数组。每个任务项至少包含：

```json
[
  {
    "task_name": "task-01",
    "priority": 1,
    "status": "TODO",
    "implment_times": 0,
    "task_file": "docs/spec/my_spec/plan/task-01.md"
  }
]
```

字段说明：

- `task_name`
  - 任务唯一标识
- `priority`
  - 数值越小越先执行
- `status`
  - 初始值必须是 `TODO`
- `implment_times`
  - 初始值必须是 `0`
- `task_file`
  - 指向实际存在的 `plan/*.md`

## 8. 插件与 `.devagent/agents`

插件不会在 `/ralph init-agent` 阶段预装自己的 subagent。

统一规则是：

- 每个插件阶段脚本在调用 `devagent --yolo` 前
- 先把该插件 `dev_subagents/*.md` 复制到工作区 `.devagent/agents/`
- 然后再启动 `devagent`

因此：

- Ralph Core 不需要知道各插件有哪些 subagent
- 插件自己对自己的 subagent 依赖负责
- 不需要单独的插件 init 命令

## 9. 常见问题

### 9.1 为什么 `frontend-workflow` 没有 `init`

因为初始化已经统一收敛到 Ralph Core：

- `/ralph init-spec`
- `/ralph init-agent`

`frontend-workflow` 只负责分阶段规划，不再负责初始化目录或复制 agent。

### 9.2 为什么中间产物放在 `artifacts/`

因为这些文档只服务于拆解过程，通常需要人工审核，但不应和最终 `plan/` 混在一起。

### 9.3 如果不使用任何插件可以吗

可以。只要你自己准备好了：

- `plan/plan.json`
- `plan/*.md`
- `.devagent/commands/ralph/implement.md`

就可以直接运行 Ralph。

### 9.4 `code-reviewer` 和 `qa-validation-engineer` 一定存在吗

不一定。`implement.md` 已经按可选依赖处理：

- 如果工作区存在这些 subagent，可以在执行阶段调用
- 如果不存在，则跳过并继续执行
