# Ralph Loop 使用指南

## 1. Ralph Loop 是什么

Ralph Loop 是一套面向“任务批量执行”的显式工作流。

它和主 Agent 的对话式工具调用链不是一回事。Ralph 的入口是 `/ralph` 这一组 Slash 命令，执行核心是仓库根目录下的 `ralph_loop.py`。

你可以把 Ralph 理解为两层能力：

1. 前置规划层
   - `init-spec`
   - `init-agent`
   - `ta`
   - `decompose`
2. 执行层
   - `dry-run`
   - `run`
   - `status`
   - `cancel`

其中，真正负责“按任务队列执行任务”的是执行层。

## 2. 两种使用方式

Ralph Loop 支持两种常见用法：

### 方式一：使用内置拆解能力

适合你只有原始需求，还没有 `plan.json` 和任务文档的情况。

典型流程：

```text
/ralph init-spec my_spec
/ralph init-agent
# 手动将原始需求文档放入 docs/spec/my_spec/requirement/
/ralph ta my_spec
/ralph decompose my_spec
/ralph dry-run my_spec
/ralph run my_spec --silent
/ralph status
```

### 方式二：不使用内置拆解能力

适合你已经自己拆好了任务，准备手工维护 `plan.json` 和 `plan/*.md` 的情况。

典型流程：

```text
/ralph init-spec my_spec
/ralph init-agent
# 手动编写 docs/spec/my_spec/plan/*.md
# 手动编写 docs/spec/my_spec/plan/plan.json
/ralph dry-run my_spec
/ralph run my_spec --silent
/ralph status
```

如果你采用方式二，可以完全跳过 `/ralph ta` 和 `/ralph decompose`。

## 3. 运行前提

- 当前工作目录就是 Ralph 要操作的项目根目录

## 4. 目录约定

Ralph 默认围绕 `docs/spec/<spec_name>/` 这套结构工作：

```text
docs/spec/<spec_name>/
├── requirement/
├── plan/
│   └── plan.json
├── implement/
└── logs/
```

### 4.1 `requirement/`

用途：

- 存放原始需求文档
- 存放 `ta` 阶段输出的标准化前置文档

如果使用方式一，通常会包含：

```text
docs/spec/<spec_name>/requirement/
├── <spec_name>-requirements.md
├── <spec_name>-design.md
└── <spec_name>-task.md
```

如果使用方式二，这个目录不是必须参与执行流程，但保留它通常更清晰。

### 4.2 `plan/`

用途：

- 存放可直接执行的任务规划文档 `plan/*.md`
- 存放 Ralph 真正消费的任务清单 `plan.json`

### 4.3 `implement/`

用途：

- 存放执行阶段生成的实现记录文档
- Ralph 在执行任务时，会通过 `implement.md` prompt 指导 `devagent` 在这里生成或更新 `<task_name>_implement.md`

### 4.4 `logs/`

用途：

- 存放 `ta`、`decompose`、`run` 阶段产生的日志
- 存放每个任务自己的执行日志

### 4.5 `.devagent/`

工作区根目录下还需要有：

```text
.devagent/
├── agents/
│   ├── code-reviewer.md
│   └── qa-validation-engineer.md
└── commands/
    └── ralph/
        ├── TA.md
        ├── DECOMPOSE_TASK.md
        ├── decompose.toml
        └── implement.md
```

其中最关键的是：

- `TA.md`
- `DECOMPOSE_TASK.md`
- `implement.md`

如果缺少 `implement.md`，`run` 和 `dry-run` 都会失败。

## 5. 每个命令实际会做什么

下面按命令说明 Ralph 的真实行为。

## 5.1 `/ralph init-spec`

命令：

```text
/ralph init-spec <spec_name> [--target-dir <path>] [--force]
```

作用：

- 初始化 spec 目录骨架
- 为后续 `ta`、`decompose`、`run` 准备标准目录

实际发生的事情：

- 调用仓库根目录下的 `ralph_init.py`
- 从 `template/` 复制模板文件到目标目录
- 默认情况下，目标路径是：
  - `./docs/spec/<spec_name>`
- 如果传了 `--target-dir <path>` 且不是 `.`，目标路径会变成：
  - `<path>/<spec_name>`
- 创建以下目录：
  - `requirement/`
  - `plan/`
  - `implement/`
  - `logs/`
- 创建空的 `plan/plan.json`

注意：

- 当前仓库的 `template/` 里主要只有 `plan/plan.json` 模板，其他目录由初始化脚本创建
- 如果目标目录非空且未加 `--force`，初始化会失败

## 5.2 `/ralph init-agent`

命令：

```text
/ralph init-agent [--target-dir <path>]
```

作用：

- 将仓库中的 `.devagent/` 复制到工作区
- 为 `ta`、`decompose`、`run` 提供 prompt 和 subagent 定义

实际发生的事情：

- 调用 `ralph_init.py`
- 把仓库根目录的 `.devagent/` 整体复制到目标工作区
- 主要复制内容包括：
  - `.devagent/commands/ralph/TA.md`
  - `.devagent/commands/ralph/DECOMPOSE_TASK.md`
  - `.devagent/commands/ralph/implement.md`
  - `.devagent/agents/code-reviewer.md`
  - `.devagent/agents/qa-validation-engineer.md`

建议：

- 即使你不使用 `ta` 和 `decompose`，也建议执行一次 `/ralph init-agent`
- 因为 `run` 阶段会直接依赖 `implement.md`

## 5.3 `/ralph ta`

命令：

```text
/ralph ta <spec_name> [description...]
```

作用：

- 把原始需求整理成 3 个标准化前置文档
- 为后续 `decompose` 提供更稳定的输入

实际发生的事情：

- 读取：
  - `docs/spec/<spec_name>/requirement/` 下现有文档
  - `.devagent/commands/ralph/TA.md`
- 在 prompt 末尾附加运行时上下文：
  - `workspace`
  - `spec_name`
  - `spec_dir`
  - `requirement_dir`
  - `requirements_file`
  - `design_file`
  - `task_file`
- 如果命令后面还有描述文本，会追加到 `Additional Request`
- 调用：
  - `devagent --yolo`
- 期望 `devagent` 生成或覆盖以下文件：
  - `docs/spec/<spec_name>/requirement/<spec_name>-requirements.md`
  - `docs/spec/<spec_name>/requirement/<spec_name>-design.md`
  - `docs/spec/<spec_name>/requirement/<spec_name>-task.md`
- 将本次执行结果写入：
  - `docs/spec/<spec_name>/logs/ta.log`

它不会做什么：

- 不会生成 `plan.json`
- 不会生成 `plan/*.md`
- 不会执行任何实现任务

## 5.4 `/ralph decompose`

命令：

```text
/ralph decompose <spec_name> [description...]
```

作用：

- 把需求继续拆解成可直接执行的原子任务
- 生成 `plan/*.md` 和 `plan/plan.json`

实际发生的事情：

- 读取：
  - `.devagent/commands/ralph/DECOMPOSE_TASK.md`
  - `docs/spec/<spec_name>/requirement/` 下的需求材料
- 在 prompt 中显式传入：
  - `requirements_file`
  - `design_file`
  - `task_file`
  - `plan_dir`
  - `plan_file`
- 优先消费以下 3 个标准化输入：
  - `<spec_name>-requirements.md`
  - `<spec_name>-design.md`
  - `<spec_name>-task.md`
- 如果上述文件不存在，则回退为直接读取 `requirement/` 下其他原始文档
- 调用：
  - `devagent --yolo`
- 期望 `devagent` 生成：
  - `docs/spec/<spec_name>/plan/*.md`
  - `docs/spec/<spec_name>/plan/plan.json`
- 将本次执行结果写入：
  - `docs/spec/<spec_name>/logs/decompose.log`

它不会做什么：

- 不会执行代码实现
- 不会修改业务代码
- 不会修改 `implement/`

## 5.5 `/ralph dry-run`

命令：

```text
/ralph dry-run [<spec_name>] [--plan-file <path>] [--log-dir <path>] [--max-retry <n>] [--delay <seconds>] [--enable-git] [--main-branch <name>] [--work-branch <name>] [--silent]
```

作用：

- 预览当前任务计划
- 不真正执行 `plan.json` 中的任务

实际发生的事情：

- `RalphService` 会先检查：
  - `ralph_loop.py` 是否存在
  - `plan.json` 是否存在
  - `.devagent/commands/ralph/implement.md` 是否存在
  - `devagent` 是否在 `PATH`
- 前台执行：
  - `python ralph_loop.py ... --dry-run`
- `ralph_loop.py` 会读取 `plan.json`
- 筛选所有 `status != DONE` 的任务
- 按 `priority` 从小到大排序
- 展示：
  - 总任务数
  - 待处理任务数
  - 每个待处理任务的名称、优先级、状态、执行次数、任务文件
  - 已完成任务列表

重要说明：

- `dry-run` 不会实际调用 `devagent` 执行任务
- 但当前实现中，`dry-run` 依然要求 `devagent` 和 `implement.md` 已存在

## 5.6 `/ralph run`

命令：

```text
/ralph run [<spec_name>] [--plan-file <path>] [--log-dir <path>] [--max-retry <n>] [--delay <seconds>] [--enable-git] [--main-branch <name>] [--work-branch <name>] [--silent]
```

作用：

- 后台启动 Ralph Loop
- 按 `plan.json` 中的任务顺序持续执行任务

实际发生的事情：

- `RalphService` 会先做和 `dry-run` 一样的前置检查
- 组装命令：
  - `python ralph_loop.py ...`
- 把命令交给 `RalphProcessManager`
- `RalphProcessManager` 会：
  - 创建 `run_id`
  - 启动后台子进程
  - 将 stdout/stderr 写入 `.bu_agent/ralph/process_logs/`
  - 将运行记录写入 `.bu_agent/ralph/runs.json`
- 后台 `ralph_loop.py` 进程会循环做这些事：
  - 读取 `plan.json`
  - 找出所有 `status != DONE` 的任务
  - 按 `priority` 升序排序
  - 每轮只取第一个任务执行
  - `TODO` 任务走首次执行逻辑
  - `FAILED` 任务走重试逻辑
  - 把任务信息填进 `.devagent/commands/ralph/implement.md`
  - 调用 `devagent --yolo`
  - 把执行结果写入任务日志
  - 更新 `plan.json` 中该任务的状态字段
  - 等待 `delay` 秒后进入下一轮

任务执行后会更新的 `plan.json` 字段：

- `implment_times`
- `complete_time`
- `last_output`
- `log_file`
- `status`

### 5.6.1 `run` 支持的可选参数

- `--max-retry`
  - FAILED 任务最大重试次数
- `--delay`
  - 每轮处理之间的等待秒数
- `--enable-git`
  - 启用 Git 分支工作流
- `--main-branch`
  - 主分支名称，默认 `main`
- `--work-branch`
  - 工作分支名称，默认 `devagent-work`
- `--silent`
  - 静默模式，只显示简化进度

### 5.6.2 `run` 的 Git 行为

只有你显式传了 `--enable-git` 才会启用。

启用后，Ralph 会尝试：

- 确保工作分支存在
- 每个任务创建子工作分支
- 成功后合并回工作分支
- 失败时删除子工作分支

如果不传 `--enable-git`，这些 Git 分支动作都不会发生。

## 5.7 `/ralph status`

命令：

```text
/ralph status [run_id]
```

作用：

- 查看后台运行状态

实际发生的事情：

- 如果不传 `run_id`
  - 列出当前工作区全部 Ralph 运行记录
  - 每条记录附带 `done/failed/todo` 摘要
- 如果传入 `run_id`
  - 返回该次运行的详细记录
  - 包括 `status`、`pid`、`plan_file`、`log_dir`、`stdout_log`、`stderr_log`

状态信息来源：

- `.bu_agent/ralph/runs.json`
- 对应 `plan.json`

## 5.8 `/ralph cancel`

命令：

```text
/ralph cancel <run_id>
```

作用：

- 停止某次后台 Ralph 运行

实际发生的事情：

- 定位对应后台进程
- Windows 下使用 `taskkill /PID ... /T /F`
- 更新该次运行记录为 `cancelled`

注意：

- 这会停止 Ralph loop 进程
- 但不会自动回滚已经完成的代码改动

## 6. 方式一：使用内置拆解能力的详细步骤

这种方式适合：

- 你还没有 `plan.json`
- 你希望 Ralph 帮你从原始需求一路拆到可执行任务

### 步骤 1：初始化 spec

```text
/ralph init-spec my_spec
```

产物：

```text
docs/spec/my_spec/
├── requirement/
├── plan/
│   └── plan.json
├── implement/
└── logs/
```

### 步骤 2：初始化 `.devagent`

```text
/ralph init-agent
```

产物：

- `.devagent/commands/ralph/TA.md`
- `.devagent/commands/ralph/DECOMPOSE_TASK.md`
- `.devagent/commands/ralph/implement.md`

### 步骤 3：放入原始需求文档

把你的原始需求说明手工放进：

```text
docs/spec/my_spec/requirement/
```

建议使用 Markdown 文档。

### 步骤 4：执行 `ta`

```text
/ralph ta my_spec
```

结果：

- 生成标准化前置文档
- 写入 `logs/ta.log`

### 步骤 5：执行 `decompose`

```text
/ralph decompose my_spec
```

结果：

- 生成 `plan/*.md`
- 生成 `plan/plan.json`
- 写入 `logs/decompose.log`

### 步骤 6：预览计划

```text
/ralph dry-run my_spec
```

用途：

- 确认优先级、状态、任务数、执行次数是否符合预期

### 步骤 7：后台执行

```text
/ralph run my_spec --silent
```

结果：

- 后台启动 Ralph Loop
- 返回一个 `run_id`

### 步骤 8：查看状态和日志

```text
/ralph status
/ralph status <run_id>
```

日志位置：

- `.bu_agent/ralph/process_logs/`
- `docs/spec/my_spec/logs/`

## 7. 方式二：不使用内置拆解能力的详细步骤

这种方式适合：

- 你已经自己拆好了任务
- 你只想利用 Ralph 的任务队列执行能力
- 你不需要 `ta` 和 `decompose`

### 步骤 1：初始化 spec

```text
/ralph init-spec my_spec
```

### 步骤 2：初始化 `.devagent`

```text
/ralph init-agent
```

这一步仍然建议执行，因为 `run` 依赖：

- `.devagent/commands/ralph/implement.md`

### 步骤 3：手工编写任务文档

你需要自己在：

```text
docs/spec/my_spec/plan/
```

下创建每个任务对应的 Markdown 文档，例如：

```text
docs/spec/my_spec/plan/
├── 01-auth-session-refactor.md
├── 02-login-rate-limit.md
└── plan.json
```

### 步骤 4：手工编写 `plan.json`

Ralph 真正消费的是：

```text
docs/spec/my_spec/plan/plan.json
```

你需要自己维护其中的任务列表。

### 步骤 5：执行 `dry-run`

```text
/ralph dry-run my_spec
```

先确认 Ralph 能正确识别你的任务。

### 步骤 6：执行 `run`

```text
/ralph run my_spec --silent
```

此后 Ralph 就会按照你的 `plan.json` 去逐项执行任务。

## 8. 手工维护 `plan.json` 时的格式要求

这是方式二里最重要的部分。

## 8.1 顶层结构

`plan.json` 必须是一个合法的 JSON 数组。

示例：

```json
[
  {
    "task_name": "auth_session_refactor",
    "priority": 0,
    "status": "TODO",
    "implment_times": 0,
    "task_file": "D:/project/docs/spec/payment_refactor/plan/01-auth-session-refactor.md",
    "complete_time": "",
    "last_output": "",
    "log_file": "",
    "branch": "task-01-auth-session-refactor"
  }
]
```

## 8.2 字段要求

| 字段 | 是否必需 | 含义 | 建议 |
|------|----------|------|------|
| `task_name` | 是 | 任务唯一标识 | 使用英文 `snake_case` |
| `priority` | 是 | 执行优先级，越小越先执行 | 从 `0` 开始递增，不要重复 |
| `status` | 是 | 任务状态 | 初始值写 `TODO` |
| `implment_times` | 是 | 已执行次数 | 初始值写 `0` |
| `task_file` | 是 | 任务规划 Markdown 路径 | 建议写绝对路径，文件必须存在 |
| `complete_time` | 是 | 最近完成时间 | 初始值写空字符串 |
| `last_output` | 是 | 最近日志摘要 | 初始值写空字符串 |
| `log_file` | 是 | 最近执行日志路径 | 初始值写空字符串 |
| `branch` | 建议有 | 任务分支名 | 只有 `--enable-git` 时才真正使用 |

### 8.2.1 关于 `implment_times`

请注意字段名必须是：

```text
implment_times
```

不是：

```text
implement_times
```

这是当前实现中固定使用的字段名。手工编写时必须保持一致，否则 Ralph 不会正确读取和更新执行次数。

## 8.3 状态值要求

建议只使用以下状态：

- `TODO`
- `FAILED`
- `DONE`

Ralph 的行为是：

- `DONE`
  - 跳过
- `TODO`
  - 作为首次执行任务
- `FAILED`
  - 作为重试任务

不要依赖其他自定义状态。

## 8.4 优先级要求

建议遵循以下规则：

- `priority` 使用唯一整数
- 从 `0` 开始递增
- 如果任务 B 依赖任务 A，那么 A 的 `priority` 必须更小
- 不要让多个任务共用同一优先级

Ralph 会按 `priority` 升序处理未完成任务。

## 8.5 `task_file` 要求

建议：

- 使用绝对路径
- 路径必须真实存在
- 文件扩展名使用 `.md`

实际代码执行前会检查该文件是否存在，不存在会直接记为失败。

## 9. 手工维护任务 Markdown 时的格式建议

严格来说，当前代码不会直接解析 `plan/*.md` 的固定 Markdown 结构。

Ralph 真正做的是：

- 把 `task_file` 路径填入 `implement.md` prompt
- 让 `devagent` 自己去读取并理解任务文档

所以：

- 任务 Markdown 没有“代码级强校验格式”
- 但为了让 `devagent` 稳定理解，建议遵循固定结构

推荐格式如下：

```markdown
# 实施计划: auth_session_refactor

## 目标
[用 2-3 句话说明任务目标和完成结果]

## 需求覆盖
- [需求点 1]
- [需求点 2]

## 影响范围
- [文件或模块 1]
- [文件或模块 2]

## 前置依赖
- 无 / [依赖任务名称]

## 假设与未决问题
- [假设 1]
- [未决问题 1]

## 实施步骤
1. [步骤 1]
2. [步骤 2]
3. [步骤 3]

## 验证方式
- [测试或验证方法 1]
- [测试或验证方法 2]

## 风险与注意事项
- [风险 1]：[缓解方式]
- [风险 2]：[缓解方式]

## 完成标准
- [ ] [标准 1]
- [ ] [标准 2]
```

## 9.1 建议写进任务 Markdown 的内容

建议每个任务文档至少写清楚：

- 任务目标
- 涉及模块或文件
- 前置依赖
- 实施步骤
- 验证方式
- 完成标准
- 风险与边界情况

这些内容越清晰，`devagent` 在执行阶段越稳定。

## 9.2 不建议写得过于模糊

不建议只写这种内容：

```text
修一下登录逻辑
补点测试
```

这种描述过于粗糙，容易导致执行阶段偏离预期。

## 10. Ralph Loop 实际执行行为说明

理解这部分很重要，尤其是在你手工维护 `plan.json` 时。

## 10.1 执行顺序

每轮循环：

1. 读取最新的 `plan.json`
2. 找出所有 `status != DONE` 的任务
3. 按 `priority` 升序排序
4. 只执行当前第一个任务
5. 回写结果
6. 进入下一轮

这意味着 Ralph 不是并发跑任务，而是串行逐个处理。

## 10.2 重试逻辑

- `TODO` 任务首次执行
- `FAILED` 任务按重试逻辑再次执行
- 每次执行完成后，`implment_times` 会加 `1`
- 如果失败次数达到 `--max-retry`，Ralph 会停止继续处理该轮最高优先级失败任务

注意：

- 当前实现里，如果当前最高优先级的 `FAILED` 任务已经达到最大重试次数，循环会直接结束，而不是自动跳过它继续执行后面的任务
- 所以优先级设计要合理，避免一个前置失败任务长期阻塞后续任务

## 10.3 任务日志

Ralph 会写两类日志。

### 10.3.1 进程级日志

位置：

```text
.bu_agent/ralph/process_logs/
```

用途：

- 记录后台 Ralph loop 进程的 stdout/stderr

### 10.3.2 任务级日志

位置：

```text
docs/spec/<spec_name>/logs/
```

常见文件：

- `ta.log`
- `decompose.log`
- `implement.log`
- `<task_name>_implment.log`
- `<task_name>_implment_<n>.log`

说明：

- 首次执行 TODO 任务通常是 `<task_name>_implment.log`
- FAILED 重试通常是 `<task_name>_implment_<n>.log`

## 10.4 实现记录文档

在执行任务时，`implement.md` prompt 会要求 `devagent`：

- 读取任务 Markdown
- 在 `implement/` 目录下创建或更新 `<task_name>_implement.md`
- 记录实施计划、实际变更、验证结果、风险和总结

因此，Ralph 的设计并不是只跑代码，还会同时沉淀执行记录。

## 10.5 不要手工改哪些字段

一旦开始 `run`，不建议手工修改以下字段：

- `status`
- `implment_times`
- `complete_time`
- `last_output`
- `log_file`

这些字段应由 Ralph 在执行过程中维护。

你可以在启动前手工编写任务条目，但启动后最好不要和 Ralph 同时修改同一个 `plan.json`。

## 11. 非标准路径运行方式

如果你不想使用默认的 `docs/spec/<spec_name>/plan/plan.json`，也可以直接指定计划文件。

示例：

```text
/ralph run --plan-file D:\project\custom\plan.json --log-dir D:\project\custom\logs --silent
```

建议：

- 只要使用 `--plan-file`
- 就同时显式传入 `--log-dir`

这样路径行为最清晰，也最接近你预期。

## 12. 常见问题

### 12.1 `ta` 或 `decompose` 提示 requirement 目录为空

原因：

- `init-spec` 只会创建目录，不会自动生成需求文档

处理：

- 手工将原始需求 Markdown 放入 `docs/spec/<spec_name>/requirement/`

### 12.2 `run` 或 `dry-run` 提示找不到 `devagent`

原因：

- `devagent` 不在 `PATH`

处理：

- 确认终端能直接执行 `devagent --yolo`

### 12.3 `run` 或 `dry-run` 提示找不到 `implement.md`

原因：

- 还没有执行 `/ralph init-agent`
- 或工作区缺少 `.devagent/commands/ralph/implement.md`

处理：

- 执行 `/ralph init-agent`
- 或手工补齐 `.devagent/commands/ralph/implement.md`

### 12.4 我已经自己拆了任务，还要不要执行 `ta` 和 `decompose`

结论：

- 不需要

你只需要：

1. 执行 `/ralph init-spec`
2. 执行 `/ralph init-agent`
3. 手工写好 `plan/*.md`
4. 手工写好 `plan/plan.json`
5. 执行 `/ralph dry-run`
6. 执行 `/ralph run`

### 12.5 手工拆任务时，`requirement/` 是不是必须有内容

结论：

- 对 `run` 不是必须

只要 `plan.json`、任务 Markdown、`.devagent/commands/ralph/implement.md`、`devagent` 都满足要求，就可以运行。

### 12.6 手工任务文档有没有强制格式

结论：

- 没有代码级强制格式
- 但强烈建议使用本指南推荐结构

因为执行阶段真正依赖的是 `devagent` 对任务文档的理解质量。

## 13. 推荐做法总结

如果你想最稳妥地使用 Ralph Loop，建议如下：

### 13.1 只有原始需求时

使用完整流程：

```text
/ralph init-spec my_spec
/ralph init-agent
/ralph ta my_spec
/ralph decompose my_spec
/ralph dry-run my_spec
/ralph run my_spec --silent
```

### 13.2 已经自己拆好任务时

使用手工 plan 流程：

```text
/ralph init-spec my_spec
/ralph init-agent
# 手工维护 plan/*.md 和 plan.json
/ralph dry-run my_spec
/ralph run my_spec --silent
```
