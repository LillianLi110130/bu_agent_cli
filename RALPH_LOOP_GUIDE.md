# Ralph Loop Guide

## 概述

Ralph Loop 是一套面向需求拆解和批量实现的命令式工作流。它不走主 Agent 的工具调用链，而是通过 `/ralph` 这一组 Slash 命令显式触发。

当前推荐将 Ralph 视为两阶段拆解流程：

1. `ta` 阶段：调用 `devagent` 生成标准化前置文档
2. `decompose` 阶段：继续调用 `devagent`，基于前置文档生成 `plan/*.md` 和 `plan.json`

随后再通过 `dry-run`、`run`、`status`、`cancel` 驱动 `ralph_loop.py` 批量执行任务。

## 功能概览

- 初始化 spec 目录骨架
- 初始化 `.devagent` 命令与 agent 配置
- 调用 `TA.md` 生成前置需求/设计/任务文档
- 调用 `DECOMPOSE_TASK.md` 生成 `plan/*.md` 和 `plan.json`
- 预览 `plan.json` 中的待执行任务
- 后台循环执行 `ralph_loop.py`
- 查询后台执行状态
- 取消后台执行

## 运行前提

- 工作区根目录中存在 `ralph_init.py`、`ralph_loop.py`、`template/`、`.devagent/`
- 环境变量中已配置好模型相关 API Key
- `devagent` 命令已安装并可通过 `PATH` 访问
- 当前工作目录就是 Ralph 要操作的项目根目录

## 目录约定

Ralph 默认围绕 `docs/spec/<spec_name>/` 这一套目录约定工作：

```text
docs/spec/<spec_name>/
├── requirement/     # 需求输入与 TA 阶段产物
├── plan/
│   └── plan.json    # 任务队列
├── implement/       # 实现规划与实现结果文档
└── logs/            # 任务执行日志
```

其中 `requirement/` 目录在两阶段流程下会包含：

```text
docs/spec/<spec_name>/requirement/
├── <spec_name>-requirements.md
├── <spec_name>-design.md
└── <spec_name>-task.md
```

工作区根目录下同时需要有：

```text
.devagent/
└── commands/
    └── ralph/
        ├── TA.md
        ├── DECOMPOSE_TASK.md
        ├── decompose.toml
        └── implement.md
```

## 推荐流程

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

如果你暂时不使用 `ta` 阶段，也可以直接从原始需求文档执行 `decompose`。当前 `decompose` 会优先读取标准化前置文档，若不存在则回退为直接读取 `requirement/` 下的原始文档。

## 命令详解

### 1. 初始化 spec 目录

```text
/ralph init-spec <spec_name> [--target-dir <path>] [--force]
```

作用：

- 从 `template/` 复制出 spec 目录骨架
- 创建 `requirement/`、`plan/`、`implement/`、`logs/`
- 创建一个空的 `plan/plan.json`

### 2. 初始化 `.devagent`

```text
/ralph init-agent [--target-dir <path>]
```

作用：

- 将仓库中的 `.devagent/` 复制到目标工作区根目录
- 为后续 `ta`、`decompose` 和 `run` 提供 prompt 与 agent 定义

### 3. 前置分析与标准化文档生成

```text
/ralph ta <spec_name> [description...]
```

作用：

- 读取 `docs/spec/<spec_name>/requirement/` 下的原始需求文档
- 读取 `.devagent/commands/ralph/TA.md`
- 调用 `devagent --yolo`
- 生成以下 3 个标准化文档：
  - `<spec_name>-requirements.md`
  - `<spec_name>-design.md`
  - `<spec_name>-task.md`
- 输出执行日志到 `docs/spec/<spec_name>/logs/ta.log`

说明：

- `ta` 阶段只做前置分析和文档标准化，不生成 `plan.json`
- 这 3 个文档会被后续 `decompose` 优先消费

### 4. 二次拆解为可执行计划

```text
/ralph decompose <spec_name> [description...]
```

作用：

- 优先读取：
  - `docs/spec/<spec_name>/requirement/<spec_name>-requirements.md`
  - `docs/spec/<spec_name>/requirement/<spec_name>-design.md`
  - `docs/spec/<spec_name>/requirement/<spec_name>-task.md`
- 若上述文件不存在，则回退为直接读取 `requirement/` 下的原始需求文档
- 读取 `.devagent/commands/ralph/DECOMPOSE_TASK.md`
- 调用 `devagent --yolo`
- 生成 `plan/*.md` 和 `plan/plan.json`
- 输出拆解日志到 `docs/spec/<spec_name>/logs/decompose.log`

### 5. 预览任务执行计划

```text
/ralph dry-run [<spec_name>] [--plan-file <path>] [--log-dir <path>] [--max-retry <n>] [--delay <seconds>] [--enable-git] [--main-branch <name>] [--work-branch <name>] [--silent]
```

作用：

- 前台执行 `ralph_loop.py --dry-run`
- 读取 `plan.json`
- 展示待执行任务、优先级、状态和执行次数
- 不实际调用 `devagent`

### 6. 后台运行 Ralph Loop

```text
/ralph run [<spec_name>] [--plan-file <path>] [--log-dir <path>] [--max-retry <n>] [--delay <seconds>] [--enable-git] [--main-branch <name>] [--work-branch <name>] [--silent]
```

作用：

- 后台启动 `ralph_loop.py`
- 由 `RalphProcessManager` 分配 `run_id`
- 持续执行 `plan.json` 中的 `TODO/FAILED` 任务
- 调用 `devagent --yolo`
- 更新 `plan.json` 和 `logs/`

### 7. 查询后台状态

```text
/ralph status [run_id]
```

作用：

- 不传 `run_id` 时，列出当前工作区中所有 Ralph 运行记录
- 传入 `run_id` 时，展示某次运行的详细状态
- 读取 `plan.json` 并统计 `DONE`、`FAILED`、`TODO`

### 8. 取消后台运行

```text
/ralph cancel <run_id>
```

作用：

- 终止指定的 Ralph 后台进程
- 更新运行状态为 `cancelled`

## 运行状态与日志

Ralph 会产生几类运行数据：

- 工作流状态文件：`.bu_agent/ralph/runs.json`
- 进程级日志：`.bu_agent/ralph/process_logs/`
- spec 级日志：`docs/spec/<spec_name>/logs/`
  - `ta.log`
  - `decompose.log`
  - `implement.log`
  - 各任务单独的实现日志

`/ralph status` 读取的就是这部分信息。

## 常见问题

### 1. `ta` 或 `decompose` 提示 requirement 目录为空

原因：

- `ralph_init` 只创建目录，不会自动生成需求文档

处理方式：

- 手动将原始需求说明写入 `docs/spec/<spec_name>/requirement/`
- 推荐使用 Markdown 文档

### 2. `run` 或 `dry-run` 提示找不到 `devagent`

原因：

- `devagent` 不在系统 `PATH`

处理方式：

- 确认 `devagent` 已安装
- 确认当前终端可以直接执行 `devagent --yolo`

### 3. `run` 提示找不到 `implement.md`

原因：

- 还没有执行 `/ralph init-agent`
- 或当前工作区根目录缺少 `.devagent/commands/ralph/implement.md`

### 4. `status` 看不到运行记录

原因：

- 还没有执行过 `/ralph run`
- 或当前工作区下的 `.bu_agent/ralph/runs.json` 为空

### 5. `decompose` 没有使用 TA 产物

排查点：

- `docs/spec/<spec_name>/requirement/` 下是否存在：
  - `<spec_name>-requirements.md`
  - `<spec_name>-design.md`
  - `<spec_name>-task.md`
- 是否先执行过 `/ralph ta <spec_name>`

如果这 3 个文件不存在，`decompose` 会回退为直接读取 `requirement/` 下的原始文档。
