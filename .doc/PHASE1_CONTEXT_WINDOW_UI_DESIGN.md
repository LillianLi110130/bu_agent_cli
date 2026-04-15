# Phase 1：CLI 上下文窗口展示详细设计

## 1. 设计目标

本阶段只解决一个核心问题：

让用户在 CLI 中实时看到“当前会话上下文窗口占用情况”，并在触发压缩时得到清晰反馈。

目标体验包括：

- 始终看到当前模型
- 始终看到当前会话估算上下文 token
- 始终看到模型上下文窗口大小
- 始终看到当前上下文占比
- 触发压缩时显示“压缩中”
- 压缩完成后展示更新后的窗口占用结果

## 2. 非目标

本阶段明确不做以下能力：

- 用户累计 token 账本
- 用户额度上限控制
- 大模型额度耗尽后的自动切模
- 云端同步
- 计费精度级 token 统计

本阶段关注的是“上下文窗口可见性”，不是“用户额度治理”。

## 3. 现有能力盘点

当前仓库已经具备本阶段所需的大部分基础能力。

### 3.1 上下文预算评估

`agent_core/agent/budget.py` 中的 `ContextBudgetEngine` 已经提供：

- 模型上下文窗口读取
- warn / compact / hard 阈值计算
- 当前消息列表 token 估算
- 最近一次 assessment 缓存

### 3.2 上下文压缩

`agent_core/agent/compaction/service.py` 已经提供：

- 压缩阈值判断
- 压缩前后 token 结果
- 压缩完成后的 working set 生成

### 3.3 模型上下文窗口来源

上下文窗口大小目前可以从两处获得：

- `config/model_presets.json` 中的 `max_input_tokens`
- `agent_core/tokens` 里的模型定价信息兜底

### 3.4 CLI 会话运行时目录

`cli/session_runtime.py` 已经为每次 CLI rollout 提供了独立目录，可用于保存本阶段的调试快照。

### 3.5 CLI 交互入口

`cli/app.py` 当前基于 `prompt_toolkit.PromptSession` 驱动交互，已经具备新增 `bottom_toolbar` 的技术条件。

## 4. 核心设计结论

本阶段不新造一套 token 评估逻辑，而是直接复用现有 `ContextBudgetEngine` 和 `CompactionService`。

新增内容主要分为三块：

1. CLI 状态栏展示层
2. 运行态状态机
3. 会话级快照落盘

## 5. 数据口径定义

为了避免后续产品和研发理解不一致，本阶段统一使用以下口径。

### 5.1 当前会话已使用 token

定义：

- 取当前消息列表在当前模型上的 `estimated_tokens`
- 数据来源为 `ContextBudgetEngine.assess(...)`

说明：

- 这是“本地估算 prompt token”
- 不包含未来模型将要生成的 completion token
- 也不是用户累计消耗

### 5.2 模型上下文大小

定义：

- 取当前模型的 `context_limit`
- 由模型 preset 的 `max_input_tokens` 优先提供

### 5.3 当前会话占比

定义：

- `estimated_tokens / context_limit`

显示时建议格式化为百分比整数，如 `64%`

### 5.4 阈值状态

定义：

- 低于 `warn_threshold`：正常
- 介于 `warn_threshold` 和 `compact_threshold`：接近压缩
- 大于等于 `compact_threshold` 但尚未进入压缩过程：待压缩
- 实际正在执行 compaction：压缩中

## 6. CLI 展示设计

## 6.1 展示位置

建议在 CLI 输入区下方新增 `bottom_toolbar`，而不是只在日志区域打印。

原因：

- 用户输入时也能看到状态
- 不会被历史输出刷走
- 与 `/model show` 这种一次性打印形成互补

## 6.2 展示字段

第一阶段状态栏建议固定展示四个字段：

- `模型`
- `上下文`
- `占用`
- `状态`

推荐展示样式：

- `模型: GLM-4.7 | 上下文: 82k/128k | 占用: 64% | 状态: 正常`
- `模型: GLM-4.7 | 上下文: 101k/128k | 占用: 79% | 状态: 接近压缩`
- `模型: GLM-4.7 | 上下文: 103k/128k | 占用: 80% | 状态: 压缩中`

## 6.3 字段解释

### 模型

- 取当前 `self._agent.llm.model`
- 如果后续需要，也可以补充 preset 名称

### 上下文

- 展示格式：`estimated_tokens / context_limit`
- 数值建议使用紧凑格式，例如 `82k/128k`

### 占用

- 展示格式：`context_utilization`
- 可取整到百分比

### 状态

状态值建议限定为以下集合：

- `正常`
- `接近压缩`
- `待压缩`
- `压缩中`
- `未知`

其中：

- `压缩中` 优先级最高，由运行时状态机直接驱动
- `正常 / 接近压缩 / 待压缩` 由 assessment 结果推导

## 7. 运行时状态机设计

## 7.1 目标

仅依赖 assessment 结果无法准确表达“压缩是否正在执行”，因此需要一个显式运行态状态机。

## 7.2 建议状态

本阶段建议引入以下最小状态集合：

- `idle`
- `generating`
- `compacting`

说明：

- `idle`：当前未在执行模型调用
- `generating`：正在请求模型或处理当前轮输出
- `compacting`：正在执行上下文压缩

为了后续扩展，可以预留但暂不启用：

- `switching`
- `syncing`

## 7.3 状态切换时机

- 用户提交输入后，进入 `generating`
- 检查并进入 compaction 时，切到 `compacting`
- compaction 完成后，回到 `generating` 或 `idle`
- 当前轮结束后，回到 `idle`

## 7.4 状态栏与状态机的关系

- 若状态机为 `compacting`，状态栏必须显示 `压缩中`
- 若状态机不是 `compacting`，再根据 assessment 推导：
  - `< warn_threshold` => `正常`
  - `>= warn_threshold 且 < compact_threshold` => `接近压缩`
  - `>= compact_threshold` => `待压缩`

## 8. 刷新策略

状态栏必须在关键时点刷新，避免展示过期信息。

建议刷新时机如下：

1. CLI 启动后初始化一次
2. 用户输入提交前刷新一次
3. 当前轮模型响应完成后刷新一次
4. compaction 开始时刷新一次
5. compaction 完成后刷新一次
6. 手动切换模型后刷新一次
7. 会话重置后刷新一次

说明：

- 第一阶段无需持续高频轮询
- 采用“事件驱动刷新”即可满足需求

## 9. 数据流设计

本阶段建议的数据流如下：

1. CLI 初始化时，构造一个会话级 `ContextWindowViewState`
2. 当用户提交输入时：
   - 读取当前模型
   - 对当前消息做 budget assessment
   - 更新状态栏
3. 当前轮执行过程中：
   - 如果进入 compaction，则切换运行态为 `compacting`
   - 状态栏显示 `压缩中`
4. 当前轮结束后：
   - 再次 assessment
   - 更新最新 `estimated_tokens / context_limit / utilization`
   - 写入会话快照

## 10. 本地落盘设计

## 10.1 目标

第一阶段落盘只服务于：

- 调试
- 问题排查
- 会话后分析

不承担计费、配额、跨会话累计职责。

## 10.2 存储位置

建议落在当前 CLI rollout 目录下，即：

- `~/.tg_agent/sessions/<yyyy>/<mm>/<dd>/rollout-.../context_window_status.json`

该目录体系已由 `cli/session_runtime.py` 管理。

## 10.3 建议内容

建议快照包含以下字段：

- `session_id`
- `model`
- `context_limit`
- `warn_threshold`
- `compact_threshold`
- `hard_threshold`
- `estimated_tokens`
- `context_utilization`
- `status`
- `last_updated_at`
- `last_compaction_started_at`
- `last_compaction_finished_at`
- `last_compaction_original_tokens`
- `last_compaction_new_tokens`

## 10.4 设计原则

- 该文件是会话级调试快照，不是权威 token ledger
- 每次刷新直接覆盖即可，不需要追加历史
- 历史分析可依赖日志和 rollout 目录下其他上下文文件

## 11. 实现落点建议

虽然本阶段先不落代码，但后续实现建议优先改动以下位置：

- `cli/app.py`
  - 增加 toolbar 展示
  - 管理运行态状态机
  - 在关键时点触发刷新
- `cli/session_runtime.py`
  - 扩展会话级状态快照文件定义
- `agent_core/agent/budget.py`
  - 复用现有 assessment 输出
- `agent_core/agent/compaction/service.py`
  - 复用 compaction 前后 token 信息

## 12. 异常与降级策略

## 12.1 assessment 失败

如果预算评估失败：

- 状态栏仍保留当前模型
- 上下文与占比显示为 `unknown`
- 状态显示为 `未知`

## 12.2 模型上下文窗口缺失

如果模型没有显式窗口配置：

- 使用当前默认上下文窗口兜底
- 同时在日志中输出告警，方便后续补 preset

## 12.3 compaction 失败

如果 compaction 失败：

- 状态从 `压缩中` 回退到根据 assessment 推导的状态
- 不应让状态栏永久停留在 `压缩中`
- 快照中可记录最近一次 compaction 失败时间与错误摘要

## 13. 测试建议

第一阶段建议覆盖以下测试点：

- CLI 初始化时能生成默认状态栏状态
- 普通会话中上下文数字会随消息增加而变化
- 接近 warn 阈值时状态变为 `接近压缩`
- 达到 compact 阈值时能显示 `待压缩`
- 执行 compaction 时状态为 `压缩中`
- compaction 完成后 token 数下降
- 模型切换后 `context_limit` 随之变化
- 会话快照文件能正确写入 rollout 目录

## 14. 验收标准

本阶段完成的判断标准如下：

1. 用户在 CLI 中无需执行额外命令即可看到当前模型和上下文窗口状态。
2. 状态栏能正确展示当前会话 `estimated_tokens / context_limit / utilization`。
3. 在接近压缩阈值和实际压缩时，用户能明显感知状态变化。
4. 压缩完成后，状态栏能反映新的上下文占用结果。
5. 每次会话都有独立的调试快照文件可供排查。

## 15. 结论

Phase 1 的正确定位不是“实现完整 token 治理系统”，而是先把现有上下文预算与压缩能力变成稳定、可观察、可解释的 CLI 产品体验。

只要这一层做好，后续 Phase 2 的用户配额展示和 Phase 3 的云端同步都可以在同一展示框架上继续演进，而不需要推翻第一阶段的设计。
