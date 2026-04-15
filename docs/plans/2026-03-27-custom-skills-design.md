# Custom Skill Discovery Design

**Date:** 2026-03-27
**Status:** Approved

## Goal

为 agent 增加用户可自定义 skill 的加载能力，并保证 CLI 与 gateway / IM runtime 在 skill 可见性和覆盖规则上保持一致。

本期新增两类自定义 skill 来源：

1. 用户级：`~/.tgagent/skills/`
2. 项目级：`<workspace_root>/skills/`

当存在同名 skill 时，覆盖优先级为：**项目级 > 用户级 > 系统内置**。

## Scope

本期覆盖以下能力：

- `@<skill-name>` 调用 skill
- `/skills` 展示可用 skill
- CLI 启动时注入到 system prompt 的 skill 列表
- gateway / IM runtime 构建 system prompt 时的 skill 列表

本期不调整插件 skill 的命名空间与加载策略。插件 skill 仍按现有 `PluginManager` 流程注册。

## Current State

当前仓库中存在两条 skill 加载路径：

1. `cli/at_commands.py`
   - `AtCommandRegistry` 默认只扫描源码内置目录 `agent_core/skills`
   - 直接影响 `@skill`、`/skills`、补全能力
2. `agent_core/bootstrap/agent_factory.py`
   - `build_system_prompt()` 通过 `load_skills(_SKILLS_DIR)` 只读取系统内置 skills
   - 直接影响 gateway / IM runtime 的 system prompt

这导致即使 CLI 支持自定义 skill，bootstrap 侧仍可能不可见，最终行为不一致。

## Design Principles

1. **单一来源规则**：skill 来源目录与覆盖优先级只定义一遍，避免 CLI 与 bootstrap 各自实现。
2. **最小完整改动**：不做 skill 模型大重构，只补齐多来源发现与合并。
3. **向后兼容**：保留 `AtCommand.from_file()` 与单目录初始化能力，避免影响插件与现有测试。
4. **容错优先**：坏 skill 文件不阻塞启动，按现有风格跳过。

## Proposed Architecture

### 1. Shared discovery layer

新增共享 discovery 模块，负责：

- 解析 workspace 对应的 skill 来源目录
- 扫描多个 skill 根目录下的 `SKILL.md` / `skill.md`
- 按 skill frontmatter 的 `name` 去重合并
- 输出最终生效 skill 集合

目录顺序固定为：

1. 系统内置：`agent_core/skills`
2. 用户级：`~/.tgagent/skills`
3. 项目级：`<workspace_root>/skills`

按此顺序加载，并允许后者覆盖前者，因此最终优先级为：

**项目级 > 用户级 > 系统内置**

### 2. CLI registry integration

`AtCommandRegistry` 扩展为支持多 skill 根目录发现：

- 兼容当前单目录模式
- 支持从共享 discovery 层接收合并后的结果
- 最终 registry 中只保留每个 skill name 的最高优先级版本

这样 `@skill`、`/skills`、Tab 补全都基于同一套最终结果。

### 3. System prompt alignment

CLI 的 system prompt 目前已从 `skill_registry.get_all()` 取数，因此只要 registry 正确，CLI prompt 会自动对齐。

bootstrap 侧则改为复用共享 discovery 结果，而不是只读 `_SKILLS_DIR`。这样 gateway / IM runtime 构建出的 prompt 与 CLI 保持一致。

## File-Level Changes

### New file

- `agent_core/skill/discovery.py`
  - 提供 skill 来源目录解析
  - 提供多目录扫描与覆盖合并
  - 提供可复用的 skill 元信息结构

### Modified files

- `cli/at_commands.py`
  - `AtCommandRegistry` 从单目录发现扩展为多目录发现
  - 兼容旧调用方式
- `tg_crab_main.py`
  - `create_runtime_registries()` 改为基于 `workspace_root` 构造合并后的 skill registry
  - CLI system prompt 自动复用新的 registry 数据
- `agent_core/bootstrap/agent_factory.py`
  - `build_system_prompt()` 改为使用共享 discovery 逻辑
- `README.md`
  - 说明自定义 skill 目录和覆盖优先级

## Conflict Resolution Rules

### Skill identity

skill 冲突以 `SKILL.md` frontmatter 中的 `name` 字段为准，而不是目录名。

### Priority

同名 skill 的最终保留规则：

1. 项目级 skill 覆盖用户级和系统内置
2. 用户级 skill 覆盖系统内置
3. 系统内置作为默认兜底

### Invalid skill files

以下情况直接跳过该 skill：

- 缺少合法 frontmatter
- 文件不存在或不可读
- YAML 解析失败
- 元信息缺失到无法形成有效 skill

跳过单个坏 skill 不影响其他 skill 注册。

## Data Flow

### CLI

1. 启动时确定 `workspace_root`
2. 解析 skill 根目录：系统内置 / 用户级 / 项目级
3. 扫描并合并最终 skill 集合
4. 构造 `AtCommandRegistry`
5. `@skill`、`/skills`、补全和 CLI system prompt 全部使用该 registry

### Gateway / IM runtime

1. 启动时确定 `workspace_root`
2. 解析 skill 根目录：系统内置 / 用户级 / 项目级
3. 扫描并合并最终 skill 集合
4. 将结果格式化后注入 bootstrap system prompt

## Error Handling

- skill 根目录不存在：忽略，不报错
- 单个 skill 文件损坏：跳过，不报错中断
- 多来源存在同名：按优先级覆盖，不做警告
- 用户目录 `~/.tgagent/skills` 无权限或不可读：按不存在处理

## Testing Strategy

### `tests/test_at_commands.py`

新增回归测试：

1. 多来源技能可同时发现
2. 同名 skill 时项目级覆盖用户级与系统内置
3. 无项目级时用户级覆盖系统内置
4. registry 中保留 skill 的路径指向最终生效版本

### `tests/test_bootstrap.py`

新增回归测试：

1. `build_system_prompt()` 能包含用户级 / 项目级自定义 skill
2. 同名 skill 时 prompt 中显示高优先级版本的描述或路径

### Verification scope

至少验证：

- `tests/test_at_commands.py`
- `tests/test_bootstrap.py`

必要时补充定向 CLI 相关回归测试。

## Compatibility Notes

- 现有单目录 `AtCommandRegistry(Path(...))` 调用保持可用
- 插件 skill 注册逻辑保持不变
- 仅新增自定义 skill 目录解析，不改变 skill 文件格式

## Assumptions

1. 项目级目录基于 agent 实际 `workspace_root`，不是进程启动目录。
2. 覆盖规则仅适用于系统内置 / 用户级 / 项目级三类普通 skill。
3. 插件 skill 由于已有命名空间机制，不纳入本期覆盖规则。

## Expected Outcome

实现完成后，用户可以将 skill 放到：

- `~/.tgagent/skills/<skill_name>/SKILL.md`
- `<workspace_root>/skills/<skill_name>/SKILL.md`

并在 CLI 中通过 `@<skill-name>` 使用、通过 `/skills` 查看；同时这些自定义 skill 也会出现在 CLI 与 gateway / IM runtime 的 system prompt 中，且冲突时按 **项目级 > 用户级 > 系统内置** 生效。
