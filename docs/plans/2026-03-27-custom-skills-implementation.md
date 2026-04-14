# Custom Skills Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 agent 增加用户级和项目级自定义 skill 加载能力，并按“项目级 > 用户级 > 系统内置”规则统一覆盖 CLI 与 gateway / IM runtime。

**Architecture:** 新增共享 skill discovery 层，统一解析系统内置、用户级和项目级 skill 根目录，并按 frontmatter 中的 `name` 合并覆盖。CLI 的 `AtCommandRegistry` 与 bootstrap 的 `build_system_prompt()` 都复用该 discovery 结果，确保 `@skill`、`/skills`、补全和 prompt 中的 skill 列表一致。

**Tech Stack:** Python 3.10, pathlib, dataclasses, pytest, uv

---

### Task 1: 先补 CLI 侧多来源 discovery 的失败测试

**Files:**
- Modify: `tests/test_at_commands.py`
- Modify: `cli/at_commands.py`

**Step 1: Write the failing test**

在 `tests/test_at_commands.py` 中新增临时技能目录构造辅助函数，并加入以下测试：

```python
def test_registry_discovers_builtin_user_and_project_skills():
    registry = AtCommandRegistry(
        skill_dirs=[builtin_skills, user_skills, project_skills]
    )

    assert set(registry.commands) == {"builtin-only", "user-only", "project-only"}


def test_registry_prefers_project_over_user_over_builtin_for_same_skill_name():
    registry = AtCommandRegistry(
        skill_dirs=[builtin_skills, user_skills, project_skills]
    )

    command = registry.get("shared")
    assert command is not None
    assert command.description == "project override"
    assert command.path == project_skills / "shared" / "SKILL.md"


def test_registry_prefers_user_over_builtin_when_project_missing():
    registry = AtCommandRegistry(skill_dirs=[builtin_skills, user_skills])

    command = registry.get("shared")
    assert command is not None
    assert command.description == "user override"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_at_commands.py -q`
Expected: FAIL，原因是 `AtCommandRegistry` 尚不支持 `skill_dirs` 多目录发现和覆盖优先级。

**Step 3: Write minimal implementation**

先只在 `cli/at_commands.py` 中给 `AtCommandRegistry` 增加多目录参数和顺序注册逻辑，最小实现形态可接近：

```python
class AtCommandRegistry:
    def __init__(self, skills_dir: Path | None = None, skill_dirs: list[Path] | None = None):
        ...

    def discover_skills(
        self,
        skills_dir: Path | None = None,
        skill_dirs: list[Path] | None = None,
    ) -> None:
        self.commands = {}
        for root in resolved_dirs:
            for skill_path in discovered_paths:
                self.register(AtCommand.from_file(skill_path))
```

这里按 `builtin -> user -> project` 顺序注册，后注册覆盖先注册。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_at_commands.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_at_commands.py cli/at_commands.py
git commit -m "test: cover custom skill registry precedence"
```

### Task 2: 抽出共享 skill discovery 模块并让 CLI 改用它

**Files:**
- Create: `agent_core/skill/discovery.py`
- Modify: `agent_core/skill/__init__.py`
- Modify: `cli/at_commands.py`
- Test: `tests/test_at_commands.py`

**Step 1: Write the failing test**

在 `tests/test_at_commands.py` 中补一个直接验证 discovery 输出顺序和元数据的测试：

```python
def test_discover_skill_files_returns_highest_priority_version():
    discovered = discover_skill_files([builtin_skills, user_skills, project_skills])

    assert [skill.name for skill in discovered] == ["shared"]
    assert discovered[0].description == "project override"
    assert discovered[0].path == project_skills / "shared" / "SKILL.md"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_at_commands.py -q`
Expected: FAIL，原因是 `agent_core.skill.discovery` 尚不存在。

**Step 3: Write minimal implementation**

新增 `agent_core/skill/discovery.py`，包含：

```python
@dataclass(frozen=True)
class DiscoveredSkill:
    name: str
    description: str
    path: Path
    category: str = "General"


def default_skill_dirs(workspace_root: Path, builtin_skills_dir: Path) -> list[Path]:
    return [
        builtin_skills_dir,
        Path("~/.tgagent/skills").expanduser(),
        workspace_root / "skills",
    ]


def discover_skill_files(skill_dirs: list[Path]) -> list[DiscoveredSkill]:
    merged: dict[str, DiscoveredSkill] = {}
    for root in skill_dirs:
        ...  # 扫描 */skill.md 和 */SKILL.md
        merged[skill.name] = skill
    return sorted(merged.values(), key=lambda item: item.name)
```

然后让 `AtCommandRegistry` 改为调用 `discover_skill_files()` 来构建 `self.commands`。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_at_commands.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add agent_core/skill/discovery.py agent_core/skill/__init__.py cli/at_commands.py tests/test_at_commands.py
git commit -m "feat: share custom skill discovery logic"
```

### Task 3: 接入 CLI runtime，让 `@skill`、`/skills` 和 prompt 看到自定义 skill

**Files:**
- Modify: `tg_crab_main.py`
- Modify: `tests/test_at_commands.py`
- Modify: `README.md`

**Step 1: Write the failing test**

在 `tests/test_at_commands.py` 中新增一个围绕 `create_runtime_registries()` 的测试，验证 workspace skills 会被注册：

```python
def test_create_runtime_registries_loads_workspace_and_user_skills(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builtin = tmp_path / "builtin_skills"
    user = tmp_path / "user_skills"
    ...

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    registry_bundle = create_runtime_registries(
        workspace_root=workspace,
        skills_dir=builtin,
    )

    assert registry_bundle.skill_registry.get("workspace-skill") is not None
    assert registry_bundle.skill_registry.get("user-skill") is not None
```

如果更适合拆到新测试文件，也可以新建 `tests/test_runtime_skills.py`，但优先保持最小文件变更。

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_at_commands.py -q`
Expected: FAIL，原因是 `create_runtime_registries()` 仍只使用单个 `skills_dir`。

**Step 3: Write minimal implementation**

在 `tg_crab_main.py` 中：

- 新增基于 `workspace_root` 解析 skill 目录列表的调用
- 将 `AtCommandRegistry(skills_dir or _SKILLS_DIR)` 改为多目录构造

目标代码形态接近：

```python
skill_dirs = default_skill_dirs(
    workspace_root=workspace_root,
    builtin_skills_dir=skills_dir or _SKILLS_DIR,
)
skill_registry = AtCommandRegistry(skill_dirs=skill_dirs)
```

同时在 `README.md` 的 skill 说明中加入：

```text
自定义 skill 支持目录：
- ~/.tgagent/skills/
- <workspace_root>/skills/
覆盖优先级：项目级 > 用户级 > 系统内置
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_at_commands.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tg_crab_main.py tests/test_at_commands.py README.md
git commit -m "feat: load custom skills in cli runtime"
```

### Task 4: 接入 bootstrap / gateway prompt，统一自定义 skill 可见性

**Files:**
- Modify: `agent_core/bootstrap/agent_factory.py`
- Modify: `tests/test_bootstrap.py`

**Step 1: Write the failing test**

在 `tests/test_bootstrap.py` 中新增：

```python
def test_build_system_prompt_includes_user_and_workspace_skills(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builtin = tmp_path / "builtin_skills"
    user = tmp_path / "home" / ".tgagent" / "skills"
    project = workspace / "skills"
    ...

    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    prompt = module.build_system_prompt(
        workspace,
        builtin_skills_dir=builtin,
    )

    assert "user-skill" in prompt
    assert "project-skill" in prompt


def test_build_system_prompt_prefers_project_skill_metadata(tmp_path: Path, monkeypatch):
    ...
    assert "project override" in prompt
    assert "builtin override" not in prompt
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bootstrap.py -q`
Expected: FAIL，原因是 `build_system_prompt()` 目前只加载 `_SKILLS_DIR`，且没有注入目录参数。

**Step 3: Write minimal implementation**

在 `agent_core/bootstrap/agent_factory.py` 中：

- 给 `build_system_prompt()` 增加可选参数 `builtin_skills_dir: Path | None = None`
- 调用共享 discovery 层解析默认三类 skill 目录
- 用 discovery 输出构造 prompt 所需的 skills 文本

目标代码形态接近：

```python
def build_system_prompt(
    working_dir: Path,
    builtin_skills_dir: Path | None = None,
) -> str:
    discovered_skills = discover_skill_files(
        default_skill_dirs(
            workspace_root=working_dir,
            builtin_skills_dir=builtin_skills_dir or _SKILLS_DIR,
        )
    )
```

保持 `create_agent()` 的调用方式不变，默认仍使用工作区目录。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bootstrap.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add agent_core/bootstrap/agent_factory.py tests/test_bootstrap.py
git commit -m "feat: expose custom skills in bootstrap prompt"
```

### Task 5: 做最终回归验证并清理文档说明

**Files:**
- Modify: `README.md`（若上一任务未完成则在此补齐）

**Step 1: Write the failing test**

无需新增测试；执行回归验证命令。

**Step 2: Run verification**

Run: `uv run pytest tests/test_at_commands.py tests/test_bootstrap.py -q`
Expected: PASS

**Step 3: Run static validation**

Run: `uv run ruff check cli/at_commands.py tg_crab_main.py agent_core/skill/discovery.py agent_core/bootstrap/agent_factory.py tests/test_at_commands.py tests/test_bootstrap.py`
Expected: All checks passed

**Step 4: Confirm README text**

确保 README 中明确包含：

- `~/.tgagent/skills/`
- `<workspace_root>/skills/`
- 覆盖优先级：`项目级 > 用户级 > 系统内置`

**Step 5: Commit**

```bash
git add README.md

git commit -m "docs: describe custom skill locations"
```
