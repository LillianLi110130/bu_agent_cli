# Merge Mem Memory Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将裁剪后的 `tg_mem` 记忆能力接入 `agent_core.server`，实现基于 `session_id + user_id` 的历史恢复与轮次增量持久化。

**Architecture:** 在当前仓库 vendoring 一个最小可运行的 `tg_mem` 包，仅保留 MySQL-only 和记忆提炼所需子集。Server 侧新增独立 memory service，负责首次会话恢复和每轮对话后的 `[user, assistant]` 增量写入，不侵入 `Agent` 核心职责。

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, SQLAlchemy, PyMySQL, pytest, pytest-asyncio

---

### Task 1: 补 server 记忆接入测试骨架

**Files:**
- Create: `tests/test_server_memory_integration.py`
- Modify: `agent_core/server/models.py`
- Modify: `agent_core/server/session.py`
- Modify: `agent_core/server/app.py`

**Step 1: Write the failing test**

```python
def test_query_requires_user_id_when_session_id_missing():
    ...


def test_session_reuses_bound_user_id_and_rejects_mismatch():
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server_memory_integration.py -q`
Expected: FAIL，原因是 `QueryRequest`/`SessionManager` 尚未支持 `user_id` 绑定逻辑。

**Step 3: Write minimal implementation**

```python
class QueryRequest(BaseModel):
    user_id: str | None = None
```

并在 `SessionManager.get_or_create_session()` 中增加 `user_id` 绑定和校验。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_server_memory_integration.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_server_memory_integration.py agent_core/server/models.py agent_core/server/session.py agent_core/server/app.py
git commit -m "test: cover server session user binding"
```

### Task 2: 引入最小 tg_mem 包并补打包依赖

**Files:**
- Create: `tg_mem/__init__.py`
- Create: `tg_mem/**`
- Modify: `pyproject.toml`

**Step 1: Write the failing test**

```python
def test_tg_mem_memory_can_be_instantiated_in_mysql_only_mode():
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server_memory_integration.py -q`
Expected: FAIL，原因是 `tg_mem` 模块不存在。

**Step 3: Write minimal implementation**

- 从 `/opt/proj/mem0/mem0` 裁剪复制最小子集到 `tg_mem/`
- 批量替换内部 import：`mem0.` -> `tg_mem.`
- 修正 `tg_mem/__init__.py`
- 在 `pyproject.toml` 中加入 `tg_mem`、`sqlalchemy`、`pymysql[rsa]`、`pytz`

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_server_memory_integration.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tg_mem pyproject.toml tests/test_server_memory_integration.py
git commit -m "feat: vendor minimal tg_mem package"
```

### Task 3: 新增 memory service 并实现首次历史恢复

**Files:**
- Create: `agent_core/server/memory_service.py`
- Modify: `agent_core/server/session.py`
- Modify: `tests/test_server_memory_integration.py`

**Step 1: Write the failing test**

```python
def test_session_loads_history_only_once_from_memory_service():
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server_memory_integration.py -q`
Expected: FAIL，原因是尚无 memory service 和 `history_loaded` 机制。

**Step 3: Write minimal implementation**

- 新增 `MemoryService.load_history()` / `append_round()`
- `AgentSession` 增加 `user_id`、`history_loaded`
- 首次 query 前调用恢复逻辑，后续同进程内不重复恢复

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_server_memory_integration.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add agent_core/server/memory_service.py agent_core/server/session.py tests/test_server_memory_integration.py
git commit -m "feat: load session history from tg_mem"
```

### Task 4: 实现非流式/流式轮次增量写入

**Files:**
- Modify: `agent_core/server/app.py`
- Modify: `tests/test_server_memory_integration.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_query_persists_single_round_after_success():
    ...


@pytest.mark.asyncio
async def test_query_stream_persists_only_after_final_event():
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server_memory_integration.py -q`
Expected: FAIL，原因是 app 尚未调用 `append_round()`。

**Step 3: Write minimal implementation**

- `/agent/query` 在成功拿到最终响应后调用 `append_round()`
- `/agent/query-stream` 与 `/agent/query-stream-delta` 在 `FinalResponseEvent` 后调用 `append_round()`
- 异常流不写 assistant 半截内容

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_server_memory_integration.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add agent_core/server/app.py tests/test_server_memory_integration.py
git commit -m "feat: persist query rounds to tg_mem"
```

### Task 5: 补充回归验证与变更记录

**Files:**
- Modify: `CHANGELOG.md`

**Step 1: Write the failing test**

无需新增测试；执行回归测试命令。

**Step 2: Run test to verify current status**

Run: `uv run pytest tests/test_server_memory_integration.py tests/test_zhaohu_channel.py tests/test_agent_hooks.py -q`
Expected: 全部 PASS

**Step 3: Write minimal implementation**

- 在 `CHANGELOG.md` 的 `## [Unreleased]` 下补 `Added`/`Changed`

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_server_memory_integration.py tests/test_zhaohu_channel.py tests/test_agent_hooks.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: record tg_mem server integration"
```
