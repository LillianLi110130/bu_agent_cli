# Server Skills System

线上 server 环境的技能系统，无需文件系统访问。

## 设计理念

**技能不是工具调用**，而是通过 API 参数直接传递。前端在调用聊天接口时传递 `skill` 参数，后端自动将技能内容注入到消息中。

```
用户请求 → API (skill参数) → 技能注入 → Agent处理 → 响应
```

## 目录结构

```
agent_core/server/skills/
├── __init__.py       # 模块导出
├── types.py          # Skill 数据类型
├── registry.py       # 技能注册表
├── loaders.py        # 技能加载器（Config/Database/API）
├── injector.py       # 技能注入器（处理消息增强）
├── builtin.py        # 内置技能定义
├── example.py        # 使用示例
└── README.md         # 本文档
```

## 快速开始

### 1. 最小配置

```python
from agent_core.server import create_server
from agent_core.server.skills import (
    SkillRegistry,
    ConfigSkillLoader,
    BUILTIN_SKILLS,
    set_global_registry,
)
from agent_core import Agent
from agent_core.llm import ChatOpenAI

# 初始化技能注册表
registry = SkillRegistry()
registry.register_loader(ConfigSkillLoader(BUILTIN_SKILLS))
set_global_registry(registry)

# 创建 Agent（不需要 use_skill 工具）
def agent_factory():
    return Agent(
        llm=ChatOpenAI(model="gpt-4o"),
        tools=[your_tools],  # 只需常规工具
    )

# 创建服务器
app = create_server(agent_factory=agent_factory)
```

### 2. API 调用方式

```bash
# 普通查询（不使用技能）
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'

# 使用 calculator 技能
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"message": "123 + 456", "skill": "calculator"}'

# 使用 brainstorming 技能（流式）
curl -X POST http://localhost:8000/agent/query-stream \
  -H "Content-Type: application/json" \
  -d '{"message": "添加一个登录功能", "skill": "brainstorming"}'
```

### 3. 前端集成示例

```typescript
// 不使用技能
const response1 = await fetch('/agent/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: '你好' })
});

// 使用 calculator 技能
const response2 = await fetch('/agent/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: '15.5 * 3',
    skill: 'calculator'  // 传递技能名称
  })
});

// 流式查询 + brainstorming 技能
const response3 = await fetch('/agent/query-stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: '设计一个用户认证系统',
    skill: 'brainstorming'
  })
});
```

## API 端点

### 聊天端点（支持 skill 参数）

| 端点 | 方法 | 描述 |
|------|------|------|
| `/agent/query` | POST | 非流式查询，支持 `skill` 参数 |
| `/agent/query-stream` | POST | SSE 流式查询，支持 `skill` 参数 |
| `/agent/query-stream-delta` | POST | Token 级流式，支持 `skill` 参数 |

### 技能管理端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/skills` | GET | 列出所有技能 |
| `/skills/by-category` | GET | 按分类列出技能 |
| `/skills/{name}` | GET | 获取指定技能详情 |
| `/skills/reload` | POST | 重新加载所有技能 |

## 内置技能

| 名称 | 描述 | 分类 |
|------|------|------|
| `calculator` | 执行算术运算 | Math |
| `brainstorming` | 头脑风暴和设计探索 | Planning |
| `code_reviewer` | 代码审查 | Development |
| `debugger` | 系统化调试 | Development |
| `writer` | 写作辅助 | Writing |

## 添加自定义技能

```python
from agent_core.server.skills.types import Skill

custom_skill = Skill(
    name="my_skill",
    display_name="我的技能",
    description="技能描述",
    content="""
# 技能标题

技能的具体指令内容...

## 使用指南

1. 第一步
2. 第二步
    """,
    category="Custom",
    tags=["custom"],
)

# 方式1: 直接添加到注册表
await registry.add(custom_skill)

# 方式2: 通过配置加载器
config_loader = ConfigSkillLoader([custom_skill.to_dict()])
registry.register_loader(config_loader)
await registry.reload()
```

## 扩展加载器

### 数据库加载器

```python
from agent_core.server.skills import DatabaseSkillLoader

db_loader = DatabaseSkillLoader(
    db_client=async_db_client,
    table_name="skills",
    cache_ttl_seconds=300,
)

registry.register_loader(db_loader)
await registry.reload()
```

### 远程 API 加载器

```python
from agent_core.server.skills import RemoteAPISkillLoader

api_loader = RemoteAPISkillLoader(
    api_url="https://api.example.com/skills",
    auth_token="your-token",
)

registry.register_loader(api_loader)
await registry.reload()
```

## 数据库表结构（PostgreSQL）

```sql
CREATE TABLE skills (
    name VARCHAR(100) PRIMARY KEY,
    display_name VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(50) DEFAULT 'General',
    enabled BOOLEAN DEFAULT TRUE,
    version VARCHAR(20) DEFAULT '1.0',
    tags JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

## 工作原理

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   前端请求    │     │  Server     │     │   Agent     │
│             │     │             │     │             │
│ message +   │────▶│ prepare_msg │────▶│ query()     │
│ skill param │     │ with_skill  │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │ Skill Registry│
                    │             │
                    │ get(skill)  │
                    └─────────────┘
```

1. 前端发送请求，包含 `message` 和可选的 `skill` 参数
2. Server 调用 `prepare_message_with_skill()`
3. 如果指定了 skill，从注册表获取技能内容
4. 将技能内容前置到用户消息
5. 处理后的消息发送给 Agent

## 与 CLI 版本的差异

| 特性 | CLI 版本 | Server 版本 |
|------|----------|-------------|
| 存储方式 | 本地 MD 文件 | 内存/数据库/API |
| 调用方式 | `@skill_name` | API 参数 `skill` |
| 技能发现 | 自动扫描目录 | 配置/查询加载 |
| 更新方式 | 编辑文件 | API 调用 |
| 适用场景 | 本地开发 | 在线服务 |
