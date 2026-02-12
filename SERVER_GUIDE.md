# BU Agent SDK HTTP 服务使用指南

## 快速开始

### 1. 设置 API Key

首先设置 OpenAI API Key（根据你的系统选择）：

**Windows CMD:**
```cmd
set OPENAI_API_KEY=sk-your-api-key-here
```

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="sk-your-api-key-here"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

---

### 2. 启动服务器

```bash
conda run -n 314 python test_server.py
```

或者直接激活环境后运行：

```bash
conda activate 314
python test_server.py
```

启动成功后会看到：

```
Starting BU Agent SDK Server on http://127.0.0.1:8000
API docs available at http://127.0.0.1:8000/docs
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

### 3. 调用 API

#### 方法 A: 使用 Python 客户端（推荐）

```python
import asyncio
from bu_agent_sdk import AgentClient

async def main():
    async with AgentClient("http://localhost:8000") as client:
        # 创建会话
        session_id = await client.create_session()
        print(f"Session: {session_id}")

        # 发送查询
        response = await client.query("什么是15+27?")
        print(f"回答: {response.response}")
        print(f"消耗tokens: {response.usage.total_tokens}")

asyncio.run(main())
```

保存为 `my_test.py` 后运行：

```bash
conda run -n 314 python my_test.py
```

#### 方法 B: 使用 curl

```bash
# 1. 创建会话
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d "{}"

# 返回: {"session_id":"xxx","created_at":"2025-02-06T..."}

# 2. 发送查询（替换 SESSION_ID）
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What is 15+27?\", \"session_id\": \"SESSION_ID\"}"
```

#### 方法 C: 使用浏览器

1. 打开 http://localhost:8000/docs
2. 在 Swagger UI 中点击任意端点
3. 点击 "Try it out"
4. 填写参数后点击 "Execute"

---

### 4. API 端点说明

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/sessions` | POST | 创建新会话 |
| `/sessions` | GET | 列出所有会话 |
| `/sessions/{id}` | GET | 获取会话详情 |
| `/sessions/{id}` | DELETE | 删除会话 |
| `/sessions/{id}/clear` | POST | 清空会话历史 |
| `/agent/query` | POST | 发送查询（非流式） |
| `/agent/query-stream` | POST | 发送查询（SSE流式） |
| `/agent/usage/{id}` | GET | 获取使用统计 |

---

### 5. 流式查询示例

```python
import asyncio
from bu_agent_sdk import AgentClient

async def main():
    async with AgentClient("http://localhost:8000") as client:
        await client.create_session()

        # 流式接收响应
        async for event in client.query_stream("今天天气怎么样？"):
            if event.type == "text":
                print(f"[文本] {event.content}")
            elif event.type == "tool_call":
                print(f"[工具调用] {event.tool}")
            elif event.type == "final":
                print(f"[完成] {event.content}")
                break

asyncio.run(main())
```

---

### 6. 配置说明

编辑 `test_server.py` 来自定义：

```python
app = create_server(
    agent_factory=create_agent,
    session_timeout_minutes=60,    # 会话超时时间
    max_sessions=1000,              # 最大会话数
    enable_cleanup_task=True,       # 启用自动清理
)
```

---

### 7. 故障排查

**问题：ImportError: cannot import name 'create_server'**
```bash
# 确保 FastAPI 已安装
conda run -n 314 pip install fastapi
```

**问题：401 Incorrect API key provided**
```bash
# 重新设置 API Key
set OPENAI_API_KEY=sk-your-correct-key
```

**问题：端口已被占用**
```bash
# 修改 test_server.py 中的端口
uvicorn.run(app, host="127.0.0.1", port=8001)  # 改成其他端口
```

---

### 8. 测试脚本

项目提供了两个测试脚本：

| 脚本 | 说明 | 需要 API Key |
|------|------|--------------|
| `test_client_simple.py` | 测试 API 结构 | 否 |
| `test_client.py` | 完整功能测试 | 是 |

运行测试：

```bash
# 简单测试（无需 API Key）
conda run -n 314 python test_client_simple.py

# 完整测试（需要设置 API Key）
set OPENAI_API_KEY=your-key
conda run -n 314 python test_client.py
```
