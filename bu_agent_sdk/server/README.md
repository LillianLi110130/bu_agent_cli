# BU Agent SDK HTTP Server

FastAPI-based HTTP server for the bu_agent_sdk, enabling remote invocation of agent capabilities via REST API.

## Features

- **Session Management**: Each conversation has its own isolated session with conversation state
- **Streaming Support**: Server-Sent Events (SSE) for real-time response streaming
- **Async/Await**: Built on FastAPI with full async support
- **Auto Cleanup**: Automatic cleanup of inactive sessions
- **Usage Tracking**: Token usage and cost tracking per session
- **Health Checks**: Built-in health check endpoint

## Installation

```bash
pip install fastapi uvicorn
```

## Quick Start

### 1. Define Your Tools and Agent

```python
from bu_agent_sdk import Agent
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.server import create_server
from bu_agent_sdk.tools import tool

@tool("Calculate")
async def calculate(a: int, b: int) -> int:
    return a + b

def agent_factory():
    return Agent(
        llm=ChatOpenAI(model="gpt-4o"),
        tools=[calculate],
    )

app = create_server(agent_factory=agent_factory)
```

### 2. Run the Server

```bash
uvicorn your_module:app --reload --port 8000
```

## API Endpoints

### Health Check
```
GET /health
```

### Sessions
```
POST   /sessions              # Create new session
GET    /sessions              # List all sessions
GET    /sessions/{session_id} # Get session info
DELETE /sessions/{session_id} # Delete session
POST   /sessions/{session_id}/clear  # Clear session history
```

### Agent Query
```
POST /agent/query          # Non-streaming query
POST /agent/query-stream   # Streaming query (SSE)
GET  /agent/usage/{session_id}  # Get usage stats
```

## Usage Examples

### Non-Streaming Query

```python
import httpx

async def query_agent(message: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/agent/query",
            json={"message": message}
        )
        return response.json()

# Returns:
# {
#   "session_id": "abc-123",
#   "response": "The answer is 5",
#   "usage": {...}
# }
```

### Streaming Query

```python
import httpx
import json

async def stream_query(message: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/agent/query-stream",
            json={"message": message},
            timeout=None
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    print(f"Event: {event['type']}", event)
```

### Session Management

```python
import httpx

# Create session
response = await client.post("http://localhost:8000/sessions")
session_id = response.json()["session_id"]

# Use session_id in queries
await client.post(
    "http://localhost:8000/agent/query",
    json={"message": "Hello", "session_id": session_id}
)
```

## Configuration

```python
from bu_agent_sdk.server import ServerConfig, create_app

config = ServerConfig(
    session_timeout_minutes=60,   # Session timeout
    max_sessions=1000,            # Max concurrent sessions
    cleanup_interval_seconds=300, # Cleanup interval
    enable_cleanup_task=True,     # Enable auto cleanup
)

app = create_app(agent_factory=agent_factory, config=config)
```

## Event Types (Streaming)

| Type | Description |
|------|-------------|
| `text` | Assistant text content |
| `thinking` | Model reasoning content |
| `tool_call` | Tool being called |
| `tool_result` | Tool execution result |
| `step_start` | Logical step starting |
| `step_complete` | Logical step completing |
| `final` | Final response (last event) |
| `hidden` | Hidden message injected by agent |
| `usage` | Usage statistics (sent after final) |
| `error` | Error occurred |

## Running with Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "bu_agent_sdk.server.example_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Production Considerations

1. **Authentication**: Add authentication middleware (e.g., OAuth2, API keys)
2. **Rate Limiting**: Use `slowapi` or similar for rate limiting
3. **CORS**: Configure CORS settings for web clients
4. **Monitoring**: Add Prometheus metrics or similar observability
5. **Persistence**: Add Redis/database for session persistence across restarts
