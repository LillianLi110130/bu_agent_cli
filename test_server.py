"""
Test server for bu_agent_sdk HTTP API.

Run with: conda run -n 314 python test_server.py

启用调试输出（查看 LLM 原始响应）：
export BU_AGENT_SDK_LLM_DEBUG=1  # Linux/Mac
set BU_AGENT_SDK_LLM_DEBUG=1     # Windows CMD
$env:BU_AGENT_SDK_LLM_DEBUG="1"  # Windows PowerShell
"""

import logging
import os

# 启用调试输出
os.environ["BU_AGENT_SDK_LLM_DEBUG"] = "1"

# 配置日志显示 DEBUG 信息
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from bu_agent_sdk import Agent
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.server import create_server
from bu_agent_sdk.tools import tool


@tool("Add two numbers")
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool("Get current weather")
async def get_weather(location: str) -> str:
    """Get the current weather for a location (mock)."""
    return f"The weather in {location} is sunny and 75°F."


@tool("Search the web")
async def search_web(query: str) -> str:
    """Search the web for information (mock)."""
    return f"Search results for '{query}': Found 5 relevant articles with helpful information."


# Create agent factory
def create_agent() -> Agent:
    """Factory function to create new Agent instances."""
    return Agent(
        llm=ChatOpenAI(
            model="GLM-4.7",
            api_key="your_key",
            base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        ),
        tools=[add, get_weather, search_web],
        system_prompt="You are a helpful assistant that can use tools to answer questions.",
        include_cost=True,
    )


# Create the FastAPI app
app = create_server(
    agent_factory=create_agent,
    session_timeout_minutes=60,
    max_sessions=1000,
    enable_cleanup_task=True,
)


if __name__ == "__main__":
    import uvicorn

    # Run the server
    print("Starting BU Agent SDK Server on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
