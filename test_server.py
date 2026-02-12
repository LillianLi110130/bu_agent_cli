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

from pydantic import BaseModel, Field

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


# Test args_schema feature
class CustodyFeeItem(BaseModel):
    """托管费计算项目"""

    datDte: str = Field(..., description="数据日期，格式 YYYY/MM/DD")
    bnkNam: str = Field(..., description="银行名称")
    typNam: str = Field(..., description="类型名称")
    sscYer: float = Field(..., description="年累计托管费收入")


@tool("Calculate custody fee", args_schema=CustodyFeeItem)
async def calculate_custody_fee(item: CustodyFeeItem) -> str:
    """Calculate custody fee based on the given parameters (mock)."""
    fee = item.sscYer * 0.01  # Simple calculation: 1% of yearly income
    return (
        f"托管费计算结果 - 银行: {item.bnkNam}, 类型: {item.typNam}, "
        f"日期: {item.datDte}, 年收入: {item.sscYer}, 计算费率: {fee:.2f}"
    )


# Create agent factory
def create_agent() -> Agent:
    """Factory function to create new Agent instances."""
    return Agent(
        llm=ChatOpenAI(
            model="GLM-4.7",
            api_key="9a7ccd2b6915401481f7caa599398658.cbmpB4oEOlQbQjKA",
            base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        ),
        tools=[add, get_weather, search_web, calculate_custody_fee],
        system_prompt="You are a helpful assistant that can use tools to answer questions.",
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
