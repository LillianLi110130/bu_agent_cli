"""
BU Agent CLI - Main Entry Point

A simplified agent CLI framework for running LLM agents with tool calling.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to sys.path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bu_agent import BUAgent, Shell
from bu_agent_sdk import Agent
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools import tool


# --- Define some basic tools ---


@tool("Calculator")
async def calculate(expression: str) -> str:
    """Calculate a math expression."""
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error: {e}"


@tool("Echo")
async def echo(message: str) -> str:
    """Echo the message back."""
    return f"Echo: {message}"


@tool("Get current time")
async def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool("Get system info")
async def get_system_info() -> str:
    """Get basic system information."""
    import platform

    info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    return "\n".join(f"{k}: {v}" for k, v in info.items())


# --- Main Entry Point ---


async def main():
    # 1. Setup LLM
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")

    # Default Zhipu API key (for testing - replace with your own)
    if not api_key:
        # For GLM models
        api_key = os.getenv("ZHIPU_API_KEY", "your_key")

    if not api_key:
        print("Please set OPENAI_API_KEY or ZHIPU_API_KEY environment variable.")
        return

    # Determine which model to use
    model = os.getenv("LLM_MODEL", "GLM-4.7")
    base_url = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4")

    # For OpenAI models
    if model.startswith("gpt-") or model.startswith("o1-") or model.startswith("o3-"):
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("OPENAI_API_KEY", api_key)

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    # 2. Setup Agent
    agent = Agent(
        llm=llm,
        tools=[calculate, echo, get_current_time, get_system_info],
        system_prompt="You are a helpful assistant. Use tools when available to provide accurate information.",
    )

    # 3. Setup BUAgent wrapper
    bu_agent = BUAgent(agent, name="BU Agent")

    # 4. Setup Shell
    welcome_info = [
        {"name": "Model", "value": model, "level": "info"},
        {"name": "Tools", "value": "calculator, echo, time, system_info", "level": "info"},
    ]

    shell = Shell(bu_agent, welcome_info=welcome_info)
    await shell.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
