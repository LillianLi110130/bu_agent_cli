"""
Example server implementation using bu_agent_sdk HTTP API.

This is a standalone example that can be run with uvicorn.
"""

import os
from bu_agent_sdk import Agent
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.server import create_server
from bu_agent_sdk.tools import tool


# Define your tools
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
    return f"Search results for '{query}': Found 5 relevant articles."


# Create agent factory
def create_agent() -> Agent:
    """Factory function to create new Agent instances."""
    return Agent(
        llm=ChatOpenAI(
            model=os.getenv("MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
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
    uvicorn.run(
        "example_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
