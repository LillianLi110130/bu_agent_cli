# BU Agent CLI

A simplified agent CLI framework for running LLM agents with tool calling.

## Features

- **Agent SDK**: Complete bu_agent_sdk for building agentic applications
- **Tool Calling**: Built-in support for OpenAI-compatible tool calling
- **Streaming Events**: Real-time event streaming for agent actions
- **Interactive Shell**: Beautiful terminal UI with rich output formatting
- **Token Tracking**: Built-in token usage and cost tracking

## Installation

```bash
# Clone the repository
cd bu_agent_cli

# Install dependencies
pip install -e .
```

## Usage

### Basic Example

```python
import asyncio
from bu_agent_sdk import Agent
from bu_agent_sdk.llm import ChatOpenAI
from bu_agent_sdk.tools import tool
from bu_agent import BUAgent, Shell

@tool("Calculator")
async def calculate(expression: str) -> str:
    """Calculate a math expression."""
    return str(eval(expression))

async def main():
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key="your-api-key",
    )

    agent = Agent(
        llm=llm,
        tools=[calculate],
        system_prompt="You are a helpful assistant.",
    )

    bu_agent = BUAgent(agent)
    shell = Shell(bu_agent)
    await shell.run()

asyncio.run(main())
```

### Running the CLI

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key"

# Run with default settings
python main.py

# Or with a custom model
export LLM_MODEL="gpt-4o"
export LLM_BASE_URL="https://api.openai.com/v1"
python main.py
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ZHIPU_API_KEY` | Zhipu AI API key | - |
| `LLM_MODEL` | Model name | `GLM-4.7` |
| `LLM_BASE_URL` | API base URL | `https://open.bigmodel.cn/api/coding/paas/v4` |

## Project Structure

```
bu_agent_cli/
├── bu_agent/              # Main agent framework
│   ├── __init__.py
│   ├── core.py            # BUAgent wrapper
│   └── shell.py           # Interactive shell UI
├── bu_agent_sdk/          # Agent SDK (source code)
│   ├── agent/             # Agent implementation
│   ├── llm/               # LLM abstractions
│   ├── tools/             # Tool decorators
│   └── tokens/            # Token tracking
├── main.py                # Entry point
├── pyproject.toml         # Project config
└── README.md              # This file
```

## Built-in Tools

The CLI comes with several built-in tools:

- **Calculator**: Evaluate math expressions
- **Echo**: Echo back messages
- **Get current time**: Get current date and time
- **Get system info**: Get system information

## License

MIT
