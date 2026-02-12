"""
Token 级别流式输出测试 - 实时打字机效果

使用 /agent/query-stream-delta 端点，逐 token 实时显示 LLM 输出
"""

import asyncio
import json

import httpx


BASE_URL = "http://localhost:8000"


async def test_token_streaming():
    """测试 Token 级别的流式输出"""

    async with httpx.AsyncClient(timeout=300) as client:

        # 1. 创建会话
        print("=== 创建会话 ===")
        response = await client.post(
            f"{BASE_URL}/sessions", headers={"Content-Type": "application/json"}, json={}
        )
        session_data = response.json()
        session_id = session_data["session_id"]
        print(f"Session ID: {session_id}\n")

        # 2. Token 级别流式查询
        print("=== Token 级别流式查询 ===")
        print("问题: 广州今天天气怎么样\n")
        print("回答: ", end="", flush=True)

        async with client.stream(
            "POST",
            f"{BASE_URL}/agent/query-stream-delta",
            headers={"Content-Type": "application/json"},
            json={"message": "广州今天天气怎么样", "session_id": session_id},
            timeout=60,
        ) as response:
            buffer = ""  # 文本缓冲区
            async for line in response.aiter_lines():
                # 调试：打印原始行
                print(f"[DEBUG] 收到行: {repr(line)}")

                if not line:
                    continue

                # 检测 SSE 注释行（如 ": done"）
                if line.startswith(":"):
                    if line.strip() == ": done":
                        print("\n[SSE 连接关闭]")

                    continue  # 其他注释行跳过

                # 处理 data 行
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        event = json.loads(data)
                        event_type = event.get("type")

                        if event_type == "text_delta":
                            # 增量文本 - 实时显示
                            delta = event.get("delta", "")
                            buffer += delta
                            print(delta, end="", flush=True)

                        elif event_type == "tool_call":
                            print(f"\n[工具调用] {event.get('tool', '')}")

                        elif event_type == "tool_result":
                            result = event.get("result", "")[:50]
                            print(f"\n[工具结果] {result}...")

                        elif event_type == "final":
                            print(f"\n[完成]")

                        elif event_type == "usage":
                            print(f"[统计] tokens: {event.get('usage', {}).get('total_tokens', 0)}")

                        elif event_type == "error":
                            print(f"\n[错误] {event.get('error', '')}")

                    except json.JSONDecodeError:
                        pass

        print(f"\n最终回答: {buffer}")


if __name__ == "__main__":
    asyncio.run(test_token_streaming())
