"""
纯 HTTP 请求调用示例 - 不依赖 SDK 客户端

使用标准库和 httpx 发送 HTTP 请求调用 API
"""

import asyncio
import json

import httpx


BASE_URL = "http://localhost:8000"


async def pure_http_call_example():
    """使用纯 HTTP 请求调用 API"""

    async with httpx.AsyncClient(timeout=300) as client:

        # ========== 1. 创建会话 ==========
        print("=== 1. 创建会话 ===")
        response = await client.post(
            f"{BASE_URL}/sessions", headers={"Content-Type": "application/json"}, json={}
        )
        session_data = response.json()
        session_id = session_data["session_id"]
        print(f"Session ID: {session_id}")
        print()

        # ========== 2. 发送查询 ==========
        # print("=== 2. 发送查询 ===")
        # response = await client.post(
        #     f"{BASE_URL}/agent/query",
        #     headers={"Content-Type": "application/json"},
        #     json={"message": "什么是15+27?", "session_id": session_id},
        # )
        # result = response.json()
        # print(f"回答: {result['response']}")
        # print(f"Tokens: {result['usage']['total_tokens']}")
        # print()

        # # ========== 3. 获取会话信息 ==========
        # print("=== 3. 获取会话信息 ===")
        # response = await client.get(f"{BASE_URL}/sessions/{session_id}")
        # info = response.json()
        # print(f"消息数: {info['message_count']}")
        # print()

        # # ========== 4. 多轮对话 ==========
        # print("=== 4. 多轮对话 ===")
        # questions = ["我叫小明", "我叫什么名字？"]
        # for q in questions:
        #     response = await client.post(
        #         f"{BASE_URL}/agent/query",
        #         headers={"Content-Type": "application/json"},
        #         json={"message": q, "session_id": session_id},
        #     )
        #     result = response.json()
        #     print(f"问: {q}")
        #     print(f"答: {result['response']}")
        #     print()

        # # ========== 5. 清空历史 ==========
        # print("=== 5. 清空历史 ===")
        # response = await client.post(
        #     f"{BASE_URL}/sessions/{session_id}/clear",
        #     headers={"Content-Type": "application/json"},
        #     json={},
        # )
        # print(f"已清空: {response.json()['cleared']}")
        # print()

        # ========== 6. 流式查询 (SSE) ==========
        print("=== 6. 流式查询 ===")
        async with client.stream(
            "POST",
            f"{BASE_URL}/agent/query-stream",
            headers={"Content-Type": "application/json"},
            json={"message": "今天广州的天气怎么样?", "session_id": session_id},
            timeout=60,  # 设置超时
        ) as response:
            async for line in response.aiter_lines():
                if not line:  # 跳过空行
                    continue

                # 检测 SSE 注释行（如 ": done"）
                if line.startswith(":"):
                    if line.strip() == ": done":
                        print("\n[SSE 连接关闭]")
                        break
                    continue  # 其他注释行跳过

                # 处理 data 行
                if line.startswith("data: "):
                    data = line[6:]  # 去掉 "data: " 前缀
                    try:
                        event = json.loads(data)
                        event_type = event.get("type")

                        if event_type == "text":
                            print(f"[文本] {event.get('content', '')}")
                        elif event_type == "tool_call":
                            print(f"[工具调用] {event.get('tool', '')}")
                        elif event_type == "tool_result":
                            print(
                                f"[工具结果] {event.get('tool', '')}: {event.get('result', '')[:50]}..."
                            )
                        elif event_type == "final":
                            print(f"\n[完成] {event.get('content', '')}")
                            # 收到 final 后继续，等待 usage 和 done 信号
                        elif event_type == "usage":
                            print(f"[统计] tokens: {event.get('usage', {}).get('total_tokens', 0)}")
                            # 收到 usage 后继续，等待 done 信号
                        elif event_type == "error":
                            print(f"[错误] {event.get('error', '')}")
                    except json.JSONDecodeError:
                        pass
        print()


if __name__ == "__main__":
    asyncio.run(pure_http_call_example())
