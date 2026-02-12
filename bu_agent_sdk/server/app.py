"""
FastAPI 应用 - bu_agent_sdk HTTP 服务器

这是 bu_agent_sdk 的 HTTP API 主文件，提供以下功能：
1. 会话管理：创建、查询、删除会话
2. Agent 查询：非流式和流式查询
3. 使用统计：Token 消耗和成本统计
4. 健康检查：服务状态监控
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Callable

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 导入 SDK 内部的事件类型（用于类型判断）
from bu_agent_sdk.agent.events import (
    TextEvent as AgentTextEvent,
    TextDeltaEvent as AgentTextDeltaEvent,
    ThinkingEvent as AgentThinkingEvent,
    ToolCallEvent as AgentToolCallEvent,
    ToolResultEvent as AgentToolResultEvent,
    FinalResponseEvent as AgentFinalResponseEvent,
    StepStartEvent as AgentStepStartEvent,
    StepCompleteEvent as AgentStepCompleteEvent,
    HiddenUserMessageEvent as AgentHiddenUserMessageEvent,
)

# 导入 API 数据模型
from bu_agent_sdk.server.models import (
    QueryRequest,
    QueryResponse,
    UsageInfo,
    StreamEvent,
    ErrorResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionInfoResponse,
    ClearHistoryRequest,
    ClearHistoryResponse,
    HealthResponse,
)
from bu_agent_sdk.server.session import SessionManager, AgentFactory


logger = logging.getLogger("bu_agent_sdk.server")


class ServerConfig(BaseModel):
    """服务器配置

    Attributes:
        session_timeout_minutes: 会话超时时间（分钟），超过此时间未使用的会话将被清理
        max_sessions: 最大并发会话数
        cleanup_interval_seconds: 会话清理任务执行间隔（秒）
        enable_cleanup_task: 是否启用自动清理任务
    """

    session_timeout_minutes: int = 60
    max_sessions: int = 1000
    cleanup_interval_seconds: int = 300
    enable_cleanup_task: bool = True


def _agent_event_to_stream_event(event) -> StreamEvent:
    """将 SDK 内部的 Agent 事件转换为 API 返回的 StreamEvent

    这个函数是事件类型转换的核心，将内部事件类型转换为对外暴露的 API 事件类型。
    这样做的好处是内部实现可以变化，而 API 接口保持稳定。

    Args:
        event: SDK 内部产生的事件（AgentTextEvent, AgentToolCallEvent 等）

    Returns:
        StreamEvent: API 对外暴露的事件类型
    """
    # 文本事件：Assistant 产生的文本内容
    if isinstance(event, AgentTextEvent):
        from bu_agent_sdk.server.models import TextEvent
        return TextEvent(content=event.content)

    # 文本增量事件：流式输出的增量文本
    if isinstance(event, AgentTextDeltaEvent):
        from bu_agent_sdk.server.models import TextDeltaEvent
        return TextDeltaEvent(delta=event.delta)

    # 思考事件：模型的推理过程（如果模型支持 thinking）
    if isinstance(event, AgentThinkingEvent):
        from bu_agent_sdk.server.models import ThinkingEvent
        return ThinkingEvent(content=event.content)

    # 工具调用事件：Assistant 决定调用某个工具
    if isinstance(event, AgentToolCallEvent):
        from bu_agent_sdk.server.models import ToolCallEvent
        return ToolCallEvent(
            tool=event.tool,
            args=event.args,
            tool_call_id=event.tool_call_id,
            display_name=event.display_name,
        )

    # 工具结果事件：工具执行完成后的结果
    if isinstance(event, AgentToolResultEvent):
        from bu_agent_sdk.server.models import ToolResultEvent
        return ToolResultEvent(
            tool=event.tool,
            result=event.result,
            tool_call_id=event.tool_call_id,
            is_error=event.is_error,
            screenshot_base64=event.screenshot_base64,
        )

    # 步骤开始事件：一个逻辑步骤的开始（用于 UI 显示进度）
    if isinstance(event, AgentStepStartEvent):
        from bu_agent_sdk.server.models import StepStartEvent
        return StepStartEvent(
            step_id=event.step_id,
            title=event.title,
            step_number=event.step_number,
        )

    # 步骤完成事件：一个逻辑步骤的完成
    if isinstance(event, AgentStepCompleteEvent):
        from bu_agent_sdk.server.models import StepCompleteEvent
        return StepCompleteEvent(
            step_id=event.step_id,
            status=event.status,
            duration_ms=event.duration_ms,
        )

    # 最终响应事件：Agent 的最终回复（流式响应的最后一个事件）
    if isinstance(event, AgentFinalResponseEvent):
        from bu_agent_sdk.server.models import FinalResponseEvent
        return FinalResponseEvent(content=event.content)

    # 隐藏消息事件：Agent 内部注入的用户消息（用于纠正等）
    if isinstance(event, AgentHiddenUserMessageEvent):
        from bu_agent_sdk.server.models import HiddenMessageEvent
        return HiddenMessageEvent(content=event.content)

    # 未知事件类型的兜底处理
    from bu_agent_sdk.server.models import TextEvent
    return TextEvent(content=str(event))


def _serialize_event(event: StreamEvent) -> str:
    """将 StreamEvent 序列化为 SSE (Server-Sent Events) 格式

    SSE 格式：每一行以 "data: " 开头，后面跟 JSON 数据，最后用两个换行符结束。

    示例输出：
        data: {"type": "text", "content": "你好", "timestamp": "2025-02-06..."}
        data: {"type": "tool_call", "tool": "add", "args": {"a": 1, "b": 2}}

    Args:
        event: 要序列化的事件

    Returns:
        str: SSE 格式的字符串
    """
    # 将 Pydantic 模型转换为字典
    data = event.model_dump()
    # 转换为 JSON 字符串，添加 SSE 前缀和结束符
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


# ================================
# 全局变量（在 lifespan 中初始化）
# ================================

_session_manager: SessionManager | None = None  # 会话管理器
_cleanup_task: asyncio.Task | None = None      # 后台清理任务


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 应用生命周期管理

    这个函数在应用启动时执行初始化，在应用关闭时执行清理。
    使用 asynccontextmanager 装饰器，可以与 FastAPI 的 lifespan 参数配合使用。

    启动时：
        1. 创建 SessionManager
        2. 启动后台清理任务

    关闭时：
        1. 取消清理任务
    """
    global _session_manager, _cleanup_task

    # 从 app.state 中获取配置和 Agent 工厂函数
    config: ServerConfig = app.state.config
    agent_factory: Callable = app.state.agent_factory

    # 初始化会话管理器
    # SessionManager 负责创建、获取、删除 Agent 会话
    _session_manager = SessionManager(
        agent_factory=agent_factory,                      # 用于创建新 Agent 的函数
        session_timeout_minutes=config.session_timeout_minutes,  # 会话超时时间
        max_sessions=config.max_sessions,                 # 最大会话数
    )

    # 如果启用了自动清理，启动后台任务
    if config.enable_cleanup_task:
        loop = asyncio.get_running_loop()
        # 创建后台任务，定期清理过期会话
        _cleanup_task = loop.create_task(
            _session_manager.cleanup_task(interval_seconds=config.cleanup_interval_seconds)
        )
        logger.info("Started session cleanup task")

    # yield 之前是启动逻辑，之后是关闭逻辑
    yield

    # 关闭时取消清理任务
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass  # 取消任务时会抛出 CancelledError，忽略它
    logger.info("Server shutdown complete")


def create_app(
    agent_factory: AgentFactory,
    config: ServerConfig | None = None,
) -> FastAPI:
    """
    创建 FastAPI 应用

    这是服务器的入口函数，负责：
    1. 创建 FastAPI 应用实例
    2. 注册所有路由
    3. 配置生命周期管理

    Args:
        agent_factory: 一个可调用对象，用于创建新的 Agent 实例
        config: 服务器配置，如果为 None 则使用默认配置

    Returns:
        FastAPI: 配置好的 FastAPI 应用实例

    Example:
        def my_agent_factory():
            return Agent(llm=ChatOpenAI(model="gpt-4o"), tools=[...])

        app = create_app(agent_factory=my_agent_factory)
    """
    # 使用默认配置（如果未提供）
    if config is None:
        config = ServerConfig()

    # 创建 FastAPI 应用
    app = FastAPI(
        title="BU Agent SDK API",
        description="HTTP API for bu_agent_sdk - Agentic applications with LLMs",
        version="1.0.0",
        lifespan=lifespan,  # 设置生命周期管理
    )

    # 将配置和工厂函数存储在 app.state 中，供后续使用
    app.state.config = config
    app.state.agent_factory = agent_factory

    # ================================
    # 健康检查端点
    # ================================

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """健康检查端点

        用于监控服务是否正常运行，以及当前活跃会话数。
        """
        return HealthResponse(
            status="healthy",
            active_sessions=_session_manager.session_count if _session_manager else 0,
        )

    # ================================
    # 会话管理端点
    # ================================

    @app.post("/sessions", response_model=SessionCreateResponse, tags=["Sessions"])
    async def create_session(request: SessionCreateRequest):
        """创建新会话

        创建一个新的 Agent 会话，每个会话有独立的对话历史。
        返回一个唯一的 session_id，后续请求需要携带此 ID。
        """
        # 获取或创建会话（不传 session_id 会创建新的）
        session = await _session_manager.get_or_create_session()
        return SessionCreateResponse(
            session_id=session.session_id,
            created_at=session.created_at,
        )

    @app.get("/sessions/{session_id}", response_model=SessionInfoResponse, tags=["Sessions"])
    async def get_session_info(session_id: str):
        """获取会话信息

        获取指定会话的详细信息，包括：
        - 创建时间
        - 消息数量
        - Token 使用统计
        """
        session = await _session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        usage_summary = await session.get_usage()
        return SessionInfoResponse(
            session_id=session.session_id,
            created_at=session.created_at,
            message_count=session.message_count,
            usage=UsageInfo.from_usage_summary(usage_summary),
        )

    @app.get("/sessions", tags=["Sessions"])
    async def list_sessions():
        """列出所有活跃会话

        返回当前所有活跃会话的列表，包括基本信息。
        """
        return {"sessions": _session_manager.list_sessions()}

    @app.delete("/sessions/{session_id}", tags=["Sessions"])
    async def delete_session(session_id: str):
        """删除会话

        删除指定的会话及其所有对话历史。
        """
        deleted = await _session_manager.delete_session(session_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )
        return {"deleted": True, "session_id": session_id}

    @app.post("/sessions/{session_id}/clear", response_model=ClearHistoryResponse, tags=["Sessions"])
    async def clear_session_history(session_id: str):
        """清空会话历史

        清空指定会话的对话历史，但保留会话本身。
        """
        session = await _session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        await session.clear_history()
        return ClearHistoryResponse(session_id=session_id, cleared=True)

    # ================================
    # Agent 查询端点
    # ================================

    @app.post("/agent/query", response_model=QueryResponse, tags=["Agent"])
    async def query(request: QueryRequest):
        """
        非流式查询

        发送消息给 Agent，等待完整响应后返回。
        适合不需要实时更新、只关注最终结果的场景。

        请求体:
            {
                "message": "用户消息",
                "session_id": "会话ID（可选，不传则创建新会话）"
            }

        响应:
            {
                "session_id": "会话ID",
                "response": "Agent的回复",
                "usage": {token使用统计}
            }
        """
        # 获取或创建会话
        session = await _session_manager.get_or_create_session(request.session_id)

        try:
            # 执行查询（等待完整响应）
            response_text = await session.query(request.message)
            # 获取使用统计
            usage_summary = await session.get_usage()

            return QueryResponse(
                session_id=session.session_id,
                response=response_text,
                usage=UsageInfo.from_usage_summary(usage_summary),
            )
        except Exception as e:
            logger.error(f"Error in query: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    @app.post("/agent/query-stream", tags=["Agent"])
    async def query_stream(request: QueryRequest):
        """
        流式查询 (Server-Sent Events)

        发送消息给 Agent，通过 SSE 实时推送响应过程。
        适合需要实时显示 Agent 思考过程、工具调用等场景。

        事件类型:
            - text: Assistant 文本内容
            - thinking: 模型推理内容
            - tool_call: 工具调用
            - tool_result: 工具执行结果
            - step_start: 步骤开始
            - step_complete: 步骤完成
            - final: 最终响应（最后一个事件）
            - usage: 使用统计
            - error: 错误信息

        SSE 格式:
            data: {"type": "text", "content": "..."}
            data: {"type": "tool_call", "tool": "add", ...}
            data: {"type": "final", "content": "..."}
            : done
        """
        session = await _session_manager.get_or_create_session(request.session_id)

        async def event_generator():
            """内部生成器函数，产生 SSE 事件"""
            try:
                # 遍历 Agent 产生的事件
                async for event in session.query_stream(request.message):
                    # 转换为 API 事件格式
                    stream_event = _agent_event_to_stream_event(event)
                    # 序列化为 SSE 格式并 yield
                    yield _serialize_event(stream_event)

                    # 如果是最终响应事件，发送 usage 和 done 后结束
                    if isinstance(event, AgentFinalResponseEvent):
                        logger.info("[SSE] FinalResponseEvent received, sending usage and done")
                        # 发送使用统计 - 直接调用 agent 方法避免死锁
                        usage_summary = await session.agent.get_usage()
                        usage_info = UsageInfo.from_usage_summary(usage_summary)
                        yield f"data: {json.dumps({'type': 'usage', 'session_id': session.session_id, 'usage': usage_info.model_dump()}, ensure_ascii=False, default=str)}\n\n"
                        # 必须在 return 前发送 done，finally 中的 yield 不会被执行
                        logger.info("[SSE] Sending done signal before return")
                        yield ": done\n\n"
                        return

                # 如果循环正常结束（没有 return），说明没有 FinalResponseEvent
                logger.info("[SSE] Stream ended without FinalResponseEvent, sending done")
                yield ": done\n\n"

            except Exception as e:
                logger.error(f"Error in stream: {e}", exc_info=True)
                # 发送错误事件
                error_event = {
                    "type": "error",
                    "error": str(e),
                }
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                yield ": done\n\n"

        # 返回流式响应
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",           # 禁用缓存
                "Connection": "close",                 # 完成后关闭连接
                "X-Accel-Buffering": "no",             # 禁用 nginx 缓冲
            },
        )

    @app.post("/agent/query-stream-delta", tags=["Agent"])
    async def query_stream_delta(request: QueryRequest):
        """
        Token 级别流式查询 (Server-Sent Events)

        与 /agent/query-stream 不同，这个端点会在 LLM 生成过程中逐 token 返回 text_delta 事件，
        实现类似 ChatGPT 的打字机效果，适合前端实时渲染文本。

        事件类型:
            - text_delta: 增量文本（每个 token/小块）
            - tool_call: 工具调用
            - tool_result: 工具执行结果
            - final: 最终响应（最后一个事件）
            - usage: 使用统计

        SSE 格式:
            data: {"type": "text_delta", "delta": "你"}
            data: {"type": "text_delta", "delta": "好"}
            data: {"type": "text_delta", "delta": "！"}
            data: {"type": "final", "content": "你好！"}
            : done

        前端处理示例:
            let buffer = "";
            for await (const line of response) {
                const event = JSON.parse(line.data);
                if (event.type === "text_delta") {
                    buffer += event.delta;
                    updateUI(buffer);  // 实时更新
                }
            }
        """
        session = await _session_manager.get_or_create_session(request.session_id)

        async def event_generator():
            """内部生成器函数，产生 SSE 事件"""
            try:
                # 使用 query_stream_delta 方法获取 Token 级别的流式输出
                async for event in session.query_stream_delta(request.message):
                    stream_event = _agent_event_to_stream_event(event)
                    yield _serialize_event(stream_event)

                    # 如果是最终响应事件，发送 usage 和 done 后结束
                    if isinstance(event, AgentFinalResponseEvent):
                        logger.info("[SSE] FinalResponseEvent received, sending usage and done")
                        # 发送使用统计 - 直接调用 agent 方法避免死锁
                        usage_summary = await session.agent.get_usage()
                        usage_info = UsageInfo.from_usage_summary(usage_summary)
                        yield f"data: {json.dumps({'type': 'usage', 'session_id': session.session_id, 'usage': usage_info.model_dump()}, ensure_ascii=False, default=str)}\n\n"
                        # 必须在 return 前发送 done，finally 中的 yield 不会被执行
                        logger.info("[SSE] Sending done signal before return")
                        yield ": done\n\n"
                        return

                # 如果循环正常结束（没有 return），说明没有 FinalResponseEvent
                logger.info("[SSE] Stream ended without FinalResponseEvent, sending done")
                yield ": done\n\n"

            except Exception as e:
                logger.error(f"Error in delta stream: {e}", exc_info=True)
                error_event = {
                    "type": "error",
                    "error": str(e),
                }
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                yield ": done\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/agent/usage/{session_id}", tags=["Agent"])
    async def get_usage(session_id: str):
        """获取会话的使用统计

        返回指定会话的 Token 消耗和成本统计。
        """
        session = await _session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        usage_summary = await session.get_usage()
        return {
            "session_id": session_id,
            "usage": UsageInfo.from_usage_summary(usage_summary).model_dump(),
        }

    return app


# ================================
# 便捷函数
# ================================

def create_server(agent_factory: AgentFactory, **config_kwargs) -> FastAPI:
    """
    创建 FastAPI 服务器的便捷函数

    这是一个简化版的创建函数，允许直接传入配置参数，
    而不需要先创建 ServerConfig 对象。

    Args:
        agent_factory: 创建 Agent 实例的工厂函数
        **config_kwargs: 配置参数（session_timeout_minutes, max_sessions 等）

    Returns:
        FastAPI: 配置好的应用实例

    Example:
        from bu_agent_sdk import Agent
        from bu_agent_sdk.llm import ChatOpenAI
        from bu_agent_sdk.tools import tool

        @tool("Add numbers")
        async def add(a: int, b: int) -> int:
            return a + b

        app = create_server(
            agent_factory=lambda: Agent(
                llm=ChatOpenAI(model="gpt-4o"),
                tools=[add],
            ),
            session_timeout_minutes=30,  # 直接传配置参数
        )

        # 运行: uvicorn example:app --reload
    """
    config = ServerConfig(**config_kwargs)
    return create_app(agent_factory=agent_factory, config=config)
