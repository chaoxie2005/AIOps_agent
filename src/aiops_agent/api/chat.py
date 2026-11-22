from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from logging import getLogger
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langgraph.checkpoint.base import BaseCheckpointSaver

from aiops_agent.api.sessions import Sessions
from aiops_agent.graph_components.graph.main_graph import MainGraph
from aiops_agent.schema.resquest.chat import ChatRequest
from aiops_agent.schema.response.chat import ChatResponse

router = APIRouter()
logger = getLogger(__name__)


@dataclass(frozen=True)
class _ChatExecution:
    request: ChatRequest # 本轮聊天请求
    checkpointer: BaseCheckpointSaver # 会话 checkpoint 存储器


async def _claim_session(request: ChatRequest, sessions: Sessions) -> AsyncIterator[_ChatExecution]:
    """取得当前会话的执行权，并在请求结束时释放。

    Args:
        request: 包含会话 ID 和用户 ID 的聊天请求。
        sessions: 用于校验会话归属并取得执行权的会话服务。

    Yields:
        包含本轮聊天请求和会话 checkpoint 存储器的执行上下文。
    """
    async with sessions.claim(request.session_id, request.user_id) as checkpointer:
        yield _ChatExecution(request, checkpointer)


ChatExecution = Annotated[_ChatExecution, Depends(_claim_session, scope="request")]


def _build_graph(execution: _ChatExecution) -> MainGraph:
    """使用本轮请求的参数构建聊天与排障主图。

    Args:
        execution: 包含本轮聊天请求和会话 checkpoint 存储器的执行上下文。

    Returns:
        以会话 ID 为 LangGraph thread_id、可读写该会话状态的主图。
    """
    request = execution.request
    kwargs = {"max_steps": request.max_steps} if request.max_steps is not None else {}
    return MainGraph(
        query=request.query,
        thread_id=request.session_id,
        checkpointer=execution.checkpointer,
        **kwargs,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(execution: ChatExecution):
    """处理用户的聊天请求，返回排障结果。

    Args:
        execution: 已校验会话归属并取得本轮执行权的聊天请求。

    Returns:
        排障结果的字典表示。
    """
    return await _build_graph(execution).ainvoke()


@router.post("/chat/stream", response_class=EventSourceResponse)
async def stream_chat(execution: ChatExecution) -> AsyncIterator[ServerSentEvent]:
    """实时聊天或故障排除的进展及其最终结果。"""

    try:
        async with aclosing(_build_graph(execution).astream()) as events:
            async for event, data in events:
                if event == "progress":
                    yield ServerSentEvent(event=event, data={"stage": data})
                else:
                    yield ServerSentEvent(event=event, data=ChatResponse.model_validate(data))
    except Exception:
        logger.exception("Chat stream failed")
        yield ServerSentEvent(event="error", data={"message": "处理请求失败"})
