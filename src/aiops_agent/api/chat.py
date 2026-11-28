from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from logging import getLogger
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langgraph.checkpoint.base import BaseCheckpointSaver

from aiops_agent.agent_components.memory.ingestion import TurnIngestion
from aiops_agent.api.sessions import Sessions
from aiops_agent.graph_components.graph.main_graph import MainGraph
from aiops_agent.schema.request.chat import ChatRequest
from aiops_agent.schema.response.chat import ChatResponse

router = APIRouter()
logger = getLogger(__name__)


@dataclass(frozen=True)
class _ChatExecution:
    request: ChatRequest # 本轮聊天请求
    checkpointer: BaseCheckpointSaver # 会话 checkpoint 存储器
    ingestion: TurnIngestion # 本轮聊天记录采集器


async def _claim_session(request: ChatRequest, sessions: Sessions) -> AsyncIterator[_ChatExecution]:
    """占用当前会话并归档本轮用户输入。

    Args:
        request: 包含会话 ID、用户 ID 和输入的聊天请求。
        sessions: 用于校验归属并取得执行权的会话服务。

    Yields:
        本轮请求、checkpointer 和会话事件采集器。

    Raises:
        SessionNotFound: 会话不存在，或记录的用户 ID 不匹配。
        SessionBusy: 同一进程中已有请求占用此会话。
        PyMongoError: 读取会话或归档用户输入失败。
    """
    async with sessions.claim(request.session_id, request.user_id) as (checkpointer, ingestion):
        await ingestion.record("user", request.query)
        yield _ChatExecution(request, checkpointer, ingestion)


ChatExecution = Annotated[_ChatExecution, Depends(_claim_session, scope="request")]


def _build_graph(execution: _ChatExecution) -> MainGraph:
    """使用本轮请求构建聊天与排障主图。

    Args:
        execution: 包含本轮请求、checkpointer 和事件采集器的执行上下文。

    Returns:
        使用会话 ID 作为 LangGraph thread_id 的主图。
    """
    request = execution.request
    kwargs = {"max_steps": request.max_steps} if request.max_steps is not None else {}
    return MainGraph(
        query=request.query,
        thread_id=request.session_id,
        checkpointer=execution.checkpointer,
        ingestion=execution.ingestion,
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
