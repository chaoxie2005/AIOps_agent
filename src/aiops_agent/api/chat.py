from collections.abc import AsyncIterator
from contextlib import aclosing
from logging import getLogger

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent

from aiops_agent.graph_components.graph.main_graph import MainGraph
from aiops_agent.schema.resquest.chat import ChatRequest
from aiops_agent.schema.response.chat import ChatResponse

router = APIRouter()
logger = getLogger(__name__)


def _build_graph(request: ChatRequest) -> MainGraph:
    if request.max_steps is None:
        return MainGraph(query=request.query)
    return MainGraph(query=request.query, max_steps=request.max_steps)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理用户的聊天请求，返回排障结果。

    Args:
        request: 包含用户问题和会话信息的请求对象。

    Returns:
        排障结果的字典表示。
    """
    return await _build_graph(request).ainvoke()


@router.post("/chat/stream", response_class=EventSourceResponse)
async def stream_chat(request: ChatRequest) -> AsyncIterator[ServerSentEvent]:
    """实时聊天或故障排除的进展及其最终结果。"""

    try:
        async with aclosing(_build_graph(request).astream()) as events:
            async for event, data in events:
                if event == "progress":
                    yield ServerSentEvent(event=event, data={"stage": data})
                else:
                    yield ServerSentEvent(event=event, data=ChatResponse.model_validate(data))
    except Exception:
        logger.exception("Chat stream failed")
        yield ServerSentEvent(event="error", data={"message": "处理请求失败"})
