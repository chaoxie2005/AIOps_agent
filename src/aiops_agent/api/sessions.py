from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from aiops_agent.schema.resquest.session import CreateSessionRequest
from aiops_agent.schema.response.session import SessionMessage, SessionResponse
from aiops_agent.service.chat_sessions import ChatSessions

router = APIRouter(prefix="/sessions")


def get_chat_sessions(request: Request) -> ChatSessions:
    """从 FastAPI 应用状态获取共用的会话服务。

    Args:
        request: 当前 HTTP 请求，用于定位处理它的应用实例。

    Returns:
        应用启动时创建的会话服务。
    """
    return request.app.state.chat_sessions


Sessions = Annotated[ChatSessions, Depends(get_chat_sessions)]
UserId = Annotated[str, Query(min_length=1, description="会话所属的用户 ID。")]


@router.post("", status_code=201, response_model=SessionResponse)
async def create_session(request: CreateSessionRequest, sessions: Sessions):
    """创建一个空会话，供后续普通或流式聊天使用。

    Args:
        request: 包含会话所属用户 ID 的请求体。
        sessions: FastAPI 注入的会话服务。

    Returns:
        新会话的 ID、所属用户和创建时间。
    """
    return await sessions.create(request.user_id)


@router.get("", response_model=list[SessionResponse])
async def list_sessions(user_id: UserId, sessions: Sessions):
    """列出指定用户的会话，最新创建的排在前面。

    Args:
        user_id: 查询参数中的用户 ID。
        sessions: FastAPI 注入的会话服务。

    Returns:
        供前端侧栏显示的会话摘要列表，包含空会话。
    """
    return await sessions.list_for_user(user_id)


@router.get("/{session_id}/messages", response_model=list[SessionMessage])
async def session_messages(session_id: str, user_id: UserId, sessions: Sessions):
    """读取会话历史，将每轮问答拆成用户消息和助手消息。

    排障回复用诊断摘要作为正文，并保留完整结构化结果。

    Args:
        session_id: 要打开的会话 ID。
        user_id: 用于核对会话归属的查询参数。
        sessions: FastAPI 注入的会话服务。

    Returns:
        按对话顺序排列的消息，空会话返回空列表。

    Raises:
        SessionNotFound: 会话不存在或用户 ID 不匹配，接口返回 404。
    """
    messages = []
    for turn in await sessions.read_turns(session_id, user_id):
        result = turn["result"]
        messages.extend([
            SessionMessage(role="user", content=turn["query"], created_at=turn["created_at"]),
            SessionMessage(
                role="assistant",
                content=turn["reply"] or (result["summary"] if result is not None else ""),
                created_at=turn["created_at"],
                intent=turn["intent"],
                result=result,
            ),
        ])
    return messages
