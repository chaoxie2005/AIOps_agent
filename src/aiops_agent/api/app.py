from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pymongo.errors import PyMongoError

from aiops_agent.api.chat import router
from aiops_agent.api.load_milvus import router as load_milvus_router
from aiops_agent.api.sessions import router as sessions_router
from aiops_agent.service.chat_sessions import ChatSessions, SessionBusy, SessionError, SessionNotFound


@asynccontextmanager
async def lifespan(app: FastAPI):
    sessions = ChatSessions()
    app.state.chat_sessions = sessions
    try:
        yield
    finally:
        await sessions.aclose()

app = FastAPI(
    title="AIops Agent API",
    description="提供普通聊天和故障排查接口。",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router, tags=["聊天与排障"])
app.include_router(sessions_router, tags=["聊天会话"])
app.include_router(load_milvus_router, tags=["向量入库"])


@app.exception_handler(SessionError)
async def session_error(request: Request, exc: SessionError) -> JSONResponse:
    status = 404 if isinstance(exc, SessionNotFound) else 409 if isinstance(exc, SessionBusy) else 503
    return JSONResponse(status_code=status, content={"detail": str(exc)})


@app.exception_handler(PyMongoError)
async def mongo_error(request: Request, exc: PyMongoError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "会话存储暂不可用"})


@app.get("/health", tags=["健康检查"])
async def health() -> dict[str, str]:
    """确认 HTTP 服务能正常响应。"""
    return {"status": "ok"}
