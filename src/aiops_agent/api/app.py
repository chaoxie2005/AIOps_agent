from fastapi import FastAPI

from aiops_agent.api.chat import router
from aiops_agent.api.upload_file import router as upload_router

app = FastAPI(
    title="AIops Agent API",
    description="提供普通聊天和故障排查接口。",
    version="0.1.0",
)
app.include_router(router, tags=["聊天与排障"])
app.include_router(upload_router, tags=["文件上传"])


@app.get("/health", tags=["健康检查"])
async def health() -> dict[str, str]:
    """确认 HTTP 服务能正常响应。"""
    return {"status": "ok"}
