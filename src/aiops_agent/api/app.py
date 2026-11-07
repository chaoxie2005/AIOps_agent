from fastapi import FastAPI

from aiops_agent.api.chat import router
from aiops_agent.api.load_milvus import router as load_milvus_router

app = FastAPI(
    title="AIops Agent API",
    description="提供普通聊天和故障排查接口。",
    version="0.1.0",
)
app.include_router(router, tags=["聊天与排障"])
app.include_router(load_milvus_router, tags=["向量入库"])


@app.get("/health", tags=["健康检查"])
async def health() -> dict[str, str]:
    """确认 HTTP 服务能正常响应。"""
    return {"status": "ok"}
