from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    user_id: str # 用户 ID
    session_id: str # 会话 ID
    query: str # 用户问题
    max_steps: int | None = Field(default=None, ge=0) # 最大执行步骤数