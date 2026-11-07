from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    query: str
    max_steps: int | None = Field(default=None, ge=0)
