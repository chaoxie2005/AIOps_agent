from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    user_id: str = Field(min_length=1, description="新会话所属的用户 ID。")
