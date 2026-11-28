from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from aiops_agent.graph_components.states.state import DiagnosisResult, Intent


class SessionResponse(BaseModel):
    session_id: str = Field(description="后端生成的会话 ID。")
    user_id: str = Field(description="会话所属的用户 ID。")
    created_at: datetime = Field(description="会话创建时间。")


class SessionMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(description="消息角色。")
    content: str = Field(description="消息正文；排障回复的正文为诊断摘要。")
    created_at: datetime = Field(description="本轮消息保存时间。")
    intent: Intent | None = Field(default=None, description="助手回复的分类。")
    result: DiagnosisResult | None = Field(default=None, description="助手回复中的结构化排障结果。")
