from pydantic import BaseModel

from aiops_agent.graph_components.states.state import DiagnosisResult, Intent


class ChatResponse(BaseModel):
    intent: Intent
    reply: str | None
    result: DiagnosisResult | None
