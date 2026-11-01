"""意图识别后的聊天与故障排查路由。"""

from aiops_agent.graph_components.states.state import OverviewState


def intent_router(state: OverviewState) -> str:
    """根据经过校验的意图选择主图分支。

    Args:
        state: 包含 intent 的主状态。

    Returns:
        普通聊天返回 chat，故障排查返回 rag。

    Raises:
        ValueError: 意图缺失或不属于支持的分类。
    """
    intent = state.get("intent")
    if intent == "chat":
        return "chat"
    if intent == "aiops":
        return "rag"
    raise ValueError("意图必须为 chat 或 aiops")
