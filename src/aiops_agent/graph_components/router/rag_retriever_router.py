"""RAG 子图的调度路由。"""

from aiops_agent.graph_components.states.state import RagState


def schedule_router(state: RagState) -> str:
    """根据调度节点写入的 next_action 决定下一步走哪个节点。

    调度节点已经完成了“选哪个子问题”的判断，这里只负责把结果翻译
    成 LangGraph 的条件边可以识别的节点名称，不再做复杂逻辑。

    Args:
        state: RAG 子图当前状态，由 schedule_node 写入 next_action。

    Returns:
        "optimize"（做 SDOM 改写）、"retrieve"（直接检索）或
        "aggregate"（所有子问题都处理完了）。

    Raises:
        ValueError: next_action 缺失或不是预期值，说明调度节点没有正常工作。
    """
    action = state.get("next_action")
    if action not in ("optimize", "retrieve", "aggregate"):
        raise ValueError(f"未知的调度动作: {action!r}")
    return action
