"""重新规划后的继续执行与结束路由。"""

from langgraph.graph import END

from aiops_agent.graph_components.states.state import OverviewState


def replan_router(state: OverviewState) -> str:
    """判断结束诊断还是继续执行。

    Args:
        state: 重新规划后的主状态。

    Returns:
        已有诊断结果时返回 END，否则返回 execute_step。

    Raises:
        ValueError: 没有诊断结果且预算耗尽，或没有待执行步骤。
    """
    if state.get("result") is not None:
        return END
    if len(state.get("step_results", [])) >= state.get("max_steps", 8):
        raise ValueError("步骤预算已耗尽，但重新规划节点未返回诊断结果")
    if not any(step["status"] == "pending" for step in state.get("steps", [])):
        raise ValueError("重新规划节点未返回待执行步骤或诊断结果")
    return "execute_step"
