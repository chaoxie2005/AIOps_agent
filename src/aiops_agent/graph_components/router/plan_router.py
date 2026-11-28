"""初始规划后的执行与收尾路由。"""

from aiops_agent.graph_components.states.state import OverviewState


def plan_router(state: OverviewState) -> str:
    """选择执行步骤或进入诊断收尾。

    Args:
        state: 包含当前计划和步骤预算的主状态。

    Returns:
        有可执行步骤和预算时返回 execute_step，否则返回 replan。
    """
    if len(state.get("step_results", [])) < state.get("max_steps", 8):
        if any(step["status"] == "pending" for step in state.get("steps", [])):
            return "execute_step"
    return "replan"
