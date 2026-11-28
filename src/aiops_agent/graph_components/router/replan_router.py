"""重新规划后的继续执行与结束路由。"""

from langgraph.graph import END

from aiops_agent.graph_components.states.state import OverviewState


def replan_router(state: OverviewState) -> str:
    """决定进入下一步执行还是结束图。

    无诊断结果、预算耗尽或没有待执行步骤时都会结束图，
    由调用方负责补全降级输出。

    Args:
        state: replan 节点处理后的主图状态。

    Returns:
        图应结束时返回 END，否则返回 execute_step。
    """
    if state.get("result") is not None:
        return END
    if len(state.get("step_results", [])) >= state.get("max_steps", 8):
        return END
    if not any(step["status"] == "pending" for step in state.get("steps", [])):
        return END
    return "execute_step"
