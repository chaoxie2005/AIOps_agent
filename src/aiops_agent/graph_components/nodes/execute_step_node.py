"""将单步骤执行子图的输出转换为主图状态更新。"""

from aiops_agent.graph_components.graph.execute_subgraph import execute_subgraph
from aiops_agent.graph_components.nodes.execute_node import ExecuteNodes
from aiops_agent.graph_components.states.state import OverviewState


async def execute_step_node(state: OverviewState, nodes: ExecuteNodes | None = None) -> dict:
    """执行第一个待执行步骤，只回传本步骤新增的结果和证据。"""
    if len(state.get("step_results", [])) >= state.get("max_steps", 8):
        raise ValueError("执行步骤预算已耗尽")
    step = next((item for item in state.get("steps", []) if item["status"] == "pending"), None)
    if step is None:
        raise ValueError("没有待执行的步骤")
    output = await execute_subgraph(
        {"steps": state.get("steps", []), "current_step_id": step["id"]}, nodes=nodes,
    )
    step_result = output.get("step_result")
    if step_result is None:
        raise ValueError("执行子图未返回步骤结果")
    return {
        "steps": [
            {**item, "status": step_result["status"]} if item["id"] == step["id"] else item
            for item in state.get("steps", [])
        ],
        "current_step_id": None,
        "step_results": [step_result],
        "evidence": output.get("evidence_candidates", []),
    }
