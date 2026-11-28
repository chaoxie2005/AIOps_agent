"""准备步骤节点的最小测试，不依赖外部服务。"""

import asyncio

from aiops_agent.graph_components.nodes.execute_node import prepare_step
from aiops_agent.graph_components.states.state import ExecutorInput


def test_prepare_step():
    """按 ID 选中步骤，初始化执行状态，并保留原始计划。"""
    state: ExecutorInput = {
        "steps": [
            {"id": "s1", "goal": "检查内存", "status": "pending"},
            {"id": "s2", "goal": "检查 CPU", "status": "pending"},
        ],
        "current_step_id": "s2",
    }
    result = asyncio.run(prepare_step(state, max_tool_calls=2))
    print(result)
    assert result["current_step"] == {"id": "s2", "goal": "检查 CPU", "status": "running"}
    assert state["steps"][1]["status"] == "pending"
    assert result["max_tool_calls"] == 2
    assert result["step_result"] is None
    for key in ("tool_calls", "tool_results", "evidence_candidates", "knowledge_candidates"):
        assert result[key] == []
