"""测试重新规划节点。"""

import asyncio

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.nodes.replan_node import replan_node
from aiops_agent.graph_components.states.state import OverviewState


def test_replan_node():
    model_client = ModelClient(
        model_name="Qwen/Qwen3-0.6B",
        model_provider="openai",
        base_url="http://127.0.0.1:8002/v1",
        api_key="EMPTY",
        max_tokens=2048,
        temperature=0,
    )
    state: OverviewState = {
        "query": "CPU突然异常高，导致系统响应缓慢",
        "steps": [
            {"id": "s1", "goal": "检查 CPU 使用率", "status": "completed"},
            {"id": "s2", "goal": "检查 CPU 占用较高的进程", "status": "pending"},
        ],
        "step_results": [{
            "step_id": "s1", "status": "completed",
            "summary": "测试场景：已确认 CPU 使用率达到 95%，尚未定位原因。",
            "evidence_ids": [], "knowledge_document_ids": [],
            "tool_call_ids": [], "error": None,
        }],
        "max_steps": 3,
    }
    output = asyncio.run(replan_node(state, model_client))
    print(output)
    assert set(output) == {"steps", "result"}
    assert isinstance(output["steps"], list)
    if output["result"] is None:
        assert output["steps"]
    else:
        assert isinstance(output["result"], dict)
        assert output["steps"] == []
