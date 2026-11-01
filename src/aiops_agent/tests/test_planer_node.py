"""测试规划节点。"""

import asyncio

from aiops_agent.graph_components.nodes.plan_node import plan_node
from aiops_agent.graph_components.states.state import OverviewState
from aiops_agent.agent_components.llm import ModelClient


def test_plan_node():
    model_client = ModelClient(
        model_name="Qwen/Qwen3-0.6B",
        model_provider="openai",
        base_url="http://127.0.0.1:8002/v1",
        api_key="EMPTY",
        max_tokens=512,
    )
    state: OverviewState = {
        "knowledge_documents": [],
        "query": "CPU突然异常高，导致系统响应缓慢",
        "max_steps": 5,
    }
    result = asyncio.run(plan_node(state, model_client))
    print(result)
    assert "steps" in result
    assert isinstance(result["steps"], list) and result["steps"]
