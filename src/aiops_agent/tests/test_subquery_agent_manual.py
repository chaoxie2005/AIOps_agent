"""子问题拆解节点集成测试：.venv/bin/python -m pytest src/aiops_agent/tests/test_subquery_agent_manual.py。"""
import asyncio

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.nodes.subquery_agent import subquery_agent_node
from aiops_agent.graph_components.states.state import RagState


def test_subquery_agent_node():
    """依赖本地 LLM（127.0.0.1:8002），返回去重后的子问题列表。"""
    model = ModelClient(
        model_name="Qwen/Qwen3-0.6B",
        model_provider="openai",
        base_url="http://127.0.0.1:8002/v1",
        api_key="EMPTY",
        max_tokens=512,
    )
    state: RagState = {
        "query": "CPU突然异常高，导致系统响应缓慢",
        "max_subqueries": 3,
    }
    result = asyncio.run(subquery_agent_node(state, model))
    assert "unique_subqueries" in result
    assert len(result["unique_subqueries"]) > 0
    assert len(result["unique_subqueries"]) <= 3
    assert len(result["unique_subqueries"]) == len(set(result["unique_subqueries"]))
