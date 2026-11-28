"""子问题拆解节点集成测试：.venv/bin/python -m pytest src/aiops_agent/tests/test_subquery_agent_manual.py。"""
import asyncio

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.nodes.rag_nodes import subquery_node
from aiops_agent.graph_components.states.state import RagState


def test_subquery_agent_node():
    """使用统一环境配置的 LLM，返回去重后的子问题列表。"""
    model = ModelClient()
    state: RagState = {
        "query": "CPU突然异常高，导致系统响应缓慢",
        "max_subqueries": 3,
    }
    result = asyncio.run(subquery_node(state, model))
    assert "unique_subqueries" in result
    assert len(result["unique_subqueries"]) > 0
    assert len(result["unique_subqueries"]) <= 3
    assert len(result["unique_subqueries"]) == len(set(result["unique_subqueries"]))
