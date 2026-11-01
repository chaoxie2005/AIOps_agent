"""rag检索tool节点集成测试：.venv/bin/python -m pytest src/aiops_agent/tests/test_rag_retriever_manual.py。"""
import asyncio

from aiops_agent.graph_components.nodes.rag_retrierver import retriever_node
from aiops_agent.graph_components.states.state import RagState


def test_retriever_node():
    """依赖本地 Milvus（localhost:19530），返回 retrieval_hits 列表。"""
    state: RagState = {
        "query": "CPU使用率过高如何排查？",
        "unique_subqueries": ["CPU使用率过高如何排查？", "mysql数据库性能问题"],
    }
    result = asyncio.run(retriever_node(state))
    assert "retrieval_hits" in result
    assert isinstance(result["retrieval_hits"], list)
