"""rag检索tool节点集成测试：.venv/bin/python -m pytest src/aiops_agent/tests/test_rag_retriever_manual.py。"""
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from aiops_agent.graph_components.nodes.rag_nodes import retriever_node
from aiops_agent.graph_components.states.state import RagState
from aiops_agent.rag_components.retriever.factory import build_retriever_from_env


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def test_retriever_node():
    """依赖本地 Milvus（localhost:19530），返回 retrieval_hits 列表。"""
    state: RagState = {
        "query": "CPU使用率过高如何排查？",
        "unique_subqueries": ["CPU使用率过高如何排查？", "mysql数据库性能问题"],
    }
    retriever = build_retriever_from_env(
        milvus_url="http://localhost:19530",
        collection_name="camel_agent",
        score_threshold=0.3,
    )
    result = asyncio.run(retriever_node(state, retriever))
    assert "retrieval_hits" in result
    assert isinstance(result["retrieval_hits"], list)
