import asyncio

from aiops_agent.graph_components.graph.rag_subgraph import rag_subgraph



def test_rag_subgraph():
    """测试 RAG 子图的功能."""
    result = asyncio.run(rag_subgraph("数据库故障怎么办"))
    assert "unique_documents" in result
    assert len(result["unique_documents"]) > 0