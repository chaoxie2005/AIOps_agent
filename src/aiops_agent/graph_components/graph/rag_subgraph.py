from pathlib import Path
import asyncio
import dotenv
from langgraph.graph import StateGraph

from aiops_agent.graph_components.nodes.rag_nodes import RagNodes
from aiops_agent.graph_components.states.state import RagInput, RagOutput, RagState
from aiops_agent.graph_components.router.rag_retriever_router import rag_retriever_router


async def rag_subgraph(query: str, nodes: RagNodes | None = None):
    """RAG 子图；子查询拆解 -> 并行检索 -> 聚合."""
    nodes = nodes or RagNodes.from_env() # 从环境变量加载节点
    builder = StateGraph(RagState, input_schema=RagInput, output_schema=RagOutput)
    builder.add_node(nodes.subquery)
    builder.add_node(nodes.retrieve)
    builder.add_node(nodes.aggregate)

    builder.set_entry_point("subquery")
    builder.add_conditional_edges("subquery", rag_retriever_router, ["retrieve"])
    builder.add_edge("retrieve", "aggregate")

    rag_agent = builder.compile()
    return await rag_agent.ainvoke({"query": query})

