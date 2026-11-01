from pathlib import Path
import asyncio
import dotenv
from langgraph.graph import StateGraph

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.states.state import RagInput, RagOutput, RagState
from aiops_agent.graph_components.nodes.aggregation_node import aggregation_node
from aiops_agent.graph_components.nodes.rag_retrierver import retriever_node
from aiops_agent.graph_components.nodes.subquery_agent import subquery_agent_node
from aiops_agent.graph_components.router.rag_retriever_router import rag_retriever_router


dotenv.load_dotenv(Path(__file__).resolve().parents[2] / ".env")

model_client = ModelClient(
    model_name="Qwen/Qwen3-0.6B",
    model_provider="openai",
    base_url="http://127.0.0.1:8002/v1",
    api_key="EMPTY",
    max_tokens=512,
)


async def subquery_node(state: RagInput):
    """包装节点：把 model_client 闭包进去，LangGraph 只传 state。"""
    return await subquery_agent_node(state, model_client)


async def rag_subgraph(query: str):
    """RAG 子图；子查询拆解 -> 并行检索 -> 聚合."""
    builder = StateGraph(RagState, input_schema=RagInput, output_schema=RagOutput)
    builder.add_node(subquery_node)
    builder.add_node(retriever_node)
    builder.add_node(aggregation_node)

    builder.set_entry_point("subquery_node")
    builder.add_conditional_edges("subquery_node", rag_retriever_router, ["retriever_node"])
    builder.add_edge("retriever_node", "aggregation_node")

    rag_agent = builder.compile()
    return await rag_agent.ainvoke({"query": query})


async def main():
    result = await rag_subgraph("数据库故障怎么办")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
