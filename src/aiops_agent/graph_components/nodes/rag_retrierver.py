"""RAG 检索节点。"""
import asyncio
from loguru import logger
from dotenv import load_dotenv

from aiops_agent.graph_components.states.state import RagState, RetrievalHit
from aiops_agent.rag_components.retriever import build_retriever_from_env


load_dotenv()


# 节点：处理单个子问题
async def retriever_node(payload: dict):
    retriever = build_retriever_from_env(
        milvus_url="http://localhost:19530",
        collection_name="camel_agent",
        score_threshold=0.3,
    )
    query = payload["query"]
    docs = await retriever.ainvoke(query)
    logger.info(f"检索工具检索的子问题：{query}")
    for doc in docs:
        logger.info(f"检索工具检索的文档：{doc}")
    return {"retrieval_hits": [RetrievalHit(subquery=query, document=doc) for doc in docs]}
