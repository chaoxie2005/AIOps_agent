"""RAG 子图的三个节点：子查询拆解、检索、聚合并按父文档去重。

节点是普通 async 函数，签名保持 (state, 依赖) 或 (state)，便于单测；
RagNodes 只负责持有依赖并在图里转发，不承载业务逻辑。
"""
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain_core.documents import Document
from loguru import logger

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.states.state import RagInput, RagState, RetrievalHit
from aiops_agent.prompts.load_texts.load_subquery_prompt import subquery_agent_prompt
from aiops_agent.rag_components.retriever.factory import build_retriever_from_env
from aiops_agent.rag_components.retriever.retriever import MilvusRetriever


async def subquery_node(state: RagInput, model_client: ModelClient) -> dict:
    """生成子问题，并返回供图合并的状态更新。"""
    messages = subquery_agent_prompt.format_messages(
        max_sub_questions=state.get("max_subqueries", 4),
        question=state["query"],
    )
    response = await model_client.ainvoke(messages)
    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    raw_content = json.loads(match.group())
    logger.info(f"子问题拆解agent生成的原始内容: {raw_content}")
    sub_questions = [item["question"] for item in raw_content["sub_questions"]]
    logger.info(f"子问题拆解agent生成的子问题: {sub_questions}")
    unique_subqueries = list(dict.fromkeys(sub_questions))
    logger.info(f"子问题拆解agent生成的去重子问题: {unique_subqueries}")
    return {"unique_subqueries": unique_subqueries}


async def retriever_node(state: RagState, retriever: MilvusRetriever) -> dict:
    """处理单个子问题：检索并按子查询记录召回结果。"""
    query = state["query"]
    docs = await retriever.ainvoke(query)
    logger.info(f"检索工具检索的子问题：{query}")
    for doc in docs:
        logger.info(f"检索工具检索的文档：{doc}")
    return {"retrieval_hits": [RetrievalHit(subquery=query, document=doc) for doc in docs]}


async def aggregation_node(state: RagState) -> dict:
    """按父块去重，旧记录缺少父块内容时回退到子块正文。"""
    selected: dict[tuple[str, str], tuple[str, bool]] = {}
    for hit in state.get("retrieval_hits", []):
        doc = hit["document"]
        parent_id = doc.metadata.get("parent_id")
        parent_content = doc.metadata.get("parent_content")
        has_parent_content = isinstance(parent_content, str) and bool(parent_content.strip())
        key = (
            ("parent", parent_id)
            if isinstance(parent_id, str) and parent_id
            else ("text", doc.page_content)
        )
        previous = selected.get(key)
        if previous is None or (has_parent_content and not previous[1]):
            selected[key] = (
                parent_content if has_parent_content else doc.page_content,
                has_parent_content,
            )
    return {"unique_documents": [content for content, _ in selected.values()]}


class RagNodes:
    """RAG 子图节点依赖容器：持有 model_client 与 retriever，整个子图只建一次。"""

    def __init__(self, model_client: ModelClient, retriever: MilvusRetriever) -> None:
        self.model_client = model_client
        self.retriever = retriever

    @classmethod
    def from_env(cls) -> "RagNodes":
        """按 .env 配置构造模型客户端与检索器。"""
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
        address = urlsplit(base_url)
        if address.scheme not in ("http", "https") or not address.hostname:
            raise ValueError("LLM_BASE_URL 必须是包含 http 或 https 的模型服务地址")
        # 仅让模型主机绕过代理，保留已有的代理绕过名单。
        for key in ("NO_PROXY", "no_proxy"):
            hosts = [host.strip() for host in os.getenv(key, "").split(",") if host.strip()]
            if address.hostname not in hosts:
                hosts.append(address.hostname)
            os.environ[key] = ",".join(hosts)
        model_client = ModelClient()
        retriever = build_retriever_from_env()
        return cls(model_client=model_client, retriever=retriever)

    async def subquery(self, state: RagInput) -> dict:
        """子查询拆解节点。"""
        return await subquery_node(state, self.model_client)

    async def retrieve(self, state: RagState) -> dict:
        """检索节点，由 router 以 Send 并行调用。"""
        return await retriever_node(state, self.retriever)

    async def aggregate(self, state: RagState) -> dict:
        """聚合去重节点。"""
        return await aggregation_node(state)


__all__ = ["RagNodes", "aggregation_node", "retriever_node", "subquery_node"]
