"""RAG 节点容器：持有模型和检索器，并把图的节点调用转发到各实现文件。"""

import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.nodes.rag.retrieval_node import aggregation_node, retriever_node
from aiops_agent.graph_components.nodes.rag.schedule_node import record_node, schedule_node
from aiops_agent.graph_components.nodes.rag.sdom_node import dependency_optimize_node
from aiops_agent.graph_components.nodes.rag.subquery_node import subquery_node
from aiops_agent.graph_components.states.state import RagInput, RagState
from aiops_agent.rag_components.retriever.factory import build_retriever_from_env
from aiops_agent.rag_components.retriever.retriever import MilvusRetriever


class RagNodes:
    """RAG 子图节点的依赖容器：持有模型和检索器，整个子图只建一次。"""

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

    def schedule(self, state: RagState) -> dict:
        """调度节点包装：挑选下一个要处理的子问题。"""
        return schedule_node(state)

    async def optimize(self, state: RagState) -> dict:
        """依赖优化节点包装：用上游信息改写当前子问题。"""
        return await dependency_optimize_node(state, self.model_client)

    async def retrieve(self, state: RagState) -> dict:
        """检索节点包装：检索当前子问题。"""
        return await retriever_node(state, self.retriever)

    def record(self, state: RagState) -> dict:
        """记录节点包装：把当前子问题标记为完成。"""
        return record_node(state)

    async def aggregate(self, state: RagState) -> dict:
        """聚合节点包装：按父文档去重。"""
        return await aggregation_node(state)
