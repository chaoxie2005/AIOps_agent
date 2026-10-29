import os

from aiops_agent.rag_components.embedding import EmbeddingModel
from aiops_agent.rag_components.rerank import RerankModel
from aiops_agent.rag_components.retriever.retriever import MilvusRetriever
from aiops_agent.rag_components.milvus import AMilvusClient


def _required_env(name: str, *aliases: str) -> str:
    for key in (name, *aliases):
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    raise ValueError(f"缺少必需的环境变量: {name}")


def build_embedding_from_env() -> EmbeddingModel:
    """入库与检索共用的向量模型配置；优先使用 MODEL_EMBEDDING_NAME。"""
    return EmbeddingModel(
        model_name=_required_env("MODEL_EMBEDDING_NAME", "MODEL_NAME"),
        base_url=_required_env("BASE_URL"),
        key=_required_env("MODEL_API_KEY"),
    )


def build_retriever_from_env(
    milvus_url: str = "http://localhost:19530",
    collection_name: str = "aiops_agent",
    score_threshold: float = 0.3,
) -> MilvusRetriever:
    """从 .env 配置构造 MilvusRetriever"""
    # 在创建网络客户端之前一次性校验必需配置。
    _required_env("MODEL_EMBEDDING_NAME", "MODEL_NAME")
    base_url = _required_env("BASE_URL")
    api_key = _required_env("MODEL_API_KEY")
    rerank_name = _required_env("RERANK_MODEL_NAME")
    embedding = build_embedding_from_env()
    rerank = RerankModel(
        model_name=rerank_name,
        base_url=base_url,
        key=api_key,
        endpoint=os.getenv("RERANK_BASE_URL"),
    )
    client = AMilvusClient(url=milvus_url)
    return MilvusRetriever(
        client=client,
        embedding_function=embedding,
        rerank_function=rerank,
        collection_name=collection_name,
        score_threshold=score_threshold,
    )
