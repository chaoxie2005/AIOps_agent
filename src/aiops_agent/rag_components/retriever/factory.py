import os
from pathlib import Path

from dotenv import load_dotenv

from aiops_agent.rag_components.embedding import EmbeddingModel
from aiops_agent.rag_components.milvus import AMilvusClient
from aiops_agent.rag_components.rerank import RerankModel
from aiops_agent.rag_components.retriever.retriever import MilvusRetriever

_DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_DEFAULT_COLLECTION = "camel_agent"


def _env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _required_env(name: str) -> str:
    value = _env(name)
    if value:
        return value
    raise ValueError(f"缺少必需的环境变量: {name}")


def build_embedding_from_env() -> EmbeddingModel:
    """为入库和检索使用同一套嵌入配置，优先沿用已有知识库的模型。"""
    load_dotenv(_DOTENV_PATH)
    model_name = _env("MODEL_EMBEDDING_NAME", "MODEL_NAME")
    base_url = _env("BASE_URL")
    api_key = _env("MODEL_API_KEY")
    if model_name and base_url and api_key:
        return EmbeddingModel(model_name=model_name, base_url=base_url, key=api_key)

    return EmbeddingModel(
        model_name=_env("EMBEDDING_MODEL") or "text-embedding-3-small",
        base_url=_env("EMBEDDING_BASE_URL") or "https://api.openai.com/v1",
        key=_required_env("EMBEDDING_API_KEY"),
    )


def get_milvus_collection_name() -> str:
    """返回入库与聊天检索共用的默认集合。"""
    return _env("MILVUS_COLLECTION_NAME") or _DEFAULT_COLLECTION


def build_milvus_client_from_env(milvus_url: str | None = None) -> AMilvusClient:
    """为入库与检索使用同一套 Milvus 连接配置。"""
    return AMilvusClient(
        url=milvus_url or _env("MILVUS_URL") or "http://localhost:19530",
        user=_env("MILVUS_USER"),
        password=_env("MILVUS_PASSWORD"),
    )


def build_retriever_from_env(
    milvus_url: str | None = None,
    collection_name: str | None = None,
    score_threshold: float = 0.3,
) -> MilvusRetriever:
    """从 .env 配置构造 MilvusRetriever"""
    embedding = build_embedding_from_env()
    base_url = _required_env("BASE_URL")
    api_key = _required_env("MODEL_API_KEY")
    rerank_name = _required_env("RERANK_MODEL_NAME")
    rerank = RerankModel(
        model_name=rerank_name,
        base_url=base_url,
        key=api_key,
        endpoint=os.getenv("RERANK_BASE_URL"),
    )
    client = build_milvus_client_from_env(milvus_url)
    return MilvusRetriever(
        client=client,
        embedding_function=embedding,
        rerank_function=rerank,
        collection_name=collection_name or get_milvus_collection_name(),
        score_threshold=score_threshold,
    )
