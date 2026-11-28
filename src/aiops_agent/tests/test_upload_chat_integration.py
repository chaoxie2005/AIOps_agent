import asyncio
import importlib
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from aiops_agent.api.app import app
from aiops_agent.graph_components.nodes import rag_nodes
from aiops_agent.rag_components.milvus import AMilvusClient
from aiops_agent.rag_components.retriever import factory
from aiops_agent.tests.test_incremental_ingestion import MemoryMilvus, MemoryMinio

upload = importlib.import_module("aiops_agent.api.load_milvus")


class SearchableMilvus(MemoryMilvus, AMilvusClient):
    async def hybrid_search(self, *, collection_name, **kwargs):
        return [[
            {
                "distance": 1.0,
                "entity": {
                    "child_content": row["child_content"],
                    "child_metadata": row["child_metadata"],
                },
            }
            for row in self.rows[collection_name].values()
        ]]


@pytest.mark.parametrize(
    ("configured_collection", "expected_collection"),
    [(None, "camel_agent"), ("ops_docs", "ops_docs")],
)
def test_uploaded_document_reaches_chat_with_parent_context(
    monkeypatch, request, configured_collection, expected_collection
):
    minio, milvus = MemoryMinio(), SearchableMilvus()
    embedding_batches = []

    class TestEmbedding(DeterministicFakeEmbedding):
        model_name: str
        base_url: str

        async def aembed_documents(self, texts):
            embedding_batches.append(texts)
            return await super().aembed_documents(texts)

    embedding_factory = Mock(side_effect=lambda model_name, base_url, key: TestEmbedding(
        size=4, model_name=model_name, base_url=base_url,
    ))
    milvus_factory = Mock(return_value=milvus)
    monkeypatch.setattr(factory, "EmbeddingModel", embedding_factory)
    monkeypatch.setattr(factory, "AMilvusClient", milvus_factory)
    monkeypatch.setattr(factory, "RerankModel", lambda **kwargs: None)
    monkeypatch.setattr(factory, "load_dotenv", lambda path: None)
    monkeypatch.setattr(rag_nodes, "load_dotenv", lambda path: None)
    monkeypatch.setattr(rag_nodes, "ModelClient", Mock())
    monkeypatch.setattr(upload, "MinioClient", Mock(return_value=minio))
    for key, value in {
        "MILVUS_URL": "http://milvus.example:19530",
        "MILVUS_USER": "offline-user",
        "MILVUS_PASSWORD": "offline-password",
        "MODEL_EMBEDDING_NAME": "existing-kb-embedding",
        "BASE_URL": "https://kb.example/v1",
        "MODEL_API_KEY": "offline-key",
        "EMBEDDING_MODEL": "old-upload-embedding",
        "EMBEDDING_API_KEY": "old-upload-key",
        "RERANK_MODEL_NAME": "offline-rerank",
        "LLM_BASE_URL": "https://llm.example/v1",
        "NO_PROXY": "",
        "no_proxy": "",
    }.items():
        monkeypatch.setenv(key, value)
    if configured_collection is None:
        monkeypatch.delenv("MILVUS_COLLECTION_NAME", raising=False)
    else:
        monkeypatch.setenv("MILVUS_COLLECTION_NAME", configured_collection)
    upload._get_ingestion_components.cache_clear()
    request.addfinalizer(upload._get_ingestion_components.cache_clear)

    file = {"file": ("guide.md", b"# Operations\n\nCheck node health.", "text/markdown")}
    with TestClient(app) as client:
        with monkeypatch.context() as previous_release:
            previous_release.setattr(upload, "_PIPELINE_REVISION", "1")
            first = client.post("/load_milvus", files=file)
        assert first.status_code == 200 and first.json()["status"] == "created", first.text
        assert set(milvus.rows) == {expected_collection}
        old_row = next(iter(milvus.rows[expected_collection].values()))
        old_row["child_metadata"].pop("parent_content")

        upgraded = client.post("/load_milvus", files=file)
        assert upgraded.status_code == 200 and upgraded.json()["status"] == "updated", upgraded.text
        repeat = client.post("/load_milvus", files=file)
        assert repeat.status_code == 200 and repeat.json()["status"] == "unchanged", repeat.text
        assert len(embedding_batches) == 1

        alternate = client.post(
            "/load_milvus", files=file, data={"collection_name": "separate_docs"}
        )
        assert alternate.status_code == 200 and set(milvus.rows["separate_docs"])

    nodes = rag_nodes.RagNodes.from_env()
    assert nodes.retriever.collection_name == expected_collection

    async def retrieve_context():
        hits = await nodes.retrieve({"query": "node health"})
        assert len(hits["retrieval_hits"]) == 1
        return await nodes.aggregate(hits)

    context = asyncio.run(retrieve_context())
    row = next(iter(milvus.rows[expected_collection].values()))
    assert row["child_chunk_id"] == old_row["child_chunk_id"]
    assert row["dense_vector"] == old_row["dense_vector"]
    parent_content = row["child_metadata"]["parent_content"]
    assert "Check node health." in parent_content
    assert context == {"unique_documents": [parent_content]}
    assert [call.kwargs for call in embedding_factory.call_args_list] == [
        {
            "model_name": "existing-kb-embedding",
            "base_url": "https://kb.example/v1",
            "key": "offline-key",
        },
    ] * 2
    assert [call.kwargs for call in milvus_factory.call_args_list] == [
        {
            "url": "http://milvus.example:19530",
            "user": "offline-user",
            "password": "offline-password",
        },
    ] * 2


def test_aggregation_uses_legacy_child_text_and_prefers_full_parent_when_available():
    documents = [
        Document(page_content="Legacy child", metadata={"parent_id": "legacy"}),
        Document(page_content="Another child", metadata={"parent_id": "legacy"}),
        Document(page_content="Partial excerpt", metadata={"parent_id": "current"}),
        Document(
            page_content="Current child",
            metadata={"parent_id": "current", "parent_content": "Full parent text"},
        ),
        Document(page_content="Legacy without parent metadata"),
    ]
    state = {"retrieval_hits": [{"document": doc, "subquery": "node health"} for doc in documents]}
    assert asyncio.run(rag_nodes.aggregation_node(state)) == {
        "unique_documents": ["Legacy child", "Full parent text", "Legacy without parent metadata"],
    }
