from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pymilvus import AnnSearchRequest, RRFRanker

from aiops_agent.rag_components.milvus import AMilvusClient
from aiops_agent.rag_components.rerank import RerankModel


class MilvusRetriever(BaseRetriever):
    """混合召回器：同步使用 invoke，异步使用 ainvoke。"""

    client: AMilvusClient
    embedding_function: Embeddings
    rerank_function: RerankModel | None = None
    collection_name: str = "aiops_agent"
    top_k: int = 3
    score_threshold: float | None = None
    partition_name: str = "_default"
    candidate_factor: int = 4

    def _search_options(self) -> dict:
        """构造各检索方式共用的 Milvus 查询参数。

        Returns:
            dict: 集合名称、分区列表和需要返回的子块字段。
        """
        return dict(
            collection_name=self.collection_name,
            partition_names=[self.partition_name],
            output_fields=["child_content", "child_metadata"],
        )

    def _hybrid_options(self, query: str, embedding: list[float]) -> dict:
        """构造稠密向量与 BM25 双路召回及 RRF 融合参数。

        Args:
            query: 用于 BM25 全文检索的查询文本。
            embedding: 查询文本对应的稠密向量。

        Returns:
            dict: 包含双路检索请求、RRF 排序器和公共查询配置的参数字典。
                每路召回及融合结果的候选数量为 top_k * candidate_factor。
        """
        candidate_k = self.top_k * self.candidate_factor
        return dict(
            **self._search_options(),
            requests=[
                AnnSearchRequest(data=[embedding], anns_field="dense_vector",
                                 param={"metric_type": "COSINE"}, limit=candidate_k),
                AnnSearchRequest(data=[query], anns_field="sparse_vector",
                                 param={"metric_type": "BM25"}, limit=candidate_k),
            ],
            ranker=RRFRanker(), top_k=candidate_k,
        )

    @staticmethod
    def _decode_hits(results) -> list[tuple[Document, float]]:
        """将 Milvus 返回的第一组查询结果转换为带分数的文档。

        Args:
            results: 按查询分组的 Milvus 命中列表。每条命中从 entity 中读取
                child_content 和 child_metadata，从 distance 中读取分数。

        Returns:
            list[tuple[Document, float]]: 保留命中顺序的文档与分数列表。
                缺失的内容、元数据和分数分别使用空字符串、空字典和 0.0；
                未命中时返回空列表。文档元数据使用浅拷贝。
        """
        docs = []
        for hit in results[0] if results else []:
            entity = hit.get("entity") or {}
            docs.append((Document(
                page_content=entity.get("child_content", ""),
                metadata=dict(entity.get("child_metadata") or {}),
            ), hit.get("distance", 0.0)))
        return docs

    def _filter(self, docs: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        """按配置的最低分数阈值筛选文档。

        Args:
            docs: 待筛选的文档与分数列表。

        Returns:
            list[tuple[Document, float]]: 分数不低于 score_threshold 的结果，
                保持原有顺序。未设置阈值时保留全部结果。
        """
        return [(doc, score) for doc, score in docs
                if self.score_threshold is None or score >= self.score_threshold]

    def embedding_search(self, query: str) -> list[tuple[Document, float]]:
        """同步执行纯向量检索，并按分数阈值过滤结果。

        Args:
            query: 待向量化并检索的查询文本。

        Returns:
            list[tuple[Document, float]]: 从最多 top_k 条召回结果中筛选出的
                文档及 Milvus 返回的向量检索分数。此路径不执行重排。
        """
        results = self.client.vector_search_sync(
            query=self.embedding_function.embed_query(query),
            top_k=self.top_k, **self._search_options(),
        )
        return self._filter(self._decode_hits(results))

    async def aembedding_search(self, query: str) -> list[tuple[Document, float]]:
        """异步执行纯向量检索，并按分数阈值过滤结果。

        Args:
            query: 待向量化并检索的查询文本。

        Returns:
            list[tuple[Document, float]]: 从最多 top_k 条召回结果中筛选出的
                文档及 Milvus 返回的向量检索分数。此路径不执行重排。
        """
        results = await self.client.vector_search(
            query=await self.embedding_function.aembed_query(query),
            top_k=self.top_k, **self._search_options(),
        )
        return self._filter(self._decode_hits(results))

    def hybrid_search(self, query: str) -> list[tuple[Document, float]]:
        """同步执行双路召回、RRF 融合及可选重排。

        Args:
            query: 用于稠密向量检索、BM25 检索和重排的查询文本。

        Returns:
            list[tuple[Document, float]]: 截取 top_k 并经阈值过滤后的文档与
                分数列表。配置重排模型时使用重排分数，否则使用 RRF 融合
                分数；无候选文档时返回空列表。
        """
        embedding = self.embedding_function.embed_query(query)
        candidates = self._decode_hits(self.client.hybrid_search_sync(
            **self._hybrid_options(query, embedding)
        ))
        if self.rerank_function is not None and candidates:
            ranked = self.rerank_function.rerank(
                query=query, documents=[doc.page_content for doc, _ in candidates],
                top_n=self.top_k,
            )
            docs = [(candidates[index][0], score) for index, score in ranked]
        else:
            docs = candidates[:self.top_k]
        return self._filter(docs)

    async def ahybrid_search(self, query: str) -> list[tuple[Document, float]]:
        """异步执行双路召回、RRF 融合及可选重排。

        Args:
            query: 用于稠密向量检索、BM25 检索和重排的查询文本。

        Returns:
            list[tuple[Document, float]]: 截取 top_k 并经阈值过滤后的文档与
                分数列表。配置重排模型时使用重排分数，否则使用 RRF 融合
                分数；无候选文档时返回空列表。
        """
        embedding = await self.embedding_function.aembed_query(query)
        candidates = self._decode_hits(await self.client.hybrid_search(
            **self._hybrid_options(query, embedding)
        ))
        if self.rerank_function is not None and candidates:
            ranked = await self.rerank_function.arerank(
                query=query, documents=[doc.page_content for doc, _ in candidates],
                top_n=self.top_k,
            )
            docs = [(candidates[index][0], score) for index, score in ranked]
        else:
            docs = candidates[:self.top_k]
        return self._filter(docs)

    @staticmethod
    def _documents(scored_docs: list[tuple[Document, float]]) -> list[Document]:
        """将分数写入文档元数据，并按分数降序排列。

        Args:
            scored_docs: 文档与对应检索或重排分数的列表。
                文档对象的 metadata["score"] 会被原地写入或覆盖。

        Returns:
            list[Document]: 按 score 降序排列的新列表，其中的文档对象
                与输入列表中的对象相同。
        """
        docs = []
        for doc, score in scored_docs:
            doc.metadata["score"] = score
            docs.append(doc)
        return sorted(docs, key=lambda doc: doc.metadata["score"], reverse=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """实现 LangChain 同步检索钩子，供 invoke 调用。

        Args:
            query: 检索查询文本。
            run_manager: LangChain 传入的同步检索回调管理器，当前未使用。

        Returns:
            list[Document]: 混合检索得到的文档，按分数降序排列，
                每个文档的 metadata["score"] 中包含对应分数。
        """
        return self._documents(self.hybrid_search(query))

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """实现 LangChain 异步检索钩子，供 ainvoke 调用。

        Args:
            query: 检索查询文本。
            run_manager: LangChain 传入的异步检索回调管理器，当前未使用。

        Returns:
            list[Document]: 混合检索得到的文档，按分数降序排列，
                每个文档的 metadata["score"] 中包含对应分数。
        """
        return self._documents(await self.ahybrid_search(query))
