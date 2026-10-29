from typing import Optional, List, Union

from pymilvus import (
    AnnSearchRequest,
    AsyncMilvusClient,
    MilvusClient,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    FunctionScore,
)
from pymilvus.client.abstract import BaseRanker
from pymilvus.milvus_client.index import IndexParams

from logger import logger


def _build_schema_and_index(
    client: AsyncMilvusClient, dim: int
) -> tuple[CollectionSchema, IndexParams]:
    """构建集合的Schema和索引参数"""
    fields = [
        FieldSchema(
            name="child_chunk_id",
            dtype=DataType.VARCHAR,
            description="子块ID",
            max_length=512,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(
            name="parent_chunk_id",
            dtype=DataType.VARCHAR,
            description="父块ID",
            max_length=512,
        ),
        FieldSchema(
            name="language",
            dtype=DataType.VARCHAR,
            description="语言标识(zh/en)，缺失时回退default分析器",
            max_length=32,
            nullable=True,
        ),
        FieldSchema(
            name="child_content",
            dtype=DataType.VARCHAR,
            description="子块内容",
            max_length=65535,
            enable_analyzer=True,
            multi_analyzer_params={
                "analyzers": {
                    "english": {"type": "english"},
                    "chinese": {"type": "chinese"},
                    "default": {"tokenizer": "icu"},
                },
                "by_field": "language",
                "alias": {
                    "zh": "chinese",
                    "en": "english",
                },
            }
        ),
        FieldSchema(
            name="child_metadata",
            dtype=DataType.JSON,
            description="子块元数据",
        ),
        FieldSchema(
            name="created_at",
            dtype=DataType.TIMESTAMPTZ,
            description="创建时间",
            nullable=True,
        ),
        FieldSchema(
            name="file_type",
            dtype=DataType.VARCHAR,
            description="文件类型",
            max_length=512,
            nullable=True,
        ),
        FieldSchema(
            name="child_minio_url",
            dtype=DataType.VARCHAR,
            description="子块MinIO URL",
            max_length=512,
            nullable=True,
        ),
        FieldSchema(
            name="dense_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
            description="稠密向量(子块内容)",
        ),
        FieldSchema(
            name="sparse_vector",
            dtype=DataType.SPARSE_FLOAT_VECTOR,
            description="稀疏向量(子块内容)",
        ),
    ]
    schema = CollectionSchema(
        fields=fields,
        enable_dynamic_field=False,
    )
    bm25_function = Function(
        name="child_content_bm25_embed",
        input_field_names=["child_content"],
        output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    )

    schema.add_function(bm25_function)

    # 构建索引
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="dense_vector", index_type="FLAT", metric_type="COSINE"
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "TAAT_NAIVE",  # 用什么算法搜索
            "bm25_k1": 1.2,  # BM25词频饱和参数，后续基于业务检索评测调优
            "bm25_b": 1,  # 文档长度影响  后续基于业务检索评测调优
        },
    )
    return schema, index_params


class AMilvusClient:
    def __init__(
        self,
        url: str = "http://localhost:19530",
        user: str = "",
        password: str = "",
        db_name: str = "",
        token: str = "",
        timeout: Optional[float] = None,
    ):
        self._connection_options = dict(
            uri=url, user=user, password=password, db_name=db_name,
            token=token, timeout=timeout,
        )
        self._sync_client = None
        self.client = AsyncMilvusClient(
            uri=url,
            user=user,
            password=password,
            db_name=db_name,
            token=token,
            timeout=timeout,
        )

    def _get_sync_client(self) -> MilvusClient:
        """仅在同步检索时建立同步连接。"""
        if self._sync_client is None:
            self._sync_client = MilvusClient(**self._connection_options)
        return self._sync_client

    def vector_search_sync(self, *, query: list[float], top_k: int = 10, **kwargs):
        """同步向量查询，参数与异步入口一致。"""
        return self._get_sync_client().search(
            data=[query], limit=top_k, anns_field="dense_vector", **kwargs
        )

    def hybrid_search_sync(self, *, requests, top_k: int = 10, **kwargs):
        """同步混合查询，参数与异步入口一致。"""
        return self._get_sync_client().hybrid_search(
            reqs=requests, limit=top_k, **kwargs
        )

    async def has_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> bool:
        """是否存在指定集合"""
        return await self.client.has_collection(
            collection_name=collection_name,
            timeout=timeout,
            **kwargs,
        )

    async def has_partition(
        self,
        collection_name: str,
        partition_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """是否存在指定分区"""
        return await self.client.has_partition(
            collection_name=collection_name,
            partition_name=partition_name,
            timeout=timeout,
            **kwargs,
        )

    async def create_collection(
        self,
        collection_name: str,
        dim: int,
        **kwargs,
    ):
        """创建指定集合"""
        schema, index_params = _build_schema_and_index(self.client, dim)
        await self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            **kwargs,
        )

    async def drop_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> None:
        """删除指定集合"""
        await self.client.drop_collection(
            collection_name=collection_name,
            timeout=timeout,
            **kwargs,
        )

    async def create_partition(
        self,
        collection_name: str,
        partition_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        """创建指定分区"""
        await self.client.create_partition(
            collection_name=collection_name,
            partition_name=partition_name,
            timeout=timeout,
            **kwargs,
        )

    async def drop_partition(
        self,
        collection_name: str,
        partition_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        """删除指定分区"""
        await self.client.drop_partition(
            collection_name=collection_name,
            partition_name=partition_name,
            timeout=timeout,
            **kwargs,
        )

    async def insert_entity(
        self,
        collection_name: str,
        data: Union[dict, List[dict]],
        partition_name: str = "_default",
    ) -> dict:
        """向 Milvus Collection 插入实体数据"""
        return await self.client.insert(
            collection_name=collection_name,
            partition_name=partition_name,
            data=data,
        )

    async def delete_entity(
        self,
        collection_name: str,
        filter: str,
        partition_name: str = "_default",
        timeout: float | None = None,
    ) -> dict:
        """按过滤表达式删除 Milvus Collection 实体数据"""
        return await self.client.delete(
            collection_name=collection_name,
            filter=filter,
            partition_name=partition_name,
            timeout=timeout,
        )

    async def update_entity(
        self,
        collection_name: str,
        data: Union[dict, List[dict]],
        partition_name: str = "_default",
    ) -> dict:
        """更新 Milvus Collection 实体数据"""
        return await self.client.upsert(
            collection_name=collection_name,
            partition_name=partition_name,
            data=data,
        )

    async def query_entity(
        self,
        collection_name: str,
        filter: str,
        partition_names: Optional[List[str]] = None,
        output_fields: list[str] | None = None,
        timeout: float | None = None,
    ) -> list[dict]:
        """按过滤表达式查询 Milvus Collection 实体数据"""
        return await self.client.query(
            collection_name=collection_name,
            filter=filter,
            partition_names=partition_names,
            output_fields=output_fields,
            timeout=timeout,
        )

    async def vector_search(
        self,
        collection_name: str,
        query: list[float],
        partition_names: Optional[List[str]] = None,
        ranker: Optional[Union[Function, FunctionScore]] = None,
        top_k: int = 10,
        anns_field: str = "dense_vector",
        output_fields: list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> List[List[dict]]:
        """在 Milvus Collection 中进行向量相似度检索"""
        return await self.client.search(
            collection_name=collection_name,
            data=[query],
            anns_field=anns_field,
            ranker=ranker,
            limit=top_k,
            output_fields=output_fields,
            partition_names=partition_names,
            timeout=timeout,
            **kwargs,
        )

    async def hybrid_search(
        self,
        collection_name: str,
        requests: List[AnnSearchRequest],
        ranker: Union[BaseRanker, Function],
        partition_names: Optional[List[str]] = None,
        top_k: int = 10,
        output_fields: list[str] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> List[List[dict]]:
        """在 Milvus Collection 中进行混合检索"""
        return await self.client.hybrid_search(
            collection_name=collection_name,
            limit=top_k,
            reqs=requests,
            ranker=ranker,
            output_fields=output_fields,
            partition_names=partition_names,
            timeout=timeout,
            **kwargs,
        )

    async def close(self) -> None:
        r"""关闭 Milvus 连接"""
        try:
            await self.client.close()
        finally:
            if self._sync_client is not None:
                self._sync_client.close()
                self._sync_client = None
        logger.info("Milvus 连接已关闭")
