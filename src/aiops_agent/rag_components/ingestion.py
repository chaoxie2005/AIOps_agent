"""负责判断文件要不要入库、哪些块要写、哪些旧块要删。"""

import asyncio
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeVar
from uuid import uuid4
from weakref import WeakValueDictionary

from aiops_agent.rag_components.chunk.types import ParentChildChunks
from aiops_agent.rag_components.milvus.milvus_client import AMilvusClient
from aiops_agent.rag_components.minio.minio import MinioClient

logger = logging.getLogger(__name__)
_WRITE_BATCH_SIZE = 128
_T = TypeVar("_T")
IngestionStatus = Literal["created", "updated", "unchanged"]


class EmptyDocumentError(ValueError):
    """文档解析后没有可入库的子块。"""


@dataclass(frozen=True)
class IngestionResult:
    status: IngestionStatus
    object_name: str
    url: str
    parent_count: int
    child_count: int


def fingerprint(*parts: str) -> str:
    """为一组有序字符串生成稳定的指纹。

    先用 JSON 保留每段字符串的边界，再计算 SHA-256。

    Args:
        *parts: 按顺序参与指纹计算的字符串。

    Returns:
        SHA-256 的十六进制字符串。
    """
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()


def _text_digest(text: str) -> str:
    """计算原始文本的 SHA-256，不改动空白或大小写。

    Args:
        text: 要计算指纹的文本。

    Returns:
        文本按 UTF-8 编码后的 SHA-256 十六进制字符串。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _batches(items: list[_T]) -> Iterator[list[_T]]:
    """按写入批次上限拆分列表，保留原顺序。

    Args:
        items: 要分批处理的元素列表。

    Yields:
        一批连续元素，最后一批可以少于批次上限。
    """
    for start in range(0, len(items), _WRITE_BATCH_SIZE):
        yield items[start : start + _WRITE_BATCH_SIZE]


def _build_rows(
    chunks: list[ParentChildChunks],
    *,
    document_id: str,
    embedding_id: str,
    file_type: str,
    url: str,
) -> list[dict]:
    """将父子块转换为尚未填充向量的 Milvus 行。

    父块和子块各自用文档 ID、文本哈希和同文出现序号生成稳定 ID，
    使重复出现的相同文本仍能对应不同记录。

    Args:
        chunks: 按文档顺序排列的父子块。
        document_id: 当前文档的稳定 ID。
        embedding_id: 当前嵌入配置的指纹。
        file_type: 不带点的文件扩展名，例如 md 或 pdf。
        url: 原文件发布后的固定 MinIO URL。

    Returns:
        每个子块对应一行，含父子块 ID、正文和元数据。
    """
    rows = []
    parents: Counter[str] = Counter()
    children: Counter[str] = Counter()
    for chunk in chunks:
        parent_content = chunk["parent"].page_content
        parent_digest = _text_digest(parent_content)
        parents[parent_digest] += 1
        parent_id = fingerprint(document_id, "parent", parent_digest, str(parents[parent_digest]))
        for child in chunk["children"]:
            text = child.page_content
            digest = _text_digest(text)
            children[digest] += 1
            child_id = fingerprint(document_id, "child", digest, str(children[digest]))
            rows.append(
                {
                    "child_chunk_id": child_id,
                    "parent_chunk_id": parent_id,
                    "child_content": text,
                    "child_metadata": {
                        **child.metadata,
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "parent_content": parent_content,
                        "document_id": document_id,
                        "content_sha256": digest,
                        "embedding_id": embedding_id,
                    },
                    "file_type": file_type,
                    "child_minio_url": url,
                    "language": "zh",
                }
            )
    return rows


class DocumentIngestor:
    """同一进程内串行更新一份文档，用 MinIO 清单标记成功和未完成的版本。"""

    def __init__(
        self,
        minio: MinioClient,
        milvus: AMilvusClient,
        embed: Callable[[list[str]], Awaitable[list[list[float]]]],
        *,
        embedding_id: str,
    ) -> None:
        """初始化增量入库服务及进程内的文档锁。

        Args:
            minio: 管理原文件、暂存对象和入库清单的 MinIO 客户端。
            milvus: 查询、写入和删除子块的 Milvus 客户端。
            embed: 异步批量生成向量的函数，向量顺序须与输入文本一致。
            embedding_id: 嵌入配置的指纹，用于判断旧向量能否复用。
        """
        self._minio = minio
        self._milvus = milvus
        self._embed = embed
        self._embedding_id = embedding_id
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    async def ingest(
        self,
        *,
        collection: str,
        bucket: str,
        file_name: str,
        file_path: Path,
        file_sha256: str,
        pipeline_id: str,
        load_chunks: Callable[[str], Awaitable[list[ParentChildChunks]]],
    ) -> IngestionResult:
        """对同一文档加锁，再执行一次增量入库。

        同一集合、存储桶和文件名对应同一把进程内的锁。
        如果调用方在入库期间取消请求，会等当前操作结束后再释放锁。

        Args:
            collection: 目标 Milvus 集合。
            bucket: 原文件和入库清单所在的 MinIO 存储桶。
            file_name: 用于识别文档和查找旧记录的原始文件名。
            file_path: 本次上传文件在本地的临时路径。
            file_sha256: 上传文件字节的 SHA-256 十六进制字符串。
            pipeline_id: 解析和切块配置的指纹。
            load_chunks: 接收暂存对象 URL 并返回父子块的异步函数。

        Returns:
            入库状态、正式对象路径和 URL，以及父子块数量。

        Raises:
            EmptyDocumentError: 解析和切块后没有可入库的子块。
            asyncio.CancelledError: 调用方取消此次请求。
        """
        document_id = fingerprint(collection, bucket, file_name)
        lock = self._locks.setdefault(document_id, asyncio.Lock())
        async with lock:
            task = asyncio.create_task(
                self._ingest(
                    collection, bucket, file_name, file_path, file_sha256,
                    pipeline_id, load_chunks, document_id,
                )
            )
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # 请求取消时仍持有锁与临时文件，直到当前更新退出。
                await asyncio.gather(task, return_exceptions=True)
                raise

    async def _ingest(
        self,
        collection: str,
        bucket: str,
        file_name: str,
        file_path: Path,
        file_sha256: str,
        pipeline_id: str,
        load_chunks: Callable[[str], Awaitable[list[ParentChildChunks]]],
        document_id: str,
    ) -> IngestionResult:
        """在持有文档锁时比较版本、更新向量并发布原文件。

        已成功提交的文件和处理配置均未变化时直接跳过。有未完成记录时，
        重新查询 Milvus 的实际数据以修复部分写入。需要更新时，先将清单
        标记为未完成，再写入新行、删除旧行并发布正式对象，最后提交清单。

        Args:
            collection: 目标 Milvus 集合。
            bucket: 保存原文件、暂存文件和清单的 MinIO 存储桶。
            file_name: 原始文件名，用于按原 URL 识别历史记录。
            file_path: 本次上传文件在本地的临时路径。
            file_sha256: 上传文件字节的 SHA-256 十六进制字符串。
            pipeline_id: 解析和切块配置的指纹。
            load_chunks: 接收暂存对象 URL 并返回父子块的异步函数。
            document_id: 当前文档的稳定 ID。

        Returns:
            入库状态、正式对象路径和 URL，以及父子块数量。

        Raises:
            EmptyDocumentError: 解析和切块后没有可入库的子块。
        """
        suffix = file_path.suffix.lower()
        prefix = f"_aiops/ingestion/{document_id}"
        object_name = f"{prefix}/source{suffix}"
        manifest_name = f"{prefix}/manifest.json"
        url = self._minio.get_minio_url(bucket, object_name)
        revision = {
            "file_sha256": file_sha256,
            "pipeline_id": fingerprint(pipeline_id, self._embedding_id),
        }
        manifest = await asyncio.to_thread(self._minio.read_json, bucket, manifest_name) or {}
        committed = manifest.get("committed")
        if (
            manifest.get("pending") is None
            and committed is not None
            and all(committed.get(key) == value for key, value in revision.items())
        ):
            return IngestionResult(
                "unchanged", object_name, url,
                committed["parent_count"], committed["child_count"],
            )

        stage = f"{prefix}/staging/{uuid4().hex}{suffix}"
        try:
            await asyncio.to_thread(self._minio.upload_file, bucket, str(file_path), stage)
            chunks = await load_chunks(self._minio.get_minio_url(bucket, stage))
            rows = _build_rows(
                chunks, document_id=document_id, embedding_id=self._embedding_id,
                file_type=suffix.lstrip("."), url=url,
            )
            if not rows:
                raise EmptyDocumentError("文件解析后无有效内容，无法入库")

            legacy_url = self._minio.get_minio_url(bucket, file_name)
            candidates = await self._milvus.query_ingestion_rows(collection, document_id, legacy_url)
            previous: dict[str, dict] = {}
            has_legacy = False
            for row in candidates:
                owner = (row.get("child_metadata") or {}).get("document_id")
                legacy = owner is None and row.get("child_minio_url") == legacy_url
                if owner == document_id or legacy:
                    previous[row["child_chunk_id"]] = row
                    has_legacy = has_legacy or legacy

            changes = await self._rows_to_write(rows, previous, document_id)
            desired_ids = {row["child_chunk_id"] for row in rows}
            stale = sorted(previous.keys() - desired_ids)
            target = {**revision, "parent_count": len(chunks), "child_count": len(rows)}

            # 先记下未完成状态，避免失败后误跳过旧版本或部分写入。
            await asyncio.to_thread(
                self._minio.write_json, bucket, manifest_name,
                {"committed": committed, "pending": target},
            )
            for batch in _batches(changes):
                await self._milvus.update_entity(collection_name=collection, data=batch)
            for ids in _batches(stale):
                await self._milvus.delete_ingestion_rows(collection, ids)
            await asyncio.to_thread(self._minio.publish_object, bucket, stage, object_name)
            await asyncio.to_thread(
                self._minio.write_json, bucket, manifest_name,
                {"committed": target, "pending": None},
            )
            status = "updated" if committed is not None or has_legacy else "created"
            return IngestionResult(status, object_name, url, len(chunks), len(rows))
        finally:
            try:
                await asyncio.to_thread(self._minio.remove_object, bucket, stage)
            except Exception:
                logger.warning("清理入库暂存对象失败: %s", stage, exc_info=True)

    async def _rows_to_write(
        self, rows: list[dict], previous: dict[str, dict], document_id: str
    ) -> list[dict]:
        """筛出需要写入的行，并为这些行复用或补齐向量。

        完全未变且已有向量的行直接跳过。需要写入的行可复用同一文档中
        文本和嵌入配置均相同的旧向量；仅为缺少向量的不同文本批量计算。

        Args:
            rows: 本次切块得到的目标行，尚未填充向量。
            previous: 以子块主键为键的已有 Milvus 行。
            document_id: 当前文档的稳定 ID，用于核对旧向量归属。

        Returns:
            需要写入的完整行，包含稠密向量和写入时间。

        Raises:
            ValueError: 嵌入函数返回的向量数量与待计算的文本数量不符。
        """
        vectors: dict[str, list[float]] = {}
        for old in previous.values():
            metadata = old.get("child_metadata") or {}
            if (
                metadata.get("document_id") == document_id
                and metadata.get("embedding_id") == self._embedding_id
                and old.get("dense_vector") is not None
            ):
                vectors[old["child_content"]] = list(old["dense_vector"])

        changed = []
        for row in rows:
            old = previous.get(row["child_chunk_id"])
            if (
                old is not None
                and old.get("dense_vector") is not None
                and all(old.get(key) == value for key, value in row.items())
            ):
                continue
            changed.append(row)

        missing = list(dict.fromkeys(
            row["child_content"] for row in changed if row["child_content"] not in vectors
        ))
        if missing:
            vectors.update(zip(missing, await self._embed(missing), strict=True))

        timestamp = datetime.now(timezone.utc).isoformat()
        return [
            {**row, "dense_vector": vectors[row["child_content"]], "created_at": timestamp}
            for row in changed
        ]
