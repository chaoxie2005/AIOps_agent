import asyncio
import os
import tempfile
from functools import lru_cache
from hashlib import sha256
from logging import getLogger
from pathlib import Path
from typing import Annotated
from zipfile import BadZipFile

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.documents import Document
from requests import RequestException

from aiops_agent.rag_components.chunk.chunk import MarkdownChunkSplitter
from aiops_agent.rag_components.chunk.types import ParentChildChunks
from aiops_agent.rag_components.ingestion import DocumentIngestor, EmptyDocumentError, fingerprint
from aiops_agent.rag_components.minio.minio import MinioClient
from aiops_agent.rag_components.parser.mineru import MineruClient
from aiops_agent.rag_components.parser.parser import Parser
from aiops_agent.rag_components.retriever.factory import (
    build_embedding_from_env,
    build_milvus_client_from_env,
    get_milvus_collection_name,
)
from aiops_agent.schema.resquest.load_milvus import LoadMilvusRequest
from aiops_agent.schema.response.load_milvus import LoadMilvusResponse

router = APIRouter()
logger = getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".md", ".pdf"}
_MINERU_POLL_INTERVAL = 5
_MINERU_POLL_TIMEOUT = int(os.environ.get("MINERU_POLL_TIMEOUT", "300"))
# 解析、切块或入库元数据变化时递增，使同一文件也会重新处理。
_PIPELINE_REVISION = "2"


@lru_cache(maxsize=1)
def _get_ingestion_components() -> tuple[DocumentIngestor, MarkdownChunkSplitter]:
    """首次上传时创建入库依赖，随后复用客户端与进程内文档锁。"""
    embedding = build_embedding_from_env()
    minio = MinioClient(
        endpoint=os.environ.get("MINIO_ENDPOINT", "127.0.0.1:9000"),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
    )
    ingestor = DocumentIngestor(
        minio, build_milvus_client_from_env(), embedding.aembed_documents,
        embedding_id=fingerprint(
            embedding.model_name, embedding.base_url,
            os.environ.get("EMBEDDING_REVISION", ""),
        ),
    )
    return ingestor, MarkdownChunkSplitter(embedding_model=embedding)


# ── 内部工具函数 ─────────────────────────────────────

def _parse_md(file_path: Path, file_name: str) -> list[Document]:
    """解析 Markdown 文件，记录原始文件名并补齐 page 字段。"""
    docs = Parser(str(file_path)).parse()
    for doc in docs:
        doc.metadata["source"] = file_name
        doc.metadata.setdefault("page", 0)
    return docs


async def _parse_pdf_via_mineru(
    minio_url: str, tmp_dir: str, file_name: str
) -> list[Document]:
    """通过 MinerU 将 PDF 解析为 Markdown，再解析成 Document 列表。"""
    mineru = MineruClient()
    batch_id = await asyncio.to_thread(mineru.upload_urls, [minio_url])

    elapsed = 0
    full_zip_url: str | None = None
    while elapsed < _MINERU_POLL_TIMEOUT:
        await asyncio.sleep(_MINERU_POLL_INTERVAL)
        elapsed += _MINERU_POLL_INTERVAL

        result = await asyncio.to_thread(mineru.get_parse_results, batch_id)
        if result.get("code") != 0:
            raise RuntimeError(f"MinerU 查询失败: {result.get('msg', '未知错误')}")
        files = result.get("data", {}).get("extract_result", [])
        if not files:
            continue

        item = files[0]
        state = item.get("state", "")
        if state == "failed":
            raise RuntimeError(
                f"MinerU 解析失败: {item.get('err_msg', '未知错误')}"
            )
        if state == "done":
            full_zip_url = item.get("full_zip_url")
            if not full_zip_url:
                raise RuntimeError("MinerU 解析结果缺少下载地址")
            break

    if full_zip_url is None:
        raise TimeoutError(f"MinerU 解析超时（>{_MINERU_POLL_TIMEOUT}s）")

    md_path = await asyncio.to_thread(
        mineru.download_md_to_local,
        full_zip_url=full_zip_url,
        save_dir=tmp_dir,
        save_file_name="full.md",
    )
    return _parse_md(md_path, file_name)


def _split_documents(
    documents: list[Document], chunker: MarkdownChunkSplitter
) -> list[ParentChildChunks]:
    return [chunk for doc in documents for chunk in chunker.split_text(doc)]


# ── 路由 ─────────────────────────────────────────────────────

@router.post("/load_milvus", response_model=LoadMilvusResponse)
async def load_milvus(
    request: Annotated[LoadMilvusRequest, Depends()],
) -> LoadMilvusResponse:
    """将 Markdown 或 PDF 上传到 MinIO，并增量更新 Milvus。"""
    file = request.file
    file_name = file.filename or "unknown"
    suffix = Path(file_name).suffix.lower()

    if suffix not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"不支持的文件格式 {suffix}，"
                f"仅支持: {', '.join(_SUPPORTED_EXTENSIONS)}"
            ),
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / f"upload{suffix}"
        digest = sha256()
        with tmp_file.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                destination.write(chunk)
                digest.update(chunk)

        async def load_chunks(staged_url: str) -> list[ParentChildChunks]:
            try:
                if suffix == ".md":
                    documents = await asyncio.to_thread(_parse_md, tmp_file, file_name)
                else:
                    documents = await _parse_pdf_via_mineru(staged_url, tmp_dir, file_name)
            except (RuntimeError, TimeoutError, RequestException, BadZipFile, ValueError) as exc:
                logger.exception("文件解析失败: %s", file_name)
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            return await asyncio.to_thread(_split_documents, documents, chunker)

        try:
            ingestor, chunker = _get_ingestion_components()
            result = await ingestor.ingest(
                collection=request.collection_name or get_milvus_collection_name(),
                bucket=request.bucket_name,
                file_name=file_name,
                file_path=tmp_file,
                file_sha256=digest.hexdigest(),
                pipeline_id=fingerprint(
                    _PIPELINE_REVISION, suffix,
                    os.environ.get("MINERU_URL", "") if suffix == ".pdf" else "",
                ),
                load_chunks=load_chunks,
            )
        except EmptyDocumentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("文档入库失败: %s", file_name)
            raise HTTPException(status_code=502, detail="文档入库失败，请重试") from exc

    return LoadMilvusResponse(
        file_name=file_name,
        object_name=result.object_name,
        url=result.url,
        parent_count=result.parent_count,
        child_count=result.child_count,
        status=result.status,
    )
