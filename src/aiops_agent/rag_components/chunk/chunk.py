from pathlib import Path
from typing import List, Any
import uuid

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .types import ParentChildChunks


def _semantic_splitter(embedding_model: Embeddings, min_size: int, threshold: float) -> SemanticChunker:
    return SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold,
        min_chunk_size=min_size,
        sentence_split_regex=r"(?<=[。！？；：.!?;:])",
    )


def _recursive_splitter(size: int, overlap_rate: float) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", "，", ",", " ", ""],
        chunk_size=size,
        chunk_overlap=int(size * overlap_rate),
        length_function=len,
    )


class ParentChildChunker:
    """父子文档切分器"""
    def __init__(
        self,
        embedding_model: Embeddings,
        parent_chunk_size: int = 2000,
        parent_min_chunk_size: int = 500,
        child_chunk_size: int = 800,
        child_min_chunk_size: int = 200,
        chunk_overlap_rate: float = 0.25,
    ):
        self.embedding_model = embedding_model

        self.parent_chunk_size = parent_chunk_size # 父块最大字符数
        self.parent_min_chunk_size = parent_min_chunk_size # 父块最小字符数

        self.child_chunk_size = child_chunk_size # 子块最大字符数
        self.child_min_chunk_size = child_min_chunk_size # 子块最小字符数

        self.chunk_overlap_rate = chunk_overlap_rate # 文本切分重叠率

        # 1. 父块语义切分器
        self.parent_semantic_splitter = _semantic_splitter(self.embedding_model, self.parent_min_chunk_size, 6.0)

        # 2. 子块语义切分器
        self.child_semantic_splitter = _semantic_splitter(self.embedding_model, self.child_min_chunk_size, 3.0)

        # 3. 父块过大时使用的兜底切分器
        self.parent_recursive_splitter = _recursive_splitter(self.parent_chunk_size, chunk_overlap_rate)

        # 4. 子块过大时使用的兜底切分器
        self.child_recursive_splitter = _recursive_splitter(self.child_chunk_size, chunk_overlap_rate)

    def split_text(self, text: Document, metadata: dict[str, Any] | None = None) -> list[ParentChildChunks]:
        """
        对文档进行父子切分, 适用于解析成普通文本文档， 先按语义切分，再按字符切分

        Args:
            text: 待切分的文档，来源信息从 text.metadata 读取（source/page）
            metadata: 额外元数据，会透传到每个分块；file_name/page 由来源自动归一化

        Returns:
            切分后的文档列表，每项包含 parent 和 children
        """
        base_metadata = {**text.metadata, **(metadata or {})}
        source = base_metadata.get("source")
        raw_page = base_metadata.get("page")

        # 归一化 file_name/page
        if source is None:
            raise ValueError("source 不能为空")
        if raw_page is None:
            raise ValueError("page 不能为空")
        chunk_metadata: dict[str, Any] = {
            key: value
            for key, value in (metadata or {}).items()
            if key not in ("file_name", "page")
        }
        chunk_metadata["file_name"] = Path(source).name if source else ""
        chunk_metadata["page"] = int(raw_page) + 1 if raw_page is not None else 1

        # 第一层：切 Parent
        semantic_parent_docs = self.parent_semantic_splitter.create_documents(
            texts=[text.page_content],
            metadatas=[chunk_metadata],
        )

        parent_docs: list[Document] = []

        for parent_doc in semantic_parent_docs:
            if len(parent_doc.page_content) >= self.parent_chunk_size:
                recursive_docs = self.parent_recursive_splitter.split_documents(
                    [parent_doc]
                )

                parent_docs.extend(recursive_docs)
            else:
                parent_docs.append(parent_doc)

        # 第二层：Parent → Child
        results: list[ParentChildChunks] = []

        for parent_doc in parent_docs:

            parent_id = str(uuid.uuid4())

            # 给 Parent 添加 ID
            parent_doc.metadata["parent_id"] = parent_id

            # 对 Parent 进行语义子切分
            semantic_child_docs = self.child_semantic_splitter.create_documents(
                texts=[parent_doc.page_content],
                metadatas=[parent_doc.metadata],
            )

            child_docs: list[Document] = []

            for child_doc in semantic_child_docs:
                if len(child_doc.page_content) >= self.child_chunk_size:
                    recursive_docs = self.child_recursive_splitter.split_documents(
                        [child_doc]
                    )
                    child_docs.extend(recursive_docs)
                else:
                    child_docs.append(child_doc)

            # 添加 Child 元数据
            for child_doc in child_docs:
                child_doc.metadata["parent_id"] = parent_id
                child_doc.metadata["child_id"] = f"{parent_id}_child_{str(uuid.uuid4())}"

            results.append(
            {
                "parent": parent_doc,
                "children": child_docs,
            }
        )
        return results


class MarkdownChunkSplitter:
    """
    Markdown 文件切分器

    按 # / ## / ### 三级标题切分父块，超长父块语义切分 + 递归兜底；
    父块内部语义切分子块，超长递归兜底，最后合并过小分块。
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        parent_chunk_size: int = 2000,
        parent_min_chunk_size: int = 500,
        child_chunk_size: int = 800,
        child_min_chunk_size: int = 200,
        chunk_overlap_rate: float = 0.25,
    ):
        self.embedding_model = embedding_model  # 嵌入模型
        self.parent_chunk_size = parent_chunk_size  # 父块最大字符数
        self.parent_min_chunk_size = parent_min_chunk_size  # 父块最小字符数
        self.child_chunk_size = child_chunk_size  # 子块最大字符数
        self.child_min_chunk_size = child_min_chunk_size  # 子块最小字符数
        self.chunk_overlap_rate = chunk_overlap_rate  # 文本切分重叠率

        # 1. 按三级标题切分
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )

        # 2. 父块语义切分器
        self.parent_semantic_splitter = _semantic_splitter(self.embedding_model, self.parent_min_chunk_size, 6.0)

        # 3. 父块过大时的兜底切分器
        self.parent_recursive_splitter = _recursive_splitter(self.parent_chunk_size, chunk_overlap_rate)

        # 4. 子块语义切分器
        self.child_semantic_splitter = _semantic_splitter(self.embedding_model, self.child_min_chunk_size, 3.0)

        # 5. 子块超长兜底切分器
        self.child_recursive_splitter = _recursive_splitter(self.child_chunk_size, chunk_overlap_rate)

    def _split_children(self, content: str) -> list[str]:
        """
        对 父块内容 进行子块切分，先按语义切分，超长部分递归兜底，最后合并过小分块

        Args:
            content: 父块文本内容

        Returns:
            list[str]: 切分后的子块文本列表
        """
        children: list[str] = []  # 子块文本列表
        semantic_docs = self.child_semantic_splitter.create_documents(texts=[content])  # 语义切分后的子块
        for semantic_doc in semantic_docs:  # 遍历语义切分出的子块
            if len(semantic_doc.page_content) >= self.child_chunk_size:
                recursive_docs = self.child_recursive_splitter.create_documents([semantic_doc.page_content])  # 过长兜底切分结果
                for doc in recursive_docs:  # 遍历兜底切分出的子块
                    children.append(doc.page_content)  # 追加子块文本
            else:
                children.append(semantic_doc.page_content)  # 追加子块文本

        return self._merge_small_children(children)

    def _merge_small_children(self, children: list[str]) -> list[str]:
        """
        合并小于最小长度的子块，向前合并，合并后不超过 child_chunk_size

        Args:
            children: 待合并的子块文本列表

        Returns:
            list[str]: 合并后的子块文本列表
        """
        merged: list[str] = []  # 合并后的子块列表
        for child in children:  # 遍历子块
            if (
                merged
                and len(child) < self.child_min_chunk_size
                and len(merged[-1]) + len(child) + 2 <= self.child_chunk_size
            ):
                merged[-1] = f"{merged[-1]}\n\n{child}"  # 向前合并
            else:
                merged.append(child)  # 保留独立分块
        return merged

    def split_text(
        self, text: Document, metadata: dict[str, Any] | None = None
    ) -> list[ParentChildChunks]:
        """
        对 Markdown 文档进行父子切分，先按三级标题切分父块，超长父块语义切分，
        父块内部语义切分子块，超长递归兜底, 最后 尽可能 合并过小分块。

        Args:
            text: 待切分的文档
            metadata: 额外元数据，会透传到每个分块

        Returns:
            切分后的文档列表，每项包含 parent 和 children
        """
        base_metadata = {**text.metadata, **(metadata or {})}  # 合并来源与传入元数据
        source = base_metadata.get("source")  # 来源文件路径
        raw_page = base_metadata.get("page")  # 原始页码

        if source is None:
            raise ValueError("source 不能为空")

        chunk_metadata: dict[str, Any] = {**text.metadata}  # 分块基础元数据
        for key, value in (metadata or {}).items():  # 遍历额外元数据
            if key not in ("file_name", "page"):  # 过滤 file_name/page
                chunk_metadata[key] = value  # 透传元数据
        chunk_metadata["file_name"] = Path(source).name if source else ""
        chunk_metadata["page"] = int(raw_page) + 1 if raw_page is not None else 1

        sections = self.markdown_splitter.split_text(text.page_content)  # 标题切分后的父块列表

        # 第一层：标题切分 → 超长语义切分 → 兜底切分，得到父块列表
        parent_docs: list[Document] = []  # 父块列表
        for section in sections:  # 遍历标题切分出的父块
            if not section.page_content.strip():
                continue

            # 未超长：直接作为父块
            if len(section.page_content) < self.parent_chunk_size:
                parent_docs.append(section)
                continue

            # 超长：语义切分
            semantic_docs = self.parent_semantic_splitter.create_documents(  # 语义切分后的父块
                texts=[section.page_content],
                metadatas=[section.metadata],
            )
            # 语义切分后的父块可能仍然过大，使用递归切分兜底
            for semantic_doc in semantic_docs:  # 遍历语义切分出的父块
                if len(semantic_doc.page_content) >= self.parent_chunk_size:
                    recursive_docs = self.parent_recursive_splitter.split_documents([semantic_doc])  # 兜底切分后的父块
                    parent_docs.extend(recursive_docs)
                else:
                    parent_docs.append(semantic_doc)

        # 第二层：Parent → Child
        results: list[ParentChildChunks] = []  # 切分结果列表

        for parent_doc in parent_docs:  # 遍历单个父块
            parent_id = str(uuid.uuid4())  # 父块唯一 ID
            parent_metadata = dict(chunk_metadata)  # 父块元数据（基础元数据副本）
            for key, value in parent_doc.metadata.items():  # 遍历来源文档元数据
                if key not in ("h1", "h2", "h3"):  # 过滤标题字段
                    parent_metadata[key] = value
            parent_doc.metadata = {**parent_metadata, "parent_id": parent_id}

            child_docs: list[Document] = []  # 子块文档列表
            for child_text in self._split_children(parent_doc.page_content):  # 段落优先切子块
                child_docs.append(  # 构造子块文档
                    Document(
                        page_content=child_text,  # 子块文本
                        metadata={
                            **parent_metadata,
                            "parent_id": parent_id,
                            "child_id": f"{parent_id}_child_{uuid.uuid4()}",
                        },
                    )
                )

            results.append({"parent": parent_doc, "children": child_docs})  # 追加父子块结果

        return results