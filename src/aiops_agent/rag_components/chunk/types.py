from typing import TypedDict

from langchain_core.documents import Document


class ParentChildChunks(TypedDict):
    """一个父文档及其子文档，保持原有字典访问方式。"""

    parent: Document
    children: list[Document]
