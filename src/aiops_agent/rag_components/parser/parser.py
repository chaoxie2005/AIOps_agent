from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

_LOADERS: dict[str, Callable[[str], BaseLoader]] = {
    ".txt": partial(TextLoader, encoding="utf-8"),
    ".md": partial(TextLoader, encoding="utf-8"),
    ".pdf": PyPDFLoader,
}


class Parser:
    """
    文件解析器
    """

    def __init__(self, file_path: str):
        """
        初始化解析器

        Args:
            file_path: 文件路径

        Returns:
            None
        """
        self.file_path = file_path

    def get_file_extension(self) -> str:
        """
        获取文件扩展名

        Returns:
            str: 文件扩展名（小写），如 ".pdf"
        """
        return Path(self.file_path).suffix.lower()

    def get_loader(self) -> BaseLoader:
        """
        获取文件解析器

        Returns:
            BaseLoader: 文件解析器
        """
        ext = self.get_file_extension()
        loader_factory = _LOADERS.get(ext)
        if loader_factory is None:
            raise ValueError(f"不支持的文件扩展名: {ext}，支持: {', '.join(_LOADERS)}")
        return loader_factory(self.file_path)

    def parse(self) -> list[Document]:
        """
        解析文件

        Returns:
            list[Document]: 文档列表
        """
        return self.get_loader().load()

    def parse_lazy(self) -> Iterator[Document]:
        """
        懒解析文件

        Returns:
            Iterator[Document]: 文档迭代器
        """
        return self.get_loader().lazy_load()


# TODO: 实现Markdown 文件解析器 使用MinerU进行解析
class MarkdownParser(Parser):
    """
    Markdown 文件解析器
    """
    