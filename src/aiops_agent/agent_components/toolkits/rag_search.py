"""将知识库检索器封装为 LangChain 工具。"""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import StructuredTool


def _format_documents(docs: list[Document]) -> str:
    if not docs:
        return "未检索到相关文档"
    return "\n\n".join(
        f"[score={doc.metadata.get('score', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def make_rag_search_tool(retriever: BaseRetriever) -> StructuredTool:
    """绑定检索器，仅向模型暴露 query，支持 invoke 和 ainvoke。"""
    def rag_search_tool(query: str) -> str:
        return _format_documents(retriever.invoke(query))

    async def arag_search_tool(query: str) -> str:
        return _format_documents(await retriever.ainvoke(query))

    return StructuredTool.from_function(
        func=rag_search_tool,
        coroutine=arag_search_tool,
        name="rag_search_tool",
        description="检索内部知识库中的相关文档。query 为检索查询文本，返回文档片段及相关性分数。",
    )
