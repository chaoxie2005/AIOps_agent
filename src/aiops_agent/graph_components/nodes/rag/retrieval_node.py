"""RAG 子图的检索与聚合节点。"""

from langchain_core.documents import Document
from loguru import logger

from aiops_agent.graph_components.states.state import RagState, RetrievalHit
from aiops_agent.rag_components.retriever.retriever import MilvusRetriever


async def retriever_node(state: RagState, retriever: MilvusRetriever) -> dict:
    """检索当前子问题，并把召回的文档按子问题记录下来。

    正常流程中，schedule_node 会把要处理的子问题放进 current_subquery，
    检索词用它里面的 effective_query（有依赖时被 SDOM 改写过的版本）。
    如果状态里没有 current_subquery，就退回用 state["query"]，保持老的
    手工测试和单节点调用方式仍然可用。

    Args:
        state: RAG 子图当前状态。
        retriever: 实际执行检索的检索器（混合召回 + 重排）。

    Returns:
        状态更新字典，retrieval_hits 里每一项记录是哪个子问题召回的文档。
    """
    current = state.get("current_subquery")
    # 有调度上下文就用改写后的查询；没有就兼容旧用法，直接用原始查询。
    query = current["effective_query"] if current else state["query"]
    docs = await retriever.ainvoke(query)
    logger.info(f"检索工具检索的子问题：{query}")
    for doc in docs:
        logger.info(f"检索工具检索的文档：{doc}")
    return {"retrieval_hits": [RetrievalHit(subquery=query, document=doc) for doc in docs]}


async def aggregation_node(state: RagState) -> dict:
    """按父块去重，旧记录缺少父块内容时回退到子块正文。

    遍历所有检索记录：有 parent_id 时按父文档去重；没有时按子块正文去重。
    最终只保留每个父文档一份正文。

    Args:
        state: RAG 子图当前状态，包含 retrieval_hits。

    Returns:
        状态更新字典，unique_documents 是去重后的文档正文列表。
    """
    selected: dict[tuple[str, str], tuple[str, bool]] = {}
    for hit in state.get("retrieval_hits", []):
        doc: Document = hit["document"]
        parent_id = doc.metadata.get("parent_id")
        parent_content = doc.metadata.get("parent_content")
        has_parent_content = isinstance(parent_content, str) and bool(parent_content.strip())
        key = (
            ("parent", parent_id)
            if isinstance(parent_id, str) and parent_id
            else ("text", doc.page_content)
        )
        previous = selected.get(key)
        if previous is None or (has_parent_content and not previous[1]):
            selected[key] = (
                parent_content if has_parent_content else doc.page_content,
                has_parent_content,
            )
    return {"unique_documents": [content for content, _ in selected.values()]}
