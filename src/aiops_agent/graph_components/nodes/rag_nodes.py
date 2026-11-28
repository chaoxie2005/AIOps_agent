"""RAG 子图的三个节点：子查询拆解、检索、聚合并按父文档去重。

节点是普通 async 函数，签名保持 (state, 依赖) 或 (state)，便于单测；
RagNodes 只负责持有依赖并在图里转发，不承载业务逻辑。
"""
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain_core.documents import Document
from loguru import logger

from aiops_agent.agent_components.llm import ModelClient
from aiops_agent.graph_components.states.state import RagInput, RagState, RetrievalHit, SubQueryItem
from aiops_agent.output_validation import SubqueryOutput, ainvoke_structured
from aiops_agent.prompts.load_texts.load_subquery_prompt import subquery_agent_prompt
from aiops_agent.rag_components.retriever.factory import build_retriever_from_env
from aiops_agent.rag_components.retriever.retriever import MilvusRetriever

analysis_logger = logging.getLogger(__name__)

# 子问题占位符正则，例如 #q1#。
_PLACEHOLDER_RE = re.compile(r"#(q\d+)#")


def _default_subquery(question: str) -> SubQueryItem:
    """构造一个无依赖的兜底子问题，子问题拆解节点失败时调用。

    Args:
        question: 用户的原始问题。

    Returns:
        一个 ID 为 ``q1``、无依赖且状态为 pending 的子问题项。
    """
    return {
        "id": "q1",
        "question": question,
        "depends_on": [],
        "purpose": "原始问题",
        "status": "pending",
        "effective_query": question,
    }


def _strip_placeholders(text: str) -> str:
    """删除问题文本中的整个 ``#qid#`` 指代标记并规整文本。

    降级为并行检索时无法解析指代指向的答案，因此直接删除整个 ``#qid#``
    标记（如 ``#q1#``），再压缩空白、清理标点前多余空格和句首悬空助词。

    Args:
        text: 可能包含 ``#qid#`` 指代标记的问题文本。

    Returns:
        指代标记已整体删除、空白与标点已规整的问题文本。
    """
    cleaned = _PLACEHOLDER_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([?!.,;:？！。，；：])", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[的之]+\s*", "", cleaned)
    return cleaned.strip()


def _to_parallel_items(parsed: SubqueryOutput, limit: int) -> list[SubQueryItem]:
    """把拆解结果降级为无依赖、可并行检索的子问题列表。

    保留每个子问题的语义文本，清空 ``depends_on``，并把问题中的整个
    ``#qid#`` 指代标记删除。模型完全失败（无 parsed 结果）时调用方
    改用 ``_default_subquery`` 兜底。

    Args:
        parsed: 模型输出的结构化拆解结果。
        limit: 保留的子问题数量上限，必须大于等于 1。

    Returns:
        无依赖、占位符已清洗的子问题项列表。

    Raises:
        ValueError: ``limit`` 小于 1，或清洗后的某个问题为空。
    """
    if limit < 1:
        raise ValueError("max_subqueries 必须大于 0")
    items: list[SubQueryItem] = [] # 无依赖的子问题项列表，当子问题依赖关系优化节点失败时使用拆解出来的子问题进行并行检索
    for item in parsed.sub_questions[:limit]:
        cleaned = _strip_placeholders(item.question)
        if not cleaned:
            raise ValueError(f"{item.id} 清洗占位符后问题为空")
        items.append({
            "id": item.id,
            "question": cleaned,
            "depends_on": [],
            "purpose": item.purpose,
            "status": "pending",
            "effective_query": cleaned,
        })
    return items


def _to_subquery_items(parsed: SubqueryOutput) -> list[SubQueryItem]:
    """把校验通过的模型输出转换为图状态中的子问题项。

    同时检查每个问题中的整个 ``#qid#`` 指代标记是否与 ``depends_on`` 完全一致。

    Args:
        parsed: 已通过校验的结构化拆解结果。

    Returns:
        携带依赖信息、状态为 pending 的子问题项列表。

    Raises:
        ValueError: 问题中的占位符与依赖关系不一致。
    """
    items: list[SubQueryItem] = []
    for item in parsed.sub_questions:
        placeholders = set(_PLACEHOLDER_RE.findall(item.question))
        declared = set(item.depends_on) # 问题中声明的依赖关系，使用集合去重
        if placeholders != declared:
            raise ValueError(
                f"{item.id} 的占位符 {sorted(placeholders)} 与依赖 {sorted(declared)} 不一致"
            )
        items.append({
            "id": item.id,
            "question": item.question,
            "depends_on": list(dict.fromkeys(item.depends_on)),
            "purpose": item.purpose,
            "status": "pending",
            "effective_query": item.question,
        })
    return items


def _truncate_with_dependencies(items: list[SubQueryItem], limit: int) -> list[SubQueryItem]:
    """当模型拆解出的子问题超过 max_subqueries 上限时，把列表砍到上限数量，并顺手清理因此产生的悬空依赖

    保留原有 ID 顺序；任何指向已被截断子问题的依赖都会被删除，以保证剩余依赖图仍然有效。

    Args:
        items: 原始顺序的子问题项列表。
        limit: 保留的子问题数量上限，必须大于等于 1。

    Returns:
        清理依赖后保留的子问题项列表。

    Raises:
        ValueError: ``limit`` 小于 1。
    """

    if limit < 1:
        raise ValueError("max_subqueries 必须大于 0")
    kept = items[:limit]
    kept_ids = {item["id"] for item in kept}
    for item in kept:
        item["depends_on"] = [dep for dep in item["depends_on"] if dep in kept_ids]
        item["effective_query"] = item["question"]
    return kept


async def subquery_node(state: RagInput, model_client: ModelClient) -> dict:
    """拆解用户问题，保留可执行的依赖关系并写入 RAG 子图状态。

    依赖校验失败时降级为“保留子问题、清空依赖、清洗占位符”的并行检索模式；
    模型完全失败时降级为单个原始问题。

    Args:
        state: RAG 输入，包含原始问题和可选的 ``max_subqueries`` 上限。
        model_client: 用于生成问题拆解的模型客户端。

    Returns:
        包含结构化 ``sub_queries`` 和兼容旧字段 ``unique_subqueries`` 的状态更新。
    """
    question = state["query"]
    max_subqueries = state.get("max_subqueries", 4)
    messages = subquery_agent_prompt.format_messages(
        max_sub_questions=max_subqueries,
        question=question,
    )
    parsed, error = await ainvoke_structured(
        model_client, messages, SubqueryOutput,
        correction="只输出符合 SubqueryOutput schema 的 JSON 对象，不要输出解释。",
    )

    if error is not None or parsed is None:
        # 模型完全不可用：没有子问题可保留，只能用原始问题兜底。
        analysis_logger.warning("子问题拆解失败，回退到原始问题: %s", error)
        items = [_default_subquery(question)]
    else:
        try:
            items = _to_subquery_items(parsed)
            if len(items) > max_subqueries:
                analysis_logger.warning(
                    "子查询数量 %d 超过上限，截断为 %d", len(items), max_subqueries,
                )
                items = _truncate_with_dependencies(items, max_subqueries)
        except ValueError as exc:
            # 依赖图非法或占位符不一致：保留子问题语义，清空依赖后并行检索。
            analysis_logger.warning("子问题依赖校验失败，降级为并行检索: %s", exc)
            try:
                items = _to_parallel_items(parsed, max_subqueries)
            except ValueError as parallel_exc:
                analysis_logger.warning("并行检索降级也失败，回退到原始问题: %s", parallel_exc)
                items = [_default_subquery(question)]

    analysis_logger.info("子问题拆解结果: %s", [item["question"] for item in items])
    return {
        "sub_queries": items,
        "unique_subqueries": [item["question"] for item in items],
    }


def schedule_node(state: RagState) -> dict:
    """挑出下一轮要处理的子问题，并告诉图下一步该走哪个节点。

    遍历 sub_queries，找到第一个还没开始
    （status=pending）且依赖都已完成的子问题，把它标记为 ready，
    放进 current_subquery 交给下一个节点处理。

    - 有依赖的子问题 → 下一步走 optimize（SDOM 改写节点）
    - 没依赖的子问题 → 下一步走 retrieve（直接检索）
    - 没有可处理的子问题 → 下一步走 aggregate（收尾聚合）

    如果某个子问题等的上游（当前子问题的 depends_on 列表里列出的那些子问题）已经失败并被跳过，它永远等不到依赖，
    这里会直接把它标记为 skipped，避免整个流程卡在这里。

    Args:
        state: RAG 子图当前状态，至少包含 sub_queries 和 completed_ids。

    Returns:
        状态更新字典，包含新的 sub_queries、current_subquery、next_action 和 completed_ids。
    """

    # 复制一份再改，避免直接修改 LangGraph 传进来的状态对象。
    items: list[SubQueryItem] = [dict(item) for item in state.get("sub_queries", [])]
    # 用集合查依赖是否完成，比每次遍历列表快，也更直观。
    completed: set[str] = set(state.get("completed_ids", []))
    skipped_now: list[str] = [] # 这轮被跳过的子问题 ID

    # 先找出被跳过的子问题 ID：它们虽然在 completed_ids 里，但没有检索结果，
    # 依赖它们的子问题不能当“依赖已满足”来处理。
    skipped_ids = {other["id"] for other in items if other["status"] == "skipped"}

    for item in items:
        if item["status"] != "pending":
            continue

        # 只要依赖里有“未完成”或“已跳过”的，就还不能执行。
        waiting_for = [
            dep for dep in item["depends_on"]
            if dep not in completed or dep in skipped_ids
        ]
        if waiting_for:
            # 如果等的上游里有已经跳过的，这个子问题也永远等不到结果，直接跳过它。
            if any(dep in skipped_ids for dep in waiting_for):
                # 按 state.py 的约定：跳过时清空依赖和查询，避免留下无效引用。
                item["status"] = "skipped"
                item["depends_on"] = []
                item["effective_query"] = ""
                skipped_now.append(item["id"])
            continue

        # 找到一个可以执行的子问题：标记 ready 并交给下一节点。
        item["status"] = "ready"
        action = "optimize" if item["depends_on"] else "retrieve"
        return {
            "sub_queries": items,
            "current_subquery": dict(item),
            "next_action": action,
            "completed_ids": [*state.get("completed_ids", []), *skipped_now],
        }

    # 没有可执行的了：可能是全部完成，也可能是剩下的都被跳过了，统一去聚合。
    return {
        "sub_queries": items,
        "current_subquery": None,
        "next_action": "aggregate",
        "completed_ids": [*state.get("completed_ids", []), *skipped_now],
    }


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


def record_node(state: RagState) -> dict:
    """把当前子问题标记为已完成，让调度器可以继续处理下一个。

    检索节点跑完后调用它：把 current_subquery 的状态从 ready 改成 done，
    并把它的 ID 加入 completed_ids。这样依赖它的后续子问题在下一次
    schedule 时才会被认为“依赖已就绪”。

    Args:
        state: RAG 子图当前状态，至少包含 sub_queries 和 current_subquery。

    Returns:
        状态更新字典，包含更新后的 sub_queries、completed_ids 和 effective_queries。

    Raises:
        ValueError: current_subquery 缺失，说明有节点跳过了调度直接调用本节点。
    """
    current = state.get("current_subquery")
    if not current:
        raise ValueError("record_node 需要 current_subquery，请先经过 schedule_node")

    current_id = current["id"]
    items: list[SubQueryItem] = [dict(item) for item in state.get("sub_queries", [])] # 复制 sub_queries 到 items
    for item in items:
        if item["id"] == current_id: # 找到当前子问题
            item["status"] = "done"
            break

    completed = list(state.get("completed_ids", []))
    if current_id not in completed:
        completed.append(current_id)

    # 记录实际用的检索词，方便后续核对 SDOM 改写是否生效。
    effective_queries = dict(state.get("effective_queries", {}))
    effective_queries[current_id] = current["effective_query"] # 记录当前子问题

    return {
        "sub_queries": items,
        "completed_ids": completed,
        "effective_queries": effective_queries,
    }


async def aggregation_node(state: RagState) -> dict:
    """按父块去重，旧记录缺少父块内容时回退到子块正文。"""
    selected: dict[tuple[str, str], tuple[str, bool]] = {}
    for hit in state.get("retrieval_hits", []):
        doc = hit["document"]
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


class RagNodes:
    """RAG 子图节点依赖容器：持有 model_client 与 retriever，整个子图只建一次。"""

    def __init__(self, model_client: ModelClient, retriever: MilvusRetriever) -> None:
        self.model_client = model_client
        self.retriever = retriever

    @classmethod
    def from_env(cls) -> "RagNodes":
        """按 .env 配置构造模型客户端与检索器。"""
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
        address = urlsplit(base_url)
        if address.scheme not in ("http", "https") or not address.hostname:
            raise ValueError("LLM_BASE_URL 必须是包含 http 或 https 的模型服务地址")
        # 仅让模型主机绕过代理，保留已有的代理绕过名单。
        for key in ("NO_PROXY", "no_proxy"):
            hosts = [host.strip() for host in os.getenv(key, "").split(",") if host.strip()]
            if address.hostname not in hosts:
                hosts.append(address.hostname)
            os.environ[key] = ",".join(hosts)
        model_client = ModelClient()
        retriever = build_retriever_from_env()
        return cls(model_client=model_client, retriever=retriever)

    async def subquery(self, state: RagInput) -> dict:
        """子查询拆解节点。"""
        return await subquery_node(state, self.model_client)

    def schedule(self, state: RagState) -> dict:
        """调度节点包装：挑选下一个要处理的子问题。"""
        return schedule_node(state)

    async def retrieve(self, state: RagState) -> dict:
        """检索节点包装：检索当前子问题。"""
        return await retriever_node(state, self.retriever)

    def record(self, state: RagState) -> dict:
        """记录节点包装：把当前子问题标记为完成。"""
        return record_node(state)

    async def aggregate(self, state: RagState) -> dict:
        """聚合节点包装：按父文档去重。"""
        return await aggregation_node(state)


__all__ = [
    "RagNodes", "aggregation_node", "record_node",
    "retriever_node", "schedule_node", "subquery_node",
]
